"""Collect once -> Test A/B -> grouped OOF -> deployable/oracle/value Test C.

Run --help. The launcher pins the source commit. Result ZIPs contain no LLM
weights; --mode train reuses a result corpus without loading either large model.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
import traceback
import zipfile

import numpy as np
import torch

from agreement_profile_data import (SCHEMA, FEATURES, Normalizer, atomic_json,
    copy_corpus, group_folds, load_corpus, pack_features, pack_targets, prefix_length)
from agreement_profile_models import ProfileProbe, loss_for
from agreement_profile_metrics import paired_comparison, prediction_row, score_rows

ARMS = {"P1": ["metadata", "distribution"],
        "P2": ["metadata", "distribution", "content"],
        "P3": ["metadata", "distribution", "hidden"],
        "P4": list(FEATURES)}


def package(output, summary, check=False):
    output = Path(output)
    atomic_json(output / "summary.json", summary)
    destination = output.with_suffix(".zip")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, path.relative_to(output),
                              compress_type=zipfile.ZIP_STORED if path.suffix in (".npz", ".pt")
                              else zipfile.ZIP_DEFLATED)
    if check:
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"Corrupt result member {bad}")
    temporary.replace(destination)
    print(f"[ZIP] {destination} status={summary['status']} stage={summary['stage']}", flush=True)


def predict(model, pairs, normalizer, device, profiles=None, batch_size=16, task="margin"):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            section = pairs[start:start + batch_size]
            sources = [p[0] for p in section]
            batch = pack_features(sources, normalizer, device, profiles)
            out = model(batch)
            for i, (source, target) in enumerate(section):
                margin = out["margin"][i, :source.length].cpu().numpy()
                K = int(out["K_logits"][i].argmax().item()) if task in ("K", "K_profile") else None
                row = prediction_row(source, target, margin, K,
                                     parent=source if source.uid != target.uid else None)
                row["margin_supervised"] = task != "K"
                if profiles is not None:
                    row["source_K_pred"] = prefix_length(np.asarray(profiles[source.uid]) >= 0)
                result.append(row)
    return result


def train_probe(name, groups, train_pairs, validation_pairs, normalizer, args,
                seed, folder, profiles=None, residual=False, task="margin", save=True):
    if not train_pairs or not validation_pairs:
        raise ValueError(f"{name} needs nonempty training/validation samples")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sizes = {g: train_pairs[0][0].features[g].shape[1] for g in FEATURES}
    model = ProfileProbe(sizes, groups, args.width, args.layers, args.max_proposal_tokens,
                         profiles is not None, residual, normalizer.margin_scale, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    grouped = defaultdict(list)
    for pair in train_pairs:
        grouped[pair[0].question].append(pair)
    prompts = sorted(grouped)
    rng = random.Random(seed)
    best, best_value, patience, logs = None, float("inf"), 0, []
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for _ in range(max(1, math.ceil(len(train_pairs) / args.batch_size))):
            # Uniform prompt -> uniform sample inside prompt, preventing long
            # trajectories from supplying most optimization minibatches.
            pairs = [rng.choice(grouped[rng.choice(prompts)]) for _ in range(args.batch_size)]
            sources, targets = [p[0] for p in pairs], [p[1] for p in pairs]
            batch = pack_features(sources, normalizer, args.device, profiles)
            labels = pack_targets(targets, batch["valid"].shape[1], args.device)
            current = pack_targets(sources, batch["valid"].shape[1], args.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = loss_for(out, labels, batch["valid"], normalizer.margin_scale,
                            task, current, args.sign_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss in {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = defaultdict(list)
        with torch.inference_mode():
            for start in range(0, len(validation_pairs), args.batch_size):
                pairs = validation_pairs[start:start + args.batch_size]
                sources, targets = [p[0] for p in pairs], [p[1] for p in pairs]
                batch = pack_features(sources, normalizer, args.device, profiles)
                labels = pack_targets(targets, batch["valid"].shape[1], args.device)
                current = pack_targets(sources, batch["valid"].shape[1], args.device)
                out = model(batch)
                # Select checkpoints using prompt-macro validation loss.
                for i, source in enumerate(sources):
                    selected = {k: v[i:i + 1] for k, v in out.items()}
                    label = {k: v[i:i + 1] for k, v in labels.items()}
                    old = {k: v[i:i + 1] for k, v in current.items()}
                    selection_task = "K" if task == "K_profile" else task
                    value = loss_for(selected, label, batch["valid"][i:i + 1],
                                     normalizer.margin_scale, selection_task, old, args.sign_weight)
                    val_losses[source.question].append(float(value))
        value = float(np.mean([np.mean(v) for v in val_losses.values()]))
        logs.append(dict(epoch=epoch + 1, train_loss=float(np.mean(train_losses)),
                         prompt_macro_validation_loss=value))
        print(f"[train] {name} seed={seed} epoch={epoch+1} val={value:.5f}", flush=True)
        if value < best_value - 1e-6:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_value, patience = value, 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    model.load_state_dict(best)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if save:
        checkpoint = dict(model_config=model.config, state_dict=best,
            normalizer=normalizer.state_dict(), name=name, seed=seed, task=task,
            train_prompts=prompts, validation_prompts=sorted({p[0].question for p in validation_pairs}),
            profile_inputs=("explicit_source_profile" if profiles is not None else "none"))
        torch.save(checkpoint, folder / f"{name}_seed{seed}.pt")
    atomic_json(folder / f"{name}_seed{seed}_training.json", dict(logs=logs,
        parameters=sum(p.numel() for p in model.parameters()),
        wall_seconds=time.perf_counter() - began, task=task,
        selection_objective="K_cross_entropy" if task in ("K", "K_profile") else "margin_loss",
        train_prompts=prompts, validation_prompts=sorted({p[0].question for p in validation_pairs})))
    return model


def write_predictions(path, arm, seed, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(arm=arm, seed=seed, **row), allow_nan=False) + "\n")


def save_reports(output, reports):
    atomic_json(Path(output) / "metrics.json", reports)
    with (Path(output) / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["test", "arm_split_seed", "states", "prompts",
            "macro_margin_mae", "macro_profile_K_mae", "macro_direct_K_mae", "status"])
        writer.writeheader()
        for test in ("A_B", "C"):
            for name, value in reports[test].items():
                writer.writerow(dict(test=test, arm_split_seed=name, states=value.get("states"),
                    prompts=value.get("prompts"), macro_margin_mae=value.get("macro", {}).get("margin_mae"),
                    macro_profile_K_mae=value.get("macro", {}).get("K_mae"),
                    macro_direct_K_mae=value.get("macro_K_head_mae"), status=value.get("status", "measured")))


def corpus_audit(records, edges, manifest):
    result = dict(states=len(records), R_edges=len(edges), splits={}, ties=0,
                  profile_target="raw Qwen candidate-vs-best-rival margin",
                  R_groups=dict(changed=0, gain=0, loss=0, same=0,
                                blocked_before_active_segment=0, native_tokens_unchanged=0))
    for split in ("train", "validation", "test"):
        values = [r for r in records.values() if r.split == split]
        pair_values = [e for e in edges if records[e["source"]].split == split]
        result["splits"][split] = dict(states=len(values), prompts=len({r.question for r in values}),
                                       R_edges=len(pair_values), R_prompts=len({records[e["source"]].question for e in pair_values}),
                                       K_histogram={str(k): sum(r.accepted == k for r in values)
                                                    for k in sorted({r.accepted for r in values})},
                                       K_zero_fraction=float(np.mean([r.accepted == 0 for r in values])) if values else None,
                                       fully_unmasked_states=sum(not np.any(r.native == 151665) for r in values))
        if not values:
            raise ValueError(f"No observed states in {split}; increase capture coverage")
    for record in records.values():
        result["ties"] += int(np.sum(record.margin == 0))
        if manifest["split"].get(record.question) != record.split:
            raise ValueError("Record disagrees with pinned prompt split")
    for edge in edges:
        a, b = records[edge["source"]], records[edge["child"]]
        result["R_groups"]["changed"] += int(np.any(a.candidate != b.candidate))
        result["R_groups"]["blocked_before_active_segment"] += int(a.accepted < a.meta.get("segment_start", 0))
        result["R_groups"]["native_tokens_unchanged"] += int(np.array_equal(a.native, b.native))
        result["R_groups"]["gain" if b.accepted > a.accepted else
                            "loss" if b.accepted < a.accepted else "same"] += 1
    return result


def baseline_rows(pairs, profiles, delta=None):
    rows = []
    for source, target in pairs:
        value = np.asarray(profiles[source.uid]).copy()
        if delta is not None:
            value += delta[:source.length]
        row = prediction_row(source, target, value,
                             parent=source if source.uid != target.uid else None)
        row["source_K_pred"] = prefix_length(np.asarray(profiles[source.uid]) >= 0)
        rows.append(row)
    return rows


def runtime_probe(model, records, norm, args):
    times, packing = [], []
    synchronize = (lambda: torch.cuda.synchronize(torch.device(args.device))) if str(args.device).startswith("cuda") else lambda: None
    model.eval()
    for record in records[:min(8, len(records))]:
        with torch.inference_mode():
            for _ in range(2):
                model(pack_features([record], norm, args.device))
            for _ in range(5):
                synchronize()
                start = time.perf_counter()
                batch = pack_features([record], norm, args.device)
                synchronize()
                packed = time.perf_counter()
                model(batch)
                synchronize()
                end = time.perf_counter()
                packing.append((packed - start) * 1000)
                times.append((end - packed) * 1000)
    return dict(batch=1, samples=len(times), cached_feature_pack_ms_p50_p95=np.quantile(packing, [.5, .95]).tolist(),
                probe_forward_ms_p50_p95=np.quantile(times, [.5, .95]).tolist(),
                excludes_native_capture_hook=True, excludes_drafter_replay_and_Qwen=True,
                production_end_to_end_speedup_established=False)


def run_tests(records, edges, manifest, args, output, checkpoint):
    output = Path(output)
    current = {s: [(r, r) for r in records.values() if r.split == s]
               for s in ("train", "validation", "test")}
    future = {s: [(records[e["source"]], records[e["child"]]) for e in edges
                   if records[e["source"]].split == s] for s in current}
    norm = Normalizer([p[0] for p in current["train"]])
    reports = {"A_B": {}, "C": {}, "comparisons": {}, "runtime": {}}
    fold_ledger = []
    model_folder = output / "models"
    prediction_path = output / "predictions.jsonl"
    # A train-only position prior; no validation/test labels fitted.
    max_length = args.max_proposal_tokens
    total, count = np.zeros(max_length), np.zeros(max_length)
    for source, _ in current["train"]:
        total[:source.length] += source.margin
        count[:source.length] += 1
    prior = total / count.clip(1)
    for split in ("validation", "test"):
        mapping = {r.uid: prior[:r.length] for r, _ in current[split]}
        rows = baseline_rows(current[split], mapping)
        reports["A_B"][f"P0_{split}"] = score_rows(rows)
        write_predictions(prediction_path, "A_P0", -1, rows)
        length_labels = defaultdict(list)
        for r, _ in current["train"]:
            length_labels[r.length].append(r.accepted)
        length_rows = []
        for source, target in current[split]:
            nearest = min(length_labels, key=lambda n: abs(n - source.length))
            K = int(round(np.median(length_labels[nearest]) * source.length / nearest))
            row = prediction_row(source, target, np.zeros(source.length), min(source.length, K))
            row["margin_supervised"] = False
            length_rows.append(row)
        reports["A_B"][f"length_K_{split}"] = score_rows(length_rows)
        write_predictions(prediction_path, "B_length_K", -1, length_rows)
    for seed in args.training_seeds:
        current_models, profiles, all_rows = {}, {}, {}
        for arm, groups in ARMS.items():
            model = train_probe("A_" + arm, groups, current["train"], current["validation"],
                                norm, args, seed, model_folder)
            for split in ("validation", "test"):
                rows = predict(model, current[split], norm, args.device, batch_size=args.batch_size)
                all_rows[arm + "_" + split] = rows
                reports["A_B"][f"{arm}_{split}_seed{seed}"] = score_rows(rows)
                write_predictions(prediction_path, "A_" + arm, seed, rows)
                if arm == "P4":
                    profiles.update({r["state_id"]: np.asarray(r["margin_pred"], np.float32) for r in rows})
            if arm == "P4":
                current_models[arm] = model
                reports["runtime"][f"P4_seed{seed}"] = runtime_probe(model,
                    [p[0] for p in current["test"]], norm, args)
            else:
                del model
        for metric in ("margin_mae", "K_mae"):
            reports["comparisons"][f"A_P4_vs_P1_{metric}_seed{seed}"] = paired_comparison(
                all_rows["P4_test"], all_rows["P1_test"], metric, args.bootstrap_draws, seed)
        current_K = train_probe("B_Konly", ARMS["P4"], current["train"], current["validation"],
                                norm, args, seed, model_folder, task="K")
        for split in ("validation", "test"):
            rows = predict(current_K, current[split], norm, args.device,
                           batch_size=args.batch_size, task="K")
            reports["A_B"][f"Konly_{split}_seed{seed}"] = score_rows(rows)
            write_predictions(prediction_path, "B_Konly", seed, rows)
            if split == "test":
                reports["comparisons"][f"B_profile_vs_direct_K_seed{seed}"] = paired_comparison(
                    all_rows["P4_test"], rows, "K_mae", args.bootstrap_draws, seed,
                    right_metric="K_head_mae")
        del current_K
        save_reports(output, reports)
        checkpoint("A_B")
        if any(not future[s] for s in future):
            reports["C"][f"seed{seed}"] = dict(status="insufficient_R_edges",
                counts={s: len(pairs) for s, pairs in future.items()})
            del current_models
            continue
        # OOF P4 predictions for every training state. No fold's held-out prompt
        # participates in its encoder training or train-only normalization.
        train_records = [p[0] for p in current["train"]]
        for fold_index, held in enumerate(group_folds(train_records, args.oof_folds, seed)):
            selected = [p for p in current["train"] if p[0].question not in held]
            held_pairs = [p for p in current["train"] if p[0].question in held]
            fold_norm = Normalizer([p[0] for p in selected])
            model = train_probe(f"OOF_P4_fold{fold_index}", ARMS["P4"], selected,
                current["validation"], fold_norm, args, seed, model_folder, save=False)
            rows = predict(model, held_pairs, fold_norm, args.device, batch_size=args.batch_size)
            profiles.update({r["state_id"]: np.asarray(r["margin_pred"], np.float32) for r in rows})
            fold_ledger.append(dict(seed=seed, fold=fold_index,
                trained_prompts=sorted({p[0].question for p in selected}),
                predicted_prompts=sorted(held), checkpoint_selection="external_validation_only"))
            del model
        atomic_json(output / "oof_provenance.json", fold_ledger)
        oracle = {r.uid: r.margin for r in records.values()}
        # Global per-position training delta. Positions not observed in train
        # get zero delta, rather than a prior fitted on held-out data.
        total, count = np.zeros(max_length), np.zeros(max_length)
        for source, target in future["train"]:
            total[:source.length] += target.margin - source.margin
            count[:source.length] += 1
        delta = total / count.clip(1)
        c_rows = {}
        for split in ("validation", "test"):
            for name, mapping, change in (("persistence", profiles, None),
                    ("mean_transition", profiles, delta), ("oracle_persistence", oracle, None)):
                rows = baseline_rows(future[split], mapping, change)
                c_rows[name + "_" + split] = rows
                reports["C"][f"{name}_{split}_seed{seed}"] = score_rows(rows)
                write_predictions(prediction_path, "C_" + name, seed, rows)
        specifications = [
            ("profile", ["metadata"], profiles, False, "margin"),
            ("direct", ARMS["P4"], None, False, "margin"),
            ("delta", ARMS["P4"], profiles, True, "margin"),
            ("oracle_profile", ["metadata"], oracle, False, "margin"),
            ("oracle_rich", ARMS["P4"], oracle, False, "margin"),
            ("Konly", ARMS["P4"], None, False, "K"),
            ("K_profile", ARMS["P4"], None, False, "K_profile")]
        for name, groups, mapping, residual, task in specifications:
            model = train_probe("C_" + name, groups, future["train"], future["validation"],
                                norm, args, seed, model_folder, mapping, residual, task)
            for split in ("validation", "test"):
                rows = predict(model, future[split], norm, args.device, mapping,
                               args.batch_size, task)
                # Diagnostic source prediction is supplied for evaluation only
                # when the predictor itself has no profile input.
                for row in rows:
                    if "source_K_pred" not in row:
                        row["source_K_pred"] = prefix_length(profiles[row["state_id"]] >= 0)
                c_rows[name + "_" + split] = rows
                reports["C"][f"{name}_{split}_seed{seed}"] = score_rows(rows)
                write_predictions(prediction_path, "C_" + name, seed, rows)
            del model
        for left, right, metric in (("direct", "persistence", "margin_mae"),
                ("direct", "persistence", "affected_margin_mae"),
                ("delta", "mean_transition", "margin_mae"),
                ("direct", "profile", "margin_mae"),
                ("oracle_rich", "oracle_profile", "margin_mae"),
                ("K_profile", "Konly", "K_head_mae")):
            reports["comparisons"][f"C_{left}_vs_{right}_{metric}_seed{seed}"] = paired_comparison(
                c_rows[left + "_test"], c_rows[right + "_test"], metric, args.bootstrap_draws, seed)
        del current_models
        save_reports(output, reports)
        checkpoint("C")
    save_reports(output, reports)
    return reports


def run(args):
    output = args.output_dir
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    atomic_json(output / "config.json", config)
    summary = dict(schema=SCHEMA, status="running", stage="setup",
                   source_revision=args.source_revision, LLMs_frozen=True,
                   tests=["A_current_profile", "B_first_rejection", "C_one_step_R"],
                   generation_to_eos_required=False, latency_scope="readout_only_no_speedup_claim")
    summary["resume_semantics"] = "completed_capture_questions_reused_probes_retrain_offline"
    began = time.perf_counter()
    def checkpoint(stage):
        summary.update(stage=stage, elapsed_seconds=time.perf_counter() - began)
        package(output, summary)
    try:
        corpus = output / "corpus"
        if args.input_path:
            copy_corpus(args.input_path, corpus)
        else:
            corpus.mkdir()
        if args.mode in ("all", "collect"):
            from collect_agreement_profiles import collect_real
            manifest = collect_real(args, corpus, lambda: checkpoint("collection"))
        else:
            manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema"] != SCHEMA:
            raise ValueError("Input is not an agreement-profile corpus")
        records, edges, completed = load_corpus(corpus)
        summary["questions_captured"] = len(completed)
        summary["questions_expected"] = len(manifest["questions"])
        if len(completed) != len(manifest["questions"]):
            raise ValueError("Corpus incomplete; resume with --mode all/collect before training")
        audit = corpus_audit(records, edges, manifest)
        atomic_json(output / "audit.json", audit)
        summary["audit"] = audit
        if args.mode != "collect":
            if max(r.length for r in records.values()) > args.max_proposal_tokens:
                raise ValueError("Training max_proposal_tokens smaller than corpus")
            reports = run_tests(records, edges, manifest, args, output, checkpoint)
            summary["scientific_tests_complete"] = not any(
                value.get("status") == "insufficient_R_edges" for value in reports["C"].values())
            summary["stage"] = "A_B_C"
        else:
            summary["scientific_tests_complete"] = False
            summary["stage"] = "collection_only"
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "partial_error"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - began
        package(output, summary, check=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "collect", "train"], default="all")
    p.add_argument("--input_path", type=Path)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--dllm_dir", type=Path, default=Path("unused"))
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--num_questions", type=int, default=100)
    p.add_argument("--validation_questions", type=int, default=20)
    p.add_argument("--test_questions", type=int, default=20)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--max_rounds_per_question", type=int, default=2)
    p.add_argument("--segments_per_round", type=int, default=8)
    p.add_argument("--states_per_question", type=int, default=32)
    p.add_argument("--max_context_tokens", type=int, default=1024)
    p.add_argument("--max_proposal_tokens", type=int, default=64)
    p.add_argument("--extend_size", type=int, default=8)
    p.add_argument("--max_refinement_steps", type=int, default=3)
    p.add_argument("--physical_block_size", type=int, default=32)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--drafter_threshold", type=float, default=.5)
    p.add_argument("--hidden_layers", nargs="+", type=int, default=[7, 14, 28])
    p.add_argument("--raw_top_k", type=int, default=32)
    p.add_argument("--causal_atol", type=float, default=.01)
    p.add_argument("--parity_repeat", action="store_true")
    p.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=1)
    p.add_argument("--target_gpu_memory_gib", type=int, default=8)
    p.add_argument("--source_revision", default="local_unpinned")
    p.add_argument("--training_seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--oof_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--sign_weight", type=float, default=0.)
    p.add_argument("--bootstrap_draws", type=int, default=2000)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    positive = ("num_questions", "validation_questions", "test_questions", "max_rounds_per_question",
        "segments_per_round", "states_per_question", "max_context_tokens", "max_proposal_tokens",
        "extend_size", "max_refinement_steps", "physical_block_size", "small_block_size",
        "epochs", "patience", "batch_size", "width", "layers", "bootstrap_draws", "raw_top_k")
    if any(getattr(args, key) < 1 for key in positive):
        p.error("Counts/budgets must be positive")
    if args.num_questions <= args.validation_questions + args.test_questions + 1:
        p.error("Need >=2 train prompts plus validation and test")
    if not 0 <= args.drafter_threshold <= 1 or args.raw_top_k < 2 or args.oof_folds < 2:
        p.error("Invalid threshold/top-k/folds")
    if args.physical_block_size % args.small_block_size or args.width % 4:
        p.error("Block size must divide sub-block geometry; width divisible by four")
    if (not 0 <= args.dropout < 1 or args.learning_rate <= 0
            or args.sign_weight < 0 or args.causal_atol <= 0):
        p.error("Invalid dropout/learning-rate/sign-weight/tolerance")
    if args.mode == "train" and not args.input_path:
        p.error("--mode train requires --input_path")
    args.capture_verifier_teacher = True
    args.verifier_teacher_top_k = 2  # Margin + exact argmax only; no full distillation target.
    return args


if __name__ == "__main__":
    run(parse_args())
