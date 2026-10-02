"""Offline Phase-1 persistent WM + gated-FiLM Phase-2 test on a collected run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from persistent_world_model_v1 import (GatedFiLMAdapter, PersistentWorldModelV1,
    expected_yield, grouped_distribution_kl, hazard_mode, hazard_nll)
from world_model_core import Observation, pack_observations


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _select_states(root, max_nodes_per_question=80, max_edges_per_question=24):
    metadata = read_jsonl(root / "states.jsonl")
    edges = read_jsonl(root / "edges.jsonl")
    rounds = read_jsonl(root / "rounds.jsonl")
    teacher_rows = read_jsonl(root / "teacher_targets.jsonl")
    label_rows = read_jsonl(root / "labels.jsonl")
    if not metadata or not edges or not teacher_rows:
        raise RuntimeError("The run must contain native experiences, R/E edges and verifier teacher targets")
    teacher = {row["state_id"]: row for row in teacher_rows}
    labels = {row["state_id"]: row for row in label_rows}
    by_question = defaultdict(list)
    for row in metadata:
        by_question[str(row["question"])].append(row)
    edges_by_question = defaultdict(list)
    for edge in edges:
        edges_by_question[str(edge["question"])].append(edge)
    final_by_round = {(str(row["question"]), int(row["round_id"])): row["final_state"]
                      for row in rounds}
    selected_ids, selected_edges = set(), []
    for question, rows in by_question.items():
        rows.sort(key=lambda row: (int(row["round_id"]), row["state_id"]))
        roots = [row for row in rows if row.get("parent_state_id") is None]
        finals = {state_id for (qid, _), state_id in final_by_round.items() if qid == question}
        qedges = edges_by_question[question][:max_edges_per_question]
        critical = {row["state_id"] for row in roots} | finals
        for edge in qedges:
            critical.update((edge["parent"], edge["child"]))
        selected = set(critical)
        for row in rows:
            if len(selected) >= max_nodes_per_question:
                break
            label = labels.get(row["state_id"], {})
            if label.get("label_valid") and row["state_id"] not in selected:
                selected.add(row["state_id"])
        selected_ids.update(selected)
        selected_edges.extend(edge for edge in qedges
                              if edge["parent"] in selected and edge["child"] in selected)

    by_shard = defaultdict(list)
    for row in metadata:
        if row["state_id"] in selected_ids:
            by_shard[row["shard"]].append(row)
    nodes = {}
    for shard, rows in by_shard.items():
        with np.load(root / shard, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for meta in rows:
            row = int(meta["row"])
            lo, hi = map(int, arrays["offsets"][row:row+2])
            def tensor(name, dtype):
                return torch.tensor(arrays[name][lo:hi], dtype=dtype)
            label = labels.get(meta["state_id"], {})
            y = label.get("accepted_len")
            if y is None and "accepted" in arrays and int(arrays["accepted"][row]) >= 0:
                y = int(arrays["accepted"][row])
            obs = Observation(
                uid=meta["state_id"], question=str(meta["question"]), round_id=int(meta["round_id"]),
                ids=tensor("ids", torch.long), hidden=tensor("hidden", torch.float16),
                gaps=tensor("gaps", torch.float16), scalars=tensor("scalars", torch.float32),
                context=torch.tensor(arrays["context"][row], dtype=torch.float32),
                accepted=None if y is None or not label.get("label_valid", y is not None) else int(y),
                prefix_ids=torch.tensor(meta["prefix_token_ids"], dtype=torch.long),
                topk_ids=tensor("aligned_topk_token_ids", torch.long),
                history=tensor("history", torch.float32))
            if "teacher_margin" in arrays and bool(arrays.get(
                    "teacher_valid", np.zeros(hi-lo, dtype=np.bool_))[lo:hi].any()):
                obs.teacher_margin = tensor("teacher_margin", torch.float32)
                if "teacher_features" in arrays:
                    obs.teacher_features = tensor("teacher_features", torch.float32)
            if "teacher_aux_features" in arrays:
                obs.teacher_aux_features = tensor("teacher_aux_features", torch.float32)
            if "teacher_topk_token_ids" in arrays:
                valid = arrays.get("teacher_distribution_valid", np.zeros(hi-lo, dtype=np.bool_))[lo:hi]
                if bool(valid.any()):
                    obs.teacher_topk_ids = tensor("teacher_topk_token_ids", torch.long)
                    obs.teacher_topk_logits = tensor("teacher_topk_logits", torch.float16)
                    obs.teacher_logsumexp = tensor("teacher_logsumexp", torch.float32)
            source = teacher.get(obs.uid, {}).get("source")
            obs.teacher_is_actual = bool(arrays.get("teacher_actual", np.zeros(len(arrays["lengths"]),
                dtype=np.bool_))[row]) or source == "actual_STOP_forward"
            if "verifier_history_offsets" in arrays:
                hlo, hhi = map(int, arrays["verifier_history_offsets"][row:row+2])
                obs.verifier_history = torch.tensor(arrays["verifier_history"][hlo:hhi], dtype=torch.float32)
            nodes[obs.uid] = obs
    selected_edges = [(edge["parent"], edge["child"], edge["action"])
                      for edge in selected_edges if edge["parent"] in nodes and edge["child"] in nodes]
    split = json.loads((root / "question_split.json").read_text(encoding="utf-8"))
    questions = {str(q["question_id"]): q for q in read_jsonl(root / "questions.jsonl")}
    return nodes, selected_edges, split, questions, final_by_round, teacher


def _batch(observations, token_table, device):
    return pack_observations(observations, token_table, device)


def _encode_cache(model, nodes, token_table, device, batch_size=16, ablation=None):
    result = {}
    obs = list(nodes.values())
    model.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            group = obs[start:start+batch_size]
            batch = _batch(group, token_table, device)
            values = model.encode(batch,
                ablate_hidden=ablation == "no_drafter_hidden",
                ablate_verifier_hidden=ablation == "no_verifier_hidden",
                ablate_verifier_logits=ablation == "no_verifier_logits")
            for i, item in enumerate(group):
                result[item.uid] = {key: value[i].detach() for key, value in values.items()}
    return result


def _auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float64)
    positive, negative = labels == 1, labels == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
                 / (positive.sum() * negative.sum()))


def _metric_rows(records):
    if not records:
        return dict(n=0)
    truth = np.asarray([r["accepted"] for r in records], dtype=np.float64)
    pred = np.asarray([r["expected_yield"] for r in records], dtype=np.float64)
    error = pred - truth
    hazards_y, hazards_p = [], []
    maxlen = max(len(r["hazards"]) for r in records)
    survival_y, survival_p, survival_counts = [], [], []
    for k in range(maxlen):
        observed = [r for r in records if len(r["hazards"]) > k]
        if not observed:
            survival_y.append(None); survival_p.append(None); survival_counts.append(0)
            continue
        survival_y.append(float(np.mean([r["accepted"] > k for r in observed])))
        survival_p.append(float(np.mean([r["survival"][k] for r in observed])))
        survival_counts.append(len(observed))
    for row in records:
        y = int(row["accepted"])
        for i, prob in enumerate(row["hazards"]):
            if i <= y:
                hazards_y.append(int(i < y))
                hazards_p.append(float(prob))
    return dict(n=len(records), expected_yield_mae=float(np.mean(np.abs(error))),
        expected_yield_rmse=float(np.sqrt(np.mean(error**2))),
        signed_bias=float(error.mean()), absolute_error_p90=float(np.quantile(np.abs(error), .9)),
        exact_K_rate=float(np.mean([r["mode"] == r["accepted"] for r in records])),
        hazard_nll=float(np.mean([r["hazard_nll"] for r in records])),
        hazard_auc=_auc(hazards_y, hazards_p),
        mean_survival_brier=float(np.mean([
            (survival_p[k]-survival_y[k])**2 for k in range(maxlen)
            if survival_y[k] is not None])),
        survival=dict(position=[i+1 for i in range(maxlen)], predicted=survival_p,
                      observed=survival_y, count=survival_counts))


@torch.no_grad()
def evaluate_current(model, nodes, token_table, device, ablation=None, output_path=None):
    records = []
    model.eval()
    for obs in nodes.values():
        if obs.accepted is None:
            continue
        batch = _batch([obs], token_table, device)
        enc = model.encode(batch,
            ablate_hidden=ablation == "no_drafter_hidden",
            ablate_verifier_hidden=ablation == "no_verifier_hidden",
            ablate_verifier_logits=ablation == "no_verifier_logits")
        z = enc["unverified"] if ablation == "drafter_only" else enc["state"]
        logits = model.hazards(z, batch["lengths"], width=obs.length)[0, :obs.length]
        probs = torch.sigmoid(logits).cpu().tolist()
        surv = torch.cumprod(torch.tensor(probs), 0).tolist()
        logp = torch.logit(torch.tensor(probs).clamp(1e-6, 1-1e-6))[None].to(device)
        nll = float(hazard_nll(logp, batch["lengths"], batch["labels"]))
        row = dict(state_id=obs.uid, question=obs.question, split="validation",
            accepted=int(obs.accepted), proposal_length=obs.length,
            expected_yield=float(expected_yield(logp, batch["lengths"])[0]),
            mode=int(hazard_mode(logits)[0]),
            hazards=probs, survival=surv, hazard_nll=nll,
            source="grounded_posterior_after_verifier" if bool(enc["has_verifier"][0]) else "drafter_prior")
        records.append(row)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
    return _metric_rows(records), records


def _sample_paths(edges, max_horizon, batch_size, rng):
    outgoing = defaultdict(list)
    for parent, child, action in edges:
        outgoing[parent].append((child, action))
    starts = list(edges)
    if not starts:
        return []
    paths = []
    for _ in range(batch_size):
        parent, child, action = rng.choice(starts)
        path = [(parent, child, action)]
        node = child
        while len(path) < max_horizon and outgoing.get(node):
            nxt, act = rng.choice(outgoing[node])
            path.append((node, nxt, act))
            node = nxt
        paths.append(path)
    return paths


def evaluate_dynamics(model, nodes, edges, token_table, device, max_horizon=3,
                      ablation=None, output_path=None):
    model.eval()
    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[edge[0]].append(edge)
    paths = []
    rng = random.Random(20261002)
    for edge in edges:
        path = [edge]
        node = edge[1]
        while len(path) < max_horizon and outgoing.get(node):
            nxt = sorted(outgoing[node], key=lambda x: (x[2], x[1]))[0]
            path.append(nxt)
            node = nxt[1]
        paths.append(path)
    if len(paths) > 1200:
        paths = rng.sample(paths, 1200)
    results = {f"h{h}": [] for h in range(1, max_horizon+1)}
    prediction_rows = []
    with torch.no_grad():
        for path in paths:
            first = nodes.get(path[0][0])
            if first is None:
                continue
            b0 = _batch([first], token_table, device)
            enc0 = model.encode(b0,
                ablate_hidden=ablation == "no_drafter_hidden",
                ablate_verifier_hidden=ablation == "no_verifier_hidden",
                ablate_verifier_logits=ablation == "no_verifier_logits")
            if ablation == "drafter_only" or not bool(enc0["has_verifier"][0]):
                z = enc0["unverified"]
            else:
                z = enc0["posterior"] if first.teacher_is_actual else enc0["unverified"]
            previous_y, previous_len = first.accepted, first.length
            for depth, (parent_id, child_id, action) in enumerate(path, start=1):
                child = nodes.get(child_id)
                if child is None or child.accepted is None:
                    break
                cb = _batch([child], token_table, device)
                d_child = model.encode_d(cb, ablate_hidden=ablation == "no_drafter_hidden")
                if ablation == "no_dynamics":
                    z = model.correct(z, d_child)
                else:
                    z = model.transition(z, action, d_child)
                logits = model.hazards(z, cb["lengths"], width=child.length)
                pred = float(expected_yield(logits, cb["lengths"])[0])
                probs = torch.sigmoid(logits[0, :child.length]).cpu().tolist()
                surv = torch.cumprod(torch.tensor(probs), 0).tolist()
                logp = logits[:, :child.length]
                nll = float(hazard_nll(logp, cb["lengths"], cb["labels"]))
                persistence = (None if previous_y is None else
                    min(child.length, previous_y * child.length / max(1, previous_len)))
                row = dict(source=parent_id, state_id=child_id, action=action,
                           depth=depth, accepted=int(child.accepted),
                           expected_yield=pred, mode=int(hazard_mode(logits)[0]),
                           hazards=probs, survival=surv, hazard_nll=nll,
                           persistence_expected_yield=persistence,
                           error=abs(pred-child.accepted))
                results[f"h{depth}"].append(row)
                prediction_rows.append(row)
                previous_y, previous_len = child.accepted, child.length
    summary = {}
    for key, rows in results.items():
        metric = _metric_rows(rows)
        eligible = [r for r in rows if r["persistence_expected_yield"] is not None]
        metric["persistence_mae"] = (None if not eligible else
            float(np.mean([abs(r["persistence_expected_yield"]-r["accepted"]) for r in eligible])))
        metric["by_action"] = {}
        for action in ("R", "E"):
            subset = [r for r in rows if r["action"] == action]
            metric["by_action"][action] = _metric_rows(subset)
        summary[key] = metric
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            for row in prediction_rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
    return summary, prediction_rows


def _train_update(model, optimizer, train_nodes, train_edges, token_table, device,
                  batch_size, horizon, rng):
    labeled = [obs for obs in train_nodes.values() if obs.accepted is not None]
    if not labeled:
        raise RuntimeError("No accepted-prefix labels available for Phase 1")
    group = rng.choices(labeled, k=batch_size)
    batch = _batch(group, token_table, device)
    encoded = model.encode(batch)
    lengths = batch["lengths"]
    logits = model.hazards(encoded["state"], lengths, width=batch["hidden"].shape[1])
    loss_acc = hazard_nll(logits, lengths, batch["labels"])
    direct = batch["teacher_valid"].any(-1)
    loss_post = (hazard_nll(model.hazards(encoded["posterior"], lengths,
        width=batch["hidden"].shape[1])[direct], lengths[direct], batch["labels"][direct])
        if bool(direct.any()) else loss_acc.detach()*0)

    path_list = _sample_paths(train_edges, horizon, max(2, batch_size//2), rng)
    latent_losses, imagined_losses = [], []
    if path_list:
        start_obs = [train_nodes[path[0][0]] for path in path_list]
        sb = _batch(start_obs, token_table, device)
        se = model.encode(sb)
        actual = sb["teacher_actual"] & se["has_verifier"]
        z = torch.where(actual[:, None], se["posterior"], se["unverified"])
        for step in range(max(len(path) for path in path_list)):
            active = [(i, path[step]) for i, path in enumerate(path_list) if len(path) > step]
            if not active:
                break
            indices = torch.tensor([item[0] for item in active], device=device)
            z = z.index_select(0, indices)
            selected_paths = [path_list[i] for i, _ in active]
            child_obs = [train_nodes[edge[1]] for _, edge in active]
            cb = _batch(child_obs, token_table, device)
            db = model.encode_d(cb)
            action_names = [edge[2] for _, edge in active]
            z = torch.cat([model.transition(z[i:i+1], action_names[i], db[i:i+1])
                           for i in range(len(action_names))], dim=0)
            future_logits = model.hazards(z, cb["lengths"], width=cb["hidden"].shape[1])
            imagined_losses.append(hazard_nll(future_logits, cb["lengths"], cb["labels"]))
            child_has_v = cb["teacher_valid"].any(-1)
            if bool(child_has_v.any()):
                post = model.encode(cb)["posterior"]
                latent_losses.append(F.smooth_l1_loss(z[child_has_v], post[child_has_v].detach()))

    loss_dyn = torch.stack(latent_losses).mean() if latent_losses else loss_acc.detach()*0
    loss_imag = torch.stack(imagined_losses).mean() if imagined_losses else loss_acc.detach()*0
    loss = loss_acc + .25*loss_post + .1*loss_dyn + .5*loss_imag
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return dict(loss=float(loss.detach()), acceptance=float(loss_acc.detach()),
        posterior_acceptance=float(loss_post.detach()), dynamics=float(loss_dyn.detach()),
        imagined_acceptance=float(loss_imag.detach()), paths=len(path_list), horizon=horizon)


@torch.no_grad()
def _make_conditioning_latents(model, nodes, edges, final_by_round,
                               metadata, token_table, device):
    """Reconstruct only causally available prior z for each visited draft node."""
    cache = _encode_cache(model, nodes, token_table, device)
    children = defaultdict(list)
    for parent, child, action in edges:
        children[parent].append((child, action))
    by_question_round = defaultdict(list)
    for obs in nodes.values():
        by_question_round[(obs.question, obs.round_id)].append(obs.uid)
    condition = {}
    current = {}
    qids = sorted({obs.question for obs in nodes.values()})
    for qid in qids:
        round_ids = sorted(r for q, r in by_question_round if q == qid)
        previous_z = torch.zeros(1, 128, device=device)
        previous_round = None
        for round_id in round_ids:
            roots = [uid for uid in by_question_round[(qid, round_id)]
                     if metadata[uid].get("parent_state_id") is None]
            root = min(roots, key=lambda uid: (nodes[uid].length, uid)) if roots else None
            if previous_round is not None:
                final_id = final_by_round.get((qid, previous_round))
                # Shadow verifier labels are training targets, not persistent
                # runtime observations. Carry only a genuine submitted STOP.
                if (final_id in nodes and nodes[final_id].teacher_is_actual
                        and cache[final_id]["has_verifier"].item()):
                    previous_z = cache[final_id]["posterior"].reshape(1, -1)
            if root is None:
                previous_round = round_id
                continue
            condition[root] = previous_z.detach()
            root_d = cache[root]["d"].reshape(1, -1)
            current[root] = model.root_observation(previous_z, root_d)
            frontier = [root]
            visited = {root}
            while frontier:
                parent = frontier.pop(0)
                for child, action in sorted(children.get(parent, []), key=lambda x: (x[1], x[0])):
                    if child in visited or child not in cache:
                        continue
                    condition[child] = current[parent].detach()
                    dchild = cache[child]["d"].reshape(1, -1)
                    current[child] = model.transition(current[parent], action, dchild)
                    visited.add(child)
                    frontier.append(child)
            # A round's realized final verifier state becomes next round's actual prior.
            previous_round = round_id
    return condition


def _make_distill_examples(nodes, metadata, condition, hidden_slot, split):
    examples = []
    for uid, obs in nodes.items():
        if metadata[uid]["split"] != split or obs.teacher_topk_ids is None or obs.accepted is None:
            continue
        z = condition.get(uid)
        if z is None or float(z.norm()) < 1e-6:
            continue  # round zero has no preceding verifier state; FiLM must be identity there
        clean = min(obs.length, obs.accepted + int(obs.accepted < obs.length))
        if clean <= 0 or obs.hidden.shape[1] <= hidden_slot:
            continue
        for pos in range(clean):
            if obs.teacher_topk_ids[pos].numel() == 0:
                continue
            examples.append((obs.hidden[pos, hidden_slot].clone(), z.reshape(-1).cpu().half(),
                obs.teacher_topk_ids[pos].clone(), obs.teacher_topk_logits[pos].clone(),
                obs.teacher_logsumexp[pos].clone()))
    return examples


def _distill_loss(adapter, drafter, examples, device, batch_tokens, steps, seed, learning_rate):
    if not examples:
        raise RuntimeError("No noninitial-round verifier top-K examples for FiLM distillation")
    rng = random.Random(seed)
    adapter.train()
    for parameter in drafter.parameters():
        parameter.requires_grad_(False)
    head = drafter.get_output_embeddings()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=0.001)
    history = []
    for step in range(steps):
        chosen = [examples[rng.randrange(len(examples))] for _ in range(batch_tokens)]
        hidden = torch.stack([x[0] for x in chosen]).to(device)
        latent = torch.stack([x[1] for x in chosen]).to(device).float()
        ids = torch.stack([x[2] for x in chosen]).to(device).long()
        teacher_logits = torch.stack([x[3] for x in chosen]).to(device).float()
        teacher_lse = torch.stack([x[4] for x in chosen]).to(device).float()
        with torch.no_grad():
            base_logits = head(hidden)
        conditioned_hidden = adapter(hidden[:, None, :], latent)[:, 0]
        conditioned_logits = head(conditioned_hidden)
        loss = grouped_distribution_kl(conditioned_logits, teacher_logits, teacher_lse,
                                       ids, preserve_logits=base_logits).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(f"[film] step={step+1}/{steps} grouped_KL={history[-1]:.5f} examples={len(examples)}", flush=True)
    adapter.eval()
    return dict(steps=steps, token_examples=len(examples), final_train_grouped_kl=history[-1],
                first_train_grouped_kl=history[0], trace=history)


@torch.no_grad()
def _offline_distill_eval(adapter, drafter, examples, device, batch_tokens=32, seed=31415):
    if not examples:
        return dict(n=0, base_kl=None, conditioned_kl=None)
    rng = random.Random(seed)
    choose = examples if len(examples) <= 512 else rng.sample(examples, 512)
    base_total, conditioned_total = 0.0, 0.0
    for start in range(0, len(choose), batch_tokens):
        group = choose[start:start+batch_tokens]
        h = torch.stack([x[0] for x in group]).to(device)
        z = torch.stack([x[1] for x in group]).to(device).float()
        ids = torch.stack([x[2] for x in group]).to(device).long()
        target = torch.stack([x[3] for x in group]).to(device).float()
        lse = torch.stack([x[4] for x in group]).to(device).float()
        base = drafter.get_output_embeddings()(h)
        changed = drafter.get_output_embeddings()(adapter(h[:, None], z)[:, 0])
        base_total += float(grouped_distribution_kl(base, target, lse, ids).sum())
        conditioned_total += float(grouped_distribution_kl(changed, target, lse, ids).sum())
    return dict(n=len(choose), base_kl=base_total/len(choose),
                conditioned_kl=conditioned_total/len(choose))


def _bootstrap_mean_ci(values, seed=42, draws=3000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=None, ci95=[None, None])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=[float(x) for x in np.quantile(sampled, [.025, .975])])


def _repack(run_dir):
    archive = run_dir.with_suffix(".zip")
    temp = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                zf.write(path, path.relative_to(run_dir),
                    compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temp) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Archive verification failed at {bad}")
    temp.replace(archive)
    print(f"[archive] refreshed {archive} ({archive.stat().st_size/2**30:.2f} GiB)", flush=True)


def run(args):
    run_dir = args.run_dir.resolve()
    out = run_dir / "persistent_v1"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Persistent V1 training/evaluation expects Kaggle GPU T4 x2")
    device = torch.device(f"cuda:{args.drafter_device}")
    nodes, edges, split, questions, final_by_round, teacher_rows = _select_states(
        run_dir, args.max_nodes_per_question, args.max_edges_per_question)
    metadata = {row["state_id"]: row for row in read_jsonl(run_dir / "states.jsonl")}
    train_questions = [str(q) for q in split["train"]]
    val_questions = [str(q) for q in split["validation"]]
    train_nodes = {uid: obs for uid, obs in nodes.items() if metadata[uid]["split"] == "train"}
    val_nodes = {uid: obs for uid, obs in nodes.items() if metadata[uid]["split"] == "validation"}
    train_edges = [edge for edge in edges if edge[0] in train_nodes and edge[1] in train_nodes]
    val_edges = [edge for edge in edges if edge[0] in val_nodes and edge[1] in val_nodes]
    if len(train_nodes) < 50 or len(val_nodes) < 10 or not train_edges:
        raise RuntimeError(f"Insufficient selected experiences: train={len(train_nodes)}, "
                           f"validation={len(val_nodes)}, edges={len(train_edges)}")
    print(f"[data] train_nodes={len(train_nodes)} train_edges={len(train_edges)} "
          f"validation_nodes={len(val_nodes)} validation_edges={len(val_edges)}", flush=True)

    from transformers import AutoModelForCausalLM
    drafter = AutoModelForCausalLM.from_pretrained(args.dllm_dir, torch_dtype=torch.float16,
        device_map={"": args.drafter_device}, trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa")
    drafter.lm_head.weight = drafter.model.embed_tokens.weight
    drafter.eval().requires_grad_(False)
    token_table = drafter.get_input_embeddings().weight.detach()
    hidden_dim = int(drafter.config.hidden_size)
    top_k = int(json.loads((run_dir / "config.json").read_text()).get("raw_top_k", 32))
    hidden_layers = json.loads((run_dir / "config.json").read_text()).get("hidden_layers", [7, 14, 28])
    model = PersistentWorldModelV1(hidden_dim, token_table.shape[-1], top_k,
        num_hidden_layers=len(hidden_layers), dropout=args.dropout).to(device)
    parameter_counts = dict(world_model=sum(p.numel() for p in model.parameters()),
        drafter=sum(p.numel() for p in drafter.parameters()),
        frozen_drafter=sum(p.numel() for p in drafter.parameters() if not p.requires_grad))
    print(f"[phase1] model params={parameter_counts['world_model']:,}; dLLM hidden={hidden_dim}; "
          f"D slots={hidden_layers}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.world_learning_rate, weight_decay=.01)
    rng = random.Random(args.seed + 313)
    milestones = sorted(set(min(len(train_questions), int(x)) for x in args.milestones
                            if int(x) > 0))
    if not milestones or milestones[-1] < len(train_questions):
        milestones.append(len(train_questions))
    curve, update_metrics = [], []
    previous_count, updates = 0, 0
    all_train_edges = train_edges
    for milestone in milestones:
        active_questions = set(train_questions[:milestone])
        active_nodes = {uid: obs for uid, obs in train_nodes.items()
                        if metadata[uid]["question"] in active_questions}
        active_edges = [edge for edge in all_train_edges
                        if edge[0] in active_nodes and edge[1] in active_nodes]
        delta = milestone - previous_count
        n_updates = max(args.min_updates_per_stage, delta * args.updates_per_question)
        stage_start = time.perf_counter()
        model.train()
        for stage_step in range(n_updates):
            horizon = 1 if updates < args.horizon_warmup_updates else args.horizon
            metric = _train_update(model, optimizer, active_nodes, active_edges,
                token_table, device, args.batch_size, horizon, rng)
            updates += 1
            metric.update(update=updates, training_questions=milestone)
            update_metrics.append(metric)
            if updates % 100 == 0:
                print(f"[wm] update={updates} q={milestone} loss={metric['loss']:.4f} "
                      f"H={horizon} dyn={metric['dynamics']:.4f}", flush=True)
        current, _ = evaluate_current(model, val_nodes, token_table, device)
        rollout, _ = evaluate_dynamics(model, val_nodes, val_edges, token_table,
                                       device, args.horizon)
        stage = dict(training_questions=milestone, updates=updates,
            stage_seconds=time.perf_counter()-stage_start, current_acceptance=current,
            rollout=rollout)
        curve.append(stage)
        print(f"[learning] q={milestone} heldout_prior_MAE="
              f"{rollout.get('h1',{}).get('expected_yield_mae')} "
              f"H2={rollout.get('h2',{}).get('expected_yield_mae')} "
              f"H3={rollout.get('h3',{}).get('expected_yield_mae')}", flush=True)
        previous_count = milestone
        write_json(out / "learning_curve.json", curve)
        write_json(out / "world_model_config.json", dict(schema=PersistentWorldModelV1.schema,
            latent_dim=128, drafter_dim=64, verifier_dim=64, acceptance="per-token conditional hazard",
            actions=["R", "E"], horizon=args.horizon, hidden_layers=hidden_layers,
            parameters=parameter_counts, training_questions=milestone, updates=updates,
            seed=args.seed, question_split=split))
        torch.save(dict(model=model.state_dict(), config=model.config, updates=updates,
                        question_count=milestone, seed=args.seed), out / "persistent_world_model.pt")

    model.eval()
    final_current, current_rows = evaluate_current(model, val_nodes, token_table, device,
        output_path=out / "evaluation" / "heldout_grounded_state_predictions.jsonl")
    final_rollout, rollout_rows = evaluate_dynamics(model, val_nodes, val_edges,
        token_table, device, args.horizon,
        output_path=out / "evaluation" / "heldout_rollout_predictions.jsonl")
    ablations = {}
    for variant in ("full", "drafter_only", "no_drafter_hidden",
                    "no_verifier_logits", "no_verifier_hidden", "no_dynamics"):
        if variant == "full":
            metrics = final_rollout
        else:
            metrics, _ = evaluate_dynamics(model, val_nodes, val_edges,
                token_table, device, args.horizon, ablation=variant)
        ablations[variant] = metrics
    write_json(out / "evaluation" / "phase1_metrics.json", dict(
        grounded_state_diagnostic=final_current, rollout=final_rollout,
        factor_sensitivity=ablations,
        interpretation="Grounded-state scores are post-verifier diagnostics; operational pre-verifier scores are R/E rollouts."))
    write_json(out / "training_metrics.json", update_metrics)
    torch.save(dict(model=model.state_dict(), config=model.config, updates=updates,
                    seed=args.seed), out / "persistent_world_model.pt")

    # Conditioning input is the prior state only. The current proposal's verifier
    # target is used for distillation labels, never as the FiLM input.
    condition = _make_conditioning_latents(model, nodes, edges,
                                           final_by_round, metadata, token_table, device)
    final_slot = max(range(len(hidden_layers)), key=lambda i: hidden_layers[i])
    train_examples = _make_distill_examples(nodes, metadata, condition, final_slot, "train")
    val_examples = _make_distill_examples(nodes, metadata, condition, final_slot, "validation")
    adapter = GatedFiLMAdapter(hidden_dim).to(device)
    pre_distill = _offline_distill_eval(adapter, drafter, val_examples, device)
    adapter_result = _distill_loss(adapter, drafter, train_examples, device,
        args.film_batch_tokens, args.film_steps, args.seed+71, args.film_learning_rate)
    post_distill = _offline_distill_eval(adapter, drafter, val_examples, device)
    adapter_result.update(validation_before=pre_distill, validation_after=post_distill,
        adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        conditioning="previous grounded/predicted z; current verifier is target only",
        objective="verifier top-K + exact OTHER-bucket KL; 0.05 base-drafter preservation KL",
        usable_train_tokens=len(train_examples), usable_validation_tokens=len(val_examples),
        clean_region="positions through first rejected token (Y+1); no post-reject suffix")
    write_json(out / "film_training.json", adapter_result)
    torch.save(dict(adapter=adapter.state_dict(), hidden_dim=hidden_dim, latent_dim=128,
                    injection="final_norm_output_before_lm_head", seed=args.seed+71),
               out / "gated_film_adapter.pt")

    # Release the standalone drafter before loading the sharded verifier for the
    # matched real-verifier structural test.
    del token_table, drafter
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = _real_fixed_policy_eval(args, run_dir, out, model, adapter,
                                         val_questions, questions, hidden_dim)
    write_json(out / "evaluation" / "real_verifier_film_comparison.json", evaluation)
    report = _report(curve, final_current, final_rollout, ablations, adapter_result, evaluation)
    (out / "READ_RESULTS.md").write_text(report, encoding="utf-8")
    root_summary_path = run_dir / "summary.json"
    root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
    root_summary["persistent_world_model_v1"] = dict(status="complete", updates=updates,
        training_questions=len(train_questions), heldout_questions=len(val_questions),
        phase1_rollout=final_rollout, film_real_verifier=evaluation,
        report="persistent_v1/READ_RESULTS.md")
    write_json(root_summary_path, root_summary)
    _repack(run_dir)
    print(json.dumps(root_summary["persistent_world_model_v1"], indent=2), flush=True)


def _real_fixed_policy_eval(args, run_dir, out, model, adapter, val_questions,
                            questions, hidden_dim):
    from types import SimpleNamespace
    from sparse_extend_world_model_collector import _load_models
    from native_elysia_graph import NativeElysiaRunner, NativeEosWithoutSnapshot
    from world_model_environment import MASK_ID
    from world_model_teacher_environment import TeacherVerifier, TeacherTrainingEnvironment
    from world_model_core import pack_observations

    class EvalArgs(SimpleNamespace):
        pass
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ev = EvalArgs(target_model_name=cfg.get("target_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        target_device=args.target_device, drafter_device=args.drafter_device,
        target_gpu_memory_gib=args.target_gpu_memory_gib, target_placement="auto",
        dllm_dir=args.dllm_dir, raw_top_k=32, hidden_layers=[7,14,28],
        physical_block_size=32, small_block_size=8, drafter_threshold=.5,
        max_refinement_steps=3, extend_size=8, max_proposal_tokens=64,
        capture_verifier_teacher=True, verifier_teacher_top_k=32,
        max_context_tokens=4096)
    tokenizer, target, drafter = _load_models(ev)
    target.eval().requires_grad_(False)
    drafter.eval().requires_grad_(False)
    device = torch.device(f"cuda:{args.drafter_device}")
    model = model.to(device).eval()
    adapter = adapter.to(device).eval()
    adapter.force_gate = None
    verifier = TeacherVerifier(target, tokenizer, ev)
    runner = NativeElysiaRunner(drafter, tokenizer, ev)
    table = drafter.get_input_embeddings().weight.detach()
    environment = TeacherTrainingEnvironment(runner, verifier, tokenizer.eos_token_id,
        hidden_dim, ev, lambda *_: None, token_table=table)
    environment.counter = 0
    root_rows = read_jsonl(run_dir / "questions.jsonl")
    selected = []
    val_set = set(map(str, val_questions))
    seen = set()
    for row in root_rows:
        qid = str(row["question_id"])
        if row.get("split") != "validation" or qid not in val_set or qid in seen:
            continue
        selected.append(row); seen.add(qid)
    if not selected:
        # Older archives omit split in questions.jsonl; the split file remains authoritative.
        selected = [row for row in root_rows if str(row["question_id"]) in val_set]
    selected = selected[:args.eval_questions]
    outcomes = []
    for index, question in enumerate(selected):
        qid = str(question["question_id"])
        prompt = question.get("prompt")
        if not prompt:
            outcomes.append(dict(question_id=qid, paired=False,
                excluded_reason="missing_saved_prompt"))
            continue
        prompt_ids = tokenizer.apply_chat_template([{"role":"user","content":prompt}],
            tokenize=True, add_generation_prompt=True)
        environment.reset_history()
        drafter._persistent_world_film_adapter = None
        drafter._persistent_world_latent = None
        try:
            first = environment.start(qid, 0, list(prompt_ids), 9)
        except NativeEosWithoutSnapshot as eos:
            accepted, _, emitted, verifier_ms = verifier.score(
                list(prompt_ids), eos.candidate_token_ids, 9)
            outcomes.append(dict(question_id=qid, paired=False,
                excluded_reason="first_native_segment_eos_without_snapshot",
                seed_accepted=int(accepted), seed_proposal_length=len(eos.candidate_token_ids),
                seed_verifier_ms=float(verifier_ms), emitted_token_ids=emitted))
            continue
        environment.submit(first, 9)
        if tokenizer.eos_token_id in first.emitted:
            outcomes.append(dict(question_id=qid, paired=False,
                excluded_reason="first_verified_round_reached_eos",
                seed_accepted=int(first.observation.accepted),
                seed_proposal_length=first.observation.length,
                emitted_token_ids=first.emitted))
            continue
        if first.observation.teacher_features is None:
            outcomes.append(dict(question_id=qid, paired=False,
                excluded_reason="missing_first_round_verifier_features",
                seed_accepted=int(first.observation.accepted),
                seed_proposal_length=first.observation.length))
            continue
        first_batch = pack_observations([first.observation], table, device)
        with torch.no_grad():
            z_prior = model.posterior_from_batch(first_batch).detach()
        common_prefix = list(prompt_ids) + list(first.emitted)
        common_history = list(environment.verifier_history)
        pair = {}
        for arm in ("baseline", "film"):
            environment.verifier_history = list(common_history)
            if arm == "film":
                drafter._persistent_world_film_adapter = adapter
                drafter._persistent_world_latent = z_prior
            else:
                drafter._persistent_world_film_adapter = None
                drafter._persistent_world_latent = None
            started = time.perf_counter()
            try:
                state = environment.start(qid, 1, common_prefix, 9)
                draft_value = state.snapshot.get("draft_latency_elapsed_ms")
                draft_ms = None if draft_value is None else float(draft_value)
                accepted, _, emitted, verifier_ms = verifier.score(
                    state.prefix, state.observation.ids[:, 1].tolist(), 9)
                proposal = state.observation.ids[:, 1].tolist()
                result = dict(accepted=int(accepted), proposal_length=len(proposal),
                    acceptance_fraction=accepted/max(1,len(proposal)), proposal_token_ids=proposal,
                    emitted_token_ids=emitted, draft_ms=draft_ms, verifier_ms=float(verifier_ms),
                    total_ms=(None if draft_ms is None else draft_ms+float(verifier_ms)), unmask_forward_index=int(
                        state.snapshot.get("unmask_forward_index", -1)), snapshot_available=True)
            except NativeEosWithoutSnapshot as eos:
                accepted, _, emitted, verifier_ms = verifier.score(common_prefix,
                    eos.candidate_token_ids, 9)
                result = dict(accepted=int(accepted), proposal_length=len(eos.candidate_token_ids),
                    acceptance_fraction=accepted/max(1,len(eos.candidate_token_ids)),
                    proposal_token_ids=eos.candidate_token_ids, emitted_token_ids=emitted,
                    draft_ms=None, verifier_ms=float(verifier_ms), total_ms=None,
                    unmask_forward_index=None, snapshot_available=False,
                    termination_reason="native_eos_without_raw_snapshot")
            result["wall_seconds"] = time.perf_counter()-started
            pair[arm] = result
        outcomes.append(dict(question_id=qid, paired=True, prefix_token_count=len(common_prefix),
            seed_accepted=int(first.observation.accepted), seed_proposal_length=first.observation.length,
            baseline=pair["baseline"], film=pair["film"],
            accepted_delta=pair["film"]["accepted"]-pair["baseline"]["accepted"],
            same_prefix=True, same_policy="one native 8-token proposal, first available unmask snapshot, threshold=0.5"))
        print(f"[real-eval] {index+1}/{len(selected)} {qid} "
              f"Y={pair['baseline']['accepted']}->{pair['film']['accepted']} "
              f"draft_ms={pair['baseline']['draft_ms']}->{pair['film']['draft_ms']}", flush=True)
        drafter._persistent_world_film_adapter = None
        drafter._persistent_world_latent = None
    paired = [row for row in outcomes if row.get("paired")]
    accepted_base = [r["baseline"]["accepted"] for r in paired]
    accepted_film = [r["film"]["accepted"] for r in paired]
    deltas = [r["accepted_delta"] for r in paired]
    result = dict(n=len(paired), questions_seen=len(selected),
        excluded_questions=len(selected)-len(paired), exclusions=[
            {"question_id": row["question_id"], "reason": row.get("excluded_reason")}
            for row in outcomes if not row.get("paired")],
        protocol="paired same question and same second-round prefix; first round grounds z with an actual verifier call; policy/action/refinement fixed; no planner",
        proposal_tokens=8, refinement_snapshot="first exposed native unmask snapshot", threshold=0.5,
        baseline_mean_accepted=(None if not accepted_base else float(np.mean(accepted_base))),
        film_mean_accepted=(None if not accepted_film else float(np.mean(accepted_film))),
        paired_delta_accepted=_bootstrap_mean_ci(deltas),
        baseline_mean_acceptance_fraction=(None if not paired else float(np.mean([
            r["baseline"]["acceptance_fraction"] for r in paired]))),
        film_mean_acceptance_fraction=(None if not paired else float(np.mean([
            r["film"]["acceptance_fraction"] for r in paired]))),
        baseline_draft_ms_median=_median([r["baseline"]["draft_ms"] for r in paired]),
        film_draft_ms_median=_median([r["film"]["draft_ms"] for r in paired]),
        baseline_verifier_ms_median=_median([r["baseline"]["verifier_ms"] for r in paired]),
        film_verifier_ms_median=_median([r["film"]["verifier_ms"] for r in paired]),
        baseline_total_ms_median=_median([r["baseline"]["total_ms"] for r in paired]),
        film_total_ms_median=_median([r["film"]["total_ms"] for r in paired]),
        per_question=outcomes,
        caveat="Small fixed heldout structural experiment; one 8-token proposal per arm is not a full-answer benchmark or proof of general improvement.")
    return result


def _median(values):
    clean = [float(x) for x in values if x is not None]
    return None if not clean else float(statistics.median(clean))


def _report(curve, current, rollout, ablations, film, real):
    lines = ["# Persistent Drafter–Verifier World Model V1", "",
        "## What was tested", "",
        "Offline Phase 1 trained a 128-D persistent latent (64-D drafter + 64-D verifier posterior), separate R/E transitions, drafter correction, and per-token conditional acceptance hazards. Horizon metrics are reported separately for H=1/2/3.", "",
        "Phase 2 froze that world model and the Fast-dLLM backbone, trained only a zero-initialized token-gated FiLM adapter using verifier top-K logits plus an exact OTHER probability bucket and a base-drafter preservation term. The current proposal's verifier data is target-only; FiLM receives the prior latent.", "",
        "## Learning curve", "", "| Train questions | Updates | H1 MAE | H2 MAE | H3 MAE |", "|---:|---:|---:|---:|---:|"]
    for stage in curve:
        h = stage["rollout"]
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(f"| {stage['training_questions']} | {stage['updates']} | "
            f"{fmt(h.get('h1',{}).get('expected_yield_mae'))} | "
            f"{fmt(h.get('h2',{}).get('expected_yield_mae'))} | "
            f"{fmt(h.get('h3',{}).get('expected_yield_mae'))} |")
    lines += ["", "## Final Phase 1 heldout", "",
        f"- Grounded/post-verifier state diagnostic: n={current.get('n')}, expected-yield MAE={current.get('expected_yield_mae')}, hazard AUC={current.get('hazard_auc')}. This is post-verifier, not an operational pre-verifier score.",
        "- Operational pre-verifier rollouts:"]
    for h in ("h1", "h2", "h3"):
        lines.append(f"  - {h}: n={rollout.get(h,{}).get('n')}, MAE={rollout.get(h,{}).get('expected_yield_mae')}, persistence MAE={rollout.get(h,{}).get('persistence_mae')}")
    lines += ["", "## Feature sensitivity (heldout input lesions)", "",
        "These are inference-time masking tests, not independently retrained causal ablations.", ""]
    for name, metrics in ablations.items():
        lines.append(f"- `{name}`: H1 MAE={metrics.get('h1',{}).get('expected_yield_mae')}; H2={metrics.get('h2',{}).get('expected_yield_mae')}; H3={metrics.get('h3',{}).get('expected_yield_mae')}")
    lines += ["", "## FiLM and real verifier comparison", "",
        f"- Adapter training: {film.get('token_examples')} train token examples; validation grouped KL {film.get('validation_before',{}).get('conditioned_kl')} after vs base {film.get('validation_after',{}).get('base_kl')} before.",
        f"- Real fixed-policy heldout test: n={real.get('n')}, mean accepted {real.get('baseline_mean_accepted')} -> {real.get('film_mean_accepted')}, paired delta {real.get('paired_delta_accepted')}.",
        f"- Median draft latency: {real.get('baseline_draft_ms_median')} -> {real.get('film_draft_ms_median')} ms; verifier latency: {real.get('baseline_verifier_ms_median')} -> {real.get('film_verifier_ms_median')} ms.",
        "- This is a paired two-round protocol: baseline first-round verifier grounds z; baseline and FiLM are compared on the same next-round prefix with the same 8-token/first-snapshot policy. It is a structural smoke, not full-generation answer accuracy.",
        "", "## Files", "",
        "`learning_curve.json`, `training_metrics.json`, `evaluation/phase1_metrics.json`, per-state/rollout JSONL predictions, `film_training.json`, `gated_film_adapter.pt`, and `evaluation/real_verifier_film_comparison.json`.", "",
        "Interpret improvement only if heldout rollout errors improve with more training questions and the paired real-verifier acceptance delta is positive without a material draft-latency penalty. The confidence interval and per-question rows should be inspected; a 20-question pilot is not decisive."]
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--dllm_dir", type=Path, required=True)
    p.add_argument("--drafter_device", type=int, default=1)
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--target_gpu_memory_gib", type=int, default=8)
    p.add_argument("--max_nodes_per_question", type=int, default=80)
    p.add_argument("--max_edges_per_question", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--updates_per_question", type=int, default=4)
    p.add_argument("--min_updates_per_stage", type=int, default=20)
    p.add_argument("--milestones", type=int, nargs="+", default=[10,20,40,60,80])
    p.add_argument("--horizon", type=int, choices=[1,2,3], default=3)
    p.add_argument("--horizon_warmup_updates", type=int, default=40)
    p.add_argument("--world_learning_rate", type=float, default=2e-4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--film_steps", type=int, default=300)
    p.add_argument("--film_batch_tokens", type=int, default=32)
    p.add_argument("--film_learning_rate", type=float, default=1e-4)
    p.add_argument("--eval_questions", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.max_edges_per_question < 3 or args.max_nodes_per_question < 8:
        p.error("Need enough per-question nodes/edges for H1-H3")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except BaseException as error:
        run_dir = arguments.run_dir.resolve()
        partial_dir = run_dir / "persistent_v1"
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["persistent_world_model_v1"] = dict(
                    status="partial", error=f"{type(error).__name__}: {error}",
                    report="persistent_v1/READ_RESULTS.md")
                write_json(summary_path, summary)
            except Exception as summary_error:
                print(f"[partial] Could not update root summary: {summary_error}", flush=True)
        try:
            _repack(run_dir)
        except Exception as package_error:
            print(f"[partial] Could not refresh ZIP: {package_error}", flush=True)
        raise
