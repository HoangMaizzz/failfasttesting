"""Exploratory behavior reports for frozen-128 Phase0 continuation experiments.

Input is one flat row per method/seed/question/edge. q_pred and q_oracle are
ALREADY cumulative survival, not conditional hazards. Primary comparisons use
selected, state-eligible checkpoints only; failed candidates remain diagnostic.
The writer returns the report dictionary and consumes one-pass inputs once.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import stdev

from factorized_wm_metrics import (
    binary_metrics, calibration, delta_report, paired_question_bootstrap,
    verifier_report, write_json, write_jsonl,
)

__all__ = ["behavior_report", "write_behavior_reports"]
_METHODS = ("A", "B1", "B2", "C")
_LOSS_KEYS = ("K_child_truth_mae", "anchored_delta_mae", "emulator_K_gap",
              "survival_profile_gap", "Qwen_survival_brier", "state_mse",
              "new_block_survival_gap", "new_block_Qwen_brier")
_SEMANTICS = {
    "scope": "exploratory continuation on reused test questions; not independent confirmation",
    "A_role": "pure state dynamics reference",
    "B1_role": "real-latent consistency dynamics",
    "B2_role": "Qwen-supervised dynamics",
    "C_role": "H1 direct prediction reference",
    "oracle_role": "frozen G on real child latent; descriptive emulator reference, not an absolute oracle",
    "survival": "q_pred/q_oracle are supplied cumulative survival; never recomputed or restarted at a region boundary",
    "Qwen_truth": "only supplied q_qwen_truth labels; missing labels remain unobserved",
    "anchored_delta": "K_pred - K_parent_true; sign is truth-anchored, not a deployment decision",
    "anchored_delta_identity": "abs((K_pred-K_parent_true)-(K_child_true-K_parent_true)) = abs(K_pred-K_child_true), on common observed support",
    "anchored_delta_is_not_independent_endpoint_evidence": True,
    "deployed_delta": "additional diagnostic K_pred - K_parent_emulator against supplied delta_K_true",
    "bootstrap": "pair seed/question_id/edge_id; average edges within question and seed, then matched seeds within question, then resample questions",
    "seed_variation": "mean and sample std of per-seed question-macro errors; one seed has undefined std",
    "free_rollout": "A/B1/B2 H2/H3 are free rollouts of H1-selected models, with no horizon retraining",
    "selection": "primary selected excludes diagnostic_only, failed state gates, unsupported selection metadata, and failed lambda selection",
}


def _number(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value, name, minimum=0, missing=False):
    number = _number(value)
    if number is None and missing:
        return None
    if number is None or not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}" + (" or missing" if missing else ""))
    return int(number)


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _boolean(value, name):
    if hasattr(value, "item"):
        value = value.item()
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean or missing")
    return value


def _vector(value, name, labels=False):
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a vector or missing")
    result = [_number(v) for v in value]
    if any(v is not None and (v not in (0, 1) if labels else not 0 <= v <= 1) for v in result):
        raise ValueError(f"{name} contains invalid {'binary labels' if labels else 'probabilities'}")
    return result


def _at(values, index):
    return values[index] if index < len(values) else None


def _selection_metadata(config, method, seed):
    """Native runner: selection={A:[...], final:[...], lambda_selection:{...}}.

    Also accept seed-keyed metadata for standalone metric use. Explicit negative
    metadata cannot be overridden by optimistic row flags.
    """
    selection = config.get("selection", {})
    if not isinstance(selection, Mapping):
        return {}, False
    if "final" in selection or "A" in selection:
        entries = selection.get("A", []) if method == "A" else selection.get("final", [])
        matched = [entry for entry in entries if entry.get("method", "A") == method
                   and str(entry.get("seed")) == str(seed)]
        return (dict(matched[-1]), True) if matched else ({}, False)
    metadata = selection.get(str(seed), selection.get(seed, {}))
    if isinstance(metadata, Mapping):
        return dict(metadata.get(method, metadata)), True
    return {}, False


def _is_selected(row, config):
    if row.get("selection_status") != "selected":
        return False
    method = row["method"]
    if method in ("B1", "B2") and row.get("state_fidelity_passed") is not True:
        return False
    metadata, supported = _selection_metadata(config, method, row["seed"])
    selection = config.get("selection", {})
    if isinstance(selection, Mapping) and ("final" in selection or "A" in selection) and not supported:
        return False
    status = metadata.get("status", metadata.get("selection_status"))
    if status is not None and status != "selected":
        return False
    state = metadata.get("state_fidelity_passed", metadata.get("state_gate_passed"))
    if method in ("B1", "B2") and ("state_fidelity_passed" in metadata or "state_gate_passed" in metadata) and state is not True:
        return False
    choices = selection.get("lambda_selection", {}) if isinstance(selection, Mapping) else {}
    if method in ("B1", "B2") and method in choices:
        choice = choices[method]
        if not isinstance(choice, Mapping) or (choice.get("status", "selected") != "selected" or
                ("state_fidelity_passed" in choice and choice["state_fidelity_passed"] is not True)):
            return False
    return True


def _prepare(rows, config):
    result, seen, references = [], set(), {}
    for original in rows:
        row = dict(original)
        row["_raw"] = dict(original)
        if row.get("method") not in _METHODS:
            raise ValueError("method must be A, B1, B2, or C")
        for name in ("question_id", "edge_id"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"{name} must be a nonempty string")
        row["seed"] = _integer(row.get("seed"), "seed")
        row["horizon"] = _integer(row.get("horizon"), "horizon", minimum=1)
        actions = row.get("actions")
        if (not isinstance(actions, str) or len(actions) != row["horizon"] or
                any(a not in "RE" for a in actions) or row["horizon"] not in (1, 2, 3) or
                row.get("action") != actions[-1]):
            raise ValueError("actions/action/horizon must describe the same full R/E sequence")
        if row["method"] == "C" and row["horizon"] != 1:
            raise ValueError("C is an H1-only direct reference")
        for name in ("parent_length", "child_length"):
            row[name] = _integer(row.get(name), name, minimum=1)
        for name, length in (("K_parent_true", row["parent_length"]), ("K_child_true", row["child_length"])):
            row[name] = _integer(row.get(name), name, missing=True)
            if row[name] is not None and row[name] > length:
                raise ValueError(f"{name} exceeds candidate length")
        delta = _number(row.get("delta_K_true"))
        if delta is not None and not delta.is_integer():
            raise ValueError("delta_K_true must be integer or missing")
        row["delta_K_true"] = None if delta is None else int(delta)
        parent, child = row["K_parent_true"], row["K_child_true"]
        if parent is not None and child is not None and delta is not None and delta != child-parent:
            raise ValueError("delta_K_true must equal observed child K minus parent K")
        for name in ("K_parent_emulator", "K_oracle_latent", "K_pred", "state_mse", "state_cos"):
            row[name] = _number(row.get(name))
        for name in ("R_changed", "R_gain", "R_loss", "E_full_prefix", "state_fidelity_passed"):
            row[name] = _boolean(row.get(name), name)
        row["_q"] = {name: _vector(row.get(name), name, labels=name == "q_qwen_truth")
                     for name in ("q_pred", "q_oracle", "q_qwen_truth", "hazards_pred", "hazards_oracle")}
        mask = row.get("region_mask_new_block")
        if mask is None:
            row["_mask"] = None
        else:
            row["_mask"] = [_boolean(v, "region_mask_new_block entry") for v in mask]
            if len(row["_mask"]) != row["child_length"] or any(v is None for v in row["_mask"]):
                raise ValueError("region_mask_new_block must contain one boolean per child position")
        row["_key"] = (row["seed"], row["question_id"], row["edge_id"])
        unique = (row["method"],) + row["_key"]
        if unique in seen:
            raise ValueError("Duplicate method/seed/question/edge row")
        seen.add(unique)
        reference = references.setdefault(row["_key"], row)
        for name in ("actions", "horizon", "parent_length", "child_length", "K_parent_true", "K_child_true", "delta_K_true", "K_oracle_latent"):
            if reference.get(name) is not None and row.get(name) is not None and reference[name] != row[name]:
                raise ValueError(f"Paired rows disagree on {name}")
        row["_selected"] = _is_selected(row, config)
        row["_losses"] = _row_losses(row)
        result.append(row)
    return result


def _profile(row, truth, indices, squared=False):
    pairs = [(_at(row["_q"]["q_pred"], i), _at(row["_q"][truth], i)) for i in indices]
    return _mean([(a-b)**2 if squared else abs(a-b) for a, b in pairs if a is not None and b is not None])


def _row_losses(row):
    pred, child, parent, delta, oracle = (row[key] for key in
        ("K_pred", "K_child_true", "K_parent_true", "delta_K_true", "K_oracle_latent"))
    endpoint = abs(pred-child) if pred is not None and child is not None else None
    positions = range(row["child_length"])
    new = [i for i, value in enumerate(row["_mask"] or []) if value]
    return {
        "K_child_truth_mae": endpoint,
        "anchored_delta_mae": endpoint if parent is not None and delta is not None else None,
        "emulator_K_gap": abs(pred-oracle) if pred is not None and oracle is not None else None,
        "oracle_K_truth_mae": abs(oracle-child) if oracle is not None and child is not None else None,
        "survival_profile_gap": _profile(row, "q_oracle", positions),
        "Qwen_survival_brier": _profile(row, "q_qwen_truth", positions, squared=True),
        "state_mse": row["state_mse"], "state_cos": row["state_cos"],
        "new_block_survival_gap": _profile(row, "q_oracle", new),
        "new_block_Qwen_brier": _profile(row, "q_qwen_truth", new, squared=True),
    }


def _loss_summary(rows, key):
    grouped, values = defaultdict(list), []
    for row in rows:
        value = row["_losses"][key]
        if value is not None:
            values.append(value)
            grouped[(row["question_id"], row["seed"])].append(value)
    questions = defaultdict(list)
    for (question, seed), observations in sorted(grouped.items()):
        questions[question].append(_mean(observations))
    return dict(count=len(values), missing_count=len(rows)-len(values),
                status="ok" if values else "missing", mean=_mean(values),
                question_macro_mean=_mean([_mean(questions[q]) for q in sorted(questions)]))


def _metrics(rows, region=False):
    endpoint = verifier_report([dict(question=r["question_id"], uid=r["edge_id"],
        length=r["child_length"], accepted=r["K_child_true"], expected_yield=r["K_pred"])
        for r in rows])["all"]["k"]
    endpoint["question_macro_mae"] = _loss_summary(rows, "K_child_truth_mae")["question_macro_mean"]
    anchored, deployed, truth, pred, oracle, hazard_truth, hazard_pred = [], [], [], [], [], [], []
    for row in rows:
        delta = row["delta_K_true"]
        available = all(row[name] is not None for name in ("K_parent_true", "K_child_true", "delta_K_true"))
        for entries, source, target in ((anchored, row["K_parent_true"], delta if available else None),
                                        (deployed, row["K_parent_emulator"], delta)):
            entries.append(dict(question=row["question_id"], action=row["action"], delta_true=target,
                delta_pred=row["K_pred"]-source if row["K_pred"] is not None and source is not None else None))
        indices = [i for i, v in enumerate(row["_mask"] or []) if v] if region else range(row["child_length"])
        for index in indices:
            truth.append(_at(row["_q"]["q_qwen_truth"], index))
            pred.append(_at(row["_q"]["q_pred"], index))
            oracle.append(_at(row["_q"]["q_oracle"], index))
            k = row["K_child_true"]
            if k is not None and index < min(row["child_length"], k+1):
                hazard_truth.append(int(index < k))
                hazard_pred.append(_at(row["_q"]["hazards_pred"], index))
    return {
        "endpoint_K": endpoint, "anchored_delta": delta_report(anchored)["all"],
        "deployed_delta": delta_report(deployed)["all"],
        "losses": {key: _loss_summary(rows, key) for key in (*_LOSS_KEYS, "oracle_K_truth_mae", "state_cos")},
        "qwen_survival": {"metrics": binary_metrics(truth, pred), "calibration": calibration(truth, pred)},
        "real_latent_qwen_survival": {"metrics": binary_metrics(truth, oracle), "calibration": calibration(truth, oracle)},
        "conditional_hazards": binary_metrics(hazard_truth, hazard_pred),
    }


def _group(rows, region=False):
    result = {"count": len(rows), "question_count": len({r["question_id"] for r in rows}),
              "region": "new_block; K remains whole-child endpoint" if region else "all", "methods": {}}
    for method in _METHODS:
        group = [r for r in rows if r["method"] == method]
        selected = [r for r in group if r["_selected"]]
        result["methods"][method] = {
            "count": len(group), "selection": {
                "selected_count": len(selected), "diagnostic_only_count": len(group)-len(selected),
                "state_failed_count": sum(r["state_fidelity_passed"] is False for r in group)},
            "metrics": _metrics(group, region), "selected_metrics": _metrics(selected, region),
            "diagnostic_metrics": _metrics([r for r in group if not r["_selected"]], region),
        }
    return result


def _paired(rows, method, metric, samples, seed):
    a = {r["_key"]: r for r in rows if r["method"] == "A"}
    b = {r["_key"]: r for r in rows if r["method"] == method}
    common = sorted(set(a) & set(b))
    grouped = defaultdict(list)
    count = 0
    for key in common:
        va, vb = a[key]["_losses"][metric], b[key]["_losses"][metric]
        if va is not None and vb is not None:
            grouped[(key[1], key[0])].append((va, vb))
            count += 1
    by_question, by_seed = defaultdict(list), defaultdict(list)
    for question, run_seed in sorted(grouped):
        values = grouped[(question, run_seed)]
        aa, bb = _mean([v[0] for v in values]), _mean([v[1] for v in values])
        by_question[question].append((aa, bb))
        by_seed[run_seed].append((aa, bb))
    macro = {q: (_mean([v[0] for v in values]), _mean([v[1] for v in values]))
             for q, values in sorted(by_question.items())}
    differences = [vb-va for va, vb in macro.values()]
    # Encode signed question-level differences as a difference of nonnegative
    # errors. abs(difference) would incorrectly erase which method improved.
    collapsed = [dict(question=q, accepted=0., a=max(vb-va, 0.), b=max(va-vb, 0.))
                 for q, (va, vb) in macro.items()]
    boot = paired_question_bootstrap(collapsed, "a", "b", samples=samples, seed=seed) if samples else None
    return {
        "mean_difference": _mean(differences), "ci95": boot["ci95"] if boot else None,
        "direction": method + "_minus_A", "status": "ok" if macro else "missing",
        "reference_mean": _mean([va for va, _ in macro.values()]),
        "method_mean": _mean([vb for _, vb in macro.values()]),
        "question_count": len(macro), "paired_seed_count": len(by_seed),
        "paired_row_count": count, "missing_count": len(common)-count,
        "unmatched_A_count": len(set(a)-set(b)), "unmatched_method_count": len(set(b)-set(a)),
        "per_question_seed_counts": {q: len(values) for q, values in sorted(by_question.items())},
        "samples": samples, "seed": seed,
        "by_seed": {str(s): {"mean_difference": _mean([vb-va for va, vb in values]),
                              "question_count": len(values)} for s, values in sorted(by_seed.items())},
    }


def _seed_variation(rows):
    result = {}
    for method in _METHODS:
        result[method] = {}
        for metric in (*_LOSS_KEYS, "state_cos"):
            values = {str(seed): _loss_summary([r for r in rows if r["method"] == method and r["seed"] == seed], metric)["question_macro_mean"]
                      for seed in sorted({r["seed"] for r in rows if r["method"] == method})}
            observed = [v for v in values.values() if v is not None]
            result[method][metric] = {"by_seed": values, "mean": _mean(observed),
                                      "std": stdev(observed) if len(observed) > 1 else None,
                                      "seed_count": len(observed)}
    return result


def _gates(rows, scopes, config):
    expected = sorted({_integer(s, "seeds entry") for s in config.get("seeds", sorted({r["seed"] for r in rows}))})
    by_seed = {}
    for seed in expected:
        candidate = [r for r in rows if r["seed"] == seed and r["method"] == "B1" and r["horizon"] == 1]
        selected = bool(candidate) and all(r["_selected"] and r["state_fidelity_passed"] is True for r in candidate)
        cohorts = {}
        for name in ("H1_R_nonzero", "H1_E_full_prefix"):
            subset = [r for r in scopes[name] if r["seed"] == seed and r["_selected"]]
            gap = _paired(subset, "B1", "emulator_K_gap", 0, seed)
            profile = _paired(subset, "B1", "survival_profile_gap", 0, seed)
            baseline, learned = gap["reference_mean"], gap["method_mean"]
            reduction = (baseline-learned)/baseline if baseline is not None and baseline > 0 and learned is not None else None
            same_support = gap["paired_row_count"] == profile["paired_row_count"] and gap["paired_row_count"] > 0
            passed = selected and reduction is not None and reduction >= .15 and same_support and profile["mean_difference"] <= 1e-12
            cohorts[name] = dict(passed=bool(passed), relative_K_gap_reduction=reduction,
                                 survival_gap_difference=profile["mean_difference"], question_count=gap["question_count"],
                                 paired_support_complete=same_support)
        metadata, supported = _selection_metadata(config, "B1", seed)
        by_seed[str(seed)] = dict(passed=selected and all(c["passed"] for c in cohorts.values()),
                                 state_and_selection_passed=selected, selection_metadata=metadata,
                                 selection_metadata_supported=supported, cohorts=cohorts)
    return {"B1": {"passed": not config.get('pipeline_check_only',False) and bool(expected) and all(v["passed"] for v in by_seed.values()),
                    "pipeline_check_only": bool(config.get('pipeline_check_only',False)),
                    "expected_seeds": expected, "by_seed": by_seed,
                    "scope": "exploratory engineering criteria, not independent confirmation",
                    "rule": "each seed: selected state gate + >=15% paired K gap reduction on R-nonzero and E-full-prefix, with no survival-profile gap increase"}}


def behavior_report(rows, config):
    """Pure report builder; config may override bootstrap_samples (default 2000).

    selection accepts the runner's A/final lists and lambda_selection mapping;
    explicit failed/unsupported final selections stay diagnostic. C is only H1.
    Missing teacher entries are omitted, not assigned zero. New-block selection
    uses the supplied boolean mask on global cumulative survival values.
    """
    config = dict(config)
    samples = _integer(config.get("bootstrap_samples", 2000), "bootstrap_samples", minimum=1)
    seed = _integer(config.get("bootstrap_seed", config.get("seed", 42)), "bootstrap_seed")
    rows = _prepare(rows, config)
    h1 = [r for r in rows if r["horizon"] == 1]
    rr, ee = [r for r in h1 if r["action"] == "R"], [r for r in h1 if r["action"] == "E"]
    r_cohorts = {"all": rr, "changed": [r for r in rr if r["R_changed"] is True],
                 "unchanged": [r for r in rr if r["R_changed"] is False],
                 "nonzero": [r for r in rr if r["delta_K_true"] is not None and abs(r["delta_K_true"]) > .5],
                 "gain": [r for r in rr if r["delta_K_true"] is not None and r["delta_K_true"] > .5],
                 "loss": [r for r in rr if r["delta_K_true"] is not None and r["delta_K_true"] < -.5]}
    e_cohorts = {"all": ee, "full_prefix": [r for r in ee if r["E_full_prefix"] is True],
                 "rejected_prefix": [r for r in ee if r["E_full_prefix"] is False],
                 "new_block": [r for r in ee if any(r["_mask"] or [])]}
    scopes = {"H1_ALL": h1, **{"H1_R_"+name: group for name, group in r_cohorts.items() if name != "all"},
              "H1_E_full_prefix": e_cohorts["full_prefix"], "H1_E_new_block": e_cohorts["new_block"],
              **{f"H{h}_ALL": [r for r in rows if r["horizon"] == h] for h in (2, 3)}}
    selected = {name: [r for r in group if r["_selected"]] for name, group in scopes.items()}
    def comparisons(groups, draws):
        return {name: {method+"_minus_A": {
            metric: _paired(group, method, metric, draws, seed) for metric in
            (("new_block_survival_gap", "new_block_Qwen_brier") if name == "H1_E_new_block" else _LOSS_KEYS[:6])}
            for method in ("B1", "B2", "C")} for name, group in groups.items()}
    all_report = _group(h1)
    report = {
        "schema_version": 1, "semantics": dict(_SEMANTICS),
        "H1_ALL": all_report,
        "H1_R_COHORTS": {name: _group(group) for name, group in r_cohorts.items()},
        "H1_E_COHORTS": {name: _group(group, region=name == "new_block") for name, group in e_cohorts.items()},
        "H1_STATE_FIDELITY": {"methods": {method: {"selection": entry["selection"],
            "all_predictions": {k: entry["metrics"]["losses"][k] for k in ("state_mse", "state_cos")},
            "selected": {k: entry["selected_metrics"]["losses"][k] for k in ("state_mse", "state_cos")}}
            for method, entry in all_report["methods"].items()}},
        "H1_QWEN_FIDELITY": all_report,
        "H1_DYNAMICS_ISOLATION": {"scope": _SEMANTICS["oracle_role"], "methods": {
            method: {"selection": entry["selection"], "losses": entry["metrics"]["losses"],
                     "real_latent_qwen_survival": entry["metrics"]["real_latent_qwen_survival"]}
            for method, entry in all_report["methods"].items()}},
        "QUESTION_BOOTSTRAP": {"unit": _SEMANTICS["bootstrap"],
            "primary_selected": comparisons(selected, samples),
            "diagnostic_all_predictions": comparisons(scopes, 0)},
        "SEED_VARIATION": {name: _seed_variation(group) for name, group in selected.items()},
        "FREE_ROLLOUT_H1_H2_H3": {
            "scope": _SEMANTICS["free_rollout"],
            "by_horizon": {f"H{h}": _group([r for r in rows if r["horizon"] == h]) for h in (1, 2, 3)},
            "by_sequence": {seq: _group([r for r in rows if r["actions"] == seq]) for seq in sorted({r["actions"] for r in rows})}},
        "gates": _gates(rows, scopes, config),
    }
    return report


def _joint_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(int(row["seed"]), row["question_id"], row["edge_id"])][row["method"]] = row
    for (seed, question, edge), methods in sorted(grouped.items()):
        first = methods[sorted(methods)[0]]
        yield dict(seed=seed, question_id=question, edge_id=edge,
                   actions=first["actions"], action=first["action"], horizon=first["horizon"],
                   parent_length=first["parent_length"], child_length=first["child_length"],
                   **{"K_"+method: methods.get(method, {}).get("K_pred") for method in _METHODS},
                   predictions={method: methods.get(method) for method in _METHODS})


def write_behavior_reports(rows, output, config):
    """Write the nine named JSON artifacts, FINAL_REPORT.md, joint_predictions.jsonl.

    Return behavior_report's complete dictionary. The report folder also contains
    gates inside H1_ALL.json so failed state/selection gates cannot be hidden.
    """
    rows = list(rows)
    report = behavior_report(rows, config)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in report.items():
        if name.startswith("H1_") or name in ("QUESTION_BOOTSTRAP", "SEED_VARIATION", "FREE_ROLLOUT_H1_H2_H3"):
            artifact = dict(content, semantics=report["semantics"])
            if name == "H1_ALL":
                artifact["gates"] = report["gates"]
            write_json(destination / (name + ".json"), artifact)
    write_jsonl(destination / "joint_predictions.jsonl", _joint_rows(rows))
    lines = ["# Behavior-aware Phase0 continuation", "", "Exploratory results on reused test questions; independent confirmation is still required.", "",
             "Frozen G on real child latents is an emulator reference. C is an H1 direct reference.", "",
             "True-parent-anchored delta MAE equals child-K MAE on common observed support and is not independent endpoint evidence. Its sign uses truth, not deployment inputs. Deployed delta subtracts G(parent) separately.", "",
             "Survival values are already cumulative. New-block masks select those global values without restarting survival. Missing Qwen labels remain missing.", "",
             "Bootstrap units are questions, after averaging paired edges within seed and matched seeds within question. Seed variation is reported separately.", "",
             "| Method | H1 selected rows | H1 diagnostic rows |", "|---|---:|---:|"]
    for method, entry in report["H1_ALL"]["methods"].items():
        selection = entry["selection"]
        lines.append(f"| {method} | {selection['selected_count']} | {selection['diagnostic_only_count']} |")
    lines += ["", "B1 engineering criteria: " + ("passed across required seeds." if report["gates"]["B1"]["passed"] else
              "not passed across required seeds; failed or unsupported selections remain diagnostic."),
              "", "Criteria require state/selection eligibility, 15% paired emulator K-gap reduction on R-nonzero and E-full-prefix, and no survival-profile gap increase. Passing these exploratory criteria does not establish an absolute performance bound or generalization.", ""]
    if config.get('pipeline_check_only',False):
        lines += ['SMOKE: pipeline check only. The update budget is insufficient for a feasibility verdict.', '']
    (destination / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
