"""Dependency-free sidecar for the paired native latent feasibility study.

Rows contain one method/seed/uid/path prediction. ``q_pred`` is already global
cumulative survival; ``hazard_pred`` is conditional ACCEPTANCE probability.
K and oracle diagnostics describe the whole endpoint, including in E_new_block;
that cohort restricts only token scoring to [parent_length, length). Truth is
only supplied ``accepted`` and ``parent_accepted`` (or the parent_K alias).

Report topology: groups[current|H1|H2|H3], then by_actions[sequence], then
by_length[str(length)]. Each node has counts, overlapping cohorts, method
metrics, and paired comparisons. Standard length keys 8,16,...,64 always exist
under an observed action sequence; additional observed lengths are preserved.
Only the public writer creates files; building a report never mutates inputs.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import stdev

from factorized_wm_metrics import binary_metrics, calibration, write_csv, write_json

__all__ = ["build_report", "write_reports"]

_COHORTS = ("all", "R", "E_all", "E_full_prefix", "E_rejected_prefix", "E_new_block")
_COMPARISONS = (
    ("Bridge_state", "Direct"), ("Bridge_behavior", "Direct"),
    ("Direct_distill", "Direct"), ("Bridge_behavior", "Bridge_state"),
)
_OPTIONAL_COMPARISONS = (("V_joint", "V_hidden_only"), ("V_joint_probe", "V_joint"))
_SEMANTICS = {
    "endpoint_target": "supplied accepted length K, not emitted tokens",
    "survival": "q_pred is global cumulative survival; never multiplied again or reset",
    "hazard": "conditional acceptance; score global i < min(length, accepted + 1)",
    "E_new_block": "token positions [parent_length, length); K remains whole-endpoint K",
    "E_full_prefix": "supplied parent_accepted (parent_K alias) == parent_length; evaluate-only, not model input",
    "cohorts": "overlapping: E_full_prefix/E_rejected_prefix/E_new_block also belong to E_all and all",
    "missing_labels": "never inferred from predictions, oracle outputs, or missing parent labels",
    "pair_key": ["uid", "seed", "horizon", "actions"],
    "paired_delta": "abs(K_pred_A - true_K) - abs(K_pred_B - true_K); negative favors A",
    "bootstrap_unit": "question: first average matched seeds per unique uid/horizon/actions, then paths per question",
    "bootstrap_interval": "95% percentile interval; question sampling uncertainty, separate from seed variation",
    "seed_variation": "per-method per-seed question-macro K MAE; sample standard deviation across seeds",
    "oracle_role": "optional descriptive reference, not a guaranteed ceiling",
    "oracle_disagreement": "paired abs(K_pred - oracle_K_pred), not lost tokens",
    "decomposition_additive": False,
    "decomposition": "true-K MAE, oracle true-K MAE and output disagreement are separate nonadditive diagnostics",
    "rollouts": "current observation is separate from diagnostic free rollouts H1/H2/H3",
}


def _number(value, name):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be numeric or missing") from None
    return result if math.isfinite(result) else None


def _integer(value, name, minimum=0, missing=False):
    result = _number(value, name)
    if result is None and missing:
        return None
    if result is None or not result.is_integer() or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _key(value, name):
    if value is None:
        raise ValueError(f"{name} is required")
    def scalar(item):
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Unsupported {name} identifier")
    return json.dumps(value, default=scalar, sort_keys=True, allow_nan=False)


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def _vector(value, name, length):
    if value is None:
        return [None] * length
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a length-L sequence or missing")
    try:
        result = [_number(v, name) for v in value]
    except TypeError:
        raise ValueError(f"{name} must be a length-L sequence or missing") from None
    if len(result) != length:
        raise ValueError(f"{name} must have length L={length}")
    if any(v is not None and not 0 <= v <= 1 for v in result):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return result


def _prepare(rows):
    prepared, seen, observation_metadata, path_questions = [], set(), {}, {}
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError("rows must be mappings")
        row = dict(original)
        for name in ("uid", "seed", "question"):
            row["_" + name] = _key(row.get(name), name)
        if not isinstance(row.get("method"), str) or not row["method"]:
            raise ValueError("method must be a nonempty string")
        row["length"] = _integer(row.get("length"), "length", minimum=1)
        row["horizon"] = _integer(row.get("horizon", 0), "horizon")
        if row["horizon"] not in (0, 1, 2, 3):
            raise ValueError("horizon must be 0, 1, 2, or 3")
        actions = row.get("actions", "" if row["horizon"] == 0 else None)
        if (not isinstance(actions, str) or len(actions) != row["horizon"]
                or any(action not in "RE" for action in actions)):
            raise ValueError("actions must be an R/E sequence of length horizon ('' for current)")
        row["actions"] = actions
        row["action"] = row.get("action", actions[-1:] or "")
        if actions and row["action"] != actions[-1]:
            raise ValueError("action must equal the final action in actions")
        if not isinstance(row["action"], str):
            raise ValueError("action must be a string")
        row["parent_length"] = _integer(row.get("parent_length"), "parent_length", missing=True)
        parent = row.get("parent_accepted")
        if _number(parent, "parent_accepted") is None:
            parent = row.get("parent_K")
        elif _number(row.get("parent_K"), "parent_K") is not None:
            if _number(parent, "parent_accepted") != _number(row["parent_K"], "parent_K"):
                raise ValueError("parent_K conflicts with parent_accepted")
        row["parent_accepted"] = _integer(parent, "parent_accepted", missing=True)
        row["accepted"] = _integer(row.get("accepted"), "accepted", missing=True)
        for target, bound in (("accepted", "length"), ("parent_accepted", "parent_length")):
            if row[target] is not None and row[bound] is not None and row[target] > row[bound]:
                raise ValueError(f"{target} cannot exceed {bound}")
        changed = row.get("parent_token_changed")
        if hasattr(changed, "item"):
            changed = changed.item()
        if changed is not None and not isinstance(changed, bool):
            raise ValueError("parent_token_changed must be boolean or missing")
        row["parent_token_changed"] = changed
        for name in ("K_pred", "oracle_K_pred", "latent_mse"):
            row[name] = _number(row.get(name), name)
        if row["latent_mse"] is not None and row["latent_mse"] < 0:
            raise ValueError("latent_mse cannot be negative")
        for name in ("q_pred", "hazard_pred"):
            row[name] = _vector(row.get(name), name, row["length"])
        # Missing elements remain missing; do not repair or derive predictions.
        observed_q = [q for q in row["q_pred"] if q is not None]
        if any(b > a + 1e-12 for a, b in zip(observed_q, observed_q[1:])):
            raise ValueError("q_pred must be nonincreasing cumulative survival")
        row["_path"] = (row["_uid"], row["horizon"], actions)
        row["_pair"] = (row["_uid"], row["_seed"], row["horizon"], actions)
        unique = (row["method"], row["_pair"])
        if unique in seen:
            raise ValueError("duplicate method/(uid, seed, horizon, actions) row")
        seen.add(unique)
        # Matched methods must describe the same native observation. Different
        # seeds may supply different native labels; never pool or fill them.
        path_question = path_questions.setdefault(row["_path"], row["_question"])
        if path_question != row["_question"]:
            raise ValueError("inconsistent question for the same uid/path")
        metadata = observation_metadata.setdefault(row["_pair"], {})
        for name in ("_question", "length", "action", "accepted", "parent_length", "parent_accepted", "parent_token_changed"):
            value = row[name]
            if value is not None:
                if name in metadata and metadata[name] != value:
                    raise ValueError(f"inconsistent {name} for the same uid/seed/path")
                metadata[name] = value
        prepared.append(row)
    return sorted(prepared, key=lambda row: (row["method"], row["_pair"]))


def _counts(rows):
    return {
        "row_count": len(rows),
        "observation_count": len({r["_path"] for r in rows}),
        "observation_seed_count": len({r["_pair"] for r in rows}),
        "question_count": len({r["_question"] for r in rows}),
        "seed_count": len({r["_seed"] for r in rows}),
    }


def _regression(entries, eligible_count):
    errors = [prediction - target for _, target, prediction in entries]
    absolute = [abs(error) for error in errors]
    questions = defaultdict(list)
    for question, target, prediction in entries:
        questions[question].append(abs(prediction - target))
    return {
        "status": "ok" if entries else "empty" if eligible_count == 0 else "missing",
        "count": len(entries), "missing_count": eligible_count - len(entries),
        "question_count": len(questions),
        "question_macro_mae": _mean([_mean(questions[q]) for q in sorted(questions)]),
        "mae": _mean(absolute), "row_mean_mae": _mean(absolute),
        "median": _quantile(absolute, .5), "p90": _quantile(absolute, .9),
        "within1": _mean([float(e <= 1) for e in absolute]),
        "within2": _mean([float(e <= 2) for e in absolute]),
        "within4": _mean([float(e <= 4) for e in absolute]),
        "bias": _mean(errors),
        "under_rate": _mean([float(e < 0) for e in errors]),
        "over_rate": _mean([float(e > 0) for e in errors]),
        "under_count": sum(e < 0 for e in errors), "over_count": sum(e > 0 for e in errors),
    }


def _k(rows, prediction="K_pred", target="accepted"):
    entries = [(r["_question"], r[target], r[prediction]) for r in rows
               if r[target] is not None and r[prediction] is not None]
    return _regression(entries, len(rows))


def _tokens(rows, new_block):
    survival_y, survival_p, hazard_y, hazard_p = [], [], [], []
    positions = censored = unknown_risk = 0
    for row in rows:
        start = row["parent_length"] if new_block else 0
        # E_new_block membership requires an observed, proper extension span.
        for i in range(start, row["length"]):
            positions += 1
            accepted = row["accepted"]
            label = None if accepted is None else int(i < accepted)
            survival_y.append(label)
            survival_p.append(row["q_pred"][i])
            if accepted is None:
                unknown_risk += 1
            elif i >= min(row["length"], accepted + 1):
                censored += 1
            else:
                hazard_y.append(label)
                hazard_p.append(row["hazard_pred"][i])
    def head(labels, probabilities):
        result = binary_metrics(labels, probabilities)
        if not result["count"] and labels:
            result["status"] = "missing"
        return {"metrics": result, "calibration": calibration(labels, probabilities)}
    return {
        "position_count": positions,
        "survival": head(survival_y, survival_p),
        "hazard": dict(head(hazard_y, hazard_p), risk_position_count=len(hazard_y),
                       censored_position_count=censored, unknown_risk_position_count=unknown_risk),
    }


def _seed_variation(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["_seed"]].append(row)
    per_seed = [{"seed": json.loads(seed_key), **_counts(group), "k": _k(group)}
                for seed_key, group in sorted(by_seed.items())]
    values = [item["k"]["question_macro_mae"] for item in per_seed
              if item["k"]["question_macro_mae"] is not None]
    return {
        "status": "ok" if values else "empty" if not rows else "missing",
        "per_seed": per_seed, "observed_seed_count": len(values),
        "mean_question_macro_mae": _mean(values),
        "std_question_macro_mae": stdev(values) if len(values) > 1 else None,
    }


def _method_report(rows, new_block):
    latent = [r["latent_mse"] for r in rows if r["latent_mse"] is not None]
    return {
        "status": "ok" if rows else "empty", **_counts(rows),
        "k": _k(rows), **_tokens(rows, new_block),
        "oracle": {
            "decomposition_additive": False,
            "output_disagreement": _k(rows, target="oracle_K_pred"),
            "true_k": _k(rows, prediction="oracle_K_pred"),
        },
        "latent_mse": {"status": "ok" if latent else "empty" if not rows else "missing",
                       "count": len(latent), "missing_count": len(rows) - len(latent), "mean": _mean(latent)},
        "seed_variation": _seed_variation(rows),
    }


def _paired(rows, method_a, method_b, samples, seed):
    a = {r["_pair"]: r for r in rows if r["method"] == method_a}
    b = {r["_pair"]: r for r in rows if r["method"] == method_b}
    matched = sorted(a.keys() & b.keys())
    paths = defaultdict(list)
    missing_labels = missing_predictions = 0
    for key in matched:
        left, right = a[key], b[key]
        if left["accepted"] is None or right["accepted"] is None:
            missing_labels += 1
            continue
        if left["K_pred"] is None or right["K_pred"] is None:
            missing_predictions += 1
            continue
        paths[(left["_question"], left["_path"])].append(
            (abs(left["K_pred"] - left["accepted"]), abs(right["K_pred"] - right["accepted"])))
    questions = defaultdict(list)
    path_means = []
    for (question, path), entries in sorted(paths.items()):
        mean_a, mean_b = (_mean([entry[i] for entry in entries]) for i in (0, 1))
        questions[question].append((mean_a, mean_b))
        path_means.append({"question": json.loads(question), "uid": json.loads(path[0]),
                           "horizon": path[1], "actions": path[2], "matched_seed_count": len(entries),
                           "mae_a": mean_a, "mae_b": mean_b, "delta_abs_error": mean_a - mean_b})
    units = []
    for question, entries in sorted(questions.items()):
        mean_a, mean_b = (_mean([entry[i] for entry in entries]) for i in (0, 1))
        units.append({"question": json.loads(question), "path_count": len(entries),
                      "mae_a": mean_a, "mae_b": mean_b, "delta_abs_error": mean_a - mean_b})
    deltas = [unit["delta_abs_error"] for unit in units]
    # Restart the same RNG seed for every comparison, including every subgroup.
    rng = random.Random(seed)
    draws = [_mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
             for _ in range(samples)] if deltas else []
    return {
        "method_a": method_a, "method_b": method_b,
        "status": "ok" if units else "empty" if not a and not b else
                  "missing_method" if not a or not b else "unmatched" if not matched else "missing",
        "direction": "A_minus_B; negative favors A",
        "method_a_count": len(a), "method_b_count": len(b),
        "matched_count": len(matched), "scored_count": sum(len(v) for v in paths.values()),
        "unmatched_a_count": len(a.keys() - b.keys()), "unmatched_b_count": len(b.keys() - a.keys()),
        "missing_label_count": missing_labels, "missing_prediction_count": missing_predictions,
        "path_count": len(paths), "question_count": len(units),
        "mae_a": _mean([u["mae_a"] for u in units]), "mae_b": _mean([u["mae_b"] for u in units]),
        "delta_abs_error": _mean(deltas),
        "ci95": [_quantile(draws, .025), _quantile(draws, .975)] if draws else None,
        "bootstrap_samples": samples, "seed": seed, "bootstrap_unit": "question",
        "path_means": path_means, "question_means": units,
    }


def _cohort_rows(rows, name):
    if name == "all":
        return rows
    action = "R" if name == "R" else "E"
    candidates = [r for r in rows if r["action"] == action]
    if name in ("R", "E_all"):
        return candidates
    if name == "E_new_block":
        return [r for r in candidates if r["parent_length"] is not None
                and r["parent_length"] < r["length"]]
    return [r for r in candidates if r["parent_accepted"] is not None and r["parent_length"] is not None
            and (r["parent_accepted"] == r["parent_length"] if name == "E_full_prefix"
                 else r["parent_accepted"] < r["parent_length"])]


def _group(rows, methods, comparisons, samples, seed):
    cohorts = {}
    for name in _COHORTS:
        subset = _cohort_rows(rows, name)
        cohorts[name] = {
            "status": "ok" if subset else "empty", **_counts(subset),
            "by_method": {method: _method_report([r for r in subset if r["method"] == method],
                                                 name == "E_new_block") for method in methods},
            "paired_comparisons": {a + "_vs_" + b: _paired(subset, a, b, samples, seed)
                                   for a, b in comparisons},
        }
    return {"status": "ok" if rows else "empty", **_counts(rows), "cohorts": cohorts}


def build_report(rows, bootstrap_samples=2000, seed=42):
    """Build strict-JSON-compatible metrics with question-cluster paired CIs.

    Missing/nonfinite labels stay missing. Duplicate pairing keys, inconsistent
    observation metadata, malformed probabilities and unsupported horizons fail
    explicitly. Prediction vectors may be wholly missing or have missing entries,
    but supplied vectors must have exactly L entries. Inputs are never modified.
    """
    samples = _integer(bootstrap_samples, "bootstrap_samples", minimum=1)
    seed = _integer(seed, "seed")
    prepared = _prepare(rows)
    methods = sorted({r["method"] for r in prepared})
    comparisons = _COMPARISONS + tuple(pair for pair in _OPTIONAL_COMPARISONS
                                      if any(method in methods for method in pair))
    groups = {}
    for horizon, name in enumerate(("current", "H1", "H2", "H3")):
        subset = [r for r in prepared if r["horizon"] == horizon]
        node = _group(subset, methods, comparisons, samples, seed)
        node["by_actions"] = {}
        actions = sorted({r["actions"] for r in subset} | ({""} if horizon == 0 else set()))
        for sequence in actions:
            selected = [r for r in subset if r["actions"] == sequence]
            action_node = _group(selected, methods, comparisons, samples, seed)
            lengths = sorted(set(range(8, 65, 8)) | {r["length"] for r in selected})
            action_node["by_length"] = {
                str(length): _group([r for r in selected if r["length"] == length],
                                    methods, comparisons, samples, seed) for length in lengths}
            node["by_actions"][sequence] = action_node
        groups[name] = node
    return {
        "schema_version": 1, "status": "ok" if prepared else "empty", **_counts(prepared),
        "semantics": dict(_SEMANTICS), "bootstrap_samples": samples, "seed": seed,
        "methods": methods, "groups": groups,
        "seed_variation": {method: {name: _seed_variation([r for r in prepared
                                                          if r["method"] == method and r["horizon"] == horizon])
                                    for horizon, name in enumerate(("current", "H1", "H2", "H3"))}
                           for method in methods},
    }


def _nodes(report):
    for horizon, node in report["groups"].items():
        yield horizon, None, None, node
        for actions, action_node in node["by_actions"].items():
            yield horizon, actions, None, action_node
            for length, length_node in action_node["by_length"].items():
                yield horizon, actions, length, length_node


def write_reports(rows, out, bootstrap_samples=2000, seed=42):
    """Write comparison.csv, metrics.json, paired_question_bootstrap.json.

    Return the report dictionary, with absolute output paths in ``paths``. A
    one-pass input is consumed once. CSV contains method and paired rows at each
    group/action/length/cohort; the bootstrap JSON preserves matching counts and
    the actual path/question units used for every CI.
    """
    report = build_report(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report["paths"] = {name: str(directory / name) for name in
                       ("comparison.csv", "metrics.json", "paired_question_bootstrap.json")}
    table, paired = [], []
    for horizon, actions, length, node in _nodes(report):
        for cohort, group in node["cohorts"].items():
            context = {"group": horizon, "actions": actions, "length": length, "cohort": cohort}
            for method, values in group["by_method"].items():
                k = values["k"]
                table.append({**context, "kind": "method", "method": method,
                              "status": values["status"],
                              **{key: values[key] for key in _counts([])},
                              **{"K_" + key: value for key, value in k.items()},
                              "survival_brier": values["survival"]["metrics"]["brier"],
                              "survival_auc": values["survival"]["metrics"]["roc_auc"],
                              "hazard_brier": values["hazard"]["metrics"]["brier"],
                              "hazard_auc": values["hazard"]["metrics"]["roc_auc"],
                              "oracle_output_disagreement_mae": values["oracle"]["output_disagreement"]["mae"],
                              "oracle_true_K_mae": values["oracle"]["true_k"]["mae"],
                              "latent_mse": values["latent_mse"]["mean"],
                              "decomposition_additive": False})
            paired.append({**context, "status": group["status"],
                           "comparisons": group["paired_comparisons"]})
            for comparison, values in group["paired_comparisons"].items():
                table.append({**context, "kind": "paired", "comparison": comparison,
                              **{key: value for key, value in values.items()
                                 if key not in ("path_means", "question_means")}})
    write_csv(report["paths"]["comparison.csv"], table)
    write_json(report["paths"]["paired_question_bootstrap.json"],
               {"semantics": report["semantics"], "bootstrap_samples": report["bootstrap_samples"],
                "seed": report["seed"], "groups": paired})
    write_json(report["paths"]["metrics.json"], report)
    return report
