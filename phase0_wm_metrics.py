"""Dependency-free reporting for the latent-only Phase0 world model.

All endpoints predict accepted length K. ``expected_yield`` is the existing
verifier's legacy field name for that prediction; no emitted-token correction
is added. Delta predictions subtract each model's own ``*_source_K``. Only
supplied ``delta_true`` values are scored, including at composition horizons.

Verifier metrics use true labels, while prediction gaps directly pair imagined
and oracle outputs on the same row and token position. Oracle is a reference
model, not a guaranteed upper bound. Missing/nonfinite observations stay
missing; in particular, missing TF labels are never inferred from K.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from factorized_wm_metrics import (
    binary_metrics, calibration, delta_report, paired_question_bootstrap,
    verifier_report, write_csv, write_json,
)

__all__ = ["phase0_report", "write_reports"]

_MODELS = ("oracle", "imagined", "prior", "direct")
_VECTOR_HEADS = {"hazards": "hazards", "TF": "tf_probs", "pV": "probability_pred",
                 "margin": "margin_pred"}
_REGION_DEFINITIONS = {
    "E_old_prefix": "Positions [0, L-8), immutable STOP token content across the final E",
    "E_new_block": "Positions [L-8, L), the eight tokens introduced by the final E",
    "R_old_prefix": "Positions below native_active_start, with L-8 fallback when metadata is absent",
    "R_frontier": "Positions [native_active_start, L), with an eight-token fallback when metadata is absent",
    "R_changed_frontier": "H1 R with parent_token_changed=True; active frontier, not an exact changed-token mask",
    "R_changed_positions": "Exact supplied STOP-ID changed_positions for H1 R only",
    "new_since_parent": "Positions [parent_length, L), all blocks introduced along the full path",
}
_SEMANTICS = {
    "endpoint_target": "accepted_length_K",
    "endpoint_prediction": "model_verifier.expected_yield (legacy name; predicts K)",
    "delta_prediction": "endpoint_K - model_source_K",
    "delta_target": "supplied delta_true; missing values remain unobserved",
    "delta_zero_tolerance_tokens": 0.5,
    "prediction_gap_direction": "imagined_minus_oracle",
    "prediction_gap_support": "paired outputs at all positions below child length L",
    "mae_difference_direction": "imagined_minus_baseline; negative favors imagined",
    "bootstrap_unit": "question; paired row errors averaged within each question",
    "gap_bootstrap_unit": "question; aligned output gaps pooled within each question",
    "token_change": "supplied flag for any native STOP token change within the original parent's length",
    "R_H1_gain_loss": "supplied delta_true > 0 / < 0; overlaps token-change cohorts",
    "E_H1": "observed parent_accepted == parent_length / < parent_length",
    "oracle_role": "descriptive reference, not a mathematically guaranteed upper bound",
    "regions": "Native E appends eight tokens; final frontier is [L-8,L), not the entire extension on composed paths",
    "regional_survival": "Unconditional product of child hazards from position zero, selected after accumulation",
    "regional_hazard_truth": "Observed only at global positions i < min(L,K+1); no labels beyond first rejection",
    "regional_TF_truth": "Only supplied teacher labels; never filled from accepted K or oracle predictions",
    "R_region_scope": "Use native_active_start if supplied; exact changed_positions are safe for H1 R only",
    "composition_parent": "parent_length and parent_token_changed refer to the original source, not the final action's parent",
}


def _number(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Expected a numeric scalar, got {value!r}") from None
    return result if math.isfinite(result) else None


def _integer(value, name, minimum=None, missing=False):
    number = _number(value)
    if number is None and missing:
        return None
    if number is None or not number.is_integer() or (minimum is not None and number < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{bound}" + (" or missing" if missing else ""))
    return int(number)


def _accepted(value, length, name):
    result = _integer(value, name, minimum=0, missing=True)
    if result is not None and length is not None and result > length:
        raise ValueError(f"{name} cannot exceed its candidate length")
    return result


def _json_scalar(value):
    for method in ("tolist", "item"):
        if hasattr(value, method):
            result = getattr(value, method)()
            if result is not value:
                return result
    raise TypeError(f"Unsupported question identifier: {type(value).__name__}")


def _question_key(row):
    if row.get("question") is None:
        raise ValueError("A question identifier is required")
    return json.dumps(row["question"], default=_json_scalar, sort_keys=True, allow_nan=False)


def _vector(verifier, key):
    values = verifier.get(key)
    if values is None:
        return []
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{key} must be a one-dimensional sequence or None")
    try:
        return list(values)
    except TypeError:
        raise ValueError(f"{key} must be a one-dimensional sequence or None") from None


def _at(values, index):
    return _number(values[index]) if index < len(values) else None


def _prepare(rows):
    prepared = []
    for original in rows:
        row = dict(original)
        row["_question"] = _question_key(row)
        actions = row.get("actions")
        if not isinstance(actions, str) or not actions or any(a not in "RE" for a in actions):
            raise ValueError("actions must be a nonempty R/E sequence")
        row["horizon"] = _integer(row.get("horizon"), "horizon", minimum=1)
        if row["horizon"] != len(actions):
            raise ValueError("horizon must equal the full action-sequence length")
        row["action"] = row.get("action", actions[-1])
        if row["action"] != actions[-1]:
            raise ValueError("action must match the last action in actions")
        row["length"] = _integer(row.get("length"), "length", minimum=1)
        row["parent_length"] = _integer(row.get("parent_length"), "parent_length",
                                         minimum=1, missing=True)
        row["accepted"] = _accepted(row.get("accepted"), row["length"], "accepted")
        row["parent_accepted"] = _accepted(row.get("parent_accepted"), row["parent_length"],
                                            "parent_accepted")
        row["delta_true"] = _integer(row.get("delta_true"), "delta_true", missing=True)
        changed = row.get("parent_token_changed")
        if hasattr(changed, "item"):
            changed = changed.item()
        if changed is not None and not isinstance(changed, bool):
            raise ValueError("parent_token_changed must be boolean or missing")
        row["parent_token_changed"] = changed
        active = _integer(row.get("native_active_start"), "native_active_start", minimum=0, missing=True)
        if active is not None and active > row["length"]:
            raise ValueError("native_active_start cannot exceed child length")
        row["native_active_start"] = active
        positions = row.get("changed_positions")
        if positions is not None:
            positions = sorted({_integer(p, "changed_positions entry", minimum=0)
                                for p in _vector(row, "changed_positions")})
            bound = min(row["length"], row["parent_length"] or row["length"])
            if any(p >= bound for p in positions):
                raise ValueError("changed_positions must refer to observed old positions")
        row["changed_positions"] = positions
        row["_verifiers"], row["_K"], row["_delta"] = {}, {}, {}
        for model in _MODELS:
            supplied = row.get(model + "_verifier")
            if supplied is not None and not isinstance(supplied, Mapping):
                raise ValueError(f"{model}_verifier must be a mapping or missing")
            verifier = dict(supplied or {})
            # The outer row is authoritative for endpoint truth and pairing.
            verifier.update(question=row["question"], length=row["length"], accepted=row["accepted"])
            for key in (*_VECTOR_HEADS.values(), "tf_truth", "probability_truth", "margin_truth"):
                verifier[key] = _vector(verifier, key)
            endpoint = _number(verifier.get("expected_yield"))
            source = _number(row.get(model + "_source_K"))
            row["_verifiers"][model] = verifier
            row["_K"][model] = endpoint
            row["_delta"][model] = endpoint - source if endpoint is not None and source is not None else None
        prepared.append(row)
    return prepared


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _scalar_summary(rows, key):
    values = [_number(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    return {"count": len(observed), "missing_count": len(rows) - len(observed),
            "status": "ok" if observed else "missing", "mean": _mean(observed)}


def _gap_summary(entries, eligible_count, samples, seed):
    by_question = defaultdict(list)
    values, row_count = [], 0
    for question, differences in entries:
        if differences:
            row_count += 1
            values.extend(differences)
            by_question[question].extend(differences)
    # One scalar per question preserves clustering without treating tokens as
    # independent bootstrap observations. Its absolute error against zero is
    # exactly that question's mean absolute paired output gap.
    macro_rows = [dict(question=q, gap=_mean([abs(v) for v in by_question[q]]),
                       zero=0., accepted=0.) for q in sorted(by_question)]
    bootstrap = paired_question_bootstrap(macro_rows, "gap", "zero", samples=samples, seed=seed)
    return {
        "count": len(values), "missing_count": eligible_count - len(values),
        "eligible_count": eligible_count, "row_count": row_count,
        "question_count": len(by_question), "status": "ok" if values else "missing",
        "bias": _mean(values), "mae": _mean([abs(v) for v in values]),
        "rmse": math.sqrt(_mean([v * v for v in values])) if values else None,
        "max_absolute_gap": max(map(abs, values)) if values else None,
        "question_macro_bias": _mean([_mean(by_question[q]) for q in sorted(by_question)]),
        "question_macro_mae": _mean([_mean([abs(v) for v in by_question[q]])
                                      for q in sorted(by_question)]),
        "question_macro_mae_ci95": bootstrap["ci95"],
        "bootstrap_samples": samples, "bootstrap_seed": seed,
    }


def _survival_values(verifier, length):
    survival, result = 1.0, []
    for index in range(length):
        probability = _at(verifier["hazards"], index)
        survival = survival * probability if survival is not None and probability is not None else None
        result.append(survival)
    return result


def _prediction_gaps(rows, samples, seed, spans=None):
    """Compare outputs without using oracle predictions as teacher labels."""
    selected = [(row, range(row["length"])) for row in rows] if spans is None else spans
    heads = {"hazards": "hazards", "survival": None, "TF": "tf_probs",
             "pV": "probability_pred", "margin": "margin_pred"}
    entries = {key: [] for key in (("K",) if spans is None else ()) + tuple(heads)}
    positions = sum(len(indices) for _, indices in selected)
    for row, indices in selected:
        question = row["_question"]
        if spans is None:
            a, b = row["_K"]["imagined"], row["_K"]["oracle"]
            entries["K"].append((question, [a - b] if a is not None and b is not None else []))
        for head, field in heads.items():
            a, b = [row["_verifiers"][model] for model in ("imagined", "oracle")]
            a, b = ([_survival_values(v, row["length"]) for v in (a, b)] if field is None
                    else [v[field] for v in (a, b)])
            differences = []
            for index in indices:
                pa, pb = _at(a, index), _at(b, index)
                if pa is not None and pb is not None:
                    differences.append(pa - pb)
            entries[head].append((question, differences))
    return {key: _gap_summary(value, len(rows) if key == "K" else positions, samples, seed)
            for key, value in entries.items()}


def _regional_verifier(spans, model):
    pairs = {head: [] for head in ("hazard", "survival", "full_tf", "probability", "margin")}
    for row, indices in spans:
        verifier, accepted = row["_verifiers"][model], row["accepted"]
        survival = _survival_values(verifier, row["length"])
        for index in indices:
            if accepted is not None:
                label = int(index < accepted)
                pairs["survival"].append((label, survival[index]))
                if index < min(row["length"], accepted + 1):
                    pairs["hazard"].append((label, _at(verifier["hazards"], index)))
            for head, prediction, truth in (("full_tf", "tf_probs", "tf_truth"),
                                            ("probability", "probability_pred", "probability_truth"),
                                            ("margin", "margin_pred", "margin_truth")):
                pairs[head].append((_at(verifier[truth], index), _at(verifier[prediction], index)))
    result = {}
    for head in ("hazard", "survival", "full_tf"):
        labels, predictions = [p[0] for p in pairs[head]], [p[1] for p in pairs[head]]
        result[head] = {"metrics": binary_metrics(labels, predictions),
                        "calibration": calibration(labels, predictions)}
    for head in ("probability", "margin"):
        observed = [(truth, prediction) for truth, prediction in pairs[head]
                    if truth is not None and prediction is not None]
        result[head] = {"count": len(observed), "missing_count": len(pairs[head]) - len(observed),
                        "status": "ok" if observed else "missing",
                        "mae": _mean([abs(prediction - truth) for truth, prediction in observed])}
    return result


def _region_indices(row, name):
    length, parent = row["length"], row["parent_length"]
    valid_geometry = parent is not None and length == parent + 8 * row["actions"].count("E")
    if name == "R_changed_positions":
        return row["changed_positions"]
    if name.startswith("R_"):
        start = row["native_active_start"]
        if start is None:
            if not valid_geometry or length < 8:
                return None
            start = length - 8
        return range(start) if name.endswith("old_prefix") else range(start, length)
    if not valid_geometry or length < 8:
        return None
    if name == "new_since_parent":
        return range(parent, length)
    return range(length - 8) if name.endswith("old_prefix") else range(length - 8, length)


def _regions(rows, samples, seed):
    selected = {name: [] for name in _REGION_DEFINITIONS}
    eligible = dict.fromkeys(selected, 0)
    for row in rows:
        action = row["action"]
        names = [action + "_old_prefix", "E_new_block" if action == "E" else "R_frontier"]
        # The runner's composed change flag excludes newly appended positions;
        # it cannot identify changed final-R cohorts after an E. Only H1 is safe.
        if action == "R" and row["horizon"] == 1 and row["parent_token_changed"] is True:
            names.append("R_changed_frontier")
        if action == "R" and row["horizon"] == 1:
            names.append("R_changed_positions")
        if "E" in row["actions"]:
            names.append("new_since_parent")
        for name in names:
            eligible[name] += 1
            indices = _region_indices(row, name)
            if indices is not None:
                selected[name].append((row, indices))
    result = {}
    for name, spans in selected.items():
        positions = sum(len(indices) for _, indices in spans)
        result[name] = {
            "definition": _REGION_DEFINITIONS[name],
            "count": len(spans), "question_count": len({row["_question"] for row, _ in spans}),
            "position_count": positions, "eligible_row_count": eligible[name],
            "unavailable_count": eligible[name] - len(spans),
            "status": "ok" if positions else "empty_region" if spans else "unavailable" if eligible[name] else "empty",
            "unavailable_reason": "missing region metadata or inconsistent eight-token path geometry"
                                  if eligible[name] > len(spans) else None,
            "models": {model: _regional_verifier(spans, model) for model in _MODELS},
            "paired_prediction_gaps": _prediction_gaps(rows, samples, seed, spans=spans),
        }
    return result


def _mae_differences(rows, samples, seed):
    pairs = [dict(question=row["question"], accepted=row["accepted"], delta_true=row["delta_true"],
                  **{model + "_K": row["_K"][model] for model in _MODELS},
                  **{model + "_delta": row["_delta"][model] for model in _MODELS}) for row in rows]
    return {
        target: {
            "imagined_minus_" + baseline: paired_question_bootstrap(
                pairs, "imagined_" + suffix, baseline + "_" + suffix,
                truth=truth, samples=samples, seed=seed)
            for baseline in ("prior", "direct")
        }
        for target, suffix, truth in (("endpoint_K", "K", "accepted"),
                                      ("delta_K", "delta", "delta_true"))
    }


def _group(rows, samples, seed):
    models = {}
    for model in _MODELS:
        delta_rows = [dict(question=row["question"], action=row["action"],
                           delta_true=row["delta_true"], delta_pred=row["_delta"][model]) for row in rows]
        models[model] = {
            "verifier": verifier_report([row["_verifiers"][model] for row in rows])["all"],
            "delta": delta_report(delta_rows)["all"],
        }
    return {
        "count": len(rows), "question_count": len({row["_question"] for row in rows}),
        "status": "ok" if rows else "empty", "models": models,
        "latent": {key: _scalar_summary(rows, key) for key in ("latent_cosine", "latent_mse")},
        "paired_prediction_gaps": _prediction_gaps(rows, samples, seed),
        "paired_mae_differences": _mae_differences(rows, samples, seed),
        "regions": _regions(rows, samples, seed),
    }


def phase0_report(rows, bootstrap_samples=500, seed=42):
    """Return strict JSON data with overall, horizon, sequence, and cohort groups.

    Each group has ``models[oracle|imagined|prior|direct]`` containing aggregate
    ``verifier`` and ``delta`` reports, optional scalar ``latent`` diagnostics,
    ``paired_prediction_gaps`` (imagined minus oracle), and question-bootstrap
    ``paired_mae_differences`` against prior/direct for endpoint K and delta K.
    Gap MAE is mean absolute *paired output difference*, not a difference of
    aggregate MAEs. Gap vectors include all aligned predictions below child L;
    verifier hazard truth remains censored at the first rejected token. Gap
    question-macro MAEs also have question-bootstrap percentile CI95 intervals.

    ``regions`` separates the final E's eight-token new block and old prefix,
    R's active frontier and old prefix, H1 changed-R frontiers, and all blocks
    added since the original parent. Regional hazards retain global censoring;
    survival is accumulated from position zero before selecting a region. TF,
    probability, and margin use only supplied true teachers. No regional K is
    invented: K remains the whole-state endpoint target. Inconsistent/missing
    eight-token geometry is counted as unavailable without altering full-state
    scores. R frontiers use evaluation-only ``native_active_start`` when
    provided, otherwise the eight-token convention. ``R_changed_positions``
    uses exact supplied STOP-ID differences for H1 R. Composed flags/positions
    refer only to the original source's old positions, so they cannot identify
    final-R changes at newly appended positions and are not used as such.

    H2/H3 groups retain the entire R/E sequence. R_H1 unchanged/changed use the
    supplied native STOP token-change flag; gain/loss use supplied true delta.
    E_H1 cohorts require observed parent K and L. Cohorts may overlap. Missing
    endpoint labels, token teachers, delta labels, or source predictions are
    never filled or inferred. Outer question/length/accepted override nested
    verifier metadata, so all models score the same true endpoint target.
    """
    samples = _integer(bootstrap_samples, "bootstrap_samples", minimum=1)
    seed = _integer(seed, "seed")
    rows = _prepare(rows)
    horizons, sequences = defaultdict(list), defaultdict(list)
    cohorts = {"R_H1": {name: [] for name in ("unchanged", "changed", "gain", "loss")},
               "E_H1": {name: [] for name in ("parent_K_eq_L", "parent_K_lt_L")}}
    for row in rows:
        horizons[row["horizon"]].append(row)
        sequences[row["actions"]].append(row)
        if row["horizon"] == 1 and row["actions"] == "R":
            changed, delta = row["parent_token_changed"], row["delta_true"]
            if changed is not None:
                cohorts["R_H1"]["changed" if changed else "unchanged"].append(row)
            if delta is not None and delta != 0:
                cohorts["R_H1"]["gain" if delta > 0 else "loss"].append(row)
        if row["horizon"] == 1 and row["actions"] == "E":
            k, length = row["parent_accepted"], row["parent_length"]
            if k is not None and length is not None:
                cohorts["E_H1"]["parent_K_eq_L" if k == length else "parent_K_lt_L"].append(row)
    return {
        "schema_version": 1, "semantics": dict(_SEMANTICS),
        "overall": _group(rows, samples, seed),
        "by_horizon": {f"H{horizon}": _group(horizons[horizon], samples, seed)
                       for horizon in sorted({1, 2, 3} | set(horizons))},
        "by_sequence": {sequence: _group(sequences[sequence], samples, seed)
                        for sequence in sorted(sequences)},
        "cohorts": {family: {name: _group(group, samples, seed) for name, group in groups.items()}
                    for family, groups in cohorts.items()},
    }


def _report_groups(report):
    yield "overall", "all", report["overall"]
    for scope in ("by_horizon", "by_sequence"):
        for name, group in report[scope].items():
            yield scope, name, group
    for family, cohorts in report["cohorts"].items():
        for name, group in cohorts.items():
            yield "cohorts", family + "/" + name, group


def _compact_rows(report):
    for scope, name, group in _report_groups(report):
        for model, metrics in group["models"].items():
            verifier, delta = metrics["verifier"], metrics["delta"]
            yield {
                "scope": scope, "group": name, "region": "all", "model": model,
                "row_count": group["count"], "question_count": group["question_count"],
                "K_count": verifier["k"]["count"], "K_mae": verifier["k"]["mae"],
                "K_bias": verifier["k"]["bias"], "K_question_macro_mae": verifier["k"]["question_macro_mae"],
                "delta_count": delta["count"], "delta_mae": delta["mae"],
                "delta_sign_accuracy": delta["sign_accuracy"],
                "delta_useful_auc": delta["useful"]["roc_auc"],
                "delta_useful_auc_status": delta["useful"]["ranking_status"],
                "hazard_brier": verifier["hazard"]["metrics"]["brier"],
                "survival_brier": verifier["survival"]["metrics"]["brier"],
                "TF_count": verifier["full_tf"]["metrics"]["count"],
                "TF_brier": verifier["full_tf"]["metrics"]["brier"],
                "pV_count": verifier["probability"]["count"], "pV_mae": verifier["probability"]["mae"],
                "margin_count": verifier["margin"]["count"], "margin_mae": verifier["margin"]["mae"],
                "imagined_oracle_K_gap_mae": group["paired_prediction_gaps"]["K"]["mae"],
                "latent_cosine": group["latent"]["latent_cosine"]["mean"],
                "latent_mse": group["latent"]["latent_mse"]["mean"],
            }
        for region_name, region in group["regions"].items():
            for model, verifier in region["models"].items():
                gaps = region["paired_prediction_gaps"]
                yield {
                    "scope": scope, "group": name, "region": region_name, "model": model,
                    "row_count": region["count"], "question_count": region["question_count"],
                    "position_count": region["position_count"], "unavailable_count": region["unavailable_count"],
                    "hazard_count": verifier["hazard"]["metrics"]["count"],
                    "hazard_brier": verifier["hazard"]["metrics"]["brier"],
                    "survival_count": verifier["survival"]["metrics"]["count"],
                    "survival_brier": verifier["survival"]["metrics"]["brier"],
                    "TF_count": verifier["full_tf"]["metrics"]["count"],
                    "TF_brier": verifier["full_tf"]["metrics"]["brier"],
                    "pV_count": verifier["probability"]["count"], "pV_mae": verifier["probability"]["mae"],
                    "margin_count": verifier["margin"]["count"], "margin_mae": verifier["margin"]["mae"],
                    **{"paired_" + head + "_gap_mae": gap["mae"] for head, gap in gaps.items()},
                }


def _paired_sections(report, key, transform=None):
    def value(group):
        return transform(group[key]) if transform is not None else group[key]
    return {
        "overall": value(report["overall"]),
        "by_horizon": {name: value(group) for name, group in report["by_horizon"].items()},
        "by_sequence": {name: value(group) for name, group in report["by_sequence"].items()},
        "cohorts": {family: {name: value(group) for name, group in groups.items()}
                    for family, groups in report["cohorts"].items()},
    }


def write_reports(rows, out, bootstrap_samples=500, seed=42):
    """Write three JSON reports and phase0_comparison.csv; return the full report.

    Input may be a one-pass iterator. Reports are built before writing; each
    individual file is written atomically by the existing metrics utilities.
    """
    report = phase0_report(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "phase0_comparison.json", report)
    write_json(directory / "challenge_cohorts.json",
               {"schema_version": 1, "semantics": report["semantics"], "cohorts": report["cohorts"]})
    bootstrap = {"schema_version": 1, "semantics": report["semantics"],
                 **_paired_sections(report, "paired_mae_differences"),
                 "paired_prediction_gaps": _paired_sections(report, "paired_prediction_gaps"),
                 "regional_prediction_gaps": _paired_sections(report, "regions", transform=lambda regions:
                     {name: region["paired_prediction_gaps"] for name, region in regions.items()})}
    write_json(directory / "paired_question_bootstrap.json", bootstrap)
    write_csv(directory / "phase0_comparison.csv", _compact_rows(report))
    return report
