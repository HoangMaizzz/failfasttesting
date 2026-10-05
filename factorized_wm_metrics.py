"""Independent, dependency-free metrics for factorized world-model evaluation.

Public functions accept Python iterables, including NumPy arrays/scalars; NumPy,
sklearn, torch, and LLM inference are not dependencies. Missing/nonfinite paired
observations are omitted and counted, never replaced with zero teacher labels.
All undefined statistics are None. Binary decisions use score >= threshold.

Conventions:
* hazards q_i are CONDITIONAL ACCEPTANCE probabilities, matching this repo.
  Hazard labels are i < K, observed only for i < min(L, K + 1). Survival labels
  are i < K at every position, with predictions prod(q_0, ..., q_i).
* full-TF labels are distinct teacher labels, including the rejected suffix.
* calibration uses equal-width [0, 1] bins, left closed and right open except
  the last bin includes 1; NLL clips probabilities to [1e-12, 1 - 1e-12].
* delta sign accuracy has a +/-0.5-token inclusive zero band. Useful actions
  have delta_true > 0, independently of that descriptive sign tolerance.
* action rows are the units of threshold/ranking metrics. They are not expanded
  into independent tokens. Bootstrap uncertainty resamples whole questions.

These are descriptive diagnostics. Neither correlations nor these reports
establish causal effects, a literal performance ceiling, or online speedup.
"""

from __future__ import annotations

import csv
import json
import math
import numbers
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "binary_metrics", "calibration", "verifier_report", "threshold_sweep",
    "select_operating_points", "delta_report", "paired_question_bootstrap",
    "dynamics_report", "write_json", "write_jsonl", "write_csv",
]

_EPS = 1e-12
_LENGTH_BUCKETS = ("8", "16", "24", "32", "40+")
_QUARTILES = ("Q1", "Q2", "Q3", "Q4")
_DYNAMICS_METRICS = (
    "mse_hidden", "cosine_hidden", "mse_surface", "mask_brier",
    "token_agreement", "top5_recall", "top10_recall", "jaccard5",
    "conditional_topk_kl", "conditional_topk_js", "top1_confidence_error",
    "native_top1_agreement", "conditional_topk_coverage",
    "token_in_vocabulary_coverage", "linear_CKA_hidden", "jaccard10",
    "fresh_hidden_tokens", "fresh_logits_tokens",
)


def _number(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Expected a numeric scalar, got {value!r}") from None
    return result if math.isfinite(result) else None


def _integer(value, name, minimum=0):
    result = _number(value)
    if result is None or result < minimum or not result.is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _probability(value, name="probability"):
    result = _number(value)
    if result is not None and not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _label(value):
    result = _number(value)
    if result is not None and result not in (0, 1):
        raise ValueError("Binary truth must be 0 or 1")
    return None if result is None else int(result)


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _question(row):
    if row.get("question") is None:
        raise ValueError("A question identifier is required for question-level metrics")
    # Stable ordering supports reproducible bootstrap draws even if rows reorder.
    return json.dumps(_json_safe(row["question"]), sort_keys=True, allow_nan=False)


def _binary_pairs(truth, scores):
    labels, predictions = list(truth), list(scores)
    if len(labels) != len(predictions):
        raise ValueError("truth and scores must have equal lengths")
    pairs = []
    for target, score in zip(labels, predictions):
        target, score = _label(target), _number(score)
        if target is not None and score is not None:
            pairs.append((target, score))
    return pairs, len(labels) - len(pairs)


def _threshold(value):
    if value == "-inf":
        return -math.inf
    if value == "+inf":
        return math.inf
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("threshold must be numeric, '-inf', or '+inf'") from None
    if math.isnan(result):
        raise ValueError("threshold cannot be NaN")
    return result


def _safe_threshold(value):
    return ("+inf" if value > 0 else "-inf") if math.isinf(value) else value


def _decision_metrics(count, positives, tp, fp):
    negatives = count - positives
    fn, tn = positives - tp, negatives - fp
    selected = tp + fp
    precision = tp / selected if selected else None
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    denominator = 2 * tp + fp + fn
    recalls = [r for r in (recall, specificity) if r is not None]
    return {
        "count": count, "positives": positives, "negatives": negatives,
        "selected_count": selected, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "base_rate": positives / count if count else None,
        "accuracy": (tp + tn) / count if count else None,
        "balanced_accuracy": _mean(recalls),
        "precision": precision, "recall": recall,
        "f1": 2 * tp / denominator if denominator else None,
        "fpr": fp / negatives if negatives else None,
        "fnr": fn / positives if positives else None,
    }


def _score_groups(pairs):
    groups = []
    for label, score in sorted(pairs, key=lambda pair: pair[1], reverse=True):
        if not groups or groups[-1][0] != score:
            groups.append([score, 0, 0])
        groups[-1][1 if label else 2] += 1
    return groups


def _ranking(pairs):
    """Tie-aware AUROC and non-interpolated average precision (sklearn rule)."""
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    tp = fp = 0
    concordant = ap_sum = 0.0
    for _, group_positive, group_negative in _score_groups(pairs):
        concordant += group_negative * (tp + 0.5 * group_positive)
        tp += group_positive
        fp += group_negative
        ap_sum += group_positive * tp / (tp + fp)
    return (
        concordant / (positives * negatives) if positives and negatives else None,
        ap_sum / positives if positives else None,
    )


def calibration(truth, probabilities, bins=10):
    """Return all bins as {mean_probability, empirical_rate, count, lo, hi}.

    Empty bins have None means. Missing/nonfinite pairs are omitted; finite
    out-of-range probabilities or nonbinary labels raise ValueError.
    """
    bins = _integer(bins, "bins", minimum=1)
    pairs, _ = _binary_pairs(truth, probabilities)
    grouped = [[] for _ in range(bins)]
    for target, probability in pairs:
        probability = _probability(probability)
        grouped[min(bins - 1, int(probability * bins))].append((target, probability))
    return [
        {
            "mean_probability": _mean([p for _, p in group]),
            "empirical_rate": _mean([y for y, _ in group]),
            "count": len(group), "lo": index / bins, "hi": (index + 1) / bins,
        }
        for index, group in enumerate(grouped)
    ]


def _binary_metrics(pairs, threshold, missing_count=0, allow_probabilities=True):
    cutoff = _threshold(threshold)
    positives = sum(y for y, _ in pairs)
    tp = sum(y == 1 and score >= cutoff for y, score in pairs)
    fp = sum(y == 0 and score >= cutoff for y, score in pairs)
    result = _decision_metrics(len(pairs), positives, tp, fp)
    result.update(
        status="empty" if not pairs else "ok",
        missing_count=missing_count, threshold=_safe_threshold(cutoff),
        roc_auc=None, average_precision=None, brier=None, nll=None, ece=None,
    )
    result["roc_auc"], result["average_precision"] = _ranking(pairs)
    result["ranking_status"] = (
        "empty" if not pairs else "ok" if 0 < positives < len(pairs) else "single_class"
    )
    if not allow_probabilities:
        result["probability_status"] = "scores_not_probabilities"
    elif not pairs:
        result["probability_status"] = "empty"
    elif not all(0 <= p <= 1 for _, p in pairs):
        result["probability_status"] = "scores_not_probabilities"
    else:
        result["probability_status"] = "ok"
        probabilities = [min(1.0, max(0.0, p)) for _, p in pairs]
        labels = [y for y, _ in pairs]
        result["brier"] = _mean([(p - y) ** 2 for y, p in zip(labels, probabilities)])
        clipped = [min(1 - _EPS, max(_EPS, p)) for p in probabilities]
        result["nll"] = _mean([
            -math.log(p) if y else -math.log1p(-p)
            for y, p in zip(labels, clipped)
        ])
        curve = calibration(labels, probabilities)
        result["ece"] = math.fsum(
            b["count"] * abs(b["mean_probability"] - b["empirical_rate"])
            for b in curve if b["count"]
        ) / len(pairs)
    return result


def binary_metrics(truth, scores, threshold=.5):
    """Flat binary classification/ranking metrics plus counts and status.

    Brier, NLL, and ECE are computed only if ALL valid scores lie in [0, 1].
    A mixed unbounded-score cohort is never partially clipped into probabilities.
    Precision is None if nothing is selected; recall/AUPRC are None without
    positives; AUROC is None without both classes. Balanced accuracy averages
    recall over the classes actually present, as in sklearn. ECE uses 10 bins.
    """
    pairs, missing_count = _binary_pairs(truth, scores)
    return _binary_metrics(pairs, threshold, missing_count)


def _vector(row, key):
    value = row.get(key)
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{key} must be a one-dimensional sequence or None")
    try:
        return list(value)
    except TypeError:
        raise ValueError(f"{key} must be a one-dimensional sequence or None") from None


def _at(values, index):
    return values[index] if index < len(values) else None


def _length_bucket(length):
    return "40+" if length >= 40 else str(length)


def _token_accumulator():
    heads = ("hazard", "survival", "full_tf", "probability", "margin")
    return dict({key: [] for key in heads}, _eligible=dict.fromkeys(heads, 0))


def _add_tokens(accumulator, pairs, eligible):
    for key in eligible:
        accumulator["_eligible"][key] += 1
    for key, pair in pairs.items():
        accumulator[key].append(pair)


def _regression_summary(pairs, eligible_count):
    return {
        "count": len(pairs), "status": "ok" if pairs else "missing",
        "missing_count": eligible_count - len(pairs),
        "mae": _mean([abs(pred - target) for target, pred in pairs]),
    }


def _token_summary(accumulator):
    result = {}
    for key in ("hazard", "survival", "full_tf"):
        pairs = accumulator[key]
        result[key] = {
            "metrics": _binary_metrics(pairs, .5, accumulator["_eligible"][key] - len(pairs)),
            "calibration": calibration([y for y, _ in pairs], [p for _, p in pairs]),
        }
    for key in ("probability", "margin"):
        result[key] = _regression_summary(accumulator[key], accumulator["_eligible"][key])
    return result


def _k_summary(rows):
    errors, by_question, modes = [], defaultdict(list), []
    for row in rows:
        target = row["_accepted"]
        prediction = _number(row.get("expected_yield"))
        if target is not None and prediction is not None:
            error = prediction - target
            errors.append(error)
            by_question[_question(row)].append(abs(error))
        mode = _number(row.get("mode"))
        if mode is not None:
            if not mode.is_integer() or not 0 <= mode <= row["_length"]:
                raise ValueError("mode must be an integer in [0, length]")
            if target is not None:
                modes.append(float(mode == target))
    absolute = [abs(error) for error in errors]
    return {
        "count": len(errors), "missing_count": len(rows) - len(errors),
        "status": "ok" if errors else "missing", "mae": _mean(absolute),
        "rmse": math.sqrt(_mean([e * e for e in errors])) if errors else None,
        "median": _quantile(absolute, .5), "p90": _quantile(absolute, .9),
        "within1": _mean([float(e <= 1) for e in absolute]),
        "within2": _mean([float(e <= 2) for e in absolute]),
        "within4": _mean([float(e <= 4) for e in absolute]),
        "bias": _mean(errors), "mode_count": len(modes), "exact_mode": _mean(modes),
        "question_count": len(by_question),
        "question_macro_mae": _mean([_mean(v) for v in by_question.values()]),
    }


def _verifier_tokens(row):
    length, accepted = row["_length"], row["_accepted"]
    hazards = _vector(row, "hazards")
    tf_probs, tf_truth = _vector(row, "tf_probs"), _vector(row, "tf_truth")
    p_pred, p_truth = _vector(row, "probability_pred"), _vector(row, "probability_truth")
    m_pred, m_truth = _vector(row, "margin_pred"), _vector(row, "margin_truth")
    survival = 1.0
    for index in range(length):
        pairs = {}
        eligible = ["full_tf", "probability", "margin"]
        q = _probability(_at(hazards, index), "hazards")
        survival = survival * q if survival is not None and q is not None else None
        if accepted is not None:
            eligible.append("survival")
            label = int(index < accepted)
            if index < min(length, accepted + 1):
                eligible.append("hazard")
                if q is not None:
                    pairs["hazard"] = (label, q)
            if survival is not None:
                pairs["survival"] = (label, survival)
        tf_p = _probability(_at(tf_probs, index), "tf_probs")
        tf_y = _label(_at(tf_truth, index))
        if tf_p is not None and tf_y is not None:
            pairs["full_tf"] = (tf_y, tf_p)
        probability = _probability(_at(p_pred, index), "probability_pred")
        teacher_probability = _probability(_at(p_truth, index), "probability_truth")
        if probability is not None and teacher_probability is not None:
            pairs["probability"] = (teacher_probability, probability)
        margin, teacher_margin = _number(_at(m_pred, index)), _number(_at(m_truth, index))
        if margin is not None and teacher_margin is not None:
            pairs["margin"] = (teacher_margin, margin)
        yield index, pairs, eligible


def _verifier_group(rows):
    accumulator = _token_accumulator()
    for row in rows:
        for _, pairs, eligible in _verifier_tokens(row):
            _add_tokens(accumulator, pairs, eligible)
    result = _token_summary(accumulator)
    result.update(count=len(rows), status="ok" if rows else "empty", k=_k_summary(rows))
    return result


def verifier_report(rows):
    """Return {all, by_length, by_relative_position} with separate head reports.

    Group reports contain `k` (expected_yield errors and separately exact_mode),
    `hazard`, `survival`, `full_tf` ({metrics, calibration}), and `probability`/
    `margin` ({mae, count, missing_count, status}). Relative-position Q1..Q4 use floor(4*i/L),
    with zero-based i, and contain token reports only: K is a state-level target.
    Length keys 8/16/24/32/40+ always exist; other observed lengths retain their
    own key. Arrays are truncated at L; missing entries remain unobserved.
    None/NaN teacher entries are omitted even when their predictions exist.
    """
    prepared, lengths = [], defaultdict(list)
    quartiles = {key: _token_accumulator() for key in _QUARTILES}
    positions = dict.fromkeys(_QUARTILES, 0)
    for original in rows:
        row = dict(original)
        row["_length"] = _integer(row.get("length"), "length", minimum=1)
        accepted = _number(row.get("accepted"))
        if accepted is not None and (not accepted.is_integer() or not 0 <= accepted <= row["_length"]):
            raise ValueError("accepted must be an integer in [0, length] or missing")
        row["_accepted"] = None if accepted is None else int(accepted)
        _question(row)
        prepared.append(row)
        lengths[_length_bucket(row["_length"])].append(row)
        for index, pairs, eligible in _verifier_tokens(row):
            key = _QUARTILES[min(3, 4 * index // row["_length"])]
            positions[key] += 1
            _add_tokens(quartiles[key], pairs, eligible)
    keys = list(_LENGTH_BUCKETS) + sorted(set(lengths) - set(_LENGTH_BUCKETS))
    return {
        "all": _verifier_group(prepared),
        "by_length": {key: _verifier_group(lengths[key]) for key in keys},
        "by_relative_position": {
            key: dict(_token_summary(quartiles[key]), positions=positions[key],
                      status="ok" if positions[key] else "empty")
            for key in _QUARTILES
        },
    }


def _action_pairs(rows, score_key, label_key):
    pairs, missing = [], 0
    for row in rows:
        label, score = _label(row.get(label_key)), _number(row.get(score_key))
        if label is None or score is None:
            missing += 1
        else:
            pairs.append((label, score))
    return pairs, missing


def threshold_sweep(rows, score_key="score", label_key="real_useful"):
    """Action-level sweep, ascending thresholds, using score >= threshold.

    Returns one flat metrics dict per distinct finite score plus '-inf' (all)
    and '+inf' (none). Boundary strings are JSON safe and accepted as threshold
    arguments by binary_metrics and delta_report. This function only evaluates;
    it does not choose an operating point. Missing pairs are counted separately.
    """
    pairs, missing = _action_pairs(rows, score_key, label_key)
    positives, tp, fp = sum(y for y, _ in pairs), 0, 0
    def entry(threshold):
        return dict(_decision_metrics(len(pairs), positives, tp, fp),
                    threshold=threshold, missing_count=missing,
                    status="ok" if pairs else "empty")
    descending = [entry("+inf")]
    for score, group_positive, group_negative in _score_groups(pairs):
        tp += group_positive
        fp += group_negative
        descending.append(entry(score))
    descending.append(entry("-inf"))
    return list(reversed(descending))


def select_operating_points(validation_rows):
    """Pure validation-only selection; apply returned thresholds unchanged to test.

    Uses `score` and `real_useful`. If a row supplies `split`, it must be val,
    validation, or dev. Untagged rows are the caller's validation responsibility;
    no helper can infer split provenance from numbers. No global state is kept.
    Results high_precision/balanced/high_recall are full sweep records or None.
    Constraints: recall >= .2; maximum F1; precision >= base_rate + .1.
    Ties prefer recall then F1 for precision, precision then recall for F1, and
    precision then F1 for recall, finally the highest threshold in each case.
    """
    rows = list(validation_rows)
    for row in rows:
        if "split" in row and str(row["split"]).lower() not in ("val", "validation", "dev"):
            raise ValueError("Operating points must be selected on validation rows only")
    sweep = threshold_sweep(rows)
    count, base_rate = sweep[0]["count"], sweep[0]["base_rate"]
    def choose(candidates, fields):
        candidates = [r for r in candidates if all(r[f] is not None for f in fields)]
        return max(candidates, key=lambda r: tuple(r[f] for f in fields) +
                   (_threshold(r["threshold"]),)) if candidates else None
    return {
        "split": "validation", "count": count, "base_rate": base_rate,
        "high_precision": choose(
            [r for r in sweep if r["recall"] is not None and r["recall"] >= .2],
            ("precision", "recall", "f1")),
        "balanced": choose(sweep, ("f1", "precision", "recall")),
        "high_recall": choose(
            [r for r in sweep if base_rate is not None and r["precision"] is not None
             and r["precision"] >= base_rate + .1], ("recall", "precision", "f1")),
    }


def _ranks(values):
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _pearson(x, y):
    if len(x) < 2:
        return None
    # Scaling avoids overflow when score magnitudes are large but finite.
    sx, sy = max(abs(v) for v in x), max(abs(v) for v in y)
    if sx == 0 or sy == 0:
        return None
    x, y = [v / sx for v in x], [v / sy for v in y]
    mx, my = _mean(x), _mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v*v for v in dx), math.fsum(v*v for v in dy)
    if xx == 0 or yy == 0:
        return None
    correlation = math.fsum(a*b for a, b in zip(dx, dy)) / math.sqrt(xx * yy)
    return min(1.0, max(-1.0, correlation))


def _delta_group(rows, threshold):
    pairs = []
    for row in rows:
        target, prediction = _number(row.get("delta_true")), _number(row.get("delta_pred"))
        if target is not None and prediction is not None:
            pairs.append((target, prediction))
    targets, predictions = [t for t, _ in pairs], [p for _, p in pairs]
    def sign(value):
        return 0 if abs(value) <= .5 else 1 if value > 0 else -1
    return {
        "count": len(pairs), "missing_count": len(rows) - len(pairs),
        "status": "ok" if pairs else "missing",
        "pearson": _pearson(targets, predictions),
        "spearman": _pearson(_ranks(targets), _ranks(predictions)),
        "mae": _mean([abs(p - t) for t, p in pairs]),
        "sign_accuracy": _mean([float(sign(t) == sign(p)) for t, p in pairs]),
        "zero_tolerance_tokens": .5,
        "useful": _binary_metrics([(int(t > 0), p) for t, p in pairs], threshold,
                                  len(rows) - len(pairs), allow_probabilities=False),
    }


def delta_report(rows, threshold=0.):
    """Return {all, by_action: {R, E}} for paired true/predicted delta-K.

    `useful` contains AUROC/AUPRC and threshold precision/recall/F1/FPR/FNR.
    delta_pred is ALWAYS a score, including cohorts numerically within [0, 1];
    Brier/NLL/ECE stay None. Constant/single-pair correlations stay None.
    """
    rows = list(rows)
    actions = {"R": [], "E": []}
    for row in rows:
        if row.get("action") not in actions:
            raise ValueError("delta action must be R or E")
        actions[row["action"]].append(row)
    return {
        "all": _delta_group(rows, threshold),
        "by_action": {key: _delta_group(group, threshold) for key, group in actions.items()},
    }


def paired_question_bootstrap(rows, prediction_a, prediction_b, truth="accepted",
                              samples=1000, seed=42):
    """Paired A-minus-B question-macro MAE and percentile CI95.

    Prediction arguments and truth identify numeric row keys. Only rows with
    BOTH predictions and truth are used for either arm. Average absolute error
    per question first, then resample questions uniformly with replacement.
    A negative delta favors A. The interval is conditional on these paired
    questions and predictions, not a causal proof or seed-variation interval.
    """
    samples = _integer(samples, "samples", minimum=1)
    grouped, count, missing = defaultdict(list), 0, 0
    for row in rows:
        target = _number(row.get(truth))
        a, b = _number(row.get(prediction_a)), _number(row.get(prediction_b))
        if target is None or a is None or b is None:
            missing += 1
            continue
        grouped[_question(row)].append(abs(a - target) - abs(b - target))
        count += 1
    deltas = [_mean(grouped[key]) for key in sorted(grouped)]
    rng = random.Random(seed)
    draws = [_mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
             for _ in range(samples)] if deltas else []
    return {
        "delta_macro_mae": _mean(deltas),
        "ci95": [_quantile(draws, .025), _quantile(draws, .975)] if draws else None,
        "count": count, "missing_count": missing, "question_count": len(deltas),
        "samples": samples, "seed": seed, "direction": "A_minus_B",
        "status": "ok" if deltas else "missing",
    }


def _action_sequence(row):
    actions = row.get("actions")
    if not isinstance(actions, str) or not actions or any(a not in "RE" for a in actions):
        raise ValueError("actions must be a nonempty string of R/E, e.g. 'RE'")
    return actions


def _change_tags(row):
    tags = set()
    if row.get("change") in ("changed", "unchanged"):
        tags.add(row["change"])
    supplied = row.get("tags", ())
    if isinstance(supplied, str):
        supplied = [supplied]
    tags.update(tag for tag in (supplied or ()) if tag in ("changed", "unchanged"))
    for name in ("changed", "unchanged"):
        value = row.get(name)
        if value is not None and isinstance(_json_safe(value), bool):
            tags.add(name if bool(value) else "unchanged" if name == "changed" else "changed")
    if len(tags) > 1:
        raise ValueError("A dynamics row cannot be both changed and unchanged")
    return tags


def _region_summary(entries):
    result = {"count": len(entries), "status": "ok" if entries else "missing"}
    counts, statuses = {}, {}
    for key in _DYNAMICS_METRICS:
        values = [_number(region.get(key)) for _, region in entries]
        values = [value for value in values if value is not None]
        result[key] = _mean(values)
        counts[key], statuses[key] = len(values), "ok" if values else "missing"
    result.update(metric_counts=counts, metric_status=statuses)
    counters, e_blocks = [], []
    for row, region in entries:
        correct, total = _number(region.get("correct_tokens")), _number(region.get("token_count"))
        if correct is None or total is None:
            continue
        correct = _integer(correct, "correct_tokens")
        total = _integer(total, "token_count")
        if correct > total:
            raise ValueError("correct_tokens cannot exceed token_count")
        counters.append((correct, total))
        if row["actions"][-1] == "E" and total == 8:
            e_blocks.append(correct)
    correct_sum, token_sum = sum(c for c, _ in counters), sum(t for _, t in counters)
    result.update(
        correct_tokens=correct_sum if counters else None,
        token_count=token_sum if counters else None,
        counter_count=len(counters), counter_status="ok" if counters else "missing",
        pooled_token_agreement=correct_sum / token_sum if token_sum else None,
    )
    result["e_block"] = {
        "count": len(e_blocks), "status": "ok" if e_blocks else "missing",
        "ge4_of8": _mean([float(c >= 4) for c in e_blocks]),
        "ge6_of8": _mean([float(c >= 6) for c in e_blocks]),
        "ge7_of8": _mean([float(c >= 7) for c in e_blocks]),
        "eq8_of8": _mean([float(c == 8) for c in e_blocks]),
    }
    return result


def _dynamics_group(rows, region_names):
    entries = defaultdict(list)
    for row in rows:
        for name, region in row["regions"].items():
            if region is not None:
                entries[name].append((row, region))
    return {
        "count": len(rows), "question_count": len({_question(row) for row in rows}),
        "status": "ok" if rows else "empty",
        "regions": {name: _region_summary(entries[name]) for name in region_names},
    }


def dynamics_report(rows):
    """Group supplied region diagnostics without inventing missing targets.

    Returns all/by_horizon (H1,H2,H3 and any further observed horizons),
    by_action_sequence, by_action (FINAL action, not duplicated per step),
    by_length (when supplied), and by_change (optional changed/unchanged tags).
    Each region has scalar means, per-metric counts/status, pooled counters,
    and E-block success rates. Scalar means weight observed rows equally;
    pooled_token_agreement is separately weighted by supplied token counters.

    E-block rates are computed per supplied region ONLY for E-ending rows with
    explicit correct_tokens and token_count == 8. Regions must identify the
    intended block; no block identity or token counters are inferred from names,
    token_agreement, missing metrics, or larger regions. Missing region/metric
    entries remain None with status='missing' and count zero in every group.
    Optional tags: change='changed'/'unchanged', tags containing these strings,
    or boolean changed/unchanged fields. No change is inferred from errors.
    """
    prepared, horizons, sequences, lengths, changes = [], defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    actions, region_names = {"R": [], "E": []}, set()
    for original in rows:
        row = dict(original)
        horizon = _integer(row.get("horizon"), "horizon", minimum=1)
        sequence = _action_sequence(row)
        if len(sequence) != horizon:
            raise ValueError("horizon must equal the action-sequence length")
        _question(row)
        regions = row.get("regions")
        if not isinstance(regions, Mapping):
            raise ValueError("regions must be a mapping")
        for name, region in regions.items():
            if not isinstance(name, str) or (region is not None and not isinstance(region, Mapping)):
                raise ValueError("regions must map string names to metric mappings or None")
            region_names.add(name)
        prepared.append(row)
        horizons[f"H{horizon}"].append(row)
        sequences[sequence].append(row)
        actions[sequence[-1]].append(row)
        if row.get("length") is not None:
            length = _integer(row["length"], "length", minimum=1)
            lengths[_length_bucket(length)].append(row)
        for tag in _change_tags(row):
            changes[tag].append(row)
    names = sorted(region_names)
    horizon_keys = ["H1", "H2", "H3"] + sorted(set(horizons) - {"H1", "H2", "H3"})
    length_keys = list(_LENGTH_BUCKETS) + sorted(set(lengths) - set(_LENGTH_BUCKETS)) if lengths else []
    def groups(mapping, keys):
        return {key: _dynamics_group(mapping[key], names) for key in keys}
    return {
        "all": _dynamics_group(prepared, names),
        "by_horizon": groups(horizons, horizon_keys),
        "by_action_sequence": groups(sequences, sorted(sequences)),
        "by_action": groups(actions, ("R", "E")),
        "by_length": groups(lengths, length_keys),
        "by_change": groups(changes, ("unchanged", "changed")) if changes else {},
    }


def _json_safe(value):
    """Convert NumPy-compatible scalars/arrays and nonfinite numbers recursively."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = _json_safe(key)
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError("JSON object keys must be scalar")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return _json_safe(converted)
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return _json_safe(scalar)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _atomic_write(path, emit):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            emit(stream)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def write_json(path, data):
    """Atomically write strict UTF-8 JSON; NaN/Inf and undefined stats become null.

    Returns the destination Path. The parent directory must exist. NumPy
    scalars/arrays are supported without importing NumPy. Unsupported values
    raise TypeError, preserving an existing destination.
    """
    safe = _json_safe(data)
    def emit(stream):
        json.dump(safe, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return _atomic_write(path, emit)


def write_jsonl(path, rows):
    """Atomically write strict JSON, one sanitized object per line, streaming rows."""
    def emit(stream):
        for row in rows:
            stream.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    return _atomic_write(path, emit)


def write_csv(path, rows):
    """Atomically write dict rows with a union of columns in first-seen order.

    Missing/nonfinite values produce empty cells; nested data use strict JSON
    cells. An empty row sequence writes an empty file. Returns destination Path.
    """
    safe = []
    columns = {}
    for row in rows:
        if not isinstance(row, Mapping) or not all(isinstance(k, str) for k in row):
            raise ValueError("CSV rows must be mappings with string column names")
        converted = _json_safe(row)
        safe.append(converted)
        columns.update(dict.fromkeys(converted))
    def emit(stream):
        if not columns:
            return
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in safe:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    return _atomic_write(path, emit)
