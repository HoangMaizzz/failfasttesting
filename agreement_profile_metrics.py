"""Prompt-macro scores, dense-tail/change diagnostics and paired uncertainty."""
from collections import defaultdict
import numpy as np

from agreement_profile_data import prefix_length


def binary_metrics(y, scores):
    y, scores = np.asarray(y, bool), np.asarray(scores, float)
    if not len(y):
        return dict(accuracy=None, balanced_accuracy=None, auroc=None, mismatch_auprc=None)
    p = scores >= 0
    result = dict(accuracy=float(np.mean(p == y)))
    if not y.any() or y.all():
        return dict(result, balanced_accuracy=None, auroc=None, mismatch_auprc=None)
    result["balanced_accuracy"] = float((p[y].mean() + (~p[~y]).mean()) / 2)
    # Average ranks handle tied scores correctly.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(y), float)
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(y)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[a:b]] = (a + 1 + b) / 2
    pos, neg = int(y.sum()), int((~y).sum())
    result["auroc"] = float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
    # Average precision for mismatch, grouping ties at one common threshold.
    order = np.argsort(scores, kind="stable")  # smaller margin = greater mismatch score
    bad = (~y)[order]
    endpoints = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(y)]
    cum_bad = np.cumsum(bad)
    last_recall, ap = 0., 0.
    for end in endpoints:
        recall = cum_bad[end - 1] / neg
        ap += (recall - last_recall) * cum_bad[end - 1] / end
        last_recall = recall
    result["mismatch_auprc"] = float(ap)
    return result


def prediction_row(source, target, margin, K_head=None, parent=None):
    margin = np.asarray(margin, np.float32)
    K_margin = prefix_length(margin >= 0)
    row = dict(state_id=source.uid, target_id=target.uid, question=source.question,
               split=source.split, length=target.length,
               margin_pred=margin.tolist(), margin_true=target.margin.tolist(),
               local_match=target.matches.tolist(), K_true=target.accepted,
               K_margin=K_margin, K_head=K_head,
               source_K_true=source.accepted,
               masked=(source.native == 151665).tolist(),
               refinement_index=source.meta.get("refinement_index", 0),
               feature_stale=(source.features["metadata"][:, 5:7].max(1) > 0).tolist())
    if parent is not None:
        changed = parent.candidate != target.candidate
        where = np.flatnonzero(changed)
        first = int(where[0]) if len(where) else target.length
        influenced = np.arange(target.length) >= first
        row.update(candidate_changed=changed.tolist(), prefix_affected=influenced.tolist(),
                   yield_group=("blocked_before_active_segment" if parent.accepted <
                                parent.meta.get("segment_start", 0) else
                                "first_rejection_reachable_in_active_segment"),
                   change_group="changed" if changed.any() else "unchanged",
                   delta_K_true=target.accepted - parent.accepted,
                   source_margin_true=parent.margin.tolist(),
                   gain_group="gain" if target.accepted > parent.accepted else
                              "loss" if target.accepted < parent.accepted else "same")
    return row


def per_prompt(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        pred_K = row["K_head"] if key == "K_head_mae" else row["K_margin"]
        if key in ("K_mae", "K_head_mae"):
            if pred_K is None:
                continue
            value = abs(pred_K - row["K_true"])
        elif key in ("margin_mae", "affected_margin_mae"):
            errors = np.abs(np.asarray(row["margin_pred"]) - row["margin_true"])
            if key == "affected_margin_mae":
                affected = np.asarray(row.get("prefix_affected", []), bool)
                if not affected.any():
                    continue
                errors = errors[affected]
            value = errors.mean()
        else:
            raise ValueError(key)
        grouped[row["question"]].append(float(value))
    return {q: float(np.mean(values)) for q, values in grouped.items()}


def score_rows(rows):
    if not rows:
        return dict(states=0, prompts=0, status="no_samples")
    if all(row.get("margin_supervised") is False for row in rows):
        byq = per_prompt(rows, "K_head_mae")
        errors = [r["K_head"] - r["K_true"] for r in rows]
        return dict(states=len(rows), prompts=len(byq),
            macro_K_head_mae=float(np.mean(list(byq.values()))),
            K_head_exact=float(np.mean(np.asarray(errors) == 0)),
            K_head_bias=float(np.mean(errors)), margin_metrics="not_applicable_K_only_head",
            future_gain=_future_gain(rows, "K_head"), yield_groups=_yield_groups(rows))
    per_position = defaultdict(lambda: ([], []))
    y, pred = [], []
    K_errors, all_accepted, signed_errors = [], [], []
    for row in rows:
        true_margin = np.asarray(row["margin_true"])
        p = np.asarray(row["margin_pred"])
        matches = np.asarray(row["local_match"], bool)
        K = row["K_true"]
        masks = np.asarray(row["masked"], bool)
        regions = dict(before_rejection=np.arange(len(p)) < K,
                       first_rejection=np.arange(len(p)) == K,
                       after_rejection=np.arange(len(p)) > K,
                       masked=masks, committed=~masks,
                       stale=np.asarray(row["feature_stale"], bool),
                       near_zero=np.abs(true_margin) <= row.get("near_zero_threshold", .25))
        if "candidate_changed" in row:
            regions.update(candidate_changed=np.asarray(row["candidate_changed"], bool),
                           prefix_affected=np.asarray(row["prefix_affected"], bool))
        for name, mask in regions.items():
            per_position[name][0].extend(np.abs(p[mask] - true_margin[mask]).tolist())
            per_position[name][1].extend((p[mask] >= 0) == matches[mask])
        y.extend(matches)
        pred.extend(p)
        K_errors.append(abs(row["K_margin"] - K))
        signed_errors.append(row["K_margin"] - K)
        if K == row["length"]:
            all_accepted.append(row["K_margin"] == K)
    macro = {name: float(np.mean(list(per_prompt(rows, name).values())))
             for name in ("margin_mae", "K_mae")}
    k_head = per_prompt(rows, "K_head_mae")
    groups = {}
    for field in ("length", "refinement_index", "gain_group", "change_group"):
        values = sorted({row[field] for row in rows if field in row}, key=str)
        groups[field] = {}
        for value in values:
            subset = [row for row in rows if row.get(field) == value]
            byq = per_prompt(subset, "K_mae")
            groups[field][str(value)] = dict(states=len(subset), prompts=len(byq),
                macro_K_mae=float(np.mean(list(byq.values()))),
                macro_margin_mae=float(np.mean(list(per_prompt(subset, "margin_mae").values()))))
    return dict(states=len(rows), prompts=len({r["question"] for r in rows}),
                macro=macro, macro_K_head_mae=float(np.mean(list(k_head.values()))) if k_head else None,
                micro=dict(K_mae=float(np.mean(K_errors)), K_bias=float(np.mean(signed_errors)),
                    first_rejection_exact=float(np.mean(np.asarray(K_errors) == 0)),
                    within_one=float(np.mean(np.asarray(K_errors) <= 1)),
                    overestimate_rate=float(np.mean(np.asarray(signed_errors) > 0)),
                    all_accepted_exact=float(np.mean(all_accepted)) if all_accepted else None,
                    **binary_metrics(y, pred)),
                future_gain=_future_gain(rows, "K_margin"),
                future_gain_K_head=_future_gain(rows, "K_head") if k_head else None,
                regions={name: dict(tokens=len(errors), margin_mae=float(np.mean(errors)) if errors else None,
                    sign_accuracy=float(np.mean(signs)) if signs else None)
                    for name, (errors, signs) in per_position.items()}, groups=groups,
                yield_groups=_yield_groups(rows))


def _yield_groups(rows):
    """Teacher-derived evaluation strata; never inputs or collection filters."""
    result = {}
    for group in sorted({r["yield_group"] for r in rows if "yield_group" in r}):
        subset = [r for r in rows if r.get("yield_group") == group]
        value = dict(states=len(subset), prompts=len({r["question"] for r in subset}))
        for metric in ("K_head_mae", "K_mae", "affected_margin_mae"):
            if metric != "K_head_mae" and all(r.get("margin_supervised") is False for r in subset):
                continue
            byq = per_prompt(subset, metric)
            if byq:
                value["macro_" + metric] = float(np.mean(list(byq.values())))
        key = "K_head" if all(r.get("margin_supervised") is False for r in subset) else "K_margin"
        value["future_gain"] = _future_gain(subset, key)
        result[group] = value
    return result


def _future_gain(rows, K_key):
    subset = [r for r in rows if "delta_K_true" in r and "source_K_pred" in r
              and r.get(K_key) is not None]
    if not subset:
        return None
    errors, baseline_errors, truth, predicted, deltas = defaultdict(list), defaultdict(list), [], [], []
    for r in subset:
        delta = r[K_key] - r["source_K_pred"]
        errors[r["question"]].append(abs(delta - r["delta_K_true"]))
        baseline_errors[r["question"]].append(abs(r["delta_K_true"]))
        deltas.append(r["delta_K_true"])
        truth.append(r["delta_K_true"] > 0)
        predicted.append(delta)
    binary = binary_metrics(truth, np.asarray(predicted) - .5)
    return dict(prompts=len(errors), macro_delta_K_mae=float(np.mean([np.mean(v) for v in errors.values()])),
                gain_accuracy=binary["accuracy"], gain_balanced_accuracy=binary["balanced_accuracy"],
                gain_auroc=binary["auroc"], gain_count=int(np.sum(truth)),
                loss_count=int(np.sum(np.asarray(deltas) < 0)),
                same_count=int(np.sum(np.asarray(deltas) == 0)),
                zero_delta_baseline_macro_mae=float(np.mean([np.mean(v) for v in baseline_errors.values()])),
                always_no_gain_accuracy=float(np.mean(~np.asarray(truth, bool))),
                scope="yield_change_prediction_no_cost_or_policy")


def paired_comparison(left, right, metric="margin_mae", draws=2000, seed=42, right_metric=None):
    a, b = per_prompt(left, metric), per_prompt(right, right_metric or metric)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return dict(status="no_common_prompts")
    # Require matched target rows, not merely matched prompt IDs.
    ka = {(r["state_id"], r["target_id"]) for r in left}
    kb = {(r["state_id"], r["target_id"]) for r in right}
    if ka != kb:
        raise ValueError("Paired comparison has mismatched evaluation states")
    delta = np.asarray([a[q] - b[q] for q in keys])
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(1)
    return dict(metric=metric, right_metric=right_metric or metric,
                difference="left_minus_right_negative_is_better",
                prompts=len(keys), mean=float(delta.mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                exploratory=len(keys) < 30)
