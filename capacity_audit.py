import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


COMPACT6 = (
    "bias",
    "prefix_advance_ratio",
    "failfast_margin",
    "accumulated_spec_ratio",
    "draft_verify_latency_ratio",
    "ema_tokens_per_verifier_ratio",
)
COMPACT4 = COMPACT6[:4]


def assign_native_oracle_label(delta_j_ms, epsilon_ms=1.0):
    delta_j_ms = float(delta_j_ms)
    epsilon_ms = float(epsilon_ms)
    if abs(delta_j_ms) <= epsilon_ms:
        return "tie"
    return "stop" if delta_j_ms > 0.0 else "continue"


def action_regret_ms(delta_j_ms, action):
    delta_j_ms = float(delta_j_ms)
    if action == "stop":
        return max(0.0, -delta_j_ms)
    if action == "continue":
        return max(0.0, delta_j_ms)
    raise ValueError(f"unknown action: {action}")


def local_regret_capture(model_regret_ms, failfast_regret_ms, atol=1e-9):
    denominator = float(failfast_regret_ms)
    if denominator <= atol:
        return math.nan
    return 1.0 - float(model_regret_ms) / denominator


def _probe_matrix(frame, probe):
    if probe == "P0_failfast_margin":
        names = ("failfast_margin",)
    elif probe == "P1_compact6_linear":
        names = COMPACT6[1:]
    elif probe == "P2_compact4_linear":
        names = COMPACT4[1:]
    elif probe == "P3_compact6_interactions":
        base = frame[[f"feature_{name}" for name in COMPACT6[1:]]].to_numpy(float)
        prefix, margin, length = base[:, 0], base[:, 1], base[:, 2]
        interactions = np.column_stack((
            prefix * margin,
            prefix * length,
            margin * length,
        ))
        return np.column_stack((base, interactions))
    elif probe == "P4_compact6_mlp":
        names = COMPACT6[1:]
    else:
        raise ValueError(f"unknown probe: {probe}")
    return frame[[f"feature_{name}" for name in names]].to_numpy(float)


def _candidate_parameters(probe):
    if probe == "P4_compact6_mlp":
        return (1e-4, 1e-3, 1e-2)
    return (0.01, 0.1, 1.0, 10.0)


def _make_estimator(probe, parameter, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if probe == "P4_compact6_mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            alpha=float(parameter),
            learning_rate_init=0.005,
            solver="lbfgs",
            max_iter=500,
            early_stopping=False,
            random_state=int(seed),
        )
    else:
        model = LogisticRegression(
            C=float(parameter),
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )
    return Pipeline((
        ("scale", StandardScaler()),
        ("model", model),
    ))


def _group_splits(groups, requested=5):
    from sklearn.model_selection import GroupKFold

    unique = np.unique(groups)
    splits = min(int(requested), len(unique))
    if splits < 2:
        raise ValueError("capacity audit requires at least two problem groups")
    return GroupKFold(n_splits=splits)


def _safe_auc(y_true, score):
    from sklearn.metrics import roc_auc_score

    return (
        float(roc_auc_score(y_true, score))
        if len(np.unique(y_true)) == 2
        else math.nan
    )


def _select_parameter(probe, x, y, groups, seed):
    best = None
    splitter = _group_splits(groups, requested=3)
    for parameter in _candidate_parameters(probe):
        scores = []
        for train, valid in splitter.split(x, y, groups):
            if len(np.unique(y[train])) < 2:
                continue
            estimator = _make_estimator(probe, parameter, seed)
            estimator.fit(x[train], y[train])
            scores.append(_safe_auc(y[valid], estimator.predict_proba(x[valid])[:, 1]))
        finite = [value for value in scores if math.isfinite(value)]
        score = float(np.mean(finite)) if finite else -math.inf
        candidate = (score, -float(parameter), float(parameter))
        if best is None or candidate > best:
            best = candidate
    return best[2]


def _fit_platt(raw_probability, labels):
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if len(np.unique(labels)) < 2:
        return None
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(logits, labels)
    return calibrator


def _apply_platt(calibrator, raw_probability):
    if calibrator is None:
        return np.asarray(raw_probability, dtype=float)
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _cross_fitted_probe(frame, probe, seed):
    x = _probe_matrix(frame, probe)
    y = frame["oracle_stop"].to_numpy(int)
    groups = frame["problem_group"].to_numpy(str)
    probabilities = np.full(len(frame), np.nan)
    selected_parameters = []
    outer = _group_splits(groups, requested=5)
    for fold, (train, test) in enumerate(outer.split(x, y, groups)):
        parameter = _select_parameter(
            probe, x[train], y[train], groups[train], seed + fold
        )
        selected_parameters.append(parameter)
        inner_probability = np.full(len(train), np.nan)
        inner = _group_splits(groups[train], requested=3)
        for inner_train, inner_valid in inner.split(
            x[train], y[train], groups[train]
        ):
            estimator = _make_estimator(probe, parameter, seed + fold)
            estimator.fit(x[train][inner_train], y[train][inner_train])
            inner_probability[inner_valid] = estimator.predict_proba(
                x[train][inner_valid]
            )[:, 1]
        calibrator = _fit_platt(inner_probability, y[train])
        estimator = _make_estimator(probe, parameter, seed + fold)
        estimator.fit(x[train], y[train])
        raw_test = estimator.predict_proba(x[test])[:, 1]
        probabilities[test] = _apply_platt(calibrator, raw_test)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"probe {probe} did not produce complete OOF predictions")
    return probabilities, selected_parameters


def _metric_row(frame, probe, probabilities, scope):
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
    )

    y = frame["oracle_stop"].to_numpy(int)
    predicted_stop = probabilities >= 0.5
    weights = frame["DeltaJ_ms"].abs().to_numpy(float)
    weighted_accuracy = float(
        np.average(predicted_stop == y, weights=np.maximum(weights, 1e-9))
    )
    model_actions = np.where(predicted_stop, "stop", "continue")
    model_regret = sum(
        action_regret_ms(delta, action)
        for delta, action in zip(frame["DeltaJ_ms"], model_actions)
    )
    failfast_regret = sum(
        action_regret_ms(delta, "continue") for delta in frame["DeltaJ_ms"]
    )
    return {
        "scope": scope,
        "probe": probe,
        "states": int(len(frame)),
        "problems": int(frame["problem_group"].nunique()),
        "roc_auc": _safe_auc(y, probabilities),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted_stop)),
        "brier": float(brier_score_loss(y, probabilities)),
        "margin_weighted_accuracy": weighted_accuracy,
        "model_local_regret_ms": float(model_regret),
        "failfast_local_regret_ms": float(failfast_regret),
        "statewise_local_regret_capture": local_regret_capture(
            model_regret, failfast_regret
        ),
    }


def _safe_stop_rows(frame, probe, probabilities, scope):
    y = frame["oracle_stop"].to_numpy(int)
    rows = []
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        selected = probabilities >= threshold
        rows.append({
            "scope": scope,
            "probe": probe,
            "stop_probability_threshold": threshold,
            "stop_coverage": float(np.mean(selected)),
            "stop_precision": (
                float(np.mean(y[selected])) if selected.any() else math.nan
            ),
            "selected_states": int(selected.sum()),
        })
    return rows


def _reliability_rows(frame, probe, probabilities, scope):
    y = frame["oracle_stop"].to_numpy(int)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    rows = []
    for bucket in range(10):
        selected = bins == bucket
        if not selected.any():
            continue
        rows.append({
            "scope": scope,
            "probe": probe,
            "probability_bin": bucket,
            "states": int(selected.sum()),
            "mean_predicted_stop_probability": float(probabilities[selected].mean()),
            "empirical_stop_rate": float(y[selected].mean()),
        })
    return rows


def _group_bootstrap(frame, probabilities, seed, repetitions):
    rng = np.random.default_rng(seed)
    groups = frame["problem_group"].unique()
    y = frame["oracle_stop"].to_numpy(int)
    delta = frame["DeltaJ_ms"].to_numpy(float)
    group_values = frame["problem_group"].to_numpy(str)
    aucs, captures = [], []
    for _ in range(int(repetitions)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([
            np.flatnonzero(group_values == group) for group in sampled
        ])
        if len(np.unique(y[indices])) == 2:
            aucs.append(_safe_auc(y[indices], probabilities[indices]))
        actions = np.where(probabilities[indices] >= 0.5, "stop", "continue")
        model_regret = sum(
            action_regret_ms(value, action)
            for value, action in zip(delta[indices], actions)
        )
        failfast_regret = sum(
            action_regret_ms(value, "continue") for value in delta[indices]
        )
        capture = local_regret_capture(model_regret, failfast_regret)
        if math.isfinite(capture):
            captures.append(capture)
    def interval(values):
        return (
            float(np.percentile(values, 2.5)) if values else math.nan,
            float(np.percentile(values, 97.5)) if values else math.nan,
        )
    auc_low, auc_high = interval(aucs)
    capture_low, capture_high = interval(captures)
    return {
        "roc_auc_ci95_low": auc_low,
        "roc_auc_ci95_high": auc_high,
        "local_regret_capture_ci95_low": capture_low,
        "local_regret_capture_ci95_high": capture_high,
    }


def _profile_validation(decisions):
    rows = []
    for branch in ("stop", "continue"):
        predicted = decisions[f"{branch}_predicted_verify_ms"].to_numpy(float)
        measured = decisions[f"{branch}_measured_verify_ms"].to_numpy(float)
        error = predicted - measured
        rows.append({
            "branch": branch,
            "observations": int(len(error)),
            "bias_ms": float(error.mean()),
            "mae_ms": float(np.abs(error).mean()),
            "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
            "p95_absolute_error_ms": float(np.percentile(np.abs(error), 95)),
            "pearson": (
                float(np.corrcoef(predicted, measured)[0, 1])
                if np.std(predicted) > 0 and np.std(measured) > 0
                else math.nan
            ),
        })
    if "selected_path_measured_verify_ms" in decisions.columns:
        first_per_round = decisions.drop_duplicates(
            ["dataset", "sample_id", "round_id"]
        )
        repeated_error = (
            first_per_round["continue_measured_verify_ms"].to_numpy(float)
            - first_per_round["selected_path_measured_verify_ms"].to_numpy(float)
        )
        rows.append({
            "branch": "continue_repeated_measurement_noise",
            "observations": int(len(repeated_error)),
            "bias_ms": float(repeated_error.mean()),
            "mae_ms": float(np.abs(repeated_error).mean()),
            "rmse_ms": float(np.sqrt(np.mean(repeated_error ** 2))),
            "p95_absolute_error_ms": float(
                np.percentile(np.abs(repeated_error), 95)
            ),
            "pearson": math.nan,
        })
    return pd.DataFrame(rows)


def run_capacity_audit(decisions, output_dir, epsilon_ms=1.0, seed=42, bootstrap=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = decisions.copy()
    frame["oracle_label"] = frame["DeltaJ_ms"].map(
        lambda value: assign_native_oracle_label(value, epsilon_ms)
    )
    frame["problem_group"] = (
        frame["dataset"].astype(str) + ":" + frame["sample_id"].astype(str)
    )
    frame["oracle_stop"] = frame["oracle_label"].eq("stop").astype(int)
    frame.to_csv(output_dir / "capacity_states_all.csv", index=False)
    evaluated = frame.loc[frame["oracle_label"] != "tie"].reset_index(drop=True)
    if evaluated["oracle_stop"].nunique() < 2:
        raise RuntimeError("capacity audit requires both STOP and CONTINUE labels")

    probes = (
        "P0_failfast_margin",
        "P1_compact6_linear",
        "P2_compact4_linear",
        "P3_compact6_interactions",
        "P4_compact6_mlp",
    )
    metric_rows, safe_rows, reliability_rows, oof_frames = [], [], [], []
    for index, probe in enumerate(probes):
        probabilities, parameters = _cross_fitted_probe(
            evaluated, probe, seed + 100 * index
        )
        scopes = [("all_datasets", np.ones(len(evaluated), dtype=bool))]
        scopes.extend(
            (dataset, evaluated["dataset"].eq(dataset).to_numpy())
            for dataset in sorted(evaluated["dataset"].unique())
        )
        for scope, selected in scopes:
            scoped = evaluated.loc[selected].reset_index(drop=True)
            scoped_probabilities = probabilities[selected]
            metric = _metric_row(scoped, probe, scoped_probabilities, scope)
            metric.update(_group_bootstrap(
                scoped,
                scoped_probabilities,
                seed + index,
                bootstrap,
            ))
            metric["selected_parameters"] = json.dumps(parameters)
            metric_rows.append(metric)
            safe_rows.extend(_safe_stop_rows(
                scoped, probe, scoped_probabilities, scope
            ))
            reliability_rows.extend(_reliability_rows(
                scoped, probe, scoped_probabilities, scope
            ))
        oof = evaluated[[
            "dataset", "sample_id", "round_id", "decision_id",
            "state_key", "oracle_label", "DeltaJ_ms",
        ]].copy()
        oof["probe"] = probe
        oof["oof_stop_probability"] = probabilities
        oof["oof_action"] = np.where(probabilities >= 0.5, "stop", "continue")
        oof_frames.append(oof)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "probe_metrics.csv", index=False)
    pd.DataFrame(safe_rows).to_csv(
        output_dir / "safe_stop_precision_coverage.csv", index=False
    )
    pd.DataFrame(reliability_rows).to_csv(
        output_dir / "reliability.csv", index=False
    )
    pd.concat(oof_frames, ignore_index=True).to_csv(
        output_dir / "probe_oof_predictions.csv", index=False
    )
    _profile_validation(frame).to_csv(
        output_dir / "frozen_profile_validation.csv", index=False
    )
    label_summary = frame.groupby(["dataset", "oracle_label"], as_index=False).size()
    label_summary.to_csv(output_dir / "native_label_summary.csv", index=False)
    return metrics
