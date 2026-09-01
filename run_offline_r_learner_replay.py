#!/usr/bin/env python3
"""Replay a cross-fitted R-learner on factual adaptive-controller logs.

Exact oracle labels are used only after each held-out prediction has been made.
The nuisance outcome model and treatment-effect model are trained exclusively on
factual behavior rows from other problem IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


BLOCK_SIZE = 8
ACTION_CONTINUE = "continue"
ACTION_STOP = "stop"
FEATURE_NAMES = (
    "active_mask_ratio",
    "masked_entropy_std",
    "previous_resolved_margin_mean",
    "previous_resolved_top1_mean",
    "previous_resolved_entropy_max",
    "ema_tokens_per_verifier_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--nuisance-alpha", type=float, default=10.0)
    parser.add_argument("--effect-alpha", type=float, default=10.0)
    parser.add_argument(
        "--outcome-mode",
        choices=("logged_factual", "frozen_rho_measured_latency"),
        default="logged_factual",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-scope",
        choices=("pooled", "dataset"),
        default="pooled",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def aggregate_features(frame: pd.DataFrame) -> np.ndarray:
    required = ["ema_tokens_per_verifier_ratio"]
    for phase in ("current", "previous"):
        for position in range(BLOCK_SIZE):
            required.extend(
                [
                    f"{phase}_position_{position}_mask_indicator",
                    f"{phase}_position_{position}_observed_indicator",
                    f"{phase}_position_{position}_top1_probability",
                    f"{phase}_position_{position}_top2_probability",
                    f"{phase}_position_{position}_normalized_entropy",
                ]
            )
    require_columns(frame, required, "raw-state frame")

    current_mask = np.column_stack(
        [
            frame[f"current_position_{position}_mask_indicator"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )
    current_entropy = np.column_stack(
        [
            frame[f"current_position_{position}_normalized_entropy"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )
    previous_mask = np.column_stack(
        [
            frame[f"previous_position_{position}_mask_indicator"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )
    previous_top1 = np.column_stack(
        [
            frame[f"previous_position_{position}_top1_probability"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )
    previous_top2 = np.column_stack(
        [
            frame[f"previous_position_{position}_top2_probability"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )
    previous_entropy = np.column_stack(
        [
            frame[f"previous_position_{position}_normalized_entropy"].to_numpy(float)
            for position in range(BLOCK_SIZE)
        ]
    )

    observed_columns = [
        f"{phase}_position_{position}_observed_indicator"
        for phase in ("current", "previous")
        for position in range(BLOCK_SIZE)
    ]
    if not (frame[observed_columns].to_numpy(float) == 1.0).all():
        raise ValueError("R-learner replay requires complete raw-state observations")

    def selected_stat(
        values: np.ndarray,
        selected: np.ndarray,
        function,
    ) -> np.ndarray:
        result = []
        for row, row_selected in zip(values, selected):
            chosen = row[row_selected]
            result.append(function(chosen) if chosen.size else np.nan)
        return np.asarray(result, dtype=float)

    current_unresolved = current_mask == 1.0
    previous_resolved = previous_mask == 0.0
    active_mask_ratio = current_mask.sum(axis=1) / BLOCK_SIZE
    masked_entropy_std = selected_stat(
        current_entropy,
        current_unresolved,
        np.std,
    )
    previous_resolved_margin_mean = selected_stat(
        previous_top1 - previous_top2,
        previous_resolved,
        np.mean,
    )
    previous_resolved_top1_mean = selected_stat(
        previous_top1,
        previous_resolved,
        np.mean,
    )
    previous_resolved_entropy_max = selected_stat(
        previous_entropy,
        previous_resolved,
        np.max,
    )
    verifier_yield = frame["ema_tokens_per_verifier_ratio"].to_numpy(float)
    return np.column_stack(
        [
            active_mask_ratio,
            masked_entropy_std,
            previous_resolved_margin_mean,
            previous_resolved_top1_mean,
            previous_resolved_entropy_max,
            verifier_yield,
        ]
    )


def nuisance_features(frame: pd.DataFrame) -> np.ndarray:
    """Use the behavior policy's full observable state for outcome residuals."""
    columns = []
    for phase in ("previous", "current"):
        for position in range(BLOCK_SIZE):
            columns.extend(
                [
                    f"{phase}_position_{position}_mask_indicator",
                    f"{phase}_position_{position}_observed_indicator",
                    f"{phase}_position_{position}_top1_probability",
                    f"{phase}_position_{position}_top2_probability",
                    f"{phase}_position_{position}_normalized_entropy",
                    f"{phase}_position_{position}_normalized_position",
                ]
            )
    columns.extend(
        [
            "accumulated_spec_ratio",
            "draft_verify_latency_ratio",
            "ema_tokens_per_verifier_ratio",
            "normalized_refinement_step",
            "has_previous_state",
            "continue_propensity",
        ]
    )
    require_columns(frame, columns, "nuisance state")
    return frame[columns].to_numpy(float)


def load_behavior(input_dir: Path, outcome_mode: str) -> pd.DataFrame:
    files = sorted(
        input_dir.glob("raw/*/raw_behavior/adaptive_td_decisions.csv")
    )
    if not files:
        raise FileNotFoundError("no factual adaptive_td_decisions.csv files found")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "dataset" not in frame:
            frame["dataset"] = path.parent.parent.name
        frames.append(frame)
    behavior = pd.concat(frames, ignore_index=True)
    require_columns(
        behavior,
        [
            "dataset",
            "problem_id",
            "executed_action",
            "behavior_stop_probability",
            "selected_action_probability",
            "factual_target",
            "factual_delta_time_ms",
            "emitted_tokens",
            "factual_boundary_id",
            "factual_update_applied",
            "raw_state_complete",
        ],
        "factual behavior log",
    )
    behavior = behavior[
        behavior["executed_action"].isin([ACTION_STOP, ACTION_CONTINUE])
        & behavior["factual_update_applied"].astype(bool)
        & behavior["raw_state_complete"].astype(bool)
        & behavior["factual_target"].notna()
    ].copy()
    behavior["group"] = (
        behavior["dataset"].astype(str)
        + "/"
        + behavior["problem_id"].astype(str)
    )
    behavior["continue_action"] = behavior["executed_action"].eq(
        ACTION_CONTINUE
    ).astype(float)
    behavior["continue_propensity"] = (
        1.0 - behavior["behavior_stop_probability"].astype(float)
    )
    reconstructed = np.where(
        behavior["continue_action"].eq(1.0),
        behavior["selected_action_probability"].astype(float),
        1.0 - behavior["selected_action_probability"].astype(float),
    )
    if not np.allclose(
        reconstructed,
        behavior["continue_propensity"].to_numpy(float),
        atol=1e-8,
    ):
        raise ValueError("logged action propensity is inconsistent")
    propensity = behavior["continue_propensity"].to_numpy(float)
    if np.any((propensity <= 0.0) | (propensity >= 1.0)):
        raise ValueError("R-learner replay requires overlap for both actions")
    behavior["boundary_key"] = (
        behavior["group"]
        + "/"
        + behavior["factual_boundary_id"].astype(str)
    )
    boundary_size = behavior.groupby("boundary_key")["boundary_key"].transform(
        "size"
    )
    behavior["boundary_weight"] = 1.0 / boundary_size.astype(float)
    if outcome_mode == "logged_factual":
        behavior["replay_outcome"] = behavior["factual_target"].astype(float)
    else:
        profile_rho = {}
        for dataset in behavior["dataset"].unique():
            profile_path = input_dir / str(dataset) / "frozen_hardware_profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile_rho[str(dataset)] = float(profile["rho_tokens_per_ms"])
        rho = behavior["dataset"].astype(str).map(profile_rho).to_numpy(float)
        behavior["replay_outcome"] = (
            behavior["emitted_tokens"].to_numpy(float)
            - rho * behavior["factual_delta_time_ms"].to_numpy(float)
        )
    return behavior.reset_index(drop=True)


def load_oracle(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "final_report" / "all_exact_alignment.csv"
    oracle = pd.read_csv(path)
    require_columns(
        oracle,
        [
            "dataset",
            "problem_id",
            "oracle_label",
            "oracle_advantage_tokens",
            "behavior_advantage",
        ],
        "exact oracle alignment",
    )
    oracle = oracle[
        oracle["oracle_label"].isin([ACTION_STOP, ACTION_CONTINUE])
    ].copy()
    oracle["group"] = (
        oracle["dataset"].astype(str)
        + "/"
        + oracle["problem_id"].astype(str)
    )
    oracle["continue_label"] = oracle["oracle_label"].eq(
        ACTION_CONTINUE
    ).astype(int)
    oracle["oracle_continue_advantage"] = -oracle[
        "oracle_advantage_tokens"
    ].astype(float)
    expected_label = oracle["oracle_continue_advantage"] > 0.0
    if not np.array_equal(
        expected_label.to_numpy(),
        oracle["continue_label"].astype(bool).to_numpy(),
    ):
        raise ValueError("oracle advantage sign does not match oracle label")
    oracle["shared_va_continue_advantage"] = -oracle[
        "behavior_advantage"
    ].astype(float)
    return oracle.reset_index(drop=True)


def cross_fitted_nuisance(
    features: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    folds: int,
    alpha: float,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    split_count = min(int(folds), len(unique_groups))
    if split_count < 2:
        raise ValueError("at least two factual problem groups are required")
    predictions = np.full(len(outcomes), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=split_count)
    for train_index, validation_index in splitter.split(
        features,
        outcomes,
        groups,
    ):
        imputer = SimpleImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(features[train_index]))
        x_validation = scaler.transform(imputer.transform(features[validation_index]))
        model = Ridge(alpha=alpha)
        model.fit(
            x_train,
            outcomes[train_index],
            sample_weight=weights[train_index],
        )
        predictions[validation_index] = model.predict(x_validation)
    if not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitted nuisance predictions are incomplete")
    return predictions


def fit_effect_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    inner_folds: int,
    nuisance_alpha: float,
    effect_alpha: float,
) -> np.ndarray:
    x_train = aggregate_features(train)
    x_test = aggregate_features(test)
    nuisance_train = nuisance_features(train)
    outcome = train["replay_outcome"].to_numpy(float)
    action = train["continue_action"].to_numpy(float)
    propensity = train["continue_propensity"].to_numpy(float)
    weights = train["boundary_weight"].to_numpy(float)
    nuisance = cross_fitted_nuisance(
        nuisance_train,
        outcome,
        train["group"].to_numpy(),
        weights,
        inner_folds,
        nuisance_alpha,
    )
    residual_outcome = outcome - nuisance
    residual_action = action - propensity

    imputer = SimpleImputer()
    scaler = StandardScaler()
    standardized_train = scaler.fit_transform(imputer.fit_transform(x_train))
    standardized_test = scaler.transform(imputer.transform(x_test))
    phi_train = np.column_stack([np.ones(len(train)), standardized_train])
    phi_test = np.column_stack([np.ones(len(test)), standardized_test])
    effect_design = residual_action[:, None] * phi_train
    effect = Ridge(alpha=effect_alpha, fit_intercept=False)
    effect.fit(
        effect_design,
        residual_outcome,
        sample_weight=weights,
    )
    return phi_test @ effect.coef_


def replay(
    behavior: pd.DataFrame,
    oracle: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    prediction = np.full(len(oracle), np.nan, dtype=float)
    fold_id = np.full(len(oracle), -1, dtype=int)
    scopes = (
        [("pooled", np.arange(len(oracle)))]
        if args.training_scope == "pooled"
        else [
            (dataset, np.flatnonzero(oracle["dataset"].eq(dataset).to_numpy()))
            for dataset in sorted(oracle["dataset"].unique())
        ]
    )
    for scope_name, scope_index in scopes:
        scope_oracle = oracle.iloc[scope_index]
        labels = scope_oracle["continue_label"].to_numpy(int)
        groups = scope_oracle["group"].to_numpy()
        split_count = min(args.outer_folds, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(
            n_splits=split_count,
            shuffle=True,
            random_state=args.seed,
        )
        scope_behavior = (
            behavior
            if args.training_scope == "pooled"
            else behavior[behavior["dataset"].eq(scope_name)]
        )
        for fold, (_, local_test_index) in enumerate(
            splitter.split(scope_oracle, labels, groups)
        ):
            test_index = scope_index[local_test_index]
            test_groups = set(oracle.iloc[test_index]["group"])
            factual_train = scope_behavior[
                ~scope_behavior["group"].isin(test_groups)
            ].copy()
            if factual_train["continue_action"].nunique() != 2:
                raise RuntimeError(
                    f"outer fold {scope_name}/{fold} lost factual action overlap"
                )
            prediction[test_index] = fit_effect_and_predict(
                factual_train,
                oracle.iloc[test_index],
                args.inner_folds,
                args.nuisance_alpha,
                args.effect_alpha,
            )
            fold_id[test_index] = fold
    if not np.isfinite(prediction).all() or np.any(fold_id < 0):
        raise RuntimeError("outer cross-fitted R-learner predictions are incomplete")
    result = oracle.copy()
    result["r_learner_continue_advantage"] = prediction
    result["r_learner_action"] = np.where(
        prediction > 0.0,
        ACTION_CONTINUE,
        ACTION_STOP,
    )
    result["outer_fold"] = fold_id
    return result


def score_method(
    frame: pd.DataFrame,
    score_column: str,
    method: str,
    dataset: str,
) -> dict:
    selected = frame if dataset == "all" else frame[frame["dataset"].eq(dataset)]
    label = selected["continue_label"].to_numpy(int)
    score = selected[score_column].to_numpy(float)
    prediction = score > 0.0
    oracle_advantage = selected["oracle_continue_advantage"].to_numpy(float)
    wrong = prediction != label.astype(bool)
    total_margin = np.abs(oracle_advantage).sum()
    regret = np.where(wrong, np.abs(oracle_advantage), 0.0)
    correlation = spearmanr(score, oracle_advantage).statistic
    return {
        "dataset": dataset,
        "method": method,
        "states": len(selected),
        "continue_states": int(label.sum()),
        "auc": roc_auc_score(label, score),
        "balanced_accuracy": balanced_accuracy_score(label, prediction),
        "stop_recall": recall_score(label, prediction, pos_label=0),
        "continue_recall": recall_score(label, prediction, pos_label=1),
        "advantage_spearman": correlation,
        "sign_accuracy": float(np.mean(~wrong)),
        "mean_oracle_regret_tokens": float(np.mean(regret)),
        "oracle_margin_capture": (
            float(1.0 - regret.sum() / total_margin)
            if total_margin > 0.0
            else np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    behavior = load_behavior(args.input_dir, args.outcome_mode)
    oracle = load_oracle(args.input_dir)
    result = replay(behavior, oracle, args)
    summaries = []
    for dataset in ("all", "math", "gsm8k"):
        summaries.append(
            score_method(
                result,
                "r_learner_continue_advantage",
                "cross_fitted_r_learner",
                dataset,
            )
        )
        summaries.append(
            score_method(
                result,
                "shared_va_continue_advantage",
                "logged_shared_va",
                dataset,
            )
        )
    summary = pd.DataFrame(summaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "r_learner_oracle_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "r_learner_replay_summary.csv", index=False)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "factual_rows": len(behavior),
        "factual_problems": int(behavior["group"].nunique()),
        "factual_continue_rows": int(behavior["continue_action"].sum()),
        "oracle_states_without_ties": len(oracle),
        "oracle_continue_states": int(oracle["continue_label"].sum()),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "nuisance_alpha": args.nuisance_alpha,
        "effect_alpha": args.effect_alpha,
        "outcome_mode": args.outcome_mode,
        "oracle_labels_used_for_training": False,
        "boundary_cluster_weighting": True,
        "nuisance_state": "full_raw_state_plus_logged_propensity",
        "training_scope": args.training_scope,
    }
    (args.output_dir / "r_learner_replay_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Saved replay report to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
