import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARK_VERSION = "online_unmask_calibration_validation_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("math", "aime", "gsm8k", "humaneval"),
        default=("math", "aime", "gsm8k", "humaneval"),
    )
    parser.add_argument("--num_questions", type=int, default=15)
    parser.add_argument("--warmup_questions", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dllm_dir", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--spec_len", type=int, default=8)
    parser.add_argument("--incr_len", type=int, default=8)
    parser.add_argument("--drafter_threshold", type=float, default=0.05)
    parser.add_argument("--lowconf_threshold", type=float, default=0.45)
    parser.add_argument("--max_spec_len", type=int, default=60)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_draft_latency_ms", type=float, default=40.0)
    parser.add_argument("--initial_verify_latency_ms", type=float, default=140.0)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument(
        "--output_dir",
        default="/content/failfasttesting/outputs_online_unmask_calibration_test15",
    )
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.num_questions <= 0:
        raise ValueError("--num_questions must be positive")
    if args.warmup_questions < 1:
        raise ValueError("--warmup_questions must be at least 1")
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema_alpha must be in (0, 1]")


def run_streaming(command, cwd):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def sigmoid(value):
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


class OnlineLogisticModel:
    def __init__(self, dimension, prior_probability=0.75, prior_precision=8.0):
        self.weights = np.zeros(dimension, dtype=np.float64)
        prior_probability = float(np.clip(prior_probability, 1e-4, 1.0 - 1e-4))
        self.weights[0] = math.log(prior_probability / (1.0 - prior_probability))
        self.precision = np.eye(dimension, dtype=np.float64) * float(prior_precision)
        self.observations = 0

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        return sigmoid(features @ self.weights)

    def update_batch(self, features, labels):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if features.size == 0:
            return
        probabilities = self.predict(features)
        curvature = np.clip(probabilities * (1.0 - probabilities), 1e-3, None)
        self.precision += features.T @ (curvature[:, None] * features)
        gradient = features.T @ (labels - probabilities)
        self.weights += np.linalg.solve(self.precision, gradient)
        self.weights = np.clip(self.weights, -12.0, 12.0)
        self.observations += len(labels)


class OnlineLinearTransition:
    def __init__(self, dimension, outputs=3, prior_precision=8.0):
        self.weights = np.zeros((dimension, outputs), dtype=np.float64)
        self.precision = np.eye(dimension, dtype=np.float64) * float(prior_precision)
        self.observations = 0

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        return features @ self.weights

    def update_batch(self, features, targets):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if features.size == 0:
            return
        residual = targets - features @ self.weights
        self.precision += features.T @ features
        self.weights += np.linalg.solve(self.precision, features.T @ residual)
        self.observations += len(targets)


class OnlineEMA:
    def __init__(self, initial_value, alpha):
        self.value = float(initial_value)
        self.alpha = float(alpha)
        self.observations = 0

    def update(self, value):
        value = float(value)
        if self.observations == 0:
            self.value = value
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * value
        self.observations += 1


def parse_vector(value, target_len, boolean=False):
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        parsed = []
    dtype = np.float64
    result = np.asarray(parsed, dtype=dtype)
    if len(result) != int(target_len):
        raise ValueError(f"Expected {target_len} token features, found {len(result)}")
    if boolean:
        result = result.astype(np.float64)
    return result


def row_state(row):
    target_len = int(row["target_len"])
    confidence = parse_vector(row["token_confidences"], target_len)
    margin = parse_vector(row["token_margins"], target_len)
    forced = parse_vector(row["token_forced"], target_len, boolean=True)
    recoverable = parse_vector(row["token_recoverable"], target_len, boolean=True)
    mask_ratio = float(recoverable.mean()) if target_len else 0.0
    recoverable_positions = np.flatnonzero(recoverable > 0.5)
    first_recoverable = (
        float(recoverable_positions[0]) / max(target_len - 1, 1)
        if len(recoverable_positions)
        else 1.0
    )
    return {
        "target_len": target_len,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "margin": np.clip(margin, 0.0, 1.0),
        "forced": forced,
        "recoverable": recoverable,
        "mask_ratio": mask_ratio,
        "first_recoverable": first_recoverable,
        "frontier_ratio": float(row.get("frontier_k", 0.0)) / max(target_len, 1),
        "unmasked_ratio": float(row.get("unmasked_this_step", 0.0)) / max(target_len, 1),
        "step": int(row["step"]),
    }


def hazard_features(state):
    target_len = state["target_len"]
    position = np.arange(target_len, dtype=np.float64) / max(target_len - 1, 1)
    confidence = state["confidence"]
    margin = state["margin"]
    return np.column_stack([
        np.ones(target_len),
        confidence,
        confidence * confidence,
        margin,
        state["recoverable"],
        position,
        np.full(target_len, state["mask_ratio"]),
        np.full(target_len, state["first_recoverable"]),
        np.full(target_len, math.log1p(state["step"]) / math.log(10.0)),
    ])


def transition_features(state):
    target_len = state["target_len"]
    position = np.arange(target_len, dtype=np.float64) / max(target_len - 1, 1)
    return np.column_stack([
        np.ones(target_len),
        state["confidence"],
        state["margin"],
        state["recoverable"],
        position,
        np.full(target_len, state["mask_ratio"]),
        np.full(target_len, state["frontier_ratio"]),
        np.full(target_len, state["unmasked_ratio"]),
        np.full(target_len, math.log1p(state["step"]) / math.log(10.0)),
        np.full(target_len, state["first_recoverable"]),
    ])


def expected_emitted_tokens(hazard_model, state):
    token_acceptance = np.clip(hazard_model.predict(hazard_features(state)), 1e-4, 1.0 - 1e-4)
    return float(1.0 + np.cumprod(token_acceptance).sum())


def predict_next_state(transition_model, state):
    prediction = transition_model.predict(transition_features(state))
    next_state = dict(state)
    recoverable = state["recoverable"]
    active = recoverable > 0.5
    next_confidence = state["confidence"].copy()
    next_margin = state["margin"].copy()
    next_recoverable = recoverable.copy()
    next_confidence[active] = np.clip(
        next_confidence[active] + prediction[active, 0],
        0.0,
        1.0,
    )
    next_margin[active] = np.clip(
        next_margin[active] + prediction[active, 1],
        0.0,
        1.0,
    )
    next_recoverable[active] = np.clip(prediction[active, 2], 0.0, 1.0)
    next_state["confidence"] = next_confidence
    next_state["margin"] = next_margin
    next_state["recoverable"] = next_recoverable
    next_state["mask_ratio"] = float(next_recoverable.mean())
    remaining_positions = np.flatnonzero(next_recoverable > 0.5)
    next_state["first_recoverable"] = (
        float(remaining_positions[0]) / max(state["target_len"] - 1, 1)
        if len(remaining_positions)
        else 1.0
    )
    next_state["step"] = state["step"] + 1
    return next_state


def update_transition_model(transition_model, current_state, next_state):
    active = current_state["recoverable"] > 0.5
    if not active.any():
        return
    features = transition_features(current_state)[active]
    targets = np.column_stack([
        next_state["confidence"] - current_state["confidence"],
        next_state["margin"] - current_state["margin"],
        next_state["recoverable"],
    ])[active]
    transition_model.update_batch(features, targets)


def update_hazard_model(hazard_model, final_row, final_state):
    accepted_len = int(final_row["accepted_len_if_stop"])
    observed = min(accepted_len + 1, final_state["target_len"])
    if observed <= 0:
        return
    labels = np.ones(observed, dtype=np.float64)
    if accepted_len < final_state["target_len"]:
        labels[-1] = 0.0
    hazard_model.update_batch(hazard_features(final_state)[:observed], labels)


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_dataset(oracle, args, dataset):
    hazard_model = OnlineLogisticModel(dimension=9)
    transition_model = OnlineLinearTransition(dimension=10)
    draft_latency = OnlineEMA(args.initial_draft_latency_ms, args.ema_alpha)
    verify_latency = OnlineEMA(args.initial_verify_latency_ms, args.ema_alpha)
    evaluation_problem_ids = set(
        oracle.loc[oracle["problem_id"] >= args.warmup_questions, "problem_id"].astype(int)
    )
    records = []

    group_columns = ["problem_id", "round_id"]
    for (problem_id, round_id), group in oracle.groupby(group_columns, sort=True):
        group = group.sort_values("step").reset_index(drop=True)
        states = [row_state(row) for _, row in group.iterrows()]
        measured = int(problem_id) in evaluation_problem_ids
        actual_verify_ms = float(group.iloc[-1]["actual_verify_latency_ms"])

        for index in range(len(group) - 1):
            current_row = group.iloc[index]
            next_row = group.iloc[index + 1]
            if int(current_row["target_len"]) != int(next_row["target_len"]):
                continue
            current_state = states[index]
            next_state = states[index + 1]
            predicted_next_state = predict_next_state(transition_model, current_state)

            predicted_current_y = expected_emitted_tokens(hazard_model, current_state)
            predicted_next_y = expected_emitted_tokens(hazard_model, predicted_next_state)
            predicted_gain = predicted_next_y - predicted_current_y
            actual_current_y = float(current_row["emitted_len_if_stop"])
            actual_next_y = float(next_row["emitted_len_if_stop"])
            actual_gain = actual_next_y - actual_current_y
            actual_next_draft_ms = max(
                0.0,
                float(next_row["draft_latency_elapsed_ms"])
                - float(current_row["draft_latency_elapsed_ms"]),
            )
            predicted_next_draft_ms = draft_latency.value
            predicted_verify_ms = verify_latency.value
            elapsed_draft_ms = float(current_row["draft_latency_elapsed_ms"])

            predicted_stop_cost = (
                elapsed_draft_ms + predicted_verify_ms
            ) / max(predicted_current_y, 1e-6)
            predicted_continue_cost = (
                elapsed_draft_ms + predicted_next_draft_ms + predicted_verify_ms
            ) / max(predicted_next_y, 1e-6)
            actual_stop_cost = (
                elapsed_draft_ms + actual_verify_ms
            ) / max(actual_current_y, 1e-6)
            actual_continue_cost = (
                elapsed_draft_ms + actual_next_draft_ms + actual_verify_ms
            ) / max(actual_next_y, 1e-6)
            predicted_continue = predicted_continue_cost < predicted_stop_cost
            oracle_continue = actual_continue_cost < actual_stop_cost
            selected_actual_cost = actual_continue_cost if predicted_continue else actual_stop_cost

            if measured:
                records.append({
                    "dataset": dataset,
                    "problem_id": int(problem_id),
                    "round_id": int(round_id),
                    "from_step": int(current_row["step"]),
                    "to_step": int(next_row["step"]),
                    "from_refinement_step": int(
                        (
                            group.loc[:index, "target_len"].astype(int)
                            == int(current_row["target_len"])
                        ).sum()
                    ),
                    "from_target_len": int(current_row["target_len"]),
                    "to_target_len": int(next_row["target_len"]),
                    "predicted_current_y": predicted_current_y,
                    "actual_current_y": actual_current_y,
                    "predicted_next_y": predicted_next_y,
                    "actual_next_y": actual_next_y,
                    "predicted_gain": predicted_gain,
                    "actual_gain": actual_gain,
                    "gain_error": predicted_gain - actual_gain,
                    "predicted_next_draft_ms": predicted_next_draft_ms,
                    "actual_next_draft_ms": actual_next_draft_ms,
                    "predicted_verify_ms": predicted_verify_ms,
                    "actual_verify_ms": actual_verify_ms,
                    "predicted_stop_ms_per_token": predicted_stop_cost,
                    "predicted_continue_ms_per_token": predicted_continue_cost,
                    "actual_stop_ms_per_token": actual_stop_cost,
                    "actual_continue_ms_per_token": actual_continue_cost,
                    "predicted_action": "continue" if predicted_continue else "stop",
                    "oracle_action": "continue" if oracle_continue else "stop",
                    "decision_correct": int(predicted_continue == oracle_continue),
                    "decision_regret_ms_per_token": selected_actual_cost
                    - min(actual_stop_cost, actual_continue_cost),
                    "hazard_observations_before": hazard_model.observations,
                    "transition_observations_before": transition_model.observations,
                    "draft_latency_observations_before": draft_latency.observations,
                    "verify_latency_observations_before": verify_latency.observations,
                })

            update_transition_model(transition_model, current_state, next_state)
            draft_latency.update(actual_next_draft_ms)

        final_row = group.iloc[-1]
        update_hazard_model(hazard_model, final_row, states[-1])
        verify_latency.update(actual_verify_ms)

    return pd.DataFrame(records)


def summarize_predictions(predictions, group_columns):
    records = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        gain_error = group["gain_error"].to_numpy(dtype=np.float64)
        record = dict(zip(group_columns, keys))
        record.update({
            "transitions": len(group),
            "problems": len(group[["dataset", "problem_id"]].drop_duplicates()),
            "predicted_gain_mean": group["predicted_gain"].mean(),
            "actual_gain_mean": group["actual_gain"].mean(),
            "gain_bias": gain_error.mean(),
            "gain_mae": np.abs(gain_error).mean(),
            "gain_rmse": math.sqrt(np.mean(gain_error * gain_error)),
            "gain_pearson": safe_correlation(group["predicted_gain"], group["actual_gain"]),
            "current_y_mae": np.abs(
                group["predicted_current_y"] - group["actual_current_y"]
            ).mean(),
            "next_y_mae": np.abs(
                group["predicted_next_y"] - group["actual_next_y"]
            ).mean(),
            "draft_latency_mae_ms": np.abs(
                group["predicted_next_draft_ms"] - group["actual_next_draft_ms"]
            ).mean(),
            "verify_latency_mae_ms": np.abs(
                group["predicted_verify_ms"] - group["actual_verify_ms"]
            ).mean(),
            "predicted_continue_rate_percent": 100.0
            * (group["predicted_action"] == "continue").mean(),
            "oracle_continue_rate_percent": 100.0
            * (group["oracle_action"] == "continue").mean(),
            "decision_accuracy_percent": 100.0 * group["decision_correct"].mean(),
            "mean_decision_regret_ms_per_token": group[
                "decision_regret_ms_per_token"
            ].mean(),
        })
        records.append(record)
    return pd.DataFrame(records)


def calibration_curve(predictions):
    bins = [-np.inf, 0.0, 0.5, 1.0, 2.0, 4.0, np.inf]
    labels = ["<0", "0-0.5", "0.5-1", "1-2", "2-4", ">=4"]
    frame = predictions.copy()
    frame["predicted_gain_bin"] = pd.cut(
        frame["predicted_gain"],
        bins=bins,
        labels=labels,
        right=False,
    )
    records = []
    for (dataset, gain_bin), group in frame.groupby(
        ["dataset", "predicted_gain_bin"],
        observed=True,
        sort=True,
    ):
        records.append({
            "dataset": dataset,
            "predicted_gain_bin": str(gain_bin),
            "transitions": len(group),
            "predicted_gain_mean": group["predicted_gain"].mean(),
            "actual_gain_mean": group["actual_gain"].mean(),
            "positive_actual_gain_percent": 100.0 * (group["actual_gain"] > 0.0).mean(),
            "decision_accuracy_percent": 100.0 * group["decision_correct"].mean(),
        })
    return pd.DataFrame(records)


def run_oracle_collection(args, output_dir):
    oracle_output = output_dir / "oracle_profile"
    command = [
        sys.executable,
        "-u",
        "run_oracle_refinement_profile.py",
        "--datasets", *args.datasets,
        "--num_questions", str(args.num_questions),
        "--warmup_questions", str(args.warmup_questions),
        "--max_new_tokens", str(args.max_new_tokens),
        "--target_model_name", args.target_model_name,
        "--dllm_dir", args.dllm_dir,
        "--block_size", str(args.block_size),
        "--small_block_size", str(args.small_block_size),
        "--spec_len", str(args.spec_len),
        "--incr_len", str(args.incr_len),
        "--drafter_threshold", str(args.drafter_threshold),
        "--lowconf_threshold", str(args.lowconf_threshold),
        "--max_spec_len", str(args.max_spec_len),
        "--sample_seed", str(args.sample_seed),
        "--seed", str(args.seed),
        "--output_dir", str(oracle_output),
        "--log_level", args.log_level,
        "--skip_archive",
    ]
    if args.resume:
        command.append("--resume")
    run_streaming(command, Path(__file__).resolve().parent)
    return oracle_output


def write_manifest(args, output_dir):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = None
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": vars(args),
        "purpose": (
            "Chronological validation of online acceptance, denoising transition, "
            "latency, and stop/continue cost predictions for same-length unmask "
            "transitions. Intermediate verifier labels are used only as diagnostic "
            "ground truth."
        ),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_output = run_oracle_collection(args, output_dir)
    oracle = pd.read_csv(oracle_output / "oracle_refinement_snapshots.csv")

    required = {
        "token_confidences",
        "token_margins",
        "token_forced",
        "token_recoverable",
        "actual_verify_latency_ms",
    }
    missing = required.difference(oracle.columns)
    if missing:
        raise ValueError(f"Oracle diagnostics are missing columns: {sorted(missing)}")

    prediction_frames = []
    for dataset in args.datasets:
        dataset_oracle = oracle[oracle["dataset"] == dataset].copy()
        predictions = evaluate_dataset(dataset_oracle, args, dataset)
        prediction_frames.append(predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    dataset_summary = summarize_predictions(predictions, ["dataset"])
    step_summary = summarize_predictions(
        predictions,
        ["dataset", "from_refinement_step"],
    )
    overall_summary = summarize_predictions(
        predictions.assign(scope="all_datasets"),
        ["scope"],
    )
    curve = calibration_curve(predictions)

    predictions.to_csv(output_dir / "online_calibration_predictions.csv", index=False)
    dataset_summary.to_csv(output_dir / "online_calibration_dataset_summary.csv", index=False)
    step_summary.to_csv(output_dir / "online_calibration_step_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "online_calibration_overall_summary.csv", index=False)
    curve.to_csv(output_dir / "online_calibration_curve.csv", index=False)
    write_manifest(args, output_dir)

    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print("\nONLINE CALIBRATION DATASET SUMMARY")
    print(dataset_summary.to_string(index=False))
    print("\nONLINE CALIBRATION OVERALL SUMMARY")
    print(overall_summary.to_string(index=False))
    print(f"\nSaved report: {output_dir}")
    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
