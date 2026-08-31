import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_nonlinear import OnlineNonlinearVA
from raw_state_experiment import load_raw_npz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_raw_shared_va_math_gsm8k_test25"
        ),
    )
    parser.add_argument(
        "--oracle_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_raw_state_oracle_math_gsm8k_test10/raw_capacity_dataset"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/maihoang/failfasttesting/"
            "outputs_raw_checkpoint_oracle_math_gsm8k"
        ),
    )
    return parser.parse_args()


def metrics(qs, qc, advantage):
    actual = qs - qc
    truth_stop = actual > 0
    predicted_stop = advantage > 0
    stop_recall = predicted_stop[truth_stop].mean() if truth_stop.any() else np.nan
    continue_recall = (~predicted_stop[~truth_stop]).mean() if (~truth_stop).any() else np.nan
    selected = np.where(predicted_stop, qs, qc)
    regret = np.maximum(qs, qc) - selected
    return {
        "states": len(qs),
        "sign_accuracy": float((truth_stop == predicted_stop).mean()),
        "balanced_accuracy": float(0.5 * (stop_recall + continue_recall)),
        "stop_recall": float(stop_recall),
        "continue_recall": float(continue_recall),
        "advantage_spearman": float(
            pd.Series(advantage).rank().corr(pd.Series(actual).rank())
        ),
        "mean_oracle_regret": float(regret.mean()),
    }


def predict(snapshot, model_type, X):
    learner = OnlineNonlinearVA(
        model_type,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        huber_delta=32.0,
        seed=42,
        feature_dim=X.shape[1],
    )
    learner.load_snapshot(snapshot)
    return np.asarray([learner.predict(row)[1] for row in X], dtype=np.float32)


def main():
    args = parse_args()
    online_root = Path(args.online_dir)
    oracle = load_raw_npz(Path(args.oracle_dir) / "raw_oracle_states.npz")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_type in ("raw_linear", "raw_mlp"):
        method_root = online_root / model_type
        manifest = json.loads((method_root / "benchmark_manifest.json").read_text())
        method = manifest["method"]
        for dataset in ("math", "gsm8k"):
            state_path = (
                method_root / "raw" / dataset / method
                / "adaptive_td_runtime_state.json"
            )
            state = json.loads(state_path.read_text())
            checkpoints = [
                item for item in state.get("weight_snapshots", [])
                if item.get("nonlinear_value")
            ]
            checkpoints.append({
                "learning_update_count": int(
                    state["nonlinear_value"]["update_count"]
                ),
                "decision_count": int(state["decision_count"]),
                "nonlinear_value": state["nonlinear_value"],
                "final": True,
            })
            mask = oracle["dataset"] == dataset
            for checkpoint in checkpoints:
                advantage = predict(
                    checkpoint["nonlinear_value"], model_type, oracle["X"][mask]
                )
                rows.append({
                    "dataset": dataset,
                    "model": model_type,
                    "learning_update_count": int(
                        checkpoint.get("learning_update_count", 0)
                    ),
                    "decision_count": int(checkpoint.get("decision_count", 0)),
                    "final": int(bool(checkpoint.get("final", False))),
                    **metrics(
                        oracle["q_stop"][mask],
                        oracle["q_continue"][mask],
                        advantage,
                    ),
                })
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "model", "learning_update_count", "final"]
    )
    frame.to_csv(output / "checkpoint_oracle_learning_curve.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
