import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from adaptive_nonlinear import RawSharedVA
from raw_state_experiment import load_raw_npz


class AdvantageModel(nn.Module):
    def __init__(self, model_type, input_dim, seed):
        super().__init__()
        torch.manual_seed(int(seed))
        self.network = (
            nn.Linear(input_dim, 1)
            if model_type == "raw_linear"
            else nn.Sequential(
                nn.Linear(input_dim, 32), nn.SiLU(),
                nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1),
            )
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser()
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
            "outputs_raw_state_capacity_math_gsm8k"
        ),
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank_corr(left, right):
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def metrics(qs, qc, predicted_advantage):
    actual = qs - qc
    truth_stop = actual > 0.0
    predicted_stop = predicted_advantage > 0.0
    stop_recall = float(predicted_stop[truth_stop].mean()) if truth_stop.any() else math.nan
    continue_recall = float((~predicted_stop[~truth_stop]).mean()) if (~truth_stop).any() else math.nan
    selected = np.where(predicted_stop, qs, qc)
    regret = np.maximum(qs, qc) - selected
    always_stop_regret = np.maximum(qs, qc) - qs
    return {
        "states": len(qs),
        "balanced_accuracy": 0.5 * (stop_recall + continue_recall),
        "stop_recall": stop_recall,
        "continue_recall": continue_recall,
        "spearman_advantage": rank_corr(predicted_advantage, actual),
        "mean_oracle_regret": float(regret.mean()),
        "always_stop_mean_regret": float(always_stop_regret.mean()),
    }


def train_predict(X_train, qs_train, qc_train, X_test, model_type, objective, epochs, seed):
    x = torch.tensor(X_train, dtype=torch.float32)
    test = torch.tensor(X_test, dtype=torch.float32)
    qs = torch.tensor(qs_train, dtype=torch.float32)
    qc = torch.tensor(qc_train, dtype=torch.float32)
    model = (
        AdvantageModel(model_type, X_train.shape[1], seed)
        if objective == "advantage_only"
        else RawSharedVA(model_type, X_train.shape[1], seed)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        if objective == "advantage_only":
            loss = torch.mean((model(x) - (qs - qc)) ** 2)
        else:
            value, advantage = model(x)
            predicted_stop = value + 0.5 * advantage
            predicted_continue = value - 0.5 * advantage
            loss = torch.mean(
                (predicted_stop - qs) ** 2
                + (predicted_continue - qc) ** 2
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        if objective == "advantage_only":
            return model(test).numpy()
        _, advantage = model(test)
        return advantage.numpy()


def grouped_folds(problem_ids, fold_count, seed):
    groups = np.asarray(sorted(set(int(value) for value in problem_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return [part for part in np.array_split(groups, min(fold_count, len(groups))) if len(part)]


def main():
    args = parse_args()
    data = load_raw_npz(Path(args.oracle_dir) / "raw_oracle_states.npz")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_type in ("raw_linear", "raw_mlp"):
        for objective in ("advantage_only", "shared_va"):
            for dataset in ("math", "gsm8k"):
                dataset_mask = data["dataset"] == dataset
                folds = grouped_folds(
                    data["problem_id"][dataset_mask], args.folds, args.seed
                )
                for fold_index, held_out in enumerate(folds):
                    test_mask = dataset_mask & np.isin(data["problem_id"], held_out)
                    train_mask = dataset_mask & ~test_mask
                    predicted = train_predict(
                        data["X"][train_mask], data["q_stop"][train_mask],
                        data["q_continue"][train_mask], data["X"][test_mask],
                        model_type, objective, args.epochs, args.seed + fold_index,
                    )
                    rows.append({
                        "scope": f"within_{dataset}", "fold": fold_index,
                        "model": model_type, "objective": objective,
                        **metrics(data["q_stop"][test_mask], data["q_continue"][test_mask], predicted),
                    })
            for target in ("math", "gsm8k"):
                test_mask = data["dataset"] == target
                train_mask = ~test_mask
                predicted = train_predict(
                    data["X"][train_mask], data["q_stop"][train_mask],
                    data["q_continue"][train_mask], data["X"][test_mask],
                    model_type, objective, args.epochs, args.seed,
                )
                rows.append({
                    "scope": f"cross_to_{target}", "fold": -1,
                    "model": model_type, "objective": objective,
                    **metrics(data["q_stop"][test_mask], data["q_continue"][test_mask], predicted),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "raw_capacity_fold_results.csv", index=False)
    summary = frame.groupby(["scope", "model", "objective"]).mean(numeric_only=True).reset_index()
    summary.to_csv(output / "raw_capacity_summary.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
