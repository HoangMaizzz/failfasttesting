import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ("math", "gsm8k", "humaneval")

# Exact measured 100-problem IDs from the current U1 fixed-threshold bundles.
PROBLEM_IDS = {
    "math": [
        285, 294, 308, 315, 319, 332, 348, 351, 375, 385, 394, 396, 398,
        402, 413, 418, 419, 432, 441, 453, 457, 459, 476, 488, 489, 299, 139,
        321, 357, 69, 433, 121, 150, 158, 40, 134, 424, 199, 21, 43, 442, 174,
        461, 141, 355, 271, 120, 328, 81, 217, 404, 213, 126, 72, 311, 431,
        140, 51, 20, 366, 214, 6, 93, 474, 491, 218, 289, 238, 388, 80, 208,
        221, 220, 100, 300, 298, 49, 467, 275, 3, 114, 26, 473, 345, 470, 283,
        437, 483, 165, 343, 409, 226, 291, 454, 314, 71, 403, 277, 198, 61,
    ],
    "gsm8k": [
        756, 771, 814, 862, 865, 904, 921, 937, 945, 1005, 1006, 1030, 1049,
        1052, 1102, 1122, 1140, 1142, 1173, 1183, 1202, 1231, 1258, 1273,
        1307, 284, 1194, 1025, 1128, 83, 441, 366, 821, 803, 282, 712, 540,
        859, 758, 386, 33, 665, 248, 120, 406, 166, 1266, 629, 236, 146, 233,
        1253, 538, 274, 273, 747, 695, 632, 624, 1241, 500, 1195, 117, 493,
        1010, 483, 382, 1149, 764, 786, 1306, 784, 1110, 605, 572, 922, 725,
        456, 365, 24, 300, 534, 835, 603, 1228, 1155, 1257, 135, 150, 280, 96,
        388, 45, 1047, 886, 1028, 84, 791, 219, 309,
    ],
    "humaneval": [
        88, 90, 91, 92, 97, 102, 108, 109, 113, 116, 123, 126, 128, 129, 132,
        135, 141, 143, 145, 147, 150, 151, 152, 154, 160, 33, 122, 111, 110, 98,
        17, 66, 41, 130, 23, 153, 6, 60, 21, 1, 5, 58, 117, 30, 96, 134, 77,
        68, 47, 139, 105, 39, 20, 43, 64, 119, 12, 118, 55, 65, 80, 140, 86,
        89, 62, 94, 162, 112, 63, 45, 127, 4, 125, 121, 67, 83, 32, 144, 156,
        18, 138, 28, 15, 36, 133, 54, 101, 37, 13, 24, 99, 78, 79, 10, 56, 7,
        59, 46, 103, 148,
    ],
}

# Previous matched FailFast values are included only as references; this runner does NOT rerun them.
PREVIOUS_FAILFAST_MS_PER_TOKEN = {
    "math": 26.240350466082997,
    "gsm8k": 23.671124799417285,
    "humaneval": 30.044611910390834,
}
PREVIOUS_F2_MS_PER_TOKEN = {
    "math": 25.25164630380439,
    "gsm8k": 22.644542620711725,
    "humaneval": 28.391911053164314,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run only U1 batch-1x + normalized current-prefix F3 on the exact "
            "100-problem IDs used by the current matched experiments."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--target_quantization", default="int8")
    parser.add_argument("--target_device", type=int, default=0)
    parser.add_argument("--drafter_device", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--dllm_dir",
        default="/home/maihoang/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/maihoang/failfasttesting/outputs_u1_f3prefix_fixed05_test100",
    )
    return parser.parse_args()


def run(command):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def command(args, dataset, destination):
    ids = PROBLEM_IDS[dataset]
    assert len(ids) == 100, (dataset, len(ids))
    return [
        sys.executable, "-u", "failfast.py",
        "--dataset_name", dataset,
        "--num_questions", "100",
        "--problem_ids", *map(str, ids),
        "--warmup_questions", "1",
        "--benchmark_modes", "dllm_ar",
        "--dllm_variant", "failfast",
        "--decoding_strategy", "greedy",
        "--max_new_tokens", str(args.max_new_tokens),
        "--spec_len", "8",
        "--block_size", "32",
        "--small_block_size", "8",
        "--target_model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--dllm_dir", args.dllm_dir,
        "--target_device", str(args.target_device),
        "--drafter_device", str(args.drafter_device),
        "--target_quantization", args.target_quantization,
        "--drafter_thresholds", "0.5",
        "--sweep_lowconf_threshold", "0.7",
        "--sweep_max_spec_len", "64",
        "--sweep_incr_len", "8",
        "--seed", "42",
        "--adaptive-td",
        "--adaptive-feature-schema", "otrc_v2_2_compact_td",
        "--adaptive-credit-assignment", "hindsight_delta_j_logistic_f2",
        "--adaptive-policy-mode", "hindsight_delta_j_logistic_f2",
        # NEW FEATURE: k_t / L_t^{target}
        "--adaptive-hindsight-logistic-use-prefix-feature",
        # Keep every U1 batch-1x hyperparameter unchanged.
        "--adaptive-hindsight-logistic-learning-rate", "0.05",
        "--adaptive-hindsight-logistic-continue-threshold", "0.5",
        "--adaptive-hindsight-logistic-tie-ms-per-token", "1.0",
        "--adaptive-hindsight-logistic-min-positive-problems", "2",
        "--adaptive-hindsight-delta-j-class-balance-alpha", "5.0",
        "--adaptive-hindsight-delta-j-max-continue-weight", "3.0",
        "--adaptive-hindsight-delta-j-min-pairs", "30",
        "--adaptive-hindsight-delta-j-min-continue-pairs", "3",
        "--adaptive-hindsight-delta-j-structural-probe", "0.08",
        "--adaptive-hindsight-delta-j-floor-probe", "0.02",
        "--adaptive-hindsight-logistic-utility-weighting", "raw_abs",
        "--adaptive-hindsight-logistic-replay-batch-size", "16",
        "--adaptive-hindsight-logistic-replay-buffer-size", "100",
        "--no-adaptive-hindsight-logistic-use-class-weight",
        "--adaptive-log-decisions",
        "--adaptive-profile-overhead",
        "--quiet_generation",
        "--disable_progress",
        "--skip_artifacts",
        "--skip_plots",
        "--overwrite",
        "--output_dir", str(destination),
        "--log_level", "INFO",
    ]


def auc(labels, scores, weights=None):
    frame = pd.DataFrame({"label": labels, "score": scores})
    if weights is not None:
        frame["weight"] = weights
    frame = frame.dropna()
    pos = frame[frame.label == 1]
    neg = frame[frame.label == 0]
    if pos.empty or neg.empty:
        return float("nan")
    if weights is None:
        # Pairwise form, ties count as one half.
        wins = 0.0
        for p in pos.score.to_numpy():
            n = neg.score.to_numpy()
            wins += float((p > n).sum()) + 0.5 * float((p == n).sum())
        return wins / (len(pos) * len(neg))
    # Utility-weighted pairwise AUC.
    ps, pw = pos.score.to_numpy(), pos.weight.to_numpy()
    ns, nw = neg.score.to_numpy(), neg.weight.to_numpy()
    num = 0.0
    den = float(pw.sum() * nw.sum())
    for score_p, weight_p in zip(ps, pw):
        cmp = (score_p > ns).astype(float) + 0.5 * (score_p == ns).astype(float)
        num += float(weight_p * (cmp * nw).sum())
    return num / den if den > 0 else float("nan")


def summarize(dataset, destination):
    benchmark = pd.read_csv(destination / "benchmark_results.csv")
    tokens = float(benchmark.output_tokens.sum())
    mspt = 1000.0 * float(benchmark.actual_algorithm_time.sum()) / max(1.0, tokens)
    row = {
        "dataset": dataset,
        "method": "u1_batch1x_f3_prefix",
        "questions": int(benchmark.problem_id.nunique()),
        "output_tokens": int(tokens),
        "ms_per_output_token": mspt,
        "draft_forwards": int(benchmark.total_num_forward_passes.sum()),
        "verifier_rounds": int(benchmark.num_speculation_rounds.sum()),
        "previous_failfast_ms_per_output_token": PREVIOUS_FAILFAST_MS_PER_TOKEN[dataset],
        "previous_f2_ms_per_output_token": PREVIOUS_F2_MS_PER_TOKEN[dataset],
        "speedup_vs_previous_failfast": PREVIOUS_FAILFAST_MS_PER_TOKEN[dataset] / mspt,
        "latency_reduction_vs_previous_failfast_percent": 100.0 * (
            1.0 - mspt / PREVIOUS_FAILFAST_MS_PER_TOKEN[dataset]
        ),
        "latency_change_vs_previous_f2_percent": 100.0 * (
            mspt / PREVIOUS_F2_MS_PER_TOKEN[dataset] - 1.0
        ),
    }

    decisions = pd.read_csv(destination / "adaptive_td_decisions.csv")
    source = decisions.get("action_source", pd.Series(dtype=str)).fillna("")
    row.update({
        "decisions": int(len(decisions)),
        "learned_stop": int((source == "learned_stop").sum()),
        "learned_continue": int((source == "learned_continue").sum()),
        "structural_probes": int((source == "structural_probe").sum()),
        "floor_probes": int((source == "floor_probe").sum()),
    })

    transitions_path = destination / "adaptive_full_stream_transitions.csv"
    if transitions_path.exists():
        transitions = pd.read_csv(transitions_path)
        applied = transitions.update_applied.map(
            lambda v: str(v).strip().lower() in {"1", "true", "yes"}
        )
        labeled = transitions[applied].copy()
        y = pd.to_numeric(labeled.binary_label_C, errors="coerce")
        score = pd.to_numeric(labeled.continue_score_before_update, errors="coerce")
        utility = pd.to_numeric(labeled.delta_J_ms_per_token, errors="coerce").abs()
        row.update({
            "resolved_non_tie_pairs": int(len(labeled)),
            "beneficial_continue_pairs": int((y == 1).sum()),
            "temporal_auc": auc(y, score),
            "utility_weighted_temporal_auc": auc(y, score, utility),
        })
        learned_c = labeled[
            (labeled.model_action == "continue")
            & (labeled.action_source == "learned_continue")
        ]
        row["sum_delta_j_learned_continue"] = float(
            pd.to_numeric(learned_c.delta_J_ms_per_token, errors="coerce").sum()
        )

    runtime_path = destination / "adaptive_td_runtime_state.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        h = runtime.get("hindsight_block_gain", {})
        row["final_hindsight_feature_names"] = json.dumps(
            h.get("feature_names", h.get("logistic_model", {}).get("feature_names", []))
        )
        model = h.get("logistic_model", {})
        if model:
            row["final_logistic_weights"] = json.dumps(model.get("weights", []))
    return row


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        destination = root / "raw" / dataset / "u1_batch1x_f3_prefix"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 92)
        print(f"RUN {dataset.upper()} | U1 BATCH-1X + F3 PREFIX | 100 EXACT MATCHED IDS")
        print("=" * 92)
        run(command(args, dataset, destination))
        rows.append(summarize(dataset, destination))

        # Persist incremental summary so a completed dataset is not lost if a later run fails.
        pd.DataFrame(rows).to_csv(root / "dataset_summary.csv", index=False)

    manifest = {
        "method": "u1_batch1x_f3_prefix",
        "new_feature": "normalized_current_prefix_length = prefix_length / proposal_length",
        "problem_ids": {d: PROBLEM_IDS[d] for d in args.datasets},
        "only_new_variant_was_run": True,
        "target_quantization": args.target_quantization,
        "target_device": args.target_device,
        "drafter_device": args.drafter_device,
        "fixed_threshold": 0.5,
        "utility_weighting": "raw_abs(delta_J_ms_per_token)",
        "replay_batch_size": 16,
        "replay_buffer_size": 100,
        "learning_rate": 0.05,
        "tie_ms_per_token": 1.0,
        "min_pairs": 30,
        "min_continue_pairs": 3,
        "min_positive_problems": 2,
        "structural_probe": 0.08,
        "floor_probe": 0.02,
        "previous_failfast_reference_ms_per_token": PREVIOUS_FAILFAST_MS_PER_TOKEN,
        "previous_f2_reference_ms_per_token": PREVIOUS_F2_MS_PER_TOKEN,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = pd.DataFrame(rows)
    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved: {root / 'dataset_summary.csv'}")
    print(f"Saved: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
