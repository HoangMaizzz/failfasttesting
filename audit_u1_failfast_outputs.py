"""Compare saved full-token hashes without running or changing either model."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    summaries = []
    for dataset in manifest["datasets"]:
        frames = []
        for method in ("failfast", "u1_batch1x"):
            path = args.root / "raw" / dataset / method / "benchmark_results.csv"
            data = pd.read_csv(path)
            if data.problem_id.duplicated().any():
                raise ValueError(f"Duplicate problem IDs in {path}")
            print(dataset, method, "quantization:", data.target_quantization.unique().tolist())
            columns = ["problem_id", "output_tokens", "output_token_hash", "actual_algorithm_time"]
            frames.append(data[columns].rename(columns={c: f"{method}_{c}" for c in columns if c != "problem_id"}))
        paired = frames[0].merge(frames[1], on="problem_id", how="outer", validate="one_to_one", indicator=True)
        paired["exact_match"] = (
            paired["_merge"].eq("both")
            & paired.failfast_output_tokens.eq(paired.u1_batch1x_output_tokens)
            & paired.failfast_output_token_hash.eq(paired.u1_batch1x_output_token_hash)
        )
        expected = set(manifest["problem_ids"][dataset])
        complete = set(paired.problem_id) == expected and bool(paired["_merge"].eq("both").all())
        passed = complete and bool(paired.exact_match.all())
        summary = {
            "dataset": dataset, "questions": len(paired), "complete": complete,
            "matched": int(paired.exact_match.sum()), "passed": passed,
            "mismatch_ids": paired.loc[~paired.exact_match, "problem_id"].tolist(),
            "failfast_seconds": float(paired.failfast_actual_algorithm_time.sum()),
            "u1_seconds": float(paired.u1_batch1x_actual_algorithm_time.sum()),
            "lossless_speedup": (float(paired.failfast_actual_algorithm_time.sum() / paired.u1_batch1x_actual_algorithm_time.sum()) if passed else None),
        }
        paired.to_csv(args.root / f"{dataset}_u1_failfast_output_audit.csv", index=False)
        summaries.append(summary)
    (args.root / "u1_failfast_output_audit.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
