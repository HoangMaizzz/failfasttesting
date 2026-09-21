#!/usr/bin/env python3
"""Validate the streamed raw-state schema and latency-per-token labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def validate_dataset(root: Path, dataset: str) -> dict:
    dataset_root = root / "raw" / dataset
    index_path = dataset_root / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    items = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
    ]
    states = len(items)
    masked_states = 0
    masked_positions = 0
    continue_labels = sum(
        int(item.get("one_step_latency_continue_label") == 1) for item in items
    )
    continue_available = sum(int(bool(item.get("continue_available"))) for item in items)
    for shard_name in sorted({item["shard"] for item in items}):
        shard = dataset_root / shard_name
        shard_items = [item for item in items if item["shard"] == shard_name]
        with np.load(shard, allow_pickle=False) as arrays:
            required = {
                "proposal_mask_before_fill",
                "proposal_token_ids_before_fill",
                "proposal_token_ids_after_fill",
                "hidden_states",
                "topk_token_ids",
                "topk_logits",
            }
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise RuntimeError(f"{dataset}: {shard.name} missing {missing}")
            for item in shard_items:
                mask = arrays["proposal_mask_before_fill"][int(item["row"])].astype(bool)
                count = int(mask.sum())
                masked_positions += count
                masked_states += int(count > 0)

    return {
        "dataset": dataset,
        "states": states,
        "states_with_mask_before_fill": masked_states,
        "masked_positions": masked_positions,
        "continue_available_states": continue_available,
        "one_step_latency_continue_labels": continue_labels,
    }


def main() -> None:
    args = parse_args()
    results = [
        validate_dataset(args.root, dataset)
        for dataset in ("math", "gsm8k")
        if (args.root / "raw" / dataset).exists()
    ]
    print(json.dumps({"ok": True, "datasets": results}, indent=2))


if __name__ == "__main__":
    main()
