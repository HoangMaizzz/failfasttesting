"""Streaming compressed raw-state writer used by the Kaggle collector."""
from __future__ import annotations

import atexit
import hashlib
import json
from pathlib import Path

import numpy as np


_WRITERS: list["RawShardWriter"] = []


def _state_id(dataset: str, row: dict) -> str:
    raw = f"{dataset}|{row.get('problem_id')}|{row.get('round_id')}|{row.get('step')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


class RawShardWriter:
    def __init__(self, root: str | Path, dataset: str, shard_rows: int = 128):
        self.dataset = dataset
        self.shard_rows = int(shard_rows)
        self.out = Path(root) / dataset
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.shard_index = 0
        self.metadata_path = self.out / "index.jsonl"
        self.metadata_path.write_text("", encoding="utf-8")
        _WRITERS.append(self)

    def append(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        rows = self.rows
        self.rows = []
        width = max(len(row["proposal_token_ids"]) for row in rows)

        def pad(values, fill):
            return list(values) + [fill] * (width - len(values))

        prefix_offsets = [0]
        prefix_flat = []
        for row in rows:
            prefix = [int(value) for value in row.get("prefix_token_ids", [])]
            prefix_flat.extend(prefix)
            prefix_offsets.append(len(prefix_flat))

        name = f"shard_{self.shard_index:05d}.npz"
        np.savez_compressed(
            self.out / name,
            proposal_token_ids=np.asarray(
                [pad(row["proposal_token_ids"], 0) for row in rows], dtype=np.int64
            ),
            proposal_token_ids_before_fill=np.asarray(
                [pad(row.get("proposal_token_ids_before_fill", row["proposal_token_ids"]), 151665)
                 for row in rows], dtype=np.int64
            ),
            drafter_observed_prob=np.asarray(
                [pad(row["drafter_observed_prob"], 0.0) for row in rows], dtype=np.float16
            ),
            proposal_mask=np.asarray(
                [pad(row.get("proposal_mask", []), False) for row in rows], dtype=np.bool_
            ),
            proposal_mask_before_fill=np.asarray(
                [pad(row.get("proposal_mask_before_fill", row.get("proposal_mask", [])), False)
                 for row in rows], dtype=np.bool_
            ),
            committed_position_mask=np.asarray(
                [pad(row.get("committed_position_mask", []), False) for row in rows], dtype=np.bool_
            ),
            proposal_token_ids_after_fill=np.asarray(
                [pad(row.get("proposal_token_ids_after_fill", row["proposal_token_ids"]), 0)
                 for row in rows], dtype=np.int64
            ),
            prefix_token_ids_flat=np.asarray(prefix_flat, dtype=np.int32),
            prefix_token_ids_offsets=np.asarray(prefix_offsets, dtype=np.int64),
            hidden_states=np.asarray([row["hidden_states"] for row in rows], dtype=np.float16),
            hidden_layer_indices=np.asarray(
                [row["hidden_layer_indices"] for row in rows], dtype=np.int64
            ),
            topk_token_ids=np.asarray(
                [row["topk_token_ids"] for row in rows], dtype=np.int64
            ),
            topk_logits=np.asarray(
                [row["topk_logits"] for row in rows], dtype=np.float16
            ),
        )
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            for offset, row in enumerate(rows):
                metadata = dict(row["metadata"])
                metadata.update({"shard": name, "row": offset})
                handle.write(json.dumps(metadata) + "\n")
        self.shard_index += 1


def flush_all() -> None:
    for writer in _WRITERS:
        writer.flush()


atexit.register(flush_all)
