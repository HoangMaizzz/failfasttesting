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

        max_feature_tokens = max(
            max(len(layer) for layer in row["hidden_states"])
            for row in rows
        )

        def pad_layered_features(layers, target_tokens, fill):
            padded = []
            for layer in layers:
                layer = [list(token_values) for token_values in layer]
                if layer:
                    feature_width = len(layer[0])
                else:
                    feature_width = 0
                layer.extend(
                    [[fill] * feature_width] * (target_tokens - len(layer))
                )
                padded.append(layer)
            return padded

        max_topk_tokens = max(len(row["topk_token_ids"]) for row in rows)

        prefix_offsets = [0]
        prefix_flat = []
        for row in rows:
            prefix = [int(value) for value in row.get("prefix_token_ids", [])]
            prefix_flat.extend(prefix)
            prefix_offsets.append(len(prefix_flat))

        name = f"shard_{self.shard_index:05d}.npz"
        payload = dict(
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
            hidden_states=np.asarray([
                pad_layered_features(row["hidden_states"], max_feature_tokens, 0.0)
                for row in rows
            ], dtype=np.float16),
            hidden_layer_indices=np.asarray(
                [row["hidden_layer_indices"] for row in rows], dtype=np.int64
            ),
            topk_token_ids=np.asarray([
                pad_layered_features([row["topk_token_ids"]], max_topk_tokens, 0)[0]
                for row in rows
            ], dtype=np.int64),
            topk_logits=np.asarray([
                pad_layered_features([row["topk_logits"]], max_topk_tokens, 0.0)[0]
                for row in rows
            ], dtype=np.float16),
        )
        if any("native_active_hidden_states" in row for row in rows):
            layer_count = max(len(row.get("native_active_hidden_states", [])) for row in rows)
            native_tokens = max((len(layer)
                for row in rows for layer in row.get("native_active_hidden_states", [])),
                default=0)
            hidden_width = max((len(token)
                for row in rows for layer in row.get("native_active_hidden_states", [])
                for token in layer), default=0)
            topk_tokens = max((len(row.get("native_active_topk_token_ids", []))
                for row in rows), default=0)
            topk_width = max((len(token)
                for row in rows for token in row.get("native_active_topk_token_ids", [])),
                default=0)
            native_hidden = np.zeros((len(rows), layer_count, native_tokens, hidden_width),
                                     dtype=np.float16)
            native_hidden_valid = np.zeros((len(rows), layer_count, native_tokens),
                                           dtype=np.bool_)
            native_layers = np.full((len(rows), layer_count), -1, dtype=np.int64)
            native_ids = np.zeros((len(rows), topk_tokens, topk_width), dtype=np.int64)
            native_logits = np.zeros((len(rows), topk_tokens, topk_width), dtype=np.float16)
            native_topk_valid = np.zeros((len(rows), topk_tokens), dtype=np.bool_)
            for ri, row in enumerate(rows):
                layers = row.get("native_active_hidden_states", [])
                indices = row.get("native_active_hidden_layer_indices", [])
                for li, layer in enumerate(layers):
                    if li < len(indices):
                        native_layers[ri, li] = int(indices[li])
                    for ti, token in enumerate(layer):
                        native_hidden[ri, li, ti, :len(token)] = token
                        native_hidden_valid[ri, li, ti] = True
                ids = row.get("native_active_topk_token_ids", [])
                logits = row.get("native_active_topk_logits", [])
                if len(ids) != len(logits):
                    raise ValueError("Native top-k token and logit rows have different lengths")
                for ti, (token_ids, values) in enumerate(zip(ids, logits)):
                    if len(token_ids) != len(values):
                        raise ValueError("Native top-k token and logit widths differ")
                    native_ids[ri, ti, :len(token_ids)] = token_ids
                    native_logits[ri, ti, :len(values)] = values
                    native_topk_valid[ri, ti] = True
            payload.update(native_active_hidden_states=native_hidden,
                native_active_hidden_valid=native_hidden_valid,
                native_active_hidden_layer_indices=native_layers,
                native_active_topk_token_ids=native_ids,
                native_active_topk_logits=native_logits,
                native_active_topk_valid=native_topk_valid)
        np.savez_compressed(self.out / name, **payload)
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
