from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_td import RAW_STATE_FEATURE_NAMES, build_raw_state_features


DATASETS = ("math", "gsm8k")


def parse_nested(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def load_exact_rows(root: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        for path in sorted((root / "raw" / dataset / "exact").glob(
            "id_*/greedy_local_oracle_decisions.csv"
        )):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame.insert(0, "dataset", dataset)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no exact oracle rows found below {root}")
    return pd.concat(frames, ignore_index=True)


def _profile_globals(root: Path, dataset: str) -> dict:
    path = root / dataset / "frozen_hardware_profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    return {
        "draft_ms": float(profile["mean_draft_forward_latency_ms"]),
        "verify_ms": float(profile["mean_verify_latency_ms"]),
        "tokens": float(profile["mean_tokens_per_verify"]),
    }


def build_raw_oracle_dataset(root: Path, output_dir: Path) -> tuple[Path, Path]:
    rows = load_exact_rows(root)
    vectors = []
    metadata = []
    profiles = {dataset: _profile_globals(root, dataset) for dataset in DATASETS}
    for row in rows.itertuples(index=False):
        previous = parse_nested(row.raw_previous_state)
        current = parse_nested(row.raw_current_state)
        profile = profiles[str(row.dataset)]
        vector = build_raw_state_features(
            raw_previous_state=previous,
            raw_current_state=current,
            has_previous_state=bool(int(row.has_previous_state)),
            proposal_length=int(row.accumulated_proposal_length),
            max_spec_len=64,
            refinement_step=int(row.refinement_step),
            max_refinement_steps=16,
            factual_draft_latency_ema_ms=profile["draft_ms"],
            factual_verifier_latency_ema_ms=profile["verify_ms"],
            factual_tokens_per_verifier_ema=profile["tokens"],
        )
        if any(
            not 0.0 <= float(token[1]) <= 1.0
            or not 0.0 <= float(token[2]) <= float(token[1])
            or not 0.0 <= float(token[3]) <= 1.0
            for token in current
        ):
            raise RuntimeError("invalid probability or entropy in raw snapshot")
        if not bool(int(row.has_previous_state)) and previous != current:
            raise RuntimeError("first raw state must copy current into previous")
        if int(row.active_block_end) - int(row.active_block_start) != 8:
            raise RuntimeError("raw oracle state is not a full 8-token block")
        vectors.append(vector)
        metadata.append({
            "dataset": str(row.dataset),
            "problem_id": int(row.sample_id),
            "round_id": int(row.round_id),
            "decision_id": int(row.decision_id),
            "decision_depth": int(row.decision_depth),
            "on_behavior_path": int(row.on_behavior_path),
            "oracle_action": str(row.oracle_action),
            "oracle_label": str(row.oracle_label),
            "q_stop": float(row.q_stop_profile_tokens),
            "q_continue": float(row.q_continue_profile_tokens),
            "oracle_advantage": float(row.oracle_advantage_tokens),
            "state_key": str(row.state_key),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_frame = pd.DataFrame(metadata)
    metadata_path = output_dir / "raw_oracle_metadata.csv"
    dataset_path = output_dir / "raw_oracle_states.npz"
    metadata_frame.to_csv(metadata_path, index=False)
    np.savez_compressed(
        dataset_path,
        X=np.asarray(vectors, dtype=np.float32),
        q_stop=metadata_frame.q_stop.to_numpy(np.float32),
        q_continue=metadata_frame.q_continue.to_numpy(np.float32),
        problem_id=metadata_frame.problem_id.to_numpy(np.int64),
        on_behavior_path=metadata_frame.on_behavior_path.to_numpy(np.int8),
        dataset=metadata_frame.dataset.to_numpy(str),
        feature_names=np.asarray(RAW_STATE_FEATURE_NAMES),
    )
    summary = metadata_frame.groupby("dataset").agg(
        problems=("problem_id", "nunique"),
        states=("problem_id", "size"),
        behavior_states=("on_behavior_path", "sum"),
    ).reset_index()
    summary.to_csv(output_dir / "raw_oracle_dataset_summary.csv", index=False)
    return dataset_path, metadata_path


def load_raw_npz(path: Path, behavior_only: bool = True):
    payload = np.load(path)
    mask = np.ones(len(payload["X"]), dtype=bool)
    if behavior_only:
        mask &= payload["on_behavior_path"].astype(bool)
    return {
        name: payload[name][mask]
        for name in (
            "X", "q_stop", "q_continue", "problem_id", "dataset"
        )
    }
