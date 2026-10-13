"""Immutable prompt-grouped observations and teacher targets for the A/B/C study.

No LLM imports. Features are selected by an explicit allowlist; teachers and
future observations never enter the feature packer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import numpy as np
import torch

SCHEMA = "agreement_profile_abc_v1"
FEATURES = ("metadata", "distribution", "content", "hidden")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def prefix_length(matches):
    matches = np.asarray(matches, dtype=bool)
    bad = np.flatnonzero(~matches)
    return int(bad[0]) if len(bad) else len(matches)


@dataclass
class Record:
    uid: str
    question: str
    split: str
    meta: dict
    features: dict[str, np.ndarray]
    margin: np.ndarray
    matches: np.ndarray
    candidate: np.ndarray
    native: np.ndarray
    accepted: int

    @property
    def length(self):
        return len(self.candidate)


def validate_record(record, mask_id=151665):
    n = record.length
    if not n or record.split not in ("train", "validation", "test"):
        raise ValueError("Invalid record length/split")
    if any(len(x) != n for x in (record.native, record.margin, record.matches)):
        raise ValueError("Record alignment mismatch")
    if mask_id in record.candidate:
        raise ValueError("STOP candidate contains MASK")
    committed = record.native != mask_id
    if np.any(record.native[committed] != record.candidate[committed]):
        raise ValueError("STOP changed a committed token")
    if prefix_length(record.matches) != record.accepted:
        raise ValueError("Teacher local matches disagree with production K")
    if set(record.features) != set(FEATURES):
        raise ValueError("Unexpected feature schema")
    metadata = record.features["metadata"]
    if metadata.shape[1] < 14 or not np.array_equal(metadata[:, 0], record.native == mask_id):
        raise ValueError("Observation MASK flag disagrees with live tokens")
    if not np.array_equal(metadata[:, 1], record.native != mask_id):
        raise ValueError("Observation commit flag disagrees with live tokens")
    for name, array in record.features.items():
        if array.ndim != 2 or len(array) != n or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} observation")
    if not np.isfinite(record.margin).all():
        raise ValueError("Nonfinite teacher margin")


def validate_edge(source, child, atol=0.01, mask_id=151665):
    if not np.any(source.native == mask_id):
        raise ValueError("R source has no live MASK")
    if (source.question != child.question or source.split != child.split
            or source.length != child.length
            or source.meta["context_hash"] != child.meta["context_hash"]):
        raise ValueError("R crossed prompt/split/context/length")
    if child.meta["native_forward_index"] != source.meta["native_forward_index"] + 1:
        raise ValueError("R endpoints are not adjacent native forwards")
    committed = source.native != mask_id
    if np.any(child.native[committed] != source.native[committed]):
        raise ValueError("R changed a committed native token")
    if np.sum(child.native == mask_id) > np.sum(source.native == mask_id):
        raise ValueError("R increased mask count")
    changed = np.flatnonzero(source.candidate != child.candidate)
    frontier = int(changed[0]) if len(changed) else source.length
    if not np.allclose(source.margin[:frontier], child.margin[:frontier],
                       atol=atol, rtol=0):
        raise ValueError("Causal-prefix margin invariance failed")
    if np.any(source.matches[:frontier] != child.matches[:frontier]):
        raise ValueError("Causal-prefix argmax invariance failed")
    return frontier


def split_questions(questions, validation_count, test_count):
    if validation_count < 1 or test_count < 1 or len(questions) <= validation_count + test_count:
        raise ValueError("Need nonempty train, validation and test prompt splits")
    ids = [q["question_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prompt IDs")
    # Exact duplicate prompt text must not masquerade as independent questions.
    hashes = [identity(q["prompt"]) for q in questions]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate prompt text; deduplicate before splitting")
    a, b = len(ids) - validation_count - test_count, len(ids) - test_count
    return {q: ("train" if i < a else "validation" if i < b else "test")
            for i, q in enumerate(ids)}


def write_question(folder, question_id, records, edges, ledger):
    """One atomic question file: interruption cannot publish a half question."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    for index, record in enumerate(records):
        validate_record(record)
        tag = f"s{index}_"
        for key, value in record.features.items():
            arrays[tag + key] = value.astype(np.float16 if key in ("hidden", "content")
                                             else np.float32)
        for key, value in (("margin", record.margin), ("matches", record.matches),
                           ("candidate", record.candidate), ("native", record.native)):
            arrays[tag + key] = value
        rows.append(dict(uid=record.uid, question=record.question, split=record.split,
                         meta=record.meta, accepted=record.accepted, tag=tag))
    payload = dict(schema=SCHEMA, question_id=question_id, records=rows,
                   edges=edges, ledger=ledger)
    arrays["index_json"] = np.asarray(json.dumps(payload, allow_nan=False))
    destination = folder / (identity(question_id)[:20] + ".npz")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def load_corpus(folder):
    folder = Path(folder)
    manifest = folder / "manifest.json"
    atol = (json.loads(manifest.read_text(encoding="utf-8")).get("collection_config", {}).get("causal_atol", .01)
            if manifest.is_file() else .01)
    records, edges, completed = {}, [], []
    for path in sorted((folder / "questions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        payload = json.loads(str(data.pop("index_json")))
        if payload["schema"] != SCHEMA:
            raise ValueError("Unsupported corpus schema")
        completed.append(payload["question_id"])
        for row in payload["records"]:
            tag = row.pop("tag")
            record = Record(**row,
                            features={key: data[tag + key].astype(np.float32)
                                      for key in FEATURES},
                            margin=data[tag + "margin"].astype(np.float32),
                            matches=data[tag + "matches"].astype(bool),
                            candidate=data[tag + "candidate"].astype(np.int64),
                            native=data[tag + "native"].astype(np.int64))
            validate_record(record)
            if record.uid in records:
                raise ValueError("Duplicate state ID")
            records[record.uid] = record
        edges.extend(payload["edges"])
    seen_splits = {}
    for record in records.values():
        if seen_splits.setdefault(record.question, record.split) != record.split:
            raise ValueError("Prompt leaked across splits")
    seen_edges = set()
    for edge in edges:
        pair = edge["source"], edge["child"]
        if pair in seen_edges or edge["action"] != "R":
            raise ValueError("Duplicate/non-R study edge")
        seen_edges.add(pair)
        validate_edge(records[pair[0]], records[pair[1]], atol)
    return records, edges, completed


def copy_corpus(source, destination):
    """Copy only the corpus subtree from a previous result; never checkpoints."""
    source, destination = Path(source), Path(destination)
    if source.is_dir():
        root = source / "corpus" if (source / "corpus").is_dir() else source
        if not (root / "manifest.json").is_file():
            raise ValueError("Input has no agreement-profile corpus manifest")
        if destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("Destination must not be inside source corpus")
        destination.mkdir(parents=True, exist_ok=False)
        for path in root.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                name = PurePosixPath(entry.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename
                        or any(":" in part for part in name.parts)):
                    raise ValueError("Unsafe archive member")
                if not name.parts or name.parts[0] != "corpus" or entry.is_dir():
                    continue
                target = destination.joinpath(*name.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    if not (destination / "manifest.json").is_file():
        raise ValueError("Input has no agreement-profile corpus manifest")


def group_folds(records, folds, seed):
    prompts = sorted({r.question for r in records})
    folds = min(folds, len(prompts))
    if folds < 2:
        raise ValueError("OOF needs at least two training prompts")
    np.random.default_rng(seed).shuffle(prompts)
    return [{q for i, q in enumerate(prompts) if i % folds == f}
            for f in range(folds)]


class Normalizer:
    def __init__(self, records):
        self.stats = {}
        for name in FEATURES:
            # Accumulate sufficient statistics without concatenating all hidden rows.
            arrays = [r.features[name] for r in records]
            count = sum(len(a) for a in arrays)
            total = sum((a.astype(np.float64).sum(0) for a in arrays))
            square = sum(((a.astype(np.float64) ** 2).sum(0) for a in arrays))
            mean = total / count
            std = np.sqrt(np.maximum(square / count - mean ** 2, 0)).clip(1e-3)
            self.stats[name] = mean.astype(np.float32), std.astype(np.float32)
        self.margin_scale = max(1., float(np.median(np.abs(np.concatenate([r.margin for r in records])))))

    def state_dict(self):
        return {"stats": {k: (torch.tensor(a), torch.tensor(b))
                          for k, (a, b) in self.stats.items()},
                "margin_scale": self.margin_scale}


def pack_features(records, normalizer, device, profiles=None):
    """Inputs ONLY. Mutating any teacher field cannot change this output."""
    width = max(r.length for r in records)
    batch = {"lengths": torch.tensor([r.length for r in records], device=device)}
    batch["valid"] = torch.arange(width, device=device)[None] < batch["lengths"][:, None]
    for name in FEATURES:
        dim = records[0].features[name].shape[1]
        values = np.zeros((len(records), width, dim), dtype=np.float32)
        mean, std = normalizer.stats[name]
        for i, record in enumerate(records):
            values[i, :record.length] = ((record.features[name] - mean) / std).clip(-20, 20)
        batch[name] = torch.as_tensor(values, device=device)
    if profiles is not None:
        values = np.zeros((len(records), width), dtype=np.float32)
        for i, record in enumerate(records):
            profile = np.asarray(profiles[record.uid], dtype=np.float32)
            if len(profile) != record.length:
                raise ValueError("Source profile length mismatch")
            values[i, :record.length] = profile
        batch["profile"] = torch.as_tensor(values, device=device)
    return batch


def pack_targets(targets, width, device):
    margins = np.zeros((len(targets), width), dtype=np.float32)
    matches = np.zeros_like(margins)
    for i, record in enumerate(targets):
        margins[i, :record.length] = record.margin
        matches[i, :record.length] = record.matches
    return {"margin": torch.as_tensor(margins, device=device),
            "match": torch.as_tensor(matches, device=device),
            "K": torch.tensor([r.accepted for r in targets], device=device)}
