"""Offline, CPU-only access to original TwoSource experiences; no LLM imports.

``resolve_input`` accepts a run, an arbitrarily named ZIP, or a directory of
downloads. Exactly one complete original run must be present, even inside ZIP
wrappers. ZIPs are read in place, never extracted; checkpoints are never read.
``load_dataset`` loads each needed NPZ member once per shard, then slices it.

Labels and teachers are targets, not model inputs. Missing acceptance is None
(mask it for verifier supervision); it does not remove drafter trajectories.
Native scalar columns 3/4 are hidden/top-k validity and 5/6 are age clocks in
units of 1/8 transition. Advancing those clocks is bookkeeping, not dynamics.
Optional caches are loader-created torch.save dictionaries in a trusted local
directory, read with weights_only=True and bound to a source fingerprint.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import tempfile
from typing import Iterable
import zipfile

import numpy as np
import torch


AVAILABLE_AT_PARENT = "AVAILABLE_AT_PARENT"
ACTION_KNOWN = "ACTION_KNOWN"
TEACHER_ONLY_DRAFTER = "TEACHER_ONLY_DRAFTER"
TEACHER_ONLY_VERIFIER = "TEACHER_ONLY_VERIFIER"
FORBIDDEN_FUTURE = "FORBIDDEN_FUTURE"
_SCHEMA = "interactive_acceptance_two_source_v1"
_REQUIRED = ("states.jsonl", "edges.jsonl", "labels.jsonl",
             "teacher_targets.jsonl", "summary.json")
_ARRAYS = ("ids", "hidden", "gaps", "history", "scalars", "context",
           "offsets", "aligned_topk_token_ids")
_OPTIONAL_ARRAYS = ("lengths", "teacher_valid", "teacher_margin", "teacher_features")
_CACHE_VERSION = 2  # Audit schema includes per-action denominators and E lengths.


@dataclass
class RawState:
    uid: str
    question: str
    round_id: int
    ids: torch.Tensor                 # long [L, 2]: native, same-forward STOP
    hidden: torch.Tensor              # half [L, 3, 1536]
    surface: torch.Tensor             # float [L, 36]: gaps32, history4
    scalars: torch.Tensor             # float [L, 16]
    context: torch.Tensor             # float [8]
    prefix_ids: torch.Tensor          # long [n], from states.jsonl
    topk_ids: torch.Tensor            # long [L, 32]
    gaps: torch.Tensor                # float [L, 32]
    accepted: int | None
    teacher_features: torch.Tensor | None = None  # float [L, 35], target only
    teacher_margin: torch.Tensor | None = None    # float [L], target only

    @property
    def length(self) -> int:
        return int(self.ids.shape[0])


@dataclass
class Dataset:
    states: dict[str, RawState]
    edges: list[tuple[str, str, str]]
    audit: dict
    source: str

    @property
    def question_ids(self) -> list[str]:
        return sorted({state.question for state in self.states.values()})


def feature_registry() -> dict[str, str]:
    """Field -> classification. Identifier metadata is also withheld from inputs.

    ``surface`` is the public gaps/history input; the separate raw ``gaps``
    tensor is retained for inspecting targets and is not an additional input.
    Teacher classifications describe supervision ownership, never input access.
    """
    registry = {name: AVAILABLE_AT_PARENT for name in
                ("hidden", "surface", "scalars", "context", "prefix_ids", "ids", "topk_ids")}
    registry.update(actions=ACTION_KNOWN,
                    teacher_features=TEACHER_ONLY_DRAFTER,
                    teacher_margin=TEACHER_ONLY_VERIFIER,
                    accepted=TEACHER_ONLY_VERIFIER,
                    question=FORBIDDEN_FUTURE, uid=FORBIDDEN_FUTURE,
                    child_hidden=FORBIDDEN_FUTURE, child_ids=FORBIDDEN_FUTURE)
    return registry


def assert_input_features(names: Iterable[str], module: str = "drafter") -> None:
    if module not in ("drafter", "verifier"):
        raise ValueError(f"Unknown module {module!r}; expected drafter or verifier")
    if isinstance(names, str):
        names = [names]
    registry = feature_registry()
    allowed = {AVAILABLE_AT_PARENT}
    if module == "drafter":
        allowed.add(ACTION_KNOWN)
    for name in names:
        if not isinstance(name, str) or name not in registry:
            raise ValueError(f"Unknown input feature: {name!r}")
        if registry[name] not in allowed:
            raise ValueError(f"Invalid {module} input {name!r}: {registry[name]}")


def _safe_name(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe relative member path: {name!r}")
    return PurePosixPath(name).as_posix()


@dataclass(frozen=True)
class _Run:
    path: Path
    prefix: str = ""

    @property
    def source(self) -> str:
        return str(self.path) + (f"!/{self.prefix.rstrip('/')}" if self.prefix else "")


class _Reader:
    def __init__(self, run: _Run):
        self.run = run
        self.zip = None

    def __enter__(self):
        if self.run.path.is_file():
            self.zip = zipfile.ZipFile(self.run.path)
        return self

    def __exit__(self, *args):
        if self.zip is not None:
            self.zip.close()

    def open(self, name):
        name = _safe_name(name)
        if self.zip is not None:
            try:
                return self.zip.open(self.run.prefix + name)
            except KeyError as exc:
                raise ValueError(f"Missing run member {name!r} in {self.run.source}") from exc
        target = self.run.path.joinpath(*name.split("/")).resolve()
        if not target.is_relative_to(self.run.path):
            raise ValueError(f"Member escapes extracted root: {name!r}")
        try:
            return target.open("rb")
        except FileNotFoundError as exc:
            raise ValueError(f"Missing run member {name!r} in {self.run.source}") from exc

    def json(self, name):
        try:
            with self.open(name) as stream:
                row = json.load(stream)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed {name}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {name}")
        return row

    def lines(self, name):
        with self.open(name) as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Malformed {name}:{lineno}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object in {name}:{lineno}")
                yield row


def _zip_names(archive: zipfile.ZipFile) -> set[str]:
    names = set()
    for info in archive.infolist():
        # ZipInfo normalizes Windows separators on Windows; validate the stored
        # spelling too so malformed members cannot hide behind that conversion.
        original = info.orig_filename.rstrip("/") if info.is_dir() else info.orig_filename
        _safe_name(original)
        name = info.filename.rstrip("/") if info.is_dir() else info.filename
        _safe_name(name)
        if info.filename in names:
            raise ValueError(f"Duplicate ZIP member: {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"ZIP symlink is not a run member: {info.filename!r}")
        names.add(info.filename)
    return names


def _candidates(path: Path, problems: list[str]) -> list[_Run]:
    runs = []
    if path.is_file():
        try:
            with zipfile.ZipFile(path) as archive:
                names = _zip_names(archive)
                prefixes = sorted({name[:-len("summary.json")] for name in names
                                   if not name.endswith("/")
                                   and PurePosixPath(name).name == "summary.json"},
                                  key=lambda name: (name.count("/"), name))
                for prefix in prefixes:
                    run = _Run(path, prefix)
                    try:
                        with _Reader(run) as reader:
                            summary = reader.json("summary.json")
                    except (ValueError, OSError) as exc:
                        problems.append(f"{run.source}: {exc}")
                        continue
                    missing = [name for name in _REQUIRED if prefix + name not in names]
                    has_npz = any(name.startswith(prefix + "experience/")
                                  and name.lower().endswith(".npz") for name in names)
                    if _complete(summary, missing, has_npz, run.source, problems):
                        runs.append(run)
        except (zipfile.BadZipFile, ValueError, OSError) as exc:
            # An unsafe archive must not become a candidate through another root.
            problems.append(f"{path}: {exc}")
            return []
    elif (path / "summary.json").is_file():
        run = _Run(path)
        try:
            with _Reader(run) as reader:
                summary = reader.json("summary.json")
            missing = [name for name in _REQUIRED if not (path / name).is_file()]
            has_npz = (path / "experience").is_dir() and any(
                item.is_file() and item.suffix.lower() == ".npz"
                for item in (path / "experience").rglob("*"))
            if _complete(summary, missing, has_npz, run.source, problems):
                runs.append(run)
        except (ValueError, OSError) as exc:
            problems.append(f"{path}: {exc}")
    return runs


def _complete(summary, missing, has_npz, source, problems):
    if summary.get("schema") != _SCHEMA or summary.get("status") != "complete":
        problems.append(f"{source}: expected ORIGINAL complete {_SCHEMA}, "
                        f"got schema={summary.get('schema')!r}, status={summary.get('status')!r}")
        return False
    if missing or not has_npz:
        problems.append(f"{source}: missing {missing + ([] if has_npz else ['experience NPZ'])}")
        return False
    return True


def _resolve_run(path) -> _Run:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    problems = []
    runs = _candidates(path, problems)
    if path.is_dir():
        # Inspect the named root first, then nested roots and arbitrarily named ZIPs.
        for directory, directories, files in os.walk(path, followlinks=False):
            directories[:] = sorted(name for name in directories
                                    if not (Path(directory) / name).is_symlink())
            current = Path(directory)
            if current != path and "summary.json" in files:
                runs.extend(_candidates(current.resolve(), problems))
            for name in sorted(files):
                item = current / name
                if item.suffix.lower() == ".zip" and not item.is_symlink():
                    runs.extend(_candidates(item.resolve(), problems))
    if len(runs) > 1:
        raise ValueError("Ambiguous input: multiple ORIGINAL complete runs: "
                         + "; ".join(run.source for run in runs))
    if not runs:
        detail = "; ".join(problems[:6]) or "missing summary.json and original experience members"
        raise ValueError(f"No ORIGINAL complete {_SCHEMA} run in {path}: {detail}")
    return runs[0]


def resolve_input(path) -> Path:
    """Return the unique ZIP or exact extracted root; ZIP prefix stays internal."""
    return _resolve_run(path).path


def _integer(value, description, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"Invalid integer {description}: {value!r}")
    return int(value)


def _identity(row, description) -> str:
    uid = row.get("state_id")
    if not isinstance(uid, str) or not uid:
        raise ValueError(f"Missing state_id in {description}")
    return uid


def _metadata(reader):
    metadata = {}
    shard_rows = set()
    for row in reader.lines("states.jsonl"):
        uid = _identity(row, "states.jsonl")
        if uid in metadata:
            raise ValueError(f"Duplicate state_id: {uid}")
        if not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError(f"Missing question for {uid}")
        _integer(row.get("round_id"), f"round_id for {uid}")
        index = _integer(row.get("row"), f"shard row for {uid}")
        shard = _safe_name(row.get("shard"))
        if not shard.startswith("experience/") or not shard.lower().endswith(".npz"):
            raise ValueError(f"Not an experience NPZ for {uid}: {shard}")
        if (shard, index) in shard_rows:
            raise ValueError(f"Duplicate shard row {shard}:{index}")
        shard_rows.add((shard, index))
        prefix = row.get("prefix_token_ids")
        if not isinstance(prefix, list):
            raise ValueError(f"Missing prefix_token_ids for {uid}")
        for token in prefix:
            _integer(token, f"prefix token ID for {uid}")
            if token > np.iinfo(np.int64).max:
                raise ValueError(f"Prefix token ID exceeds int64 for {uid}")
        if "length" in row:
            _integer(row["length"], f"length for {uid}", minimum=1)
        metadata[uid] = row
    if not metadata:
        raise ValueError("No states in states.jsonl")
    return metadata


def _latest(reader, filename, metadata):
    result = {}
    for row in reader.lines(filename):
        uid = _identity(row, filename)
        if uid not in metadata:
            raise ValueError(f"{filename} references missing state {uid}")
        result[uid] = row
    return result


def _graph(states, edges):
    """Check references, grouping and cycles before any path enumeration."""
    outgoing = defaultdict(list)
    indegree = dict.fromkeys(states, 0)
    seen = set()
    for parent, child, action in edges:
        if parent not in states or child not in states:
            raise ValueError(f"Edge references missing state: {(parent, child, action)}")
        if action not in ("R", "E"):
            raise ValueError(f"Invalid transition action: {action!r}")
        a, b = states[parent], states[child]
        question_a = a["question"] if isinstance(a, dict) else a.question
        question_b = b["question"] if isinstance(b, dict) else b.question
        round_a = a["round_id"] if isinstance(a, dict) else a.round_id
        round_b = b["round_id"] if isinstance(b, dict) else b.round_id
        if question_a != question_b:
            raise ValueError(f"Cross-question edge: {parent} -> {child}")
        if round_a != round_b:
            raise ValueError(f"Cross-round edge: {parent} -> {child}")
        if (parent, child, action) in seen:
            raise ValueError(f"Duplicate edge: {parent} -> {child} ({action})")
        seen.add((parent, child, action))
        outgoing[parent].append((child, action))
        indegree[child] += 1
    ready = deque(uid for uid, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        parent = ready.popleft()
        visited += 1
        for child, _ in outgoing[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(states):
        raise ValueError("Cycle in R/E state graph")
    return outgoing


def _edges(reader, metadata):
    edges = []
    for row in reader.lines("edges.jsonl"):
        if any(not isinstance(row.get(key), str) for key in ("parent", "child", "action")):
            raise ValueError(f"Malformed edge: {row}")
        if "question" in row and row["parent"] in metadata:
            if row["question"] != metadata[row["parent"]]["question"]:
                raise ValueError(f"Edge question disagrees with parent: {row}")
        edges.append((row["parent"], row["child"], row["action"]))
    _graph(metadata, edges)
    return edges


def _array(array, shape, name, integer=False):
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"Invalid {name} shape {array.shape}; expected {shape}")
    if integer:
        if array.dtype.kind not in "iu" or (array < 0).any():
            raise ValueError(f"Invalid integer token/offset values in {name}")
        if array.size and int(array.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"Integer exceeds int64 in {name}")
    elif array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"Nonfinite or nonnumeric {name}")


def _shard_arrays(reader, shard):
    try:
        with reader.open(shard) as stream:
            # A ZIP member need not be cheaply seekable for NumPy's inner ZIP.
            source = io.BytesIO(stream.read()) if reader.zip is not None else stream
            with np.load(source, allow_pickle=False) as npz:
                members = set(npz.files)
                if len(members) != len(npz.files):
                    raise ValueError(f"Duplicate NPZ members in {shard}")
                missing = set(_ARRAYS) - members
                if missing:
                    raise ValueError(f"Missing NPZ members in {shard}: {sorted(missing)}")
                # Never index NpzFile in the per-state loop: every member once.
                arrays = {name: npz[name] for name in (*_ARRAYS, *_OPTIONAL_ARRAYS)
                          if name in members}
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Malformed experience shard {shard}: {exc}") from exc
    offsets = arrays["offsets"]
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError(f"Invalid offsets in {shard}")
    _array(offsets, offsets.shape, f"{shard}/offsets", integer=True)
    if offsets[0] != 0 or (offsets[1:] <= offsets[:-1]).any():
        raise ValueError(f"Offsets must start at zero and increase in {shard}")
    tokens, rows = int(offsets[-1]), len(offsets) - 1
    shapes = {"ids": (tokens, 2), "hidden": (tokens, 3, 1536),
              "gaps": (tokens, 32), "history": (tokens, 4),
              "scalars": (tokens, 16), "context": (rows, 8),
              "aligned_topk_token_ids": (tokens, 32)}
    for name, shape in shapes.items():
        _array(arrays[name], shape, f"{shard}/{name}",
               integer=name in ("ids", "aligned_topk_token_ids"))
    if "lengths" in arrays:
        _array(arrays["lengths"], (rows,), f"{shard}/lengths", integer=True)
        if not np.array_equal(arrays["lengths"], np.diff(offsets)):
            raise ValueError(f"Lengths disagree with offsets in {shard}")
    # Teacher rows are validated after the authoritative late-JSONL join.
    # Old invalid targets must not invalidate a newer complete teacher record.
    for name, width in (("teacher_valid", None), ("teacher_margin", None),
                        ("teacher_features", 35)):
        if name in arrays:
            shape = (tokens,) if width is None else (tokens, width)
            if arrays[name].shape != shape:
                raise ValueError(f"Invalid {shard}/{name} shape; expected {shape}")
    if "teacher_valid" in arrays:
        valid = arrays["teacher_valid"]
        if valid.dtype.kind not in "biu" or not np.isin(valid, (0, 1)).all():
            raise ValueError(f"Malformed teacher_valid in {shard}")
    return arrays


def _tensor(array, dtype):
    # Copies detach each state from the large shard; release the shard promptly.
    return torch.from_numpy(np.array(array, copy=True)).to(dtype=dtype)


def _state(meta, arrays, label, teacher):
    uid, index = meta["state_id"], meta["row"]
    if index + 1 >= len(arrays["offsets"]):
        raise ValueError(f"Out-of-range shard row for {uid}: {index}")
    lo, hi = map(int, arrays["offsets"][index:index + 2])
    length = hi - lo
    if "length" in meta and meta["length"] != length:
        raise ValueError(f"Metadata length disagrees with offsets for {uid}")
    if label is not None and not isinstance(label.get("label_valid"), bool):
        raise ValueError(f"Malformed label_valid for {uid}")
    accepted = None
    if label is not None and label["label_valid"]:
        accepted = _integer(label.get("accepted_len"), f"acceptance label for {uid}")
        if accepted > length:
            raise ValueError(f"Acceptance exceeds length for {uid}")
    fields = {name: _tensor(arrays[key][lo:hi], dtype) for name, key, dtype in (
        ("ids", "ids", torch.long), ("hidden", "hidden", torch.float16),
        ("gaps", "gaps", torch.float32), ("scalars", "scalars", torch.float32),
        ("topk_ids", "aligned_topk_token_ids", torch.long))}
    history = _tensor(arrays["history"][lo:hi], torch.float32)
    state = RawState(uid=uid, question=meta["question"], round_id=meta["round_id"],
        surface=torch.cat((fields["gaps"], history), dim=1),
        context=_tensor(arrays["context"][index], torch.float32),
        prefix_ids=torch.tensor(meta["prefix_token_ids"], dtype=torch.long),
        accepted=accepted, **fields)
    if teacher is not None:
        # Latest JSONL completely overrides even a partially valid stale shard.
        if teacher.get("margin") is None:
            raise ValueError(f"Missing teacher margin for {uid}")
        try:
            state.teacher_margin = torch.tensor(teacher["margin"], dtype=torch.float32)
            if teacher.get("features") is not None:
                state.teacher_features = torch.tensor(teacher["features"], dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Malformed teacher target for {uid}: {exc}") from exc
    elif "teacher_valid" in arrays:
        valid = arrays["teacher_valid"][lo:hi].astype(bool)
        if valid.any() and not valid.all():
            raise ValueError(f"Partially valid teacher target for {uid}; per-token mask unavailable")
        if valid.all():
            if "teacher_margin" not in arrays:
                raise ValueError(f"Valid teacher missing margin for {uid}")
            state.teacher_margin = _tensor(arrays["teacher_margin"][lo:hi], torch.float32)
            if "teacher_features" in arrays:
                state.teacher_features = _tensor(arrays["teacher_features"][lo:hi], torch.float32)
    _validate_state(state)
    return state


def _segment_start(state):
    value = float(state.context[2]) * 64
    if not math.isfinite(value) or value != int(value) or not 0 <= value <= state.length:
        raise ValueError(f"Malformed immutable segment length for {state.uid}: {value}")
    return int(value)


def _validate_state(state):
    length = state.length
    if length < 1:
        raise ValueError(f"Empty state: {state.uid}")
    shapes = {"ids": ((length, 2), torch.long),
              "hidden": ((length, 3, 1536), torch.float16),
              "surface": ((length, 36), torch.float32),
              "scalars": ((length, 16), torch.float32),
              "context": ((8,), torch.float32),
              "topk_ids": ((length, 32), torch.long),
              "gaps": ((length, 32), torch.float32),
              "prefix_ids": ((state.prefix_ids.numel(),), torch.long),
              "teacher_features": ((length, 35), torch.float32),
              "teacher_margin": ((length,), torch.float32)}
    for name, (shape, dtype) in shapes.items():
        tensor = getattr(state, name)
        if tensor is None and name.startswith("teacher_"):
            continue
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != dtype or tensor.device.type != "cpu"):
            raise ValueError(f"Invalid {name} tensor for {state.uid}; expected {shape}, {dtype}, CPU")
        if not np.isfinite(tensor.detach().numpy()).all():
            raise ValueError(f"Nonfinite {name} for {state.uid}")
        if dtype == torch.long and bool((tensor < 0).any()):
            raise ValueError(f"Negative token IDs in {name} for {state.uid}")
    if state.accepted is not None:
        _integer(state.accepted, f"acceptance label for {state.uid}")
        if state.accepted > length:
            raise ValueError(f"Acceptance exceeds length for {state.uid}")
    if not torch.equal(state.surface[:, :32], state.gaps):
        raise ValueError(f"Surface gaps disagree for {state.uid}")
    _segment_start(state)
    for column in (0, 1, 3, 4):
        flag = state.scalars[:, column]
        if not bool(((flag == 0) | (flag == 1)).all()):
            raise ValueError(f"Nonbinary scalar validity/mask column {column} for {state.uid}")
    if bool((state.scalars[:, 5:7] < 0).any()):
        raise ValueError(f"Negative native cache age for {state.uid}")


def _validate_tokens(states, edges):
    contradictions = []
    for parent, child, action in edges:
        a, b = states[parent], states[child]
        immutable = _segment_start(a) if action == "R" else a.length
        expected = a.length if action == "R" else a.length + 8
        if b.length != expected:
            raise ValueError(f"{action} length invariant failed: {parent} -> {child}; expected {expected}")
        if not torch.equal(a.prefix_ids, b.prefix_ids):
            raise ValueError(f"R/E changed prefix_token_ids: {parent} -> {child}")
        if _segment_start(b) != immutable:
            raise ValueError(f"{action} segment-start invariant failed: {parent} -> {child}")
        if not torch.equal(a.ids[:immutable, 1], b.ids[:immutable, 1]):
            raise ValueError(f"Immutable STOP prefix ID violation: {parent} -> {child}")
        if not torch.equal(a.ids[:immutable, 1], b.ids[:immutable, 0]):
            raise ValueError(f"Immutable native prefix ID violation: {parent} -> {child}")
        if action == "R":
            committed = a.scalars[:, 0] == 0
            if not torch.equal(a.ids[committed, 0], b.ids[committed, 0]):
                raise ValueError(f"R changed committed native token IDs: {parent} -> {child}")
        if a.accepted is not None and b.accepted is not None and immutable:
            reason = None
            if a.accepted < immutable and b.accepted != a.accepted:
                reason = "first rejection is inside the immutable prefix"
            elif a.accepted >= immutable and b.accepted < immutable:
                reason = "previously accepted immutable prefix became rejected"
            if reason:
                contradictions.append(dict(parent=parent, child=child, action=action,
                    immutable_length=immutable, parent_K=a.accepted, child_K=b.accepted,
                    reason=reason))
    return contradictions


def enumerate_paths(dataset: Dataset, allowed_questions=None, horizon: int = 3):
    """All nonempty R/E subpaths of depths 1..horizon, starting at every state.

    Paths contain (tuple of UIDs, tuple of action integers R=0/E=1). Labels are
    never consulted. Branches are supported; paths cannot cross question/round.
    """
    horizon = _integer(horizon, "horizon")
    outgoing = _graph(dataset.states, dataset.edges)
    allowed = set(dataset.question_ids) if allowed_questions is None else set(allowed_questions)
    unknown = allowed - set(dataset.question_ids)
    if unknown:
        raise ValueError(f"Unknown allowed questions: {sorted(unknown)}")
    paths = []
    for uid in sorted(dataset.states):
        if dataset.states[uid].question not in allowed:
            continue
        pending = [((uid,), ())]
        while pending:
            nodes, actions = pending.pop()
            if len(actions) >= horizon:
                continue
            for child, action in sorted(outgoing.get(nodes[-1], ()), reverse=True):
                next_path = (nodes + (child,), actions + (int(action == "E"),))
                paths.append(next_path)
                pending.append(next_path)
    return sorted(paths, key=lambda path: (path[0], path[1]))


def split_questions(question_ids, seed=42, train_fraction=.7, val_fraction=.15):
    """Shuffle sorted unique question IDs, then split whole questions only.

    Train/validation sizes are floors; test receives the remainder. For 100
    questions and default fractions this is exactly 70/15/15. Duplicate inputs
    are an error instead of hiding a potentially ungrouped state-level split.
    """
    questions = list(question_ids)
    if any(not isinstance(q, str) or not q for q in questions):
        raise ValueError("Question IDs must be nonempty strings")
    if len(set(questions)) != len(questions):
        raise ValueError("Duplicate question IDs; split must be grouped by unique question")
    for fraction in (train_fraction, val_fraction):
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("Split fractions must be finite numbers")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Split fractions must be between zero and one")
    if train_fraction + val_fraction > 1:
        raise ValueError("Train and validation fractions exceed one")
    questions.sort()
    random.Random(seed).shuffle(questions)
    ntrain = math.floor(len(questions) * train_fraction)
    nval = math.floor(len(questions) * val_fraction)
    result = dict(train=questions[:ntrain], val=questions[ntrain:ntrain + nval],
                  test=questions[ntrain + nval:])
    groups = [set(part) for part in result.values()]
    if (any(groups[i] & groups[j] for i in range(3) for j in range(i))
            or set().union(*groups) != set(questions)):
        raise ValueError("Question grouping invariant failed")
    return result


def _audit(dataset, contradictions, member_counts):
    states = list(dataset.states.values())
    actions = Counter(action for _, _, action in dataset.edges)
    paths = enumerate_paths(dataset)
    sequences = Counter("".join("E" if a else "R" for a in action) for _, action in paths)
    horizons = {f"H{h}": sum(len(action) == h for _, action in paths) for h in (1, 2, 3)}
    teacher_states = [s for s in states if s.teacher_features is not None and s.teacher_margin is not None]
    labeled = sum(s.accepted is not None for s in states)
    native = {}
    for name, valid_col, age_col in (("hidden", 3, 5), ("topk", 4, 6)):
        valid = torch.cat([s.scalars[:, valid_col].bool() for s in states])
        ages = torch.cat([s.scalars[:, age_col] for s in states])[valid]
        native[name] = dict(valid_tokens=int(valid.sum()), invalid_tokens=int((~valid).sum()),
            fresh_tokens=int((ages == 0).sum()), cached_tokens=int((ages > 0).sum()),
            age_min=float(ages.min()) if len(ages) else None,
            age_max=float(ages.max()) if len(ages) else None,
            age_mean=float(ages.mean()) if len(ages) else None,
            age_histogram=dict(sorted(Counter(float(age) for age in ages.tolist()).items())))
    deltas = Counter()
    extension_lengths = Counter()
    for parent, child, action in dataset.edges:
        a, b = dataset.states[parent], dataset.states[child]
        if action == "E":
            extension_lengths[(a.length, b.length)] += 1
        if a.accepted is not None and b.accepted is not None:
            deltas["labeled_edges"] += 1
            deltas[f"{action}_labeled_edges"] += 1
            delta = b.accepted - a.accepted
            deltas["positive" if delta > 0 else "negative" if delta < 0 else "zero"] += 1
            if delta > 0:
                deltas[f"{action}_positive"] += 1
    return dict(states=len(states), questions=len(dataset.question_ids), edges=len(dataset.edges),
        R=actions["R"], E=actions["E"], lengths=dict(sorted(Counter(s.length for s in states).items())),
        E_transition_lengths={f"{parent}->{child}": count
                              for (parent, child), count in sorted(extension_lengths.items())},
        tokens=sum(s.length for s in states), labeled_states=labeled, missing_labels=len(states) - labeled,
        **horizons, action_sequences=dict(sorted(sequences.items())),
        full_teacher_states=len(teacher_states), full_teacher_tokens=sum(s.length for s in teacher_states),
        teacher_margin_states=sum(s.teacher_margin is not None for s in states),
        teacher_margin_tokens=sum(s.length for s in states if s.teacher_margin is not None),
        delta_K={**{key: deltas[key] for key in
                    ("labeled_edges", "positive", "negative", "zero", "R_positive", "E_positive",
                     "R_labeled_edges", "E_labeled_edges")},
                 # Only edges with BOTH endpoint labels enter each denominator.
                 **{f"{action}_positive_rate": (deltas[f"{action}_positive"] /
                      deltas[f"{action}_labeled_edges"] if deltas[f"{action}_labeled_edges"] else None)
                    for action in ("R", "E")}},
        native_cache=native, scalar_age_is_dynamics=False,
        native_age_units="1/8 transition; columns 5/6 are bookkeeping clocks",
        immutable_stop_prefix_checks=len(dataset.edges),
        immutable_prefix_K_contradictions=contradictions,
        immutable_prefix_K_contradiction_count=len(contradictions),
        npz_members_loaded=sum(member_counts.values()), npz_members_by_shard=member_counts)


def _fingerprint(reader, metadata, max_questions):
    names = sorted(set(_REQUIRED) | {row["shard"] for row in metadata.values()})
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(version=_CACHE_VERSION, source=reader.run.source,
                                  max_questions=max_questions), sort_keys=True).encode())
    for name in names:
        digest.update(name.encode())
        if reader.zip is not None:
            try:
                info = reader.zip.getinfo(reader.run.prefix + name)
            except KeyError as exc:
                raise ValueError(f"Missing experience member {name}") from exc
            # Central-directory CRC changes with contents; checkpoints excluded.
            digest.update(f"{info.CRC}:{info.file_size}:{info.compress_size}".encode())
        else:
            with reader.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path, max_questions=100, cache_dir=None) -> Dataset:
    """Load up to max_questions whole questions (sorted); None loads all.

    Latest labels.jsonl and teacher_targets.jsonl win over earlier records and
    shard targets. Unresolved labels are None. Partial shard teachers require a
    complete late JSONL replacement, since this API has no token validity mask.
    Source graph validation precedes question selection; no model is required.
    """
    if max_questions is not None:
        max_questions = _integer(max_questions, "max_questions", minimum=1)
    run = _resolve_run(path)
    with _Reader(run) as reader:
        summary = reader.json("summary.json")
        metadata = _metadata(reader)
        # Completeness covers every referenced shard, including questions that
        # max_questions will exclude. Opening a ZIP stream does not decompress it.
        for shard in sorted({row["shard"] for row in metadata.values()}):
            with reader.open(shard):
                pass
        labels = _latest(reader, "labels.jsonl", metadata)
        teachers = _latest(reader, "teacher_targets.jsonl", metadata)
        all_edges = _edges(reader, metadata)
        questions = sorted({row["question"] for row in metadata.values()})
        if "questions_completed" in summary:
            count = _integer(summary["questions_completed"], "summary questions_completed")
            if count != len(questions):
                raise ValueError("Question count disagrees with original completed summary")
        selected = set(questions[:max_questions])
        cache = None
        fingerprint = None
        if cache_dir is not None:
            fingerprint = _fingerprint(reader, metadata, max_questions)
            cache = Path(cache_dir) / f"factorized_wm_data_v{_CACHE_VERSION}_{fingerprint}.pt"
            if cache.is_file():
                payload = torch.load(cache, map_location="cpu", weights_only=True)
                if (not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint
                        or payload.get("version") != _CACHE_VERSION):
                    raise ValueError(f"Invalid loader cache fingerprint: {cache}")
                states = {uid: RawState(**fields) for uid, fields in payload["states"].items()}
                for uid, state in states.items():
                    if uid != state.uid:
                        raise ValueError("Cache state ID mismatch")
                    _validate_state(state)
                expected_uids = {uid for uid, row in metadata.items() if row["question"] in selected}
                if set(states) != expected_uids:
                    raise ValueError("Cache states disagree with source question selection")
                expected_edges = [edge for edge in all_edges if edge[0] in states]
                if payload["edges"] != expected_edges:
                    raise ValueError("Cache graph disagrees with source")
                dataset = Dataset(states, expected_edges, payload["audit"], run.source)
                _graph(states, expected_edges)
                _validate_tokens(states, expected_edges)
                return dataset
        grouped = defaultdict(list)
        for row in metadata.values():
            if row["question"] in selected:
                grouped[row["shard"]].append(row)
        states, counts = {}, {}
        for shard, rows in sorted(grouped.items()):
            arrays = _shard_arrays(reader, shard)
            counts[shard] = len(arrays)
            for row in rows:
                uid = row["state_id"]
                states[uid] = _state(row, arrays, labels.get(uid), teachers.get(uid))
            del arrays
        edges = [edge for edge in all_edges if edge[0] in states]
        _graph(states, edges)
        contradictions = _validate_tokens(states, edges)
        dataset = Dataset(states, edges, {}, run.source)
        dataset.audit = _audit(dataset, contradictions, counts)
        dataset.audit.update(source_schema=summary["schema"], source_questions=len(questions),
                             source_states=len(metadata), source_edges=len(all_edges))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(version=_CACHE_VERSION, fingerprint=fingerprint,
                states={uid: vars(state) for uid, state in states.items()},
                edges=edges, audit=dataset.audit)
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, cache)
            finally:
                temporary.unlink(missing_ok=True)
        return dataset
