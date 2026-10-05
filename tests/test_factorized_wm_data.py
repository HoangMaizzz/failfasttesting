"""Synthetic original runs; no checkpoints, LLMs, GPU, or training required."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import factorized_wm_data as data
from factorized_wm_data import (Dataset, RawState, assert_input_features,
    enumerate_paths, feature_registry, load_dataset, resolve_input, split_questions)


def write_jsonl(root, name, rows):
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(root, name):
    return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines()]


def make_run(root, questions=2):
    """One shard with R/E/R chains; roots lack labels but remain path sources."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "experience").mkdir()
    summary = dict(schema="interactive_acceptance_two_source_v1", status="complete",
                   questions_completed=questions)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    states, edges, labels, teachers = [], [], [], []
    rows = {key: [] for key in ("ids", "hidden", "gaps", "history", "scalars",
                                "context", "aligned_topk_token_ids", "teacher_margin",
                                "teacher_features", "teacher_valid")}
    offsets = [0]
    for question in range(questions):
        for step, (length, start) in enumerate(((8, 0), (8, 0), (16, 8), (16, 8))):
            uid = f"q{question}s{step}"
            state = dict(state_id=uid, question=f"gsm8k:{question}", round_id=0,
                length=length, segment_start=start, prefix_token_ids=[10, 20 + question],
                parent_state_id=None if step == 0 else f"q{question}s{step-1}",
                shard="experience/anything.npz", row=len(states))
            states.append(state)
            ids = np.arange(length, dtype=np.int64) + 100 + question * 32
            rows["ids"].append(np.stack((ids, ids), axis=-1))
            rows["hidden"].append(np.full((length, 3, 1536), step / 10, dtype=np.float16))
            rows["gaps"].append(np.tile(-np.arange(32, dtype=np.float16), (length, 1)))
            rows["history"].append(np.full((length, 4), step / 4, dtype=np.float32))
            rows["aligned_topk_token_ids"].append(np.tile(np.arange(32, dtype=np.int32), (length, 1)))
            scalars = np.zeros((length, 16), dtype=np.float32)
            scalars[:, 1] = 1
            scalars[:, 3:5] = 1
            scalars[:start, 5:7] = step / 8
            # Include invalid native caches without changing the token invariants.
            scalars[-1, 3:5] = 0
            rows["scalars"].append(scalars)
            rows["context"].append(np.array([2/1024, length/64, start/64,
                0, .5, .5, .125, 1], dtype=np.float32))
            rows["teacher_margin"].append(np.full(length, -7, dtype=np.float32))
            rows["teacher_features"].append(np.full((length, 35), -7, dtype=np.float16))
            rows["teacher_valid"].append(np.full(length, step == 1, dtype=bool))
            offsets.append(offsets[-1] + length)
            if step:
                labels.append(dict(state_id=uid, label_valid=True, accepted_len=length))
                edges.append(dict(parent=f"q{question}s{step-1}", child=uid,
                                  action="E" if step == 2 else "R", question=f"gsm8k:{question}"))
            if step == 1:
                # The final record replaces both earlier JSONL and stale NPZ.
                teachers.extend([dict(state_id=uid, margin=[1.] * length, features=[[1.] * 35] * length),
                                 dict(state_id=uid, margin=[2.] * length, features=[[3.] * 35] * length)])
    arrays = {key: np.stack(values) if key == "context" else np.concatenate(values)
              for key, values in rows.items()}
    arrays["offsets"] = np.array(offsets, dtype=np.int64)
    arrays["lengths"] = np.diff(arrays["offsets"]).astype(np.int32)
    # These stale shard labels must never fill missing JSONL acceptance.
    arrays["accepted"] = np.ones(len(states), dtype=np.int32)
    arrays["label_valid"] = np.ones(len(states), dtype=bool)
    np.savez_compressed(root / "experience/anything.npz", **arrays)
    for name, values in (("states.jsonl", states), ("edges.jsonl", edges),
                          ("labels.jsonl", labels), ("teacher_targets.jsonl", teachers)):
        write_jsonl(root, name, values)
    return root


def archive_run(root, target, prefix=""):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(root.rglob("*")):
            if item.is_file():
                archive.write(item, prefix + item.relative_to(root).as_posix())
    return target


def change_shard(root, transform):
    path = root / "experience/anything.npz"
    with np.load(path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


class FactorizedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)
        self.root = make_run(self.temp / "original")

    def tearDown(self):
        self.temporary.cleanup()

    def test_folder_schema_targets_and_audit(self):
        self.assertEqual(resolve_input(self.root), self.root.resolve())
        dataset = load_dataset(self.root)
        self.assertIsInstance(dataset, Dataset)
        state = dataset.states["q0s1"]
        self.assertIsInstance(state, RawState)
        self.assertEqual(state.length, 8)
        for name, shape, dtype in (
                ("ids", (8, 2), torch.long), ("hidden", (8, 3, 1536), torch.float16),
                ("surface", (8, 36), torch.float32), ("scalars", (8, 16), torch.float32),
                ("context", (8,), torch.float32), ("prefix_ids", (2,), torch.long),
                ("topk_ids", (8, 32), torch.long), ("gaps", (8, 32), torch.float32),
                ("teacher_features", (8, 35), torch.float32), ("teacher_margin", (8,), torch.float32)):
            with self.subTest(name=name):
                value = getattr(state, name)
                self.assertEqual(tuple(value.shape), shape)
                self.assertEqual(value.dtype, dtype)
                self.assertEqual(value.device.type, "cpu")
        torch.testing.assert_close(state.surface[:, :32], state.gaps)
        torch.testing.assert_close(state.surface[:, 32:], torch.full((8, 4), .25))
        self.assertEqual(state.prefix_ids.tolist(), [10, 20])
        self.assertEqual(state.accepted, 8)
        self.assertIsNone(dataset.states["q0s0"].accepted)
        torch.testing.assert_close(state.teacher_margin, torch.full((8,), 2.))
        torch.testing.assert_close(state.teacher_features, torch.full((8, 35), 3.))
        audit = dataset.audit
        self.assertEqual((audit["states"], audit["R"], audit["E"]), (8, 4, 2))
        self.assertEqual(audit["lengths"], {8: 4, 16: 4})
        self.assertEqual((audit["H2"], audit["H3"]), (4, 2))
        self.assertEqual(audit["action_sequences"]["RER"], 2)
        self.assertEqual((audit["full_teacher_states"], audit["full_teacher_tokens"]), (2, 16))
        self.assertEqual(audit["missing_labels"], 2)
        self.assertEqual(audit["delta_K"]["positive"], 2)
        self.assertEqual(audit["E_transition_lengths"], {"8->16": 2})
        self.assertEqual(audit["delta_K"]["R_labeled_edges"], 2)
        self.assertEqual(audit["delta_K"]["E_labeled_edges"], 2)
        self.assertEqual(audit["delta_K"]["R_positive_rate"], 0.)
        self.assertEqual(audit["delta_K"]["E_positive_rate"], 1.)
        self.assertEqual(audit["native_cache"]["hidden"]["invalid_tokens"], 8)
        self.assertGreater(audit["native_cache"]["hidden"]["cached_tokens"], 0)
        self.assertFalse(audit["scalar_age_is_dynamics"])
        self.assertEqual(audit["immutable_stop_prefix_checks"], 6)
        self.assertEqual(audit["immutable_prefix_K_contradiction_count"], 0)
        self.assertEqual(dataset.question_ids, ["gsm8k:0", "gsm8k:1"])

    def test_transition_audit_positive_rates_and_missing_labels(self):
        labels = read_jsonl(self.root, "labels.jsonl")
        labels.extend(dict(state_id=f"q{q}s0", label_valid=True, accepted_len=4)
                      for q in range(2))
        write_jsonl(self.root, "labels.jsonl", labels)
        audit = load_dataset(self.root).audit
        self.assertEqual(audit["delta_K"]["R_labeled_edges"], 4)
        self.assertEqual(audit["delta_K"]["R_positive"], 2)
        self.assertEqual(audit["delta_K"]["R_positive_rate"], .5)
        self.assertEqual(audit["delta_K"]["E_positive_rate"], 1.)
        write_jsonl(self.root, "labels.jsonl", [])
        audit = load_dataset(self.root).audit
        self.assertEqual(audit["E_transition_lengths"], {"8->16": 2})
        for action in ("R", "E"):
            self.assertEqual(audit["delta_K"][f"{action}_labeled_edges"], 0)
            self.assertEqual(audit["delta_K"][f"{action}_positive"], 0)
            self.assertIsNone(audit["delta_K"][f"{action}_positive_rate"])

    def test_npz_members_are_decompressed_once_per_shard_not_per_state(self):
        accessed = []
        original = np.lib.npyio.NpzFile.__getitem__
        def counted(npz, key):
            accessed.append(key)
            return original(npz, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", counted):
            dataset = load_dataset(self.root)
        self.assertEqual(len(accessed), len(set(accessed)))
        self.assertEqual(len(accessed), dataset.audit["npz_members_loaded"])
        self.assertEqual(len(accessed), 12)
        self.assertNotIn("accepted", accessed)

    def test_arbitrary_zip_name_wrapped_root_and_folder_discovery(self):
        downloads = self.temp / "downloads/deeper"
        downloads.mkdir(parents=True)
        target = archive_run(self.root, downloads / "renamed-unrelated.ZIP", "wrapper/two/source/")
        self.assertEqual(resolve_input(self.temp / "downloads"), target.resolve())
        dataset = load_dataset(target)
        self.assertEqual(len(dataset.states), 8)
        self.assertIn("!/wrapper/two/source", dataset.source)
        self.assertEqual(dataset.states["q0s1"].teacher_margin.tolist(), [2.] * 8)
        # Reading ZIPs must leave no extracted run next to the input.
        self.assertEqual(sorted(p.name for p in downloads.iterdir()), [target.name])

    def test_nested_extracted_root_and_no_checkpoint_required(self):
        parent = self.temp / "nested"
        nested = parent / "one/two/run"
        shutil.copytree(self.root, nested)
        # A checkpoint, when present, is deliberately unreadable as a torch file.
        (nested / "checkpoint.pt").write_bytes(b"not a checkpoint; never load this")
        with patch.object(torch, "load", side_effect=AssertionError("checkpoint load")):
            self.assertEqual(resolve_input(parent), nested.resolve())
            self.assertEqual(len(load_dataset(parent).states), 8)

    def test_ambiguity_root_and_nested_run_is_explicit(self):
        shutil.copytree(self.root, self.root / "nested")
        with self.assertRaisesRegex(ValueError, "Ambiguous.*multiple ORIGINAL"):
            resolve_input(self.root)

    def test_ambiguity_multiple_zip_roots(self):
        target = self.temp / "two.zip"
        with zipfile.ZipFile(target, "w") as archive:
            for prefix in ("one/", "two/"):
                for item in self.root.rglob("*"):
                    if item.is_file():
                        archive.write(item, prefix + item.relative_to(self.root).as_posix())
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_input(target)

    def test_unrelated_malformed_zip_summary_does_not_hide_original_run(self):
        target = archive_run(self.root, self.temp / "wrapped.zip", "original/")
        with zipfile.ZipFile(target, "a") as archive:
            archive.writestr("notes/summary.json", "{malformed")
            archive.writestr("other/summary.json/", b"")
        self.assertEqual(resolve_input(target), target.resolve())
        self.assertEqual(len(load_dataset(target).states), 8)

    def test_ambiguity_downloads_with_zip_and_folder(self):
        archive_run(self.root, self.temp / "download.zip")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_input(self.temp)

    def test_derived_and_partial_runs_are_not_original_candidates(self):
        for schema, status in (("twosource_grouped_5fold_cv_v1", "complete"),
                                ("interactive_acceptance_two_source_v1", "partial")):
            with self.subTest(schema=schema, status=status):
                (self.root / "summary.json").write_text(json.dumps(dict(schema=schema, status=status)))
                with self.assertRaisesRegex(ValueError, "No ORIGINAL complete"):
                    resolve_input(self.root)

    def test_missing_required_file_experience_and_nonexistent_input(self):
        (self.root / "teacher_targets.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing.*teacher_targets.jsonl"):
            resolve_input(self.root)
        write_jsonl(self.root, "teacher_targets.jsonl", [])
        (self.root / "experience/anything.npz").unlink()
        with self.assertRaisesRegex(ValueError, "experience NPZ"):
            resolve_input(self.root)
        with self.assertRaises(FileNotFoundError):
            resolve_input(self.temp / "does-not-exist")

    def test_bad_zip_and_unsafe_zip_paths(self):
        corrupt = self.temp / "broken.zip"
        corrupt.write_bytes(b"not ZIP")
        with self.assertRaisesRegex(ValueError, "No ORIGINAL complete"):
            resolve_input(corrupt)
        for unsafe in ("../outside", "/absolute", "C:/absolute", "a\\b", "a/../b"):
            with self.subTest(unsafe=unsafe):
                target = archive_run(self.root, self.temp / "unsafe.zip")
                with zipfile.ZipFile(target, "a") as archive:
                    info = zipfile.ZipInfo("safe")
                    info.filename = unsafe  # bypass Windows ZipInfo normalization
                    archive.writestr(info, b"ignored")
                with self.assertRaisesRegex(ValueError, "Unsafe relative member"):
                    resolve_input(target)

    def test_duplicate_zip_member_is_rejected(self):
        target = archive_run(self.root, self.temp / "duplicate.zip")
        with zipfile.ZipFile(target, "a") as archive:
            # zipfile emits a warning for the intentionally malformed fixture.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("summary.json", (self.root / "summary.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP member"):
            resolve_input(target)

    def test_shard_path_traversal_is_rejected(self):
        rows = read_jsonl(self.root, "states.jsonl")
        rows[0]["shard"] = "experience/../../outside.npz"
        write_jsonl(self.root, "states.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Unsafe relative member"):
            load_dataset(self.root)

    def test_latest_labels_and_unresolved_labels_do_not_use_shard_values(self):
        labels = read_jsonl(self.root, "labels.jsonl")
        labels.extend([dict(state_id="q0s1", label_valid=True, accepted_len=2),
                       dict(state_id="q0s2", label_valid=False, accepted_len=1000)])
        write_jsonl(self.root, "labels.jsonl", labels)
        dataset = load_dataset(self.root)
        self.assertEqual(dataset.states["q0s1"].accepted, 2)
        self.assertIsNone(dataset.states["q0s2"].accepted)
        self.assertIsNone(dataset.states["q0s0"].accepted)
        self.assertIn((("q0s0", "q0s1", "q0s2", "q0s3"), (0, 1, 0)), enumerate_paths(dataset))

    def test_shard_teacher_fallback_and_latest_jsonl_wins(self):
        write_jsonl(self.root, "teacher_targets.jsonl", [])
        dataset = load_dataset(self.root)
        self.assertEqual(dataset.states["q0s1"].teacher_margin.tolist(), [-7.] * 8)
        self.assertIsNone(dataset.states["q0s0"].teacher_features)
        self.assertEqual(dataset.audit["full_teacher_tokens"], 16)

    def test_late_teacher_overrides_nonfinite_or_partial_stale_shard(self):
        def change(arrays):
            arrays["teacher_margin"][8:16] = np.nan
            arrays["teacher_features"][8:16] = np.nan
            arrays["teacher_valid"][8:16] = False
            arrays["teacher_valid"][8] = True
        change_shard(self.root, change)
        dataset = load_dataset(self.root)
        self.assertTrue(torch.isfinite(dataset.states["q0s1"].teacher_features).all())
        write_jsonl(self.root, "teacher_targets.jsonl", [])
        with self.assertRaisesRegex(ValueError, "Partially valid teacher"):
            load_dataset(self.root)

    def test_margin_only_jsonl_does_not_reuse_stale_teacher_features(self):
        write_jsonl(self.root, "teacher_targets.jsonl", [dict(state_id="q0s1", margin=[9.] * 8)])
        dataset = load_dataset(self.root)
        self.assertIsNone(dataset.states["q0s1"].teacher_features)
        self.assertEqual(dataset.states["q0s1"].teacher_margin.tolist(), [9.] * 8)
        self.assertEqual(dataset.audit["full_teacher_states"], 1)  # second question's shard teacher

    def test_malformed_late_teachers_abort(self):
        for row in (dict(margin=[0.] * 7), dict(margin=[float("nan")] * 8),
                    dict(margin=[0.] * 8, features=[[0.] * 34] * 8),
                    dict(margin=[0.] * 8, features=[[float("inf")] * 35] * 8),
                    dict(features=[[0.] * 35] * 8), dict(margin="not numbers")):
            with self.subTest(row=str(row)[:50]):
                write_jsonl(self.root, "teacher_targets.jsonl", [dict(state_id="q0s1", **row)])
                with self.assertRaises(ValueError):
                    load_dataset(self.root)

    def test_malformed_labels_abort(self):
        for label in (dict(label_valid=True, accepted_len=9),
                      dict(label_valid=True, accepted_len=-1),
                      dict(label_valid=True, accepted_len=2.5),
                      dict(label_valid=True, accepted_len=True),
                      dict(label_valid=True), dict(label_valid="true", accepted_len=1)):
            with self.subTest(label=label):
                write_jsonl(self.root, "labels.jsonl", [dict(state_id="q0s1", **label)])
                with self.assertRaises(ValueError):
                    load_dataset(self.root)

    def test_malformed_jsonl_and_duplicate_states_abort(self):
        (self.root / "edges.jsonl").write_text("{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed edges.jsonl:1"):
            load_dataset(self.root)
        write_jsonl(self.root, "edges.jsonl", [])
        states = read_jsonl(self.root, "states.jsonl")
        write_jsonl(self.root, "states.jsonl", states + [states[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate state_id"):
            load_dataset(self.root)

    def test_malformed_metadata_and_row_offsets_abort(self):
        original = read_jsonl(self.root, "states.jsonl")
        for field, value in (("row", 999), ("row", 1.5), ("round_id", -1),
                             ("prefix_token_ids", [-1]), ("prefix_token_ids", [1.5]),
                             ("prefix_token_ids", "abc"), ("length", 7)):
            with self.subTest(field=field, value=value):
                rows = [dict(row) for row in original]
                rows[0][field] = value
                write_jsonl(self.root, "states.jsonl", rows)
                with self.assertRaises(ValueError):
                    load_dataset(self.root)
        write_jsonl(self.root, "states.jsonl", original)
        change_shard(self.root, lambda arrays: arrays["offsets"].__setitem__(1, 0))
        with self.assertRaisesRegex(ValueError, "Offsets must"):
            load_dataset(self.root)

    def test_missing_npz_member_and_wrong_shapes_abort(self):
        for key, replacement in (("hidden", np.zeros((96, 2, 1536), dtype=np.float16)),
                                 ("gaps", np.zeros((96, 31), dtype=np.float16)),
                                 ("history", np.zeros((96, 3), dtype=np.float32)),
                                 ("context", np.zeros((8, 7), dtype=np.float32))):
            with self.subTest(key=key):
                target = self.temp / key
                shutil.copytree(self.root, target)
                change_shard(target, lambda arrays: arrays.__setitem__(key, replacement))
                with self.assertRaisesRegex(ValueError, "shape"):
                    load_dataset(target)
        change_shard(self.root, lambda arrays: arrays.pop("history"))
        with self.assertRaisesRegex(ValueError, "Missing NPZ members"):
            load_dataset(self.root)

    def test_nonfinite_raw_float_overflow_and_noninteger_tokens_abort(self):
        mutations = (
            lambda a: a["hidden"].__setitem__((0, 0, 0), np.nan),
            lambda a: a["gaps"].__setitem__((0, 0), np.inf),
            lambda a: a["scalars"].__setitem__((0, 5), -1),
            lambda a: a["ids"].__setitem__((0, 0), -1),
            lambda a: a.__setitem__("ids", a["ids"].astype(float)),
            lambda a: a.__setitem__("hidden", np.full(a["hidden"].shape, 1e10, dtype=np.float32)),
            lambda a: a["context"].__setitem__((0, 2), .01),
        )
        for index, change in enumerate(mutations):
            with self.subTest(mutation=index):
                target = self.temp / f"mutation{index}"
                shutil.copytree(self.root, target)
                change_shard(target, change)
                with self.assertRaises(ValueError):
                    load_dataset(target)

    def test_missing_state_cross_question_cross_round_and_cycles_abort(self):
        original_edges = read_jsonl(self.root, "edges.jsonl")
        cases = ([dict(parent="missing", child="q0s1", action="R")],
                 [dict(parent="q0s0", child="q1s1", action="R")],
                 [dict(parent="q0s0", child="q0s1", action="S")],
                 original_edges + [dict(parent="q0s3", child="q0s0", action="R")])
        for edges, message in zip(cases, ("missing state", "Cross-question", "Invalid transition", "Cycle")):
            with self.subTest(message=message):
                write_jsonl(self.root, "edges.jsonl", edges)
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(self.root)
        write_jsonl(self.root, "edges.jsonl", original_edges)
        states = read_jsonl(self.root, "states.jsonl")
        states[1]["round_id"] = 1
        write_jsonl(self.root, "states.jsonl", states)
        with self.assertRaisesRegex(ValueError, "Cross-round"):
            load_dataset(self.root)

    def test_broken_unselected_graph_still_aborts(self):
        edges = read_jsonl(self.root, "edges.jsonl")
        edges.append(dict(parent="q1s0", child="missing", action="R"))
        write_jsonl(self.root, "edges.jsonl", edges)
        with self.assertRaisesRegex(ValueError, "missing state"):
            load_dataset(self.root, max_questions=1)

    def test_missing_unselected_shard_still_aborts(self):
        states = read_jsonl(self.root, "states.jsonl")
        for row in states:
            if row["question"] == "gsm8k:1":
                row["shard"] = "experience/missing.npz"
        write_jsonl(self.root, "states.jsonl", states)
        with self.assertRaisesRegex(ValueError, "Missing run member"):
            load_dataset(self.root, max_questions=1)

    def test_unknown_labels_and_teachers_abort(self):
        for name in ("labels.jsonl", "teacher_targets.jsonl"):
            with self.subTest(name=name):
                rows = read_jsonl(self.root, name)
                write_jsonl(self.root, name, rows + [dict(state_id="missing")])
                with self.assertRaisesRegex(ValueError, "references missing state"):
                    load_dataset(self.root)
                write_jsonl(self.root, name, rows)

    def test_immutable_stop_native_prefix_and_committed_R_ids_abort(self):
        for position, channel, message in ((16, 1, "Immutable STOP prefix"),
                                          (16, 0, "Immutable native prefix"),
                                          (32, 1, "Immutable STOP prefix"),
                                          (8, 0, "R changed committed")):
            with self.subTest(position=position, channel=channel):
                target = self.temp / f"ids{position}_{channel}"
                shutil.copytree(self.root, target)
                change_shard(target, lambda a: a["ids"].__setitem__((position, channel), 999))
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(target)

    def test_prefix_and_extend_length_invariants_abort(self):
        states = read_jsonl(self.root, "states.jsonl")
        states[1]["prefix_token_ids"] = [999]
        write_jsonl(self.root, "states.jsonl", states)
        with self.assertRaisesRegex(ValueError, "changed prefix_token_ids"):
            load_dataset(self.root)
        states[1]["prefix_token_ids"] = [10, 20]
        write_jsonl(self.root, "states.jsonl", states)
        edges = read_jsonl(self.root, "edges.jsonl")
        edges[1]["action"] = "R"
        write_jsonl(self.root, "edges.jsonl", edges)
        with self.assertRaisesRegex(ValueError, "length invariant"):
            load_dataset(self.root)

    def test_numeric_K_contradictions_are_audited_not_aborted(self):
        labels = read_jsonl(self.root, "labels.jsonl")
        labels.extend([dict(state_id="q0s1", label_valid=True, accepted_len=2),
                       dict(state_id="q0s2", label_valid=True, accepted_len=4),
                       dict(state_id="q0s3", label_valid=True, accepted_len=1)])
        write_jsonl(self.root, "labels.jsonl", labels)
        dataset = load_dataset(self.root)
        records = dataset.audit["immutable_prefix_K_contradictions"]
        self.assertEqual(dataset.audit["immutable_prefix_K_contradiction_count"], 2)
        self.assertEqual([r["action"] for r in records], ["E", "R"])
        self.assertEqual([r["immutable_length"] for r in records], [8, 8])
        self.assertEqual(records[0]["parent_K"], 2)
        self.assertEqual(records[0]["child_K"], 4)

    def test_whole_question_selection_and_paths_include_missing_labels(self):
        dataset = load_dataset(self.root, max_questions=1)
        self.assertEqual(dataset.question_ids, ["gsm8k:0"])
        self.assertEqual(len(dataset.states), 4)
        self.assertEqual(len(dataset.edges), 3)
        self.assertEqual(dataset.audit["source_questions"], 2)
        paths = enumerate_paths(dataset)
        self.assertEqual(len(paths), 6)
        self.assertIn((("q0s0", "q0s1", "q0s2", "q0s3"), (0, 1, 0)), paths)
        self.assertEqual(len(enumerate_paths(dataset, horizon=1)), 3)
        self.assertEqual(enumerate_paths(dataset, horizon=0), [])
        self.assertEqual(enumerate_paths(dataset, allowed_questions=[]), [])
        with self.assertRaisesRegex(ValueError, "Unknown allowed questions"):
            enumerate_paths(dataset, allowed_questions=["wrong"])
        with self.assertRaises(ValueError):
            enumerate_paths(dataset, horizon=-1)
        with self.assertRaises(ValueError):
            load_dataset(self.root, max_questions=0)

    def test_paths_support_branches_and_grouped_question_filters(self):
        dataset = load_dataset(self.root)
        # Add a second R child of the root; all source data remains immutable.
        dataset.edges.append(("q0s0", "q0s2", "E"))
        paths = enumerate_paths(dataset, allowed_questions=["gsm8k:0"], horizon=3)
        self.assertIn((("q0s0", "q0s2", "q0s3"), (1, 0)), paths)
        self.assertTrue(all(dataset.states[uid].question == "gsm8k:0" for nodes, _ in paths for uid in nodes))
        dataset.edges.append(("q0s3", "q0s0", "R"))
        with self.assertRaisesRegex(ValueError, "Cycle"):
            enumerate_paths(dataset)

    def test_split_exact_deterministic_disjoint_and_invalid_inputs(self):
        questions = [f"gsm8k:{i}" for i in range(100)]
        split = split_questions(questions)
        self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [70, 15, 15])
        self.assertEqual(split, split_questions(reversed(questions)))
        self.assertNotEqual(split, split_questions(questions, seed=43))
        self.assertEqual(set().union(*(set(part) for part in split.values())), set(questions))
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["val"]) & set(split["test"]))
        self.assertEqual(split_questions([]), dict(train=[], val=[], test=[]))
        for arguments in ((["q", "q"], {}), ([1], {}), (["q"], dict(train_fraction=1.1)),
                           (["q"], dict(val_fraction=float("nan"))),
                           (["q"], dict(train_fraction=.9, val_fraction=.2))):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    split_questions(arguments[0], **arguments[1])

    def test_registry_enforces_parent_only_verifier_and_known_action_drafter(self):
        registry = feature_registry()
        fields = ["hidden", "surface", "scalars", "context", "prefix_ids", "ids", "topk_ids"]
        self.assertTrue(all(registry[name] == data.AVAILABLE_AT_PARENT for name in fields))
        self.assertEqual(registry["actions"], data.ACTION_KNOWN)
        self.assertEqual(registry["teacher_features"], data.TEACHER_ONLY_DRAFTER)
        self.assertEqual(registry["teacher_margin"], data.TEACHER_ONLY_VERIFIER)
        assert_input_features(fields + ["actions"], "drafter")
        assert_input_features(fields, "verifier")
        assert_input_features("hidden")
        for module in ("drafter", "verifier"):
            for name in ("question", "uid", "accepted", "teacher_features", "teacher_margin",
                         "child_hidden", "child_ids", "unknown", "gaps"):
                with self.subTest(module=module, name=name):
                    with self.assertRaises(ValueError):
                        assert_input_features([name], module)
        with self.assertRaisesRegex(ValueError, "Invalid verifier input"):
            assert_input_features(["actions"], "verifier")
        with self.assertRaises(ValueError):
            assert_input_features(fields, "unknown")
        registry["hidden"] = data.FORBIDDEN_FUTURE
        assert_input_features(["hidden"])  # caller cannot mutate a global registry

    def test_cache_roundtrip_uses_only_own_weights_only_cache(self):
        cache = self.temp / "cache"
        original = load_dataset(self.root, cache_dir=cache)
        self.assertEqual(len(list(cache.glob("*.pt"))), 1)
        torch_load = torch.load
        with patch.object(data, "_shard_arrays", side_effect=AssertionError("NPZ decompression")), \
                patch.object(torch, "load", wraps=torch_load) as cached_load:
            cached = load_dataset(self.root, cache_dir=cache)
        cached_load.assert_called_once()
        self.assertTrue(cached_load.call_args.kwargs["weights_only"])
        self.assertEqual(cached.audit, original.audit)
        self.assertEqual(cached.edges, original.edges)
        torch.testing.assert_close(cached.states["q0s1"].hidden, original.states["q0s1"].hidden)
        labels = read_jsonl(self.root, "labels.jsonl")
        labels.append(dict(state_id="q0s1", label_valid=True, accepted_len=2))
        write_jsonl(self.root, "labels.jsonl", labels)
        changed = load_dataset(self.root, cache_dir=cache)
        self.assertEqual(changed.states["q0s1"].accepted, 2)
        self.assertEqual(len(list(cache.glob("*.pt"))), 2)
        # Question selection also participates in the cache fingerprint.
        self.assertEqual(len(load_dataset(self.root, max_questions=1, cache_dir=cache).states), 4)
        self.assertEqual(len(list(cache.glob("*.pt"))), 3)

    def test_zip_cache_fingerprint_invalidates_on_teacher_changes(self):
        target = archive_run(self.root, self.temp / "original.zip", "root/")
        cache = self.temp / "zip-cache"
        self.assertEqual(load_dataset(target, cache_dir=cache).states["q0s1"].teacher_margin[0], 2)
        write_jsonl(self.root, "teacher_targets.jsonl", [dict(state_id="q0s1", margin=[5.] * 8)])
        archive_run(self.root, target, "root/")
        self.assertEqual(load_dataset(target, cache_dir=cache).states["q0s1"].teacher_margin[0], 5)
        self.assertEqual(len(list(cache.glob("*.pt"))), 2)

    def test_summary_question_count_disagreement_aborts(self):
        summary = json.loads((self.root / "summary.json").read_text())
        summary["questions_completed"] = 100
        (self.root / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "Question count disagrees"):
            load_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
