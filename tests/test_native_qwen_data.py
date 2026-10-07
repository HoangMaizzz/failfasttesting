"""Offline native data, capture replay, provenance and leakage regression tests.

Fixtures use tiny CPU tensors and temporary NPY files, never a Hub model,
GPU query, original input download or training job.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_qwen_data as data


SIGNATURE = "a" * 64
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEPTHS = [1, 2, 3, 4]
HIDDEN_DIM = 3


def rows_fixture():
    return {
        "a": dict(length=2, accepted=1, question="q1", action="R", segment_start=0,
                  candidate=[10, 11], parent_K=0, parent_length=2),
        "b": dict(length=3, accepted=3, question="q2", action="E", segment_start=2,
                  candidate=[11, 10, 11], parent_K=2, parent_length=2),
    }


def capture_fixture(row, rerun_K=None):
    return dict(K=row["accepted"] if rerun_K is None else rerun_K,
                alignment={"passed": True, "finite": True, "compared_positions": row["length"] + 1},
                hidden={depth: (torch.arange(row["length"] * HIDDEN_DIM).reshape(
                    row["length"], HIDDEN_DIM) + depth * 10).half() for depth in DEPTHS})


class DiskFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native_qwen_data_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = rows_fixture()

    def store(self, *, root=None, rows=None, uids=None, depths=None, hidden_dim=HIDDEN_DIM,
              signature=SIGNATURE, readonly=False):
        rows = self.rows if rows is None else rows
        store = data.NativeHiddenStore(root or self.root / "capture", rows,
            list(rows) if uids is None else uids, DEPTHS if depths is None else depths,
            hidden_dim, signature, readonly=readonly)
        self.addCleanup(store.close)
        return store

    def completed_store(self):
        store = self.store()
        for uid, row in self.rows.items():
            store.record(uid, capture_fixture(row))
        store.close()
        return self.root / "capture"

    def progress(self):
        return json.loads((self.root / "capture/capture_progress.json").read_text())

    def rewrite_progress(self, mutate):
        progress = self.progress()
        mutate(progress)
        (self.root / "capture/capture_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    def candidate_table(self, ids=None, values=None):
        root = self.root / "embeddings"
        root.mkdir(exist_ok=True)
        (root / "candidate_embedding_ids.json").write_text(
            json.dumps([11, 10] if ids is None else ids), encoding="utf-8")
        np.save(root / "candidate_embeddings.npy", np.array(
            [[11, 12, 13], [20, 21, 22]], dtype=np.float16) if values is None else values)
        return root


class HiddenStoreTests(DiskFixtures):
    def test_fresh_capture_has_exact_offsets_depth_slots_and_no_qualified_zeros(self):
        store = self.store()
        self.assertEqual(store.offsets, {"a": (0, 2), "b": (2, 5)})
        self.assertEqual(store.total_positions, 5)
        self.assertEqual(store.qualified(), [])
        manifest = json.loads((store.root / "capture_manifest.json").read_text())
        self.assertEqual(manifest["signature"], SIGNATURE)
        self.assertEqual(manifest["projection"], "none")
        self.assertEqual(manifest["normalization"], "none")
        for slot, depth in zip(("layer25", "layer50", "layer75", "layer100"), DEPTHS):
            self.assertTrue((store.root / "native_hidden" / slot / "hidden.npy").is_file())
            self.assertIsInstance(store.arrays[depth], np.memmap)
            self.assertEqual(store.arrays[depth].shape, (5, HIDDEN_DIM))
            self.assertEqual(store.arrays[depth].dtype, np.float16)

    def test_progress_commits_raw_bytes_and_flushes_on_close(self):
        store = self.store()
        expected = {}
        for uid, row in self.rows.items():
            result = capture_fixture(row)
            expected[uid] = hashlib.sha256(b"".join(
                result["hidden"][depth].numpy().tobytes() for depth in DEPTHS)).hexdigest()
            store.record(uid, result)
        self.assertEqual(store.qualified(), ["a", "b"])
        store.close()
        progress = self.progress()
        self.assertEqual(set(progress), set(self.rows))
        for uid, row in self.rows.items():
            rec = progress[uid]
            self.assertEqual(rec["saved_K"], row["accepted"])
            self.assertEqual(rec["rerun_K"], row["accepted"])
            self.assertIs(rec["matches"], True)
            self.assertTrue(rec["alignment"]["passed"])
            self.assertEqual(rec["raw_hidden_sha256"], expected[uid])

    def test_valid_writable_resume_preserves_progress_and_raw_capture(self):
        root = self.completed_store()
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.npy")}
        resumed = self.store()
        resumed.verify_saved_rows()
        self.assertEqual(resumed.qualified(), ["a", "b"])
        for uid, row in self.rows.items():
            begin, end = resumed.offsets[uid]
            for depth, hidden in capture_fixture(row)["hidden"].items():
                np.testing.assert_array_equal(resumed.arrays[depth][begin:end], hidden.numpy())
        resumed.close()
        self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*.npy")})

    def test_partial_resume_does_not_treat_unrecorded_positions_as_captured(self):
        store = self.store()
        store.record("a", capture_fixture(self.rows["a"]))
        store.flush_progress()
        store.close()
        resumed = self.store()
        resumed.verify_saved_rows()
        self.assertEqual(resumed.qualified(), ["a"])
        table = data.load_candidate_table(self.candidate_table())
        with self.assertRaises(ValueError):
            data.pack_native_batch(["b"], {"kind": "raw", "depth": 1}, self.rows, {}, resumed, table, "cpu")
        resumed.record("b", capture_fixture(self.rows["b"]))
        resumed.close()
        self.assertEqual(self.store().qualified(), ["a", "b"])

    def test_legitimate_reproduction_mismatch_is_audited_but_not_qualified(self):
        store = self.store()
        store.record("a", capture_fixture(self.rows["a"], rerun_K=0))
        store.close()
        resumed = self.store()
        resumed.verify_saved_rows()
        self.assertEqual(resumed.qualified(), [])
        rec = resumed.progress["a"]
        self.assertIs(rec["matches"], False)
        self.assertEqual((rec["saved_K"], rec["rerun_K"]), (1, 0))

    def test_confirmed_native_labels_keep_history_and_work_in_readonly_worker(self):
        store=self.store()
        store.record('a',capture_fixture(self.rows['a'],rerun_K=0))
        store.record('b',capture_fixture(self.rows['b']))
        rec=store.progress['a']
        rec['label_reconciliation']=dict(policy='audited_direct',stable=True,confirmation_passes=2,
            saved_K=rec['saved_K'],rerun_K=rec['rerun_K'],signature=rec['signature'],
            raw_hidden_sha256=rec['raw_hidden_sha256'])
        rows=copy.deepcopy(self.rows)
        data.apply_capture_labels(rows,store,'audited_direct')
        self.assertEqual((rows['a']['historical_accepted'],rows['a']['accepted']),(1,0))
        self.assertEqual(store.qualified(),['b'])
        self.assertEqual(store.qualified(include_reconciled=True),['a','b'])
        store.close()
        worker=self.store(rows=rows,readonly=True)
        table=data.load_candidate_table(self.candidate_table())
        batch=data.pack_native_batch(['a','b'],dict(kind='raw',depth=1),rows,{},worker,table,'cpu')
        self.assertEqual(batch['accepted'].tolist(),[0,3])
        bad=copy.deepcopy(rows);bad['a']['accepted']=1
        with self.assertRaisesRegex(ValueError,'Supervision'):
            data.pack_native_batch(['a'],dict(kind='raw',depth=1),bad,{},worker,table,'cpu')
        # A modified audit is not sufficient to qualify arbitrary hidden bytes.
        rec=worker.progress['a'];rec['label_reconciliation']['raw_hidden_sha256']='different'
        self.assertFalse(data.reconciled_capture(rec))

    def test_unconfirmed_labels_and_tampered_reconciliation_are_rejected(self):
        store=self.store()
        store.record('a',capture_fixture(self.rows['a'],rerun_K=0))
        store.record('b',capture_fixture(self.rows['b']))
        with self.assertRaises(ValueError):data.apply_capture_labels(copy.deepcopy(self.rows),store,'strict')
        with self.assertRaises(ValueError):data.apply_capture_labels(copy.deepcopy(self.rows),store,'audited_direct')
        store.progress['a']['label_reconciliation']={'stable':True}
        store.close()
        with self.assertRaisesRegex(ValueError,'reconciliation'):self.store()

    def test_failed_alignment_and_incomplete_hidden_do_not_commit_a_row(self):
        for case in ("alignment", "missing_layer", "wrong_shape", "nan"):
            with self.subTest(case=case):
                store = self.store(root=self.root / case)
                result = capture_fixture(self.rows["a"])
                if case == "alignment":
                    result["alignment"]["passed"] = False
                elif case == "missing_layer":
                    del result["hidden"][4]
                elif case == "wrong_shape":
                    result["hidden"][1] = torch.zeros(3, HIDDEN_DIM)
                else:
                    result["hidden"][1][0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    store.record("a", result)
                self.assertNotIn("a", store.progress)
                self.assertEqual(store.qualified(), [])

    def test_capture_rerun_K_rejects_noninteger_or_out_of_range_counts(self):
        store = self.store()
        for count in (-1, 3, True, 1.0, None):
            with self.subTest(count=count):
                result = capture_fixture(self.rows["a"])
                result["K"] = count
                with self.assertRaises(ValueError):
                    store.record("a", result)
                self.assertNotIn("a", store.progress)

    def test_resume_rejects_changed_signature_subset_order_depth_or_geometry(self):
        self.completed_store()
        changed = copy.deepcopy(self.rows)
        changed["a"]["length"] = 1
        for kwargs in ({"signature": "b" * 64}, {"uids": ["a"]}, {"uids": ["b", "a"]},
                       {"depths": [1, 2, 3, 5]}, {"hidden_dim": 4}, {"rows": changed}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store(**kwargs)

    def test_resume_rejects_foreign_progress_state_and_changed_saved_label(self):
        self.completed_store()
        saved = self.progress()
        for case in ("foreign_uid", "signature", "saved_K"):
            with self.subTest(case=case):
                progress = copy.deepcopy(saved)
                if case == "foreign_uid":
                    progress["foreign"] = copy.deepcopy(progress["a"])
                elif case == "signature":
                    progress["a"]["signature"] = "b" * 64
                else:
                    progress["a"]["saved_K"] = 0
                (self.root / "capture/capture_progress.json").write_text(json.dumps(progress), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.store()

    def test_contradictory_matches_and_failed_alignment_are_rejected_on_replay(self):
        self.completed_store()
        saved = self.progress()
        corruptions = (
            {"matches": True, "rerun_K": 0},
            {"matches": False, "rerun_K": 1},
            {"alignment": {"passed": False}},
            {"alignment": {}},
        )
        for corruption in corruptions:
            for readonly in (False, True):
                with self.subTest(corruption=corruption, readonly=readonly):
                    progress = copy.deepcopy(saved)
                    progress["a"].update(corruption)
                    (self.root / "capture/capture_progress.json").write_text(json.dumps(progress), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.store(readonly=readonly)

    def test_replayed_audit_fields_require_integer_counts_boolean_match_and_geometry(self):
        self.completed_store()
        saved = self.progress()
        for corruption in ({"saved_K": 1.0}, {"rerun_K": True}, {"rerun_K": -1, "matches": False},
                           {"rerun_K": 3, "matches": False}, {"matches": "true"},
                           {"matches": 1}, {"alignment": {"passed": 1}}):
            with self.subTest(corruption=corruption):
                progress = copy.deepcopy(saved)
                progress["a"].update(corruption)
                (self.root / "capture/capture_progress.json").write_text(json.dumps(progress), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.store(readonly=True)

    def test_readonly_geometry_without_acceptance_can_verify_and_never_writes(self):
        root = self.completed_store()
        before = {p.relative_to(root): (p.stat().st_mtime_ns, p.read_bytes())
                  for p in root.rglob("*") if p.is_file()}
        geometry = {uid: {"length": row["length"]} for uid, row in self.rows.items()}
        replay = self.store(rows=geometry, readonly=True)
        replay.verify_saved_rows()
        self.assertEqual(replay.qualified(), ["a", "b"])
        self.assertTrue(all(not array.flags.writeable for array in replay.arrays.values()))
        with self.assertRaises(RuntimeError):
            replay.record("a", capture_fixture(self.rows["a"]))
        replay.close()
        self.assertEqual(before, {p.relative_to(root): (p.stat().st_mtime_ns, p.read_bytes())
                                 for p in root.rglob("*") if p.is_file()})

    def test_readonly_still_crosschecks_labels_when_provided(self):
        self.completed_store()
        changed = copy.deepcopy(self.rows)
        changed["a"]["accepted"] = 0
        with self.assertRaises(ValueError):
            self.store(rows=changed, readonly=True)

    def test_missing_labels_only_allowed_for_readonly_geometry(self):
        self.completed_store()
        geometry = {uid: {"length": row["length"]} for uid, row in self.rows.items()}
        with self.assertRaises((ValueError, KeyError)):
            self.store(rows=geometry, readonly=False)

    def test_readonly_acceptance_key_when_present_is_not_silently_ignored(self):
        self.completed_store()
        rows = {uid: {"length": row["length"], "accepted": None} for uid, row in self.rows.items()}
        with self.assertRaises(ValueError):
            self.store(rows=rows, readonly=True)

    def test_readonly_geometry_still_rejects_inconsistent_reproduction(self):
        self.completed_store()
        self.rewrite_progress(lambda p: p["a"].update(rerun_K=0, matches=True))
        geometry = {uid: {"length": row["length"]} for uid, row in self.rows.items()}
        with self.assertRaises(ValueError):
            self.store(rows=geometry, readonly=True)

    def test_readonly_missing_manifest_or_native_layer_fails_without_creating_files(self):
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            self.store(root=missing, readonly=True)
        self.assertFalse(missing.exists())
        root = self.completed_store()
        path = root / "native_hidden/layer100/hidden.npy"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store(readonly=True)
        self.assertFalse(path.exists())

    def test_bad_npy_shape_and_dtype_fail_even_with_matching_manifest(self):
        root = self.completed_store()
        path = root / "native_hidden/layer25/hidden.npy"
        for array in (np.zeros((4, HIDDEN_DIM), dtype=np.float16), np.zeros((5, HIDDEN_DIM), dtype=np.float32)):
            with self.subTest(shape=array.shape, dtype=array.dtype):
                np.save(path, array)
                with self.assertRaises(ValueError):
                    self.store(readonly=True)

    def test_checksum_detects_changed_bytes_in_every_native_layer(self):
        self.completed_store()
        for depth in DEPTHS:
            with self.subTest(depth=depth):
                store = self.store()
                value = float(store.arrays[depth][0, 0])
                store.arrays[depth][0, 0] = value + 1
                store.arrays[depth].flush()
                with self.assertRaisesRegex(ValueError, "checksum"):
                    store.verify_saved_rows()
                store.arrays[depth][0, 0] = value
                store.arrays[depth].flush()
                store.verify_saved_rows()
                store.close()

    def test_nonfinite_capture_is_rejected_even_when_checksum_is_updated(self):
        self.completed_store()
        store = self.store()
        store.arrays[4][0, 0] = np.nan
        begin, end = store.offsets["a"]
        store.progress["a"]["raw_hidden_sha256"] = hashlib.sha256(b"".join(
            np.asarray(store.arrays[depth][begin:end]).tobytes() for depth in DEPTHS)).hexdigest()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            store.verify_saved_rows()


class EmbeddingAndPackingTests(DiskFixtures):
    def test_own_embedding_lookup_preserves_ids_row_order_and_is_readonly(self):
        table = data.load_candidate_table(self.candidate_table())
        self.assertEqual(table["index"], {11: 0, 10: 1})
        self.assertEqual(table["values"].dtype, np.float16)
        self.assertFalse(table["values"].flags.writeable)
        np.testing.assert_array_equal(table["values"][table["index"][10]], [20, 21, 22])
        with self.assertRaises(ValueError):
            table["values"][0, 0] = 0

    def test_embedding_ids_reject_duplicate_negative_noninteger_and_boolean_entries(self):
        for ids in ([10, 10], [-1, 10], [True, 10], [10.0, 11], ["10", 11]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                data.load_candidate_table(self.candidate_table(ids=ids))

    def test_embedding_values_reject_wrong_rank_count_dtype_or_nonfinite(self):
        values = (np.zeros(2, dtype=np.float16), np.zeros((1, HIDDEN_DIM), dtype=np.float16),
                  np.zeros((2, HIDDEN_DIM), dtype=np.float32),
                  np.full((2, HIDDEN_DIM), np.nan, dtype=np.float16),
                  np.full((2, HIDDEN_DIM), np.inf, dtype=np.float16))
        for value in values:
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                data.load_candidate_table(self.candidate_table(values=value))

    def test_native_pack_aligns_hidden_candidate_and_padding_without_gradients(self):
        self.completed_store()
        store = self.store(readonly=True)
        table = data.load_candidate_table(self.candidate_table())
        batch = data.pack_native_batch(["b", "a"], {"kind": "raw", "depth": 3},
                                       self.rows, {}, store, table, "cpu")
        self.assertEqual(batch["lengths"].tolist(), [3, 2])
        self.assertEqual(batch["accepted"].tolist(), [3, 1])
        for index, uid in enumerate(("b", "a")):
            row = self.rows[uid]
            n = row["length"]
            torch.testing.assert_close(batch["hidden"][index, :n], capture_fixture(row)["hidden"][3].float())
            expected = torch.tensor(np.array(table["values"][[table["index"][t] for t in row["candidate"]]]), dtype=torch.float32)
            torch.testing.assert_close(batch["candidate_embedding"][index, :n], expected)
            torch.testing.assert_close(batch["structural"][index, :n], data.structural_features(row))
            for name in ("hidden", "candidate_embedding", "structural"):
                self.assertEqual(batch[name].shape[1], 64)
                self.assertFalse(batch[name].requires_grad)
                self.assertEqual(torch.count_nonzero(batch[name][index, n:]).item(), 0)
        self.assertNotIn("parent_K", batch)
        self.assertNotIn("final_logits", batch)

    def test_native_pack_rejects_missing_candidate_id_without_fallback_embedding(self):
        self.completed_store()
        store = self.store(readonly=True)
        table = data.load_candidate_table(self.candidate_table(ids=[11, 12]))
        with self.assertRaises((KeyError, ValueError)):
            data.pack_native_batch(["a"], {"kind": "raw", "depth": 1}, self.rows, {}, store, table, "cpu")

    def test_direct_pack_never_accesses_native_hidden_embedding_or_structural_fields(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("Direct accessed a native/teacher asset")
        class DirectRow(dict):
            def __getitem__(self, key):
                if key not in ("length", "accepted", "c", "context"):
                    raise AssertionError("Direct read forbidden row feature: " + key)
                return super().__getitem__(key)
        rows = {uid: DirectRow(length=n, accepted=1,
                              c=torch.arange(n * 20).reshape(n, 20).float(),
                              context=torch.arange(8).float()) for uid, n in (("a", 2), ("b", 3))}
        cache = {uid: torch.arange(n * 128).reshape(n, 128).half()
                 for uid, n in (("a", 2), ("b", 3))}
        batch = data.pack_native_batch(["a", "b"], {"kind": "direct"}, rows, cache, Forbidden(), Forbidden(), "cpu")
        self.assertEqual(set(batch), {"lengths", "accepted", "z_D", "c", "context"})
        self.assertEqual(batch["z_D"].shape, (2, 64, 128))
        self.assertEqual(batch["c"].shape, (2, 64, 20))
        self.assertEqual(batch["lengths"].tolist(), [2, 3])
        for index, uid in enumerate(("a", "b")):
            n = rows[uid]["length"]
            torch.testing.assert_close(batch["z_D"][index, :n], cache[uid].float())
            torch.testing.assert_close(batch["c"][index, :n], rows[uid]["c"])
            self.assertEqual(torch.count_nonzero(batch["z_D"][index, n:]).item(), 0)
            self.assertEqual(torch.count_nonzero(batch["c"][index, n:]).item(), 0)
        before = {k: batch[k].clone() for k in ("z_D", "c", "context", "lengths")}
        rows["a"]["accepted"] = 0
        changed = data.pack_native_batch(["a", "b"], {"kind": "direct"}, rows, cache, Forbidden(), Forbidden(), "cpu")
        for key in before:
            torch.testing.assert_close(before[key], changed[key])

    def test_direct_pack_accepts_source_padded_latents_and_crops_nonzero_tail(self):
        rows = {uid: dict(length=n, accepted=1, c=torch.zeros(n, 20), context=torch.zeros(8))
                for uid, n in (("a", 2), ("b", 3))}
        cache = {uid: torch.full((64, 128), 999., dtype=torch.float16) for uid in rows}
        for uid, row in rows.items():
            cache[uid][:row["length"]] = torch.arange(row["length"] * 128).reshape(row["length"], 128)
        batch = data.pack_native_batch(["a", "b"], {"kind": "direct"}, rows, cache, None, None, "cpu")
        self.assertEqual(batch["z_D"].shape, (2, 64, 128))
        self.assertEqual(batch["lengths"].tolist(), [2, 3])
        for index, uid in enumerate(("a", "b")):
            n = rows[uid]["length"]
            torch.testing.assert_close(batch["z_D"][index, :n], cache[uid][:n].float())
            self.assertEqual(torch.count_nonzero(batch["z_D"][index, n:]).item(), 0)
            self.assertTrue(torch.all(cache[uid][n:] == 999))

    def test_direct_pack_rejects_global_or_incorrect_tokenwise_latent_geometry(self):
        rows = {"a": dict(length=2, accepted=1, c=torch.zeros(2, 20), context=torch.zeros(8))}
        for latent in (torch.zeros(128), torch.zeros(3, 128), torch.zeros(2, 127)):
            with self.subTest(shape=latent.shape), self.assertRaises(ValueError):
                data.pack_native_batch(["a"], {"kind": "direct"}, rows, {"a": latent}, None, None, "cpu")

    def test_negative_acceptance_cannot_be_packed_as_a_rejection_label(self):
        rows = {"a": dict(length=1, accepted=-1)}
        with self.assertRaises(ValueError):
            data.pack_native_batch(["a"], {"kind": "direct"}, rows, {}, None, None, "cpu")


class StructuralAndSelectionTests(unittest.TestCase):
    def test_structure_depends_only_on_known_position_length_and_segment(self):
        row = dict(length=4, segment_start=2)
        expected = torch.tensor([[0, 4/64, 0, 0], [1/3, 4/64, 0, 0],
                                 [2/3, 4/64, 1, 0], [1, 4/64, 1, 1]])
        torch.testing.assert_close(data.structural_features(row), expected)
        labeled = dict(row, accepted=4, parent_K=2, survival_true=[1, 1, 1, 1],
                       final_logits=float("nan"), K_true=4, teacher_target="forbidden")
        torch.testing.assert_close(data.structural_features(labeled), expected)
        labeled.update(accepted=0, parent_K=0, survival_true=[0, 0, 0, 0], K_true=0)
        torch.testing.assert_close(data.structural_features(labeled), expected)

    def test_single_position_and_single_new_position_have_finite_relative_features(self):
        for row in (dict(length=1, segment_start=0), dict(length=4, segment_start=3)):
            with self.subTest(row=row):
                value = data.structural_features(row)
                self.assertTrue(torch.isfinite(value).all())
                self.assertEqual(value[-1, 3].item(), 0.)

    @staticmethod
    def selection_rows():
        return {f"q{question}_{action}{i}": dict(question=f"q{question}", action=action, accepted=i)
                for question in range(100) for action in ("root", "R", "E") for i in range(3)}

    def test_full_selects_all_and_only_labeled_states_from_all_100_questions(self):
        rows = self.selection_rows()
        rows["unlabeled"] = dict(question="q0", action="E", accepted=None)
        selected = data.select_capture_uids(rows, states_per_question=0)
        self.assertEqual(selected, sorted(set(rows) - {"unlabeled"}))
        self.assertEqual(len(selected), 900)
        self.assertEqual(len({rows[u]["question"] for u in selected}), 100)

    def test_smoke_is_deterministic_action_diverse_and_keeps_all_questions(self):
        rows = self.selection_rows()
        selected = data.select_capture_uids(rows, states_per_question=2, seed=42)
        self.assertEqual(selected, data.select_capture_uids(dict(reversed(list(rows.items()))), 2, 42))
        self.assertEqual(len(selected), 200)
        for question in range(100):
            group = [rows[u] for u in selected if rows[u]["question"] == f"q{question}"]
            self.assertEqual({r["action"] for r in group}, {"E", "R"})
        self.assertNotEqual(selected, data.select_capture_uids(rows, 2, 43))

    def test_smoke_selection_ignores_label_values_and_parent_outcomes(self):
        rows = self.selection_rows()
        before = data.select_capture_uids(rows, 2, 42)
        for row in rows.values():
            row.update(accepted=64, parent_K=64, K_true=64, gain=100)
        self.assertEqual(before, data.select_capture_uids(rows, 2, 42))

    def test_smoke_handles_sparse_action_buckets_without_duplicates(self):
        rows = {f"r{i}": dict(question="q", action="R", accepted=0) for i in range(3)}
        self.assertEqual(len(data.select_capture_uids(rows, 2)), 2)
        self.assertEqual(data.select_capture_uids(rows, 10), sorted(rows))
        for budget in (-1, True, 2.0, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                data.select_capture_uids(rows, budget)


class IdentityAndSourceTests(unittest.TestCase):
    def test_identity_comes_from_original_config_and_summary_without_latest_fallback(self):
        config = {"target_model_name": "Qwen/actual-saved-checkpoint", "target_quantization": "none"}
        summary = {"runtime": {"target_revision": REVISION, "transformers": "4.53.1"}, "verifier": "saved-mode"}
        original = copy.deepcopy((config, summary))
        identity = data.verifier_identity(config, summary)
        self.assertEqual(identity["model_id"], config["target_model_name"])
        self.assertEqual(identity["revision"], REVISION)
        self.assertEqual(identity["historical_transformers"], "4.53.1")
        self.assertEqual(identity["dtype"], "float16")
        self.assertEqual((config, summary), original)

    def test_missing_mutable_or_malformed_revisions_are_rejected(self):
        for revision in (None, "", "main", "latest", "a09a354", "g" * 40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                data.verifier_identity({"target_model_name": "saved-model"}, {"runtime": {"target_revision": revision}})
        for config in ({}, {"target_model_name": ""}, {"target_model_name": "saved-model", "target_quantization": "4bit"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                data.verifier_identity(config, {"runtime": {"target_revision": REVISION}})

    @staticmethod
    def source_fixture():
        rows = {}
        metadata = []
        for uid, n, segment, ids, accepted in (("parent", 2, 0, [10, 11], 2),
                                              ("child", 4, 2, [10, 11, 12, 13], 3)):
            rows[uid] = dict(question="q", length=n, accepted=accepted,
                ids=torch.tensor([[999, token] for token in ids]), c=torch.zeros(n, 20), context=torch.zeros(8),
                native_target="old projected teacher must not become a feature")
            metadata.append(dict(state_id=uid, prefix_token_ids=[1, 2, 3], segment_start=segment, round_id=5))
        payload = dict(rows=rows, cachez={uid: torch.zeros(64, 128) for uid in rows},
                       split={"train": ["q"], "val": [], "test": []}, provenance={"fingerprint": "frozen-source"})
        documents = dict(config={"target_model_name": "saved-model"},
                         summary={"runtime": {"target_revision": REVISION}})
        return payload, metadata, [dict(parent="parent", child="child", action="E")], documents

    def prepare(self, fixture):
        payload, metadata, edges, documents = fixture
        class Reader:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def json(self, name): return documents[name.removesuffix(".json")]
            def lines(self, name): return metadata if name == "states.jsonl" else edges
        with patch.object(data.phase0, "prepare_source", return_value=payload) as frozen, \
                patch.object(data, "_resolve_run", return_value="original-reader"), \
                patch.object(data, "_Reader", return_value=Reader()):
            result = data.prepare_native_source("original", "phase0", "temporary-cache", verification_samples=7)
        frozen.assert_called_once_with("original", "phase0", "temporary-cache", verification_samples=7, device="cpu")
        return result

    def test_source_reconstructs_exact_stop_candidates_parent_labels_and_prefix(self):
        result = self.prepare(self.source_fixture())
        child = result["rows"]["child"]
        self.assertEqual(child["candidate"], [10, 11, 12, 13])
        self.assertEqual(child["prefix"], [1, 2, 3])
        self.assertEqual((child["action"], child["parent_uid"], child["parent_length"], child["parent_K"]),
                         ("E", "parent", 2, 2))
        self.assertTrue(child["is_parent_full_prefix"])
        self.assertEqual(child["segment_start"], 2)
        self.assertNotIn("native_target", child)
        self.assertNotIn("teacher_hidden", child)
        self.assertEqual(result["identity"]["revision"], REVISION)

    def test_source_rejects_duplicate_parents_invalid_prefix_extend_geometry_or_split(self):
        for case in ("parents", "prefix", "boundary", "cross_question", "split", "missing_state"):
            fixture = self.source_fixture()
            payload, metadata, edges, _ = fixture
            if case == "parents":
                edges.append(copy.deepcopy(edges[0]))
            elif case == "prefix":
                metadata[1]["prefix_token_ids"] = []
            elif case == "boundary":
                metadata[1]["segment_start"] = 1
            elif case == "cross_question":
                payload["rows"]["child"]["question"] = "different"
            elif case == "split":
                payload["split"]["test"] = ["absent-question"]
            else:
                metadata.pop()
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.prepare(fixture)


if __name__ == "__main__":
    unittest.main()
