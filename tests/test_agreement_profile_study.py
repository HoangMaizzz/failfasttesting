"""Synthetic CPU contracts/end-to-end tests, not evidence of real Qwen quality."""
from pathlib import Path
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from agreement_profile_data import (SCHEMA, FEATURES, Record, Normalizer, atomic_json,
    copy_corpus, group_folds, load_corpus, pack_features, prefix_length, split_questions,
    validate_edge, validate_record, write_question)
from agreement_profile_metrics import binary_metrics, paired_comparison, prediction_row, score_rows
from agreement_profile_models import ProfileProbe
from collect_agreement_profiles import collect_question, weight_fingerprint
from run_agreement_profile_study import parse_args, run

torch.set_num_threads(1)


def pair(i, split):
    rng = np.random.default_rng(i)
    n = 4 if i % 2 else 8
    features = {g: rng.normal(size=(n, 5 if g != "metadata" else 23)).astype(np.float32)
                for g in FEATURES}
    candidate = np.arange(2, n + 2)
    native = np.full(n, 151665, np.int64)
    native[0] = candidate[0]
    features["metadata"][:, 0] = native == 151665
    features["metadata"][:, 1] = native != 151665
    features["metadata"][:, 2:7] = 0
    margin = np.ones(n, np.float32)
    margin[1] = -2 if i % 2 else 2
    matches = margin > 0
    a = Record(f"q{i}_a", f"q{i}", split,
        dict(context_hash=f"context{i}", native_forward_index=1, refinement_index=0),
        features, margin, matches, candidate, native, prefix_length(matches))
    b = copy.deepcopy(a)
    b.uid = f"q{i}_b"
    b.candidate[1] = 20
    b.native[1] = 20
    b.margin[1] *= -1
    b.matches[1] = not b.matches[1]
    b.accepted = prefix_length(b.matches)
    b.meta.update(native_forward_index=2, refinement_index=1)
    b.features = {g: x + .1 for g, x in features.items()}
    b.features["metadata"][:, 0] = b.native == 151665
    b.features["metadata"][:, 1] = b.native != 151665
    return a, b


class ContractTests(unittest.TestCase):
    def test_exact_argmax_tie_and_one_forward_labels(self):
        from structured_sparse_collector import FullContextVerifier
        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 2)
                self.calls = 0
            def get_input_embeddings(self): return self.emb
            def forward(self, **kwargs):
                self.calls += 1
                self.kwargs = kwargs
                return SimpleNamespace(logits=torch.tensor([[[0., 3., 3., 1.],
                    [0., 0., 4., 1.], [0., 0., 0., 5.]]]))
        model = Target()
        verifier = FullContextVerifier(model, SimpleNamespace(eos_token_id=3),
                                       SimpleNamespace(capture_verifier_teacher=True))
        result = verifier.score([0], [2, 2], 10)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.kwargs["logits_to_keep"], 3)
        self.assertFalse(model.kwargs["use_cache"])
        self.assertEqual(result[0], 0)
        self.assertEqual(verifier.last_teacher["margin"][0], 0)
        self.assertEqual(verifier.last_teacher["argmax_ids"], [1, 2])

    def test_live_candidate_and_native_R_contracts(self):
        a, b = pair(1, "train")
        validate_record(a)
        self.assertEqual(validate_edge(a, b), 1)
        bad = copy.deepcopy(b)
        bad.native[0] = 99
        with self.assertRaisesRegex(ValueError, "committed"):
            validate_edge(a, bad)
        bad = copy.deepcopy(b)
        bad.margin[0] += 1
        with self.assertRaisesRegex(ValueError, "invariance"):
            validate_edge(a, bad)

    def test_teacher_and_future_features_cannot_enter_source_input(self):
        a, b = pair(1, "train")
        norm = Normalizer([a])
        before = pack_features([a], norm, "cpu")
        a.margin[:] = 999
        a.accepted = 0
        b.features["hidden"][:] = 555
        after = pack_features([a], norm, "cpu")
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]))
        model = ProfileProbe({g: a.features[g].shape[1] for g in FEATURES}, list(FEATURES),
                             width=16, layers=1, max_length=8, dropout=0).eval()
        with torch.no_grad():
            x = model(before)["margin"]
            after["teacher_margin"] = torch.full_like(x, -999)
            after["child_hidden"] = torch.randn(1, 8, 5)
            self.assertTrue(torch.equal(x, model(after)["margin"]))

    def test_prompt_split_folds_and_duplicate_text(self):
        questions = [dict(question_id=f"q{i}", prompt=f"problem{i}") for i in range(8)]
        split = split_questions(questions, 2, 2)
        self.assertEqual(list(split.values()).count("train"), 4)
        records = [pair(i, "train")[0] for i in range(4)]
        held = group_folds(records, 3, 42)
        self.assertEqual(set.union(*held), {r.question for r in records})
        for i, fold in enumerate(held):
            self.assertFalse(fold & set.union(*(f for j, f in enumerate(held) if i != j)))
        questions[-1]["prompt"] = questions[0]["prompt"]
        with self.assertRaisesRegex(ValueError, "Duplicate prompt text"):
            split_questions(questions, 2, 2)

    def test_normalization_uses_training_only(self):
        train, _ = pair(1, "train")
        held, _ = pair(2, "test")
        norm = Normalizer([train])
        mean = norm.stats["hidden"][0].copy()
        held.features["hidden"][:] = 1e5
        pack_features([held], norm, "cpu")
        np.testing.assert_array_equal(norm.stats["hidden"][0], mean)

    def test_metrics_ties_tail_and_matched_bootstrap(self):
        self.assertEqual(prefix_length([1, 1, 0, 1]), 2)
        self.assertEqual(binary_metrics([True, False], [0., 0.])["auroc"], .5)
        a, b = pair(1, "test")
        x = prediction_row(a, b, b.margin, parent=a)
        x["source_K_pred"] = a.accepted
        y = prediction_row(a, b, np.ones(b.length), parent=a)
        comparison = paired_comparison([x], [y], "margin_mae", 20)
        self.assertLess(comparison["mean"], 0)
        self.assertEqual(score_rows([x])["macro"]["K_mae"], 0)
        self.assertEqual(score_rows([x])["future_gain"]["macro_delta_K_mae"], 0)
        y["target_id"] = "other"
        with self.assertRaisesRegex(ValueError, "mismatched"):
            paired_comparison([x], [y], draws=20)

    def test_storage_copy_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            a, b = pair(1, "train")
            atomic_json(root / "manifest.json", dict(schema=SCHEMA))
            write_question(root / "questions", "q1", [a, b],
                [dict(source=a.uid, child=b.uid, action="R")], {})
            records, edges, completed = load_corpus(root)
            self.assertEqual(len(records), 2)
            np.testing.assert_array_equal(records[a.uid].candidate, a.candidate)
            archive_path = Path(tmp) / "input.zip"
            with zipfile.ZipFile(archive_path, "w") as z:
                for path in root.rglob("*"):
                    if path.is_file(): z.write(path, Path("corpus") / path.relative_to(root))
            copy_corpus(archive_path, Path(tmp) / "copy")
            self.assertEqual(len(load_corpus(Path(tmp) / "copy")[0]), 2)
            unsafe = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(unsafe, "w") as z:
                z.writestr("corpus/../../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                copy_corpus(unsafe, Path(tmp) / "reject")
            with self.assertRaisesRegex(ValueError, "inside"):
                copy_corpus(root, root / "nested")

    def test_resume_weight_identity_and_partial_archive_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"weights-one")
            first = weight_fingerprint(root)
            (root / "model.safetensors").write_bytes(b"weights-two")
            self.assertNotEqual(first, weight_fingerprint(root))
            args = parse_args(["--mode", "train", "--input_path", str(root),
                               "--output_dir", str(root / "failed"), "--device", "cpu"])
            with self.assertRaisesRegex(ValueError, "manifest"):
                run(args)
            with zipfile.ZipFile(root / "failed.zip") as z:
                self.assertEqual(json.loads(z.read("summary.json"))["status"], "partial_error")
                self.assertIn("error.txt", z.namelist())

    def test_real_collection_shell_with_fake_LLMs_does_not_use_gains_to_choose_actions(self):
        from test_world_model_pretraining import FakeRunner, FakeVerifier, FakeTokenizer
        from world_model_environment import NativeTrainingEnvironment
        class Tokenizer(FakeTokenizer): eos_token_id = 0
        class Verifier(FakeVerifier):
            def score(self, prefix, proposal, remaining):
                result = super().score(prefix, proposal, remaining)
                expected = [(i + len(prefix) - 3) % 8 + 1 for i in range(len(proposal))]
                margin = [1. if x == y else -1. for x, y in zip(proposal, expected)]
                self.last_teacher = dict(argmax_ids=expected, margin=margin,
                    candidate_logit=margin, rival_logit=[0.]*len(proposal))
                return result
        args = parse_args(["--output_dir", "unused", "--num_questions", "8",
            "--validation_questions", "2", "--test_questions", "2",
            "--max_rounds_per_question", "1", "--segments_per_round", "2",
            "--states_per_question", "16", "--max_proposal_tokens", "16",
            "--raw_top_k", "2", "--hidden_layers", "14", "28", "--device", "cpu"])
        args.parity_repeat = True
        verifier = Verifier()
        captured = []
        env = NativeTrainingEnvironment(FakeRunner(), verifier, 0, 4, args,
                                        lambda state, *x: captured.append(state))
        rows, edges, ledger = collect_question(dict(question_id="q", prompt="problem"), "train",
            Tokenizer(), env, verifier, torch.randn(120, 4), args)
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(row.meta["label_source"].startswith("shadow") for row in rows))
        self.assertTrue(any(row.meta["label_reused_same_context_candidate"] for row in rows))
        self.assertTrue(all(not state.submitted for state in captured))
        self.assertFalse(ledger["generation_to_eos_required"])
        for edge in edges:
            table = {r.uid: r for r in rows}
            validate_edge(table[edge["source"]], table[edge["child"]])
        class WorstVerifier(Verifier):
            def score(self, prefix, proposal, remaining):
                expected = [51 + i for i in range(len(proposal))]
                self.last_teacher = dict(argmax_ids=expected, margin=[-1.]*len(proposal),
                    candidate_logit=[-1.]*len(proposal), rival_logit=[0.]*len(proposal))
                return 0, 1, [expected[0]], 1.
        worst = WorstVerifier()
        other_env = NativeTrainingEnvironment(FakeRunner(), worst, 0, 4, args, lambda *x: None)
        other_rows, other_edges, _ = collect_question(dict(question_id="q", prompt="problem"), "train",
            Tokenizer(), other_env, worst, torch.randn(120, 4), args)
        self.assertEqual([(e["source"], e["child"]) for e in edges],
                         [(e["source"], e["child"]) for e in other_edges])
        self.assertEqual([r.candidate.tolist() for r in rows], [r.candidate.tolist() for r in other_rows])
        self.assertNotEqual([r.accepted for r in rows], [r.accepted for r in other_rows])

    def test_provisional_EOS_keeps_R_and_replay_features_must_match(self):
        from test_world_model_pretraining import FakeRunner, FakeVerifier, settings
        from world_model_environment import MASK_ID, NativeTrainingEnvironment
        class EosRunner(FakeRunner):
            def segment(self, prompt, max_snapshots):
                rows = super().segment(prompt, max_snapshots)
                rows[0]["proposal_token_ids_after_fill"][-1] = 0
                return rows
        env = NativeTrainingEnvironment(EosRunner(), FakeVerifier(), 0, 4,
                                        settings(), lambda *x: None)
        state = env.start("q", 0, [10, 11, 12], 65)
        self.assertIsNone(state.terminal_reason)
        self.assertIn("R", env.actions(state))
        self.assertNotIn("E", env.actions(state))
        child = env.step(state, "R", 65)
        self.assertIsNotNone(child)
        self.assertNotIn(0, child.observation.ids[:, 1].tolist())
        # Even committed EOS cannot terminate while an earlier position is MASK.
        from test_world_model_pretraining import snap
        row = snap()
        row["proposal_token_ids_before_fill"][-1] = 0
        row["proposal_token_ids_after_fill"][-1] = 0
        pending = env.make_state("q", 0, [10], [], row, 0, None, 65, None, 0.)
        self.assertIn("R", env.actions(pending))
        row["proposal_token_ids_before_fill"] = list(row["proposal_token_ids_after_fill"])
        terminal = env.make_state("q", 0, [10], [], row, 0, None, 65, None, 0.)
        self.assertEqual(env.actions(terminal), ["S"])
        class FeatureDrift(FakeRunner):
            def segment(self, prompt, max_snapshots):
                rows = super().segment(prompt, max_snapshots)
                if max_snapshots > 1:
                    rows[0]["hidden_states"][0][0][0] += .1
                return rows
        env = NativeTrainingEnvironment(FeatureDrift(), FakeVerifier(), 0, 4,
                                        settings(), lambda *x: None)
        state = env.start("q", 0, [10], 65)
        with self.assertRaisesRegex(RuntimeError, "replay drift in hidden_states"):
            env.step(state, "R", 65)

    def test_fully_unmasked_state_never_calls_R_or_reunmasks_old_E_prefix(self):
        from test_world_model_pretraining import FakeRunner, FakeVerifier, settings, snap
        from world_model_environment import NativeTrainingEnvironment
        class CompleteRunner(FakeRunner):
            def segment(self, prompt, max_snapshots):
                self.calls.append((list(prompt), max_snapshots))
                row = snap()
                row["proposal_token_ids_before_fill"] = list(row["proposal_token_ids_after_fill"])
                row["masks_remaining"] = 0
                row["newly_unmasked_positions"] = list(range(8))
                return [row]
        runner = CompleteRunner()
        captured = []
        env = NativeTrainingEnvironment(runner, FakeVerifier(), 0, 4, settings(),
                                        lambda *x: captured.append(x))
        source = env.start("q", 0, [10, 11, 12], 65)
        self.assertNotIn("R", env.actions(source))
        calls = len(runner.calls)
        with self.assertRaises(ValueError): env.step(source, "R", 65)
        self.assertEqual(len(runner.calls), calls)
        child = env.step(source, "E", 65)
        self.assertTrue(torch.equal(child.observation.ids[:8, 0], source.observation.ids[:, 1]))
        self.assertNotIn("R", env.actions(child))
        self.assertNotIn("E", env.actions(child))  # max proposal length = 16
        env.submit(child, 65)
        self.assertEqual(env.actions(child), [])
        calls = len(runner.calls)
        for action in ("R", "E"):
            with self.assertRaises(ValueError): env.step(child, action, 65)
        with self.assertRaises(ValueError): env.submit(child, 65)
        self.assertEqual(len(runner.calls), calls)
        self.assertEqual(len(captured), 2)

    def test_collector_does_not_fabricate_R_pairs_for_complete_or_exhausted_states(self):
        from test_world_model_pretraining import FakeRunner, FakeVerifier, FakeTokenizer, snap
        from world_model_environment import NativeTrainingEnvironment
        class Tokenizer(FakeTokenizer): eos_token_id = 0
        class Verifier(FakeVerifier):
            def score(self, prefix, proposal, remaining):
                expected = list(proposal)
                self.last_teacher = dict(argmax_ids=expected, margin=[1.]*len(proposal),
                    candidate_logit=[1.]*len(proposal), rival_logit=[0.]*len(proposal))
                return len(proposal), len(proposal)+1, list(proposal)+[1], 1.
        args = parse_args(["--output_dir", "unused", "--num_questions", "8",
            "--validation_questions", "2", "--test_questions", "2",
            "--max_rounds_per_question", "1", "--segments_per_round", "2",
            "--states_per_question", "16", "--max_proposal_tokens", "16",
            "--raw_top_k", "2", "--hidden_layers", "14", "28", "--device", "cpu"])
        class Runner(FakeRunner):
            complete = False
            def segment(self, prompt, max_snapshots):
                self.calls.append((list(prompt), max_snapshots))
                row = snap()
                if self.complete:
                    row["proposal_token_ids_before_fill"] = list(row["proposal_token_ids_after_fill"])
                    row["masks_remaining"] = 0
                return [row]  # no next snapshot when incomplete
        for complete in (False, True):
            runner = Runner(); runner.complete = complete
            verifier = Verifier()
            env = NativeTrainingEnvironment(runner, verifier, 0, 4, args, lambda *x: None)
            rows, edges, ledger = collect_question(dict(question_id="q", prompt="problem"),
                "train", Tokenizer(), env, verifier, torch.randn(120, 4), args)
            self.assertEqual([row.length for row in rows], [8, 16])
            self.assertEqual(edges, [])
            self.assertEqual(ledger["R_edges"], 0)
            self.assertEqual(env.stats["executed_R"], 0)
            self.assertEqual(len(runner.calls), 2 if complete else 4)
            self.assertTrue(np.array_equal(rows[1].native[:8], rows[0].candidate))

    def test_duplicate_native_boundary_is_rejected_before_child_is_published(self):
        from test_world_model_pretraining import FakeRunner, FakeVerifier, settings
        from world_model_environment import NativeTrainingEnvironment
        class Runner(FakeRunner):
            def segment(self, prompt, max_snapshots):
                rows = super().segment(prompt, max_snapshots)
                if len(rows) > 1:
                    rows[1]["unmask_forward_index"] = rows[0]["unmask_forward_index"]
                return rows
        captured = []
        env = NativeTrainingEnvironment(Runner(), FakeVerifier(), 0, 4, settings(),
                                        lambda *x: captured.append(x))
        source = env.start("q", 0, [10], 65)
        with self.assertRaisesRegex(RuntimeError, "exactly one native forward"):
            env.step(source, "R", 65)
        self.assertEqual(len(captured), 1)
        self.assertEqual(env.stats["executed_R"], 0)

    def test_offline_complete_ABC_and_oof_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            questions = [dict(question_id=f"q{i}", prompt=f"problem{i}") for i in range(8)]
            splits = split_questions(questions, 2, 2)
            atomic_json(root / "manifest.json", dict(schema=SCHEMA, questions=questions, split=splits))
            for i in range(8):
                a, b = pair(i, splits[f"q{i}"])
                write_question(root / "questions", f"q{i}", [a, b],
                               [dict(source=a.uid, child=b.uid, action="R")], {})
            out = Path(tmp) / "result"
            args = parse_args(["--mode", "train", "--input_path", str(root), "--output_dir", str(out),
                "--num_questions", "8", "--validation_questions", "2", "--test_questions", "2",
                "--training_seeds", "42", "--epochs", "1", "--width", "16", "--layers", "1",
                "--batch_size", "2", "--oof_folds", "2", "--bootstrap_draws", "20", "--device", "cpu"])
            summary = run(args)
            self.assertEqual(summary["status"], "complete")
            self.assertTrue(summary["scientific_tests_complete"])
            report = json.loads((out / "metrics.json").read_text())
            self.assertIn("oracle_rich_test_seed42", report["C"])
            self.assertIn("C_K_profile_vs_Konly_K_head_mae_seed42", report["comparisons"])
            self.assertEqual(report["C"]["Konly_test_seed42"]["margin_metrics"], "not_applicable_K_only_head")
            provenance = json.loads((out / "oof_provenance.json").read_text())
            for fold in provenance:
                self.assertFalse(set(fold["trained_prompts"]) & set(fold["predicted_prompts"]))
            with zipfile.ZipFile(out.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("metrics.json", z.namelist())
                self.assertFalse(any("safetensors" in name for name in z.namelist()))


if __name__ == "__main__":
    unittest.main()
