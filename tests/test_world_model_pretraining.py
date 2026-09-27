"""CPU tests with explicit fake LLMs. Not a real GPU benchmark."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model_core import (AcceptanceWorldModel, ExperienceReplay, WorldModelLearner,
    acceptance_nll, expected_acceptance, prefix_log_distribution, pack_observations)
from world_model_environment import (MASK_ID, NativeTrainingEnvironment, native_observation)
from pretrain_acceptance_world_model import (ExperienceWriter, explore_questions,
    package, parse_args)

torch.set_num_threads(1)


def settings(output=None):
    args = parse_args(["--dllm_dir", "unused", "--output_dir", str(output or "unused"),
        "--max_rounds_per_question", "1", "--max_proposal_tokens", "16",
        "--latent_dim", "16", "--raw_top_k", "2", "--batch_sequences", "2",
        "--warmup_updates", "0", "--horizon_warmup_updates", "0",
        "--refine_probability", "0.8"])
    return args


def snap(index=0):
    filled = list(range(1, 9))
    if index == 0:
        filled[-1] = 99
    native = filled[:index+1] + [MASK_ID]*(7-index)
    return dict(proposal_token_ids_before_fill=native,
        proposal_token_ids_after_fill=filled, unmask_forward_index=index+1,
        masks_remaining=7-index, confidences=[0.6]*8,
        newly_unmasked_positions=[index], hidden_layer_indices=[14, 28],
        hidden_states=[[[0.1*layer, 0.2*row, 0.3*index, 0.4] for row in range(8)]
                       for layer in range(2)],
        native_hidden_start_offset=0, native_topk_start_offset=0,
        topk_logits=[[2.0, 1.0] for _ in range(8)], topk_token_ids=[[1, 2]]*8)


def observation(uid="s", index=0, question="q", accepted=7):
    result = native_observation(uid, question, 0, [10, 11, 12], [], snap(index),
                                 None, 4, 2, settings(), index)
    result.accepted = accepted
    return result


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.exhaust = False
        self.drift = False

    def segment(self, prompt, max_snapshots):
        self.calls.append((list(prompt), max_snapshots))
        rows = [snap(i) for i in range(min(max_snapshots, 1 if self.exhaust else 4))]
        if self.drift and max_snapshots > 1:
            rows[0]["proposal_token_ids_after_fill"][1] = 101
        return rows


class FakeVerifier:
    def score(self, prefix, proposal, remaining):
        expected = [i % 8 + 1 for i in range(len(proposal)+1)]
        accepted = 0
        while accepted < len(proposal) and proposal[accepted] == expected[accepted]:
            accepted += 1
        emitted = (proposal[:accepted]+[expected[accepted]])[:remaining]
        return accepted, len(emitted), emitted, 1.0


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [10, 11, 12]

    def decode(self, tokens):
        return str(tokens)


class AcceptanceTests(unittest.TestCase):
    def test_distribution_padding_normalization_and_boundaries(self):
        logits = torch.zeros(3, 8, requires_grad=True)
        lengths = torch.tensor([1, 4, 8])
        log_p = prefix_log_distribution(logits, lengths)
        self.assertTrue(torch.allclose(log_p.exp().sum(-1), torch.ones(3)))
        self.assertEqual(float(log_p.exp()[0, 2]), 0)
        nll = acceptance_nll(logits, lengths, torch.tensor([0, 4, 8]))
        self.assertTrue(torch.isfinite(nll))
        nll.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(bool((expected_acceptance(logits, lengths) <= lengths).all()))

    def test_unlabelled_is_not_zero(self):
        logits = torch.randn(1, 8, requires_grad=True)
        loss = acceptance_nll(logits, torch.tensor([8]), torch.tensor([-1]))
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_label_does_not_enter_encoder_and_dynamic_lengths(self):
        table = torch.randn(MASK_ID+1, 4)
        o = observation()
        b1 = pack_observations([o], table, "cpu")
        o.accepted = 0
        b2 = pack_observations([o], table, "cpu")
        model = AcceptanceWorldModel(4, 4, 2, dim=16, dropout=0).eval()
        z1, z2 = model.encoder(b1), model.encoder(b2)
        self.assertTrue(torch.equal(z1.tokens, z2.tokens))
        self.assertEqual(model.dynamics(z1, torch.tensor([0]), 8).tokens.shape, (1, 8, 16))
        self.assertEqual(model.dynamics(z1, torch.tensor([1]), 8).tokens.shape, (1, 16, 16))


class NativeTests(unittest.TestCase):
    def make_env(self):
        self.runner = FakeRunner()
        self.records = []
        return NativeTrainingEnvironment(self.runner, FakeVerifier(), 0, 4, settings(),
            lambda *args: self.records.append(args))

    def test_native_R_and_E_use_same_forward_commit(self):
        env = self.make_env()
        root = env.start("q", 0, [10, 11, 12], 128)
        self.assertEqual(self.runner.calls[-1][1], 1)
        child = env.step(root, "R", 128)
        self.assertEqual(self.runner.calls[-1][1], 2)
        self.assertEqual((root.observation.accepted, child.observation.accepted), (7, 8))
        extended = env.step(root, "E", 128)
        self.assertEqual(self.runner.calls[-1][0], root.prefix + root.observation.ids[:, 1].tolist())
        self.assertEqual(extended.observation.length, 16)
        self.assertEqual(extended.observation.accepted, 7)
        self.assertEqual(extended.observation.ids[:8, 0].tolist(), root.observation.ids[:, 1].tolist())

    def test_exhaustion_no_fake_edge_and_drift_raises(self):
        env = self.make_env()
        root = env.start("q", 0, [10, 11, 12], 128)
        self.runner.exhaust = True
        self.assertIsNone(env.step(root, "R", 128))
        self.assertEqual(len(self.records), 1)
        self.runner.exhaust = False
        self.runner.drift = True
        with self.assertRaisesRegex(RuntimeError, "drift"):
            env.step(root, "R", 128)

    def test_partial_capture_offsets_and_eos(self):
        snapshot = snap()
        snapshot["native_hidden_start_offset"] = 3
        snapshot["native_topk_start_offset"] = 4
        snapshot["hidden_states"] = [layer[:2] for layer in snapshot["hidden_states"]]
        snapshot["topk_logits"] = snapshot["topk_logits"][:2]
        obs = native_observation("s", "q", 0, [1]*29, [], snapshot, None, 4, 2, settings(), 0)
        self.assertEqual(torch.where(obs.scalars[:, 3].bool())[0].tolist(), [3, 4])
        self.assertEqual(torch.where(obs.scalars[:, 4].bool())[0].tolist(), [4, 5])
        env = self.make_env()
        root = env.start("q", 0, [10, 11, 12], 128)
        root.terminal_reason = "candidate_eos"
        self.assertEqual(env.actions(root), [])

    def test_extension_reaches_64_without_acceptance_gate(self):
        env = self.make_env()
        env.args.max_proposal_tokens = 64
        state = env.start("q", 0, [10, 11, 12], 128)
        for length in range(16, 65, 8):
            previous = state.observation.ids[:, 1].clone()
            state = env.step(state, "E", 128)
            self.assertEqual(state.observation.hidden.shape, (length, 2, 4))
            self.assertEqual(state.observation.gaps.shape, (length, 2))
            self.assertTrue(torch.equal(state.observation.ids[:length-8, 0], previous))
            self.assertEqual(state.observation.accepted, 7)
        self.assertNotIn("E", env.actions(state))


class ReplayTests(unittest.TestCase):
    def test_question_boundary_and_eviction(self):
        buffer = ExperienceReplay(3)
        a, b, c, d = [observation(uid=str(i), index=min(i, 3)) for i in range(4)]
        buffer.add(a, b, "R")
        other = observation("other", question="another")
        with self.assertRaises(ValueError):
            buffer.add(b, other, "R")
        buffer.add(b, c, "R")
        buffer.add(c, d, "R")
        self.assertEqual(len(buffer.nodes), 3)
        self.assertNotIn("0", buffer.nodes)
        for nodes, actions in buffer.sample(16, 3):
            for left, right, action in zip(nodes, nodes[1:], actions):
                self.assertIn((left.uid, right.uid, action), buffer.edges)

    def test_weights_update_and_rollout_receives_predicted_latent(self):
        table = torch.randn(MASK_ID+1, 4)
        model = AcceptanceWorldModel(4, 4, 2, dim=16, dropout=0)
        learner = WorldModelLearner(model, table, "cpu", warmup_updates=0, horizon_warmup=0)
        buffer = ExperienceReplay()
        a, b, c = [observation(str(i), index=i) for i in range(3)]
        buffer.add(a, b, "R")
        buffer.add(b, c, "R")
        buffer.sample = lambda *args: [([a, b, c], ["R", "R"])]
        calls, outputs = [], []
        hook = model.dynamics.register_forward_hook(lambda module, inputs, out: (
            calls.append(inputs[0].tokens.detach().clone()), outputs.append(out.tokens.detach().clone())) and None)
        original = next(model.parameters()).detach().clone()
        metrics = learner.update(buffer, 1, 3)
        hook.remove()
        self.assertTrue(torch.allclose(calls[1], outputs[0]))
        self.assertFalse(torch.equal(original, next(model.parameters())))
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertFalse(any("token_table" in key for key in learner.checkpoint()["model"]))


class FullSmokeTests(unittest.TestCase):
    def test_ten_fake_questions_interact_and_train_and_package(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            output.mkdir()
            args = settings(output)
            writer = ExperienceWriter(output)
            model = AcceptanceWorldModel(4, 4, 2, dim=16, dropout=0)
            learner = WorldModelLearner(model, torch.randn(MASK_ID+1, 4), "cpu",
                                        warmup_updates=0, horizon_warmup=0)
            env = NativeTrainingEnvironment(FakeRunner(), FakeVerifier(), 0, 4, args, None)
            questions = [dict(question_id=f"q{i}", prompt="synthetic test only") for i in range(10)]
            summary = dict(status="complete", questions_completed=0)
            explore_questions(args, questions, FakeTokenizer(), env, learner, writer, summary)
            self.assertEqual(summary["questions_completed"], 10)
            self.assertGreater(summary["updates"], 0)
            self.assertGreater(summary["parameter_l2_change"], 0)
            self.assertEqual(summary["evaluation"]["questions"], ["q8", "q9"])
            records = [json.loads(x) for x in (output / "states.jsonl").read_text().splitlines()]
            edges = [json.loads(x) for x in (output / "edges.jsonl").read_text().splitlines()]
            self.assertEqual(summary["updates"], sum(e["split"] == "train" for e in edges))
            for record in records:
                with np.load(output / record["shard"], allow_pickle=False) as shard:
                    self.assertEqual(int(shard["lengths"][record["row"]]), record["length"])
            package(output, Path(folder) / "result.zip", summary)
            import zipfile
            with zipfile.ZipFile(Path(folder) / "result.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("checkpoint.pt", archive.namelist())
                self.assertFalse(any("safetensors" in name for name in archive.namelist()))

    def test_partial_archive_is_labelled_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            output.mkdir()
            package(output, Path(folder) / "partial.zip", dict(status="partial", error="test"))
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "partial")


if __name__ == "__main__":
    unittest.main()
