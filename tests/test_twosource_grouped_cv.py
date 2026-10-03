import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_twosource_grouped_cv import (grouped_folds, load_experiences,
    make_replay, make_film_examples, evaluate_film, train_film)
from world_model_core import pack_observations
from persistent_world_model_v1 import GatedFiLMAdapter


class GroupedCrossValidationTests(unittest.TestCase):
    def test_top1_match_uses_vocabulary_id_not_topk_index(self):
        from torch import nn
        class Drafter(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Linear(4,20,bias=False)
                with torch.no_grad():
                    self.head.weight.zero_();self.head.weight[17]=1
            def get_output_embeddings(self):return self.head
        example=dict(hidden=torch.ones(4),latent=torch.zeros(8),ids=torch.tensor([17,9,6]),
            teacher_logits=torch.tensor([4.,1.,0.]),teacher_logsumexp=torch.tensor(5.),
            question='q',state_id='s',position=0)
        result=evaluate_film(GatedFiLMAdapter(4,8),Drafter(),[example],'cpu')
        self.assertEqual(result['base_top1_match'],1.0)
        self.assertEqual(result['film_top1_match'],1.0)

    def test_folds_partition_questions_without_overlap(self):
        questions = [f"gsm8k:{i}" for i in range(100)]
        folds = grouped_folds(questions, seed=7)
        self.assertEqual([len(fold) for fold in folds], [20] * 5)
        self.assertEqual(len(set(sum(folds, []))), 100)
        for i, heldout in enumerate(folds):
            train = set(questions) - set(heldout)
            self.assertFalse(train.intersection(heldout))
            self.assertEqual(len(train), 80)

    def test_loads_all_questions_then_replay_keeps_question_groups_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "experience").mkdir()
            (root / "summary.json").write_text(json.dumps({
                "schema": "interactive_acceptance_two_source_v1",
                "status": "complete", "questions_completed": 5}), encoding="utf-8")
            states, labels, edges = [], [], []
            ids, hidden, gaps, scalars, context, topk, history = [], [], [], [], [], [], []
            offsets = [0]
            for q in range(5):
                for local_state in range(2):
                    uid = f"s{q}_{local_state}"
                    row = len(states)
                    states.append({"state_id": uid,
                        "parent_state_id": None if local_state == 0 else f"s{q}_0",
                        "question": f"gsm8k:{q}", "round_id": 0, "length": 1,
                        "prefix_token_ids": [10], "shard": "experience/test.npz", "row": row})
                    labels.append({"state_id": uid, "accepted_len": 1,
                        "label_valid": True})
                    ids.append(np.zeros((1, 2), dtype=np.int64))
                    hidden.append(np.zeros((1, 2, 4), dtype=np.float16))
                    gaps.append(np.zeros((1, 3), dtype=np.float16))
                    scalars.append(np.zeros((1, 16), dtype=np.float32))
                    context.append(np.zeros(8, dtype=np.float32))
                    topk.append(np.zeros((1, 3), dtype=np.int64))
                    history.append(np.zeros((1, 48), dtype=np.float32))
                    offsets.append(offsets[-1] + 1)
                edges.append({"parent": f"s{q}_0", "child": f"s{q}_1",
                              "action": "R", "question": f"gsm8k:{q}"})
            np.savez(root / "experience/test.npz", offsets=np.asarray(offsets),
                ids=np.concatenate(ids), hidden=np.concatenate(hidden),
                gaps=np.concatenate(gaps), scalars=np.concatenate(scalars),
                context=np.stack(context), aligned_topk_token_ids=np.concatenate(topk),
                history=np.concatenate(history),
                teacher_topk_token_ids=np.tile(np.array([[3, 4, 5]], dtype=np.int32), (10, 1)),
                teacher_topk_logits=np.tile(np.array([[2, 1, 0]], dtype=np.float16), (10, 1)),
                teacher_logsumexp=np.full(10, 4.0, dtype=np.float32),
                teacher_distribution_valid=np.arange(10) % 2 == 0,
                teacher_actual=np.ones(10, dtype=np.bool_))
            for name, rows in (("states.jsonl", states), ("labels.jsonl", labels),
                               ("edges.jsonl", edges), ("teacher_targets.jsonl", [])):
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows),
                                         encoding="utf-8")

            summary, observations, loaded_edges, question_ids = load_experiences(root)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(question_ids), 5)
            self.assertEqual(len(observations), 10)
            self.assertEqual(len(loaded_edges), 5)
            heldout = {"gsm8k:0"}
            train = make_replay(observations, loaded_edges,
                                set(question_ids) - heldout, seed=11)
            validation = make_replay(observations, loaded_edges, heldout, seed=12,
                                     sampling_mode="natural")
            self.assertEqual(len(train.nodes), 8)
            self.assertEqual(len(train.edges), 4)
            self.assertEqual(len(validation.nodes), 2)
            self.assertEqual(len(validation.edges), 1)
            self.assertEqual({o.question for o in train.nodes.values()},
                             set(question_ids) - heldout)
            self.assertEqual({o.question for o in validation.nodes.values()}, heldout)
            self.assertTrue(observations["s0_0"].teacher_distribution_valid.item())
            self.assertFalse(observations["s0_1"].teacher_distribution_valid.item())
            packed = pack_observations([observations["s0_0"], observations["s0_1"]],
                                       torch.randn(32, 5), "cpu")
            self.assertEqual(packed["teacher_distribution_valid"][:, 0].tolist(), [True, False])

    def test_film_supervision_stops_at_first_rejection_and_uses_predicted_child_latent(self):
        from world_model_core import Observation
        obs = Observation("child", "q", 0,
            ids=torch.tensor([[1, 2], [1, 3], [1, 4]]),
            hidden=torch.randn(3, 2, 4).half(), gaps=torch.zeros(3, 3).half(),
            scalars=torch.zeros(3, 16), context=torch.zeros(8), accepted=1,
            prefix_ids=torch.tensor([9]), topk_ids=torch.ones(3, 3, dtype=torch.long),
            history=torch.zeros(3, 4))
        obs.teacher_topk_ids = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
        obs.teacher_topk_logits = torch.tensor([[2, 1, 0], [2, 1, 0], [2, 1, 0]]).half()
        obs.teacher_logsumexp = torch.tensor([4., 4., 4.])
        obs.teacher_distribution_valid = torch.tensor([True, True, True])
        latent = torch.arange(8).float()
        rows = make_film_examples({"child": obs}, {"q"}, {"child": latent},
                                  hidden_slot=1, vocab_size=32)
        self.assertEqual([row["position"] for row in rows], [0, 1])
        self.assertTrue(torch.equal(rows[0]["latent"], latent))

    def test_film_training_and_heldout_metric_run_without_backbone_updates(self):
        from torch import nn

        class TinyDrafter(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 20, bias=False)

            def get_output_embeddings(self):
                return self.head

        drafter = TinyDrafter()
        adapter = GatedFiLMAdapter(hidden_dim=4, latent_dim=8)
        examples = []
        for index in range(12):
            examples.append(dict(hidden=torch.randn(4).half(), latent=torch.randn(8),
                ids=torch.tensor([1, 2, 3]), teacher_logits=torch.tensor([3., 2., 1.]),
                teacher_logsumexp=torch.tensor(4.), question=f"q{index % 3}",
                state_id=str(index), position=0))
        before = evaluate_film(adapter, drafter, examples, "cpu", batch_tokens=4,
                               max_examples=8, seed=11)
        self.assertAlmostEqual(before["base_kl"], before["film_kl"], places=6)
        original = [parameter.detach().clone() for parameter in drafter.parameters()]
        trained = train_film(adapter, drafter, examples, "cpu", steps=2,
            batch_tokens=4, learning_rate=1e-4, seed=12, log_prefix="unit")
        after = evaluate_film(adapter, drafter, examples, "cpu", batch_tokens=4,
                              max_examples=8, seed=11)
        self.assertEqual(trained["steps"], 2)
        self.assertEqual(after["n"], 8)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, drafter.parameters())))


if __name__ == "__main__":
    unittest.main()
