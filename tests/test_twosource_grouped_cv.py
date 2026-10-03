import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_twosource_grouped_cv import grouped_folds, load_experiences, make_replay


class GroupedCrossValidationTests(unittest.TestCase):
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
                history=np.concatenate(history))
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


if __name__ == "__main__":
    unittest.main()
