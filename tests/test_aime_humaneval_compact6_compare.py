import unittest
from argparse import Namespace
from pathlib import Path

from run_aime_humaneval_compact6_compare import (
    DATASETS,
    experiment_commands,
    output_spec,
    validate_args,
)
from run_otrc_v2_td_benchmark import PROBLEM_IDS


class AimeHumanEvalCompact6CompareTests(unittest.TestCase):
    def args(self):
        return Namespace(
            num_questions=25,
            warmup_questions=1,
            max_new_tokens=1024,
            drafter_threshold=0.30,
            lowconf_threshold=0.50,
            target_device=0,
            drafter_device=0,
            target_quantization="int8",
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            output_dir="/tmp/four_methods",
            resume=True,
            log_level="INFO",
        )

    def test_commands_cover_four_matched_cases(self):
        args = self.args()
        validate_args(args)
        commands = dict(experiment_commands(args))
        self.assertEqual(
            tuple(commands),
            ("failfast", "always_stop", "compact6_fixed_stochastic", "compact6_annealed"),
        )
        for command in commands.values():
            joined = " ".join(command)
            self.assertIn("--datasets aime humaneval", joined)
            self.assertIn("--num_questions 25", joined)
            self.assertIn("--target_quantization int8", joined)
            self.assertIn("--drafter_threshold 0.3", joined)
            self.assertIn("--lowconf_threshold 0.5", joined)

        self.assertIn("--adaptive_policy_mode symmetric", " ".join(commands["compact6_fixed_stochastic"]))
        self.assertIn("--adaptive_policy_mode symmetric_annealed", " ".join(commands["compact6_annealed"]))
        self.assertIn("--policy_ablation frozen_stop", " ".join(commands["always_stop"]))

    def test_outputs_use_distinct_method_names(self):
        specs = output_spec(self.args())
        methods = [method for _, method in specs.values()]
        self.assertEqual(len(methods), len(set(methods)))
        self.assertTrue(all(len(PROBLEM_IDS[d][:25]) == 25 for d in DATASETS))


if __name__ == "__main__":
    unittest.main()
