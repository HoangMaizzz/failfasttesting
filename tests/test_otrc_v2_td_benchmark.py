import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from run_otrc_v2_td_benchmark import command_for, feature_diagnostics


class OTRCV2TDBenchmarkTests(unittest.TestCase):
    def args(self):
        return Namespace(
            warmup_questions=1,
            max_new_tokens=1024,
            spec_len=8,
            incr_len=8,
            max_spec_len=60,
            block_size=32,
            small_block_size=8,
            target_model_name="Qwen/Qwen2.5-7B-Instruct",
            dllm_dir="/tmp/Fast_dLLM_v2_1.5B",
            drafter_threshold=0.05,
            lowconf_threshold=0.45,
            adaptive_learning_rate=0.02,
            adaptive_mc_learning_rate=0.01,
            adaptive_mc_mix=0.5,
            adaptive_update_mode="mixed",
            adaptive_rho_alpha=0.05,
            adaptive_factual_ema_alpha=0.2,
            adaptive_risk_beta=1.0,
            adaptive_stop_probability_threshold=0.75,
            adaptive_uncertainty_prior=1.0,
            adaptive_epistemic_scale=0.1,
            adaptive_q_margin=0.0,
            adaptive_explore_epsilon=0.1,
            adaptive_explore_min=0.01,
            adaptive_explore_decay=0.998,
            adaptive_warmup_rounds=20,
            adaptive_early_stop_min_observations=32,
            adaptive_min_action_probability=0.1,
            adaptive_max_importance_weight=5.0,
            adaptive_weight_snapshot_interval=100,
            seed=42,
            log_level="INFO",
        )

    def test_command_runs_only_v2(self):
        command = command_for(
            self.args(),
            "gsm8k",
            [6, 24],
            Path("/tmp/out"),
        )
        joined = " ".join(command)
        self.assertIn("--adaptive-feature-schema otrc_v2_td", joined)
        self.assertNotIn("strict_greedy", joined)
        self.assertNotIn("causal_oracle", joined)
        self.assertNotIn("verifier_ar", joined)

    def test_feature_diagnostics_flags_constant_feature(self):
        decisions = pd.DataFrame({
            "features": [
                "[1,0.5,0.2,0.1,0.5,0.5,0.8,-0.2,0.2,1.0,0.1]",
                "[1,0.2,0.4,0.3,0.8,0.2,0.3,0.1,0.4,0.5,0.2]",
            ]
        })
        stats, pearson, spearman, conditioning = feature_diagnostics(
            "gsm8k",
            decisions,
        )
        bias = stats.loc[stats.feature.eq("bias")].iloc[0]
        self.assertEqual(bias.approximately_constant, 1)
        self.assertEqual(conditioning.iloc[0].feature_dim, 11)
        self.assertFalse(pearson.empty)
        self.assertFalse(spearman.empty)


if __name__ == "__main__":
    unittest.main()
