import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from adaptive_td import V22_FEATURE_NAMES
from run_otrc_v2_td_benchmark import (
    command_for,
    confidence_diagnostics,
    feature_diagnostics,
    snapshot_diagnostics,
)


class OTRCV2TDBenchmarkTests(unittest.TestCase):
    def args(self):
        return Namespace(
            feature_schema="otrc_v2_2_td",
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

    def test_command_runs_only_v22(self):
        command = command_for(
            self.args(),
            "gsm8k",
            [6, 24],
            Path("/tmp/out"),
        )
        joined = " ".join(command)
        self.assertIn("--adaptive-feature-schema otrc_v2_2_td", joined)
        self.assertNotIn("strict_greedy", joined)
        self.assertNotIn("causal_oracle", joined)
        self.assertNotIn("verifier_ar", joined)

    def test_feature_diagnostics_flags_constant_feature(self):
        decisions = pd.DataFrame({
            "features": [
                "[1,0.5,0.2,0.1,-0.2,0.2,1.0,0.1]",
                "[1,0.2,0.4,0.3,0.1,0.4,0.5,0.2]",
            ]
        })
        stats, pearson, spearman, conditioning = feature_diagnostics(
            "gsm8k",
            decisions,
            V22_FEATURE_NAMES,
        )
        bias = stats.loc[stats.feature.eq("bias")].iloc[0]
        self.assertEqual(bias.approximately_constant, 1)
        self.assertEqual(conditioning.iloc[0].feature_dim, 8)
        self.assertFalse(pearson.empty)
        self.assertFalse(spearman.empty)

    def test_snapshot_diagnostics_checks_post_commit_invariants(self):
        decisions = pd.DataFrame({
            "proposal_remaining_masks": [4, 2],
            "remaining_masks": [4, 2],
            "proposal_remaining_confidence_count": [4, 1],
            "proposal_remaining_confidence_coverage": [1.0, 0.5],
            "proposal_snapshot_valid": [True, True],
            "proposal_snapshot_phase": [
                "post_commit_pre_decision",
                "post_commit_pre_decision",
            ],
        })
        result = snapshot_diagnostics("math", decisions)
        self.assertEqual(result["valid_snapshot_percent"], 100.0)
        self.assertEqual(result["mask_count_match_percent"], 100.0)
        self.assertEqual(result["confidence_coverage_mean"], 0.75)

    def test_confidence_diagnostics_use_only_unresolved_states(self):
        decisions = pd.DataFrame({
            "proposal_remaining_masks": [4, 2, 0],
            "proposal_remaining_min_confidence": [0.02, 0.10, None],
        })
        result = confidence_diagnostics("math", decisions, 0.05)
        self.assertEqual(result["unresolved_decisions"], 2)
        self.assertEqual(result["confidence_observations"], 2)
        self.assertAlmostEqual(result["min_confidence_mean"], 0.06)
        self.assertEqual(result["below_drafter_threshold_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
