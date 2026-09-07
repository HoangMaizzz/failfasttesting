import unittest
from types import SimpleNamespace

from adaptive_td import AdaptiveTDConfig, OnlineTDRefinementController
from run_math50_witness_online import IDS, command


class WitnessTest(unittest.TestCase):
    def controller(self, probe_only=False, ready=True, probe=0):
        item = OnlineTDRefinementController(AdaptiveTDConfig(
            feature_dim=6, feature_schema='otrc_v2_2_compact_td',
            credit_assignment='hindsight_delta_j_logistic_f2',
            policy_mode='hindsight_delta_j_logistic_f2',
            hindsight_logistic_use_prefix_feature=False,
            hindsight_logistic_dynamic_threshold=False,
            hindsight_logistic_probe_only=probe_only,
            hindsight_delta_j_min_pairs=0 if ready else 30,
            hindsight_delta_j_min_continue_pairs=0,
            hindsight_logistic_min_positive_problems=0,
            hindsight_delta_j_structural_probe_probability=probe,
            hindsight_delta_j_floor_probe_probability=probe))
        item.begin_hindsight_problem(299)
        item.hindsight_logistic_model.weights[0] = 10
        item.prepare_hindsight_snapshot(
            draft_proposal=list(range(8)), context_len=100,
            active_block_start=100, active_block_end=108,
            raw_current_state=None,
            f2_state={'active_span_size': 8, 'current_mask_count': 4},
            remaining_masks=4,
            proposal_length=8, max_spec_len=64, refinement_step=1,
            next_forward_latency_ms=4, forward_pass_index=1, decision_eligible=True)
        return item

    def decide(self, item):
        return item.choose((1.,) * 6, allow_stop=True, refinement_step=1)

    def test_masks_only_learned_continue(self):
        self.assertEqual(self.decide(self.controller()).action, 'continue')
        decision = self.decide(self.controller(probe_only=True))
        self.assertEqual(decision.action, 'stop')
        self.assertEqual(decision.diagnostics['action_source'], 'probe_only_stop')
        self.assertEqual(self.decide(self.controller(probe_only=True, probe=1)).action, 'continue')

    def test_cold_start_unchanged(self):
        a = self.decide(self.controller(ready=False))
        b = self.decide(self.controller(probe_only=True, ready=False))
        self.assertEqual(a.action, b.action)
        self.assertEqual(a.diagnostics['action_source'], b.diagnostics['action_source'])

    def test_commands_keep_identical_learning(self):
        args = SimpleNamespace(dllm_dir='drafter', seed=44, max_new_tokens=1024,
            target_device=0, drafter_device=0, target_quantization='int8',
            drafter_threshold=.5, lowconf_threshold=.7, log_level='INFO')
        learned = command(args, 'u1', 'out')
        probe = command(args, 'probe_only', 'out')
        self.assertEqual(probe[:-1], learned)
        self.assertEqual(probe[-1], '--adaptive-hindsight-logistic-probe-only')
        self.assertEqual(len(set(IDS)), 50)


if __name__ == '__main__':
    unittest.main()
