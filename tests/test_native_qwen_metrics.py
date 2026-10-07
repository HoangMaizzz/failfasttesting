"""Independent numerical and integrity contracts for native survival metrics.

Fixtures contain no tensors, model dependencies, captured hidden, or real test
data. In particular, the synthetic PASS below is a test of the gate logic only.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import native_qwen_metrics as metrics


def row(state='s0', question='q0', method='A', seed=42, action='root',
        length=4, accepted=2, q=None, parent_length=None, parent_K=None, split='val'):
    if q is None:
        q = [float(i < accepted) for i in range(length)]
    return dict(state_id=state, question_id=question, method=method, seed=seed,
                action=action, proposal_length=length, K_true=accepted,
                K_pred=sum(q), q_pred=list(q), parent_length=parent_length,
                parent_K=parent_K, split=split,
                survival_true=[int(i < accepted) for i in range(length)],
                new_block_mask=[action == 'E' and parent_length is not None and
                                i >= parent_length for i in range(length)],
                is_E_full_prefix=action == 'E' and parent_K is not None and parent_K == parent_length)


def predicted(state, question, method, seed, kpred, *, action='E', accepted=6,
              length=8, parent_length=4, parent_K=4, split='val'):
    return row(state, question, method, seed, action, length, accepted,
               [kpred / length] * length, parent_length, parent_K, split)


def gate_config(**overrides):
    result = dict(pipeline_check_only=False, seeds=[42, 43, 44],
                  bootstrap_samples=2000, gate_E_improvement=.15,
                  gate_compression_degradation=.10, gate_advantage_retention=.80,
                  gate_max_positive_bias=1., gate_min_E_states=10, gate_min_E_questions=3)
    result.update(overrides)
    return result


def sanity_checks():
    return dict(question_splits_disjoint=True, verifier_reproduced=True,
                hidden_logit_alignment=True, no_logits_features=True,
                validation_selection_only=True, raw_capacity_matched=True,
                comparison_coverage_matched=True, latent_layer_fixed=True,
                candidate_embeddings_frozen=True, metric_counts_present=True)


def gate_rows(seeds=(42, 43, 44), questions=3, states_per_question=4,
              raw=5.5, latent=5.475, direct=2.):
    return [predicted(f'q{q}-s{s}', f'q{q}', method, seed, kpred)
            for seed in seeds for q in range(questions) for s in range(states_per_question)
            for method, kpred in [('raw', raw), ('latent128', latent), ('direct', direct)]]


def gate(rows, cfg=None, checks=None):
    return metrics.feasibility_gate(rows, 'raw', 'latent128', 'direct',
                                    gate_config() if cfg is None else cfg,
                                    sanity_checks() if checks is None else checks)


class QuestionMacroAndCohortTests(unittest.TestCase):
    def test_question_macro_mae_and_bias_do_not_weight_state_heavy_questions(self):
        rows = [row(f'a{i}', 'dense', accepted=0, q=[0.] * 4) for i in range(9)]
        rows.append(row('b', 'sparse', accepted=4, q=[0.] * 4))
        result = metrics.report(rows)['ALL']
        self.assertEqual(result['K_MAE_question_macro'], 2.)
        self.assertEqual(result['K_bias_question_macro'], -2.)
        self.assertAlmostEqual(result['K_bias_state_micro'], -.4)
        self.assertEqual((result['num_states'], result['num_positions'], result['num_questions']), (10, 40, 2))
        self.assertAlmostEqual(result['survival_brier'], .1)

    def test_exact_seven_cohorts_memberships_and_counts(self):
        rows = [row('root', 'q0'),
                row('gain', 'q1', action='R', accepted=3, parent_length=4, parent_K=1),
                row('loss', 'q1', action='R', accepted=1, parent_length=4, parent_K=3),
                row('same', 'q2', action='R', parent_length=4, parent_K=2),
                row('r_unknown', 'q2', action='R', parent_length=4),
                row('e_full', 'q3', action='E', length=6, accepted=5, parent_length=4, parent_K=4),
                row('e_blocked', 'q4', action='E', length=6, accepted=1, parent_length=4, parent_K=1),
                row('e_no_growth', 'q4', action='E', parent_length=4, parent_K=2),
                row('e_unknown', 'q5', action='E')]
        expected = dict(ALL=({r['state_id'] for r in rows}, 40, 6),
                        R_ALL=({'gain', 'loss', 'same', 'r_unknown'}, 16, 2),
                        R_GAIN=({'gain'}, 4, 1), R_LOSS=({'loss'}, 4, 1),
                        E_ALL=({'e_full', 'e_blocked', 'e_no_growth', 'e_unknown'}, 20, 3),
                        E_PARENT_FULL_PREFIX=({'e_full'}, 6, 1),
                        E_NEW_BLOCK=({'e_full', 'e_blocked'}, 4, 2))
        result = metrics.report(rows)
        self.assertEqual(set(result), set(expected))
        self.assertEqual(tuple(result), metrics.COHORTS)
        for name, (states, positions, questions) in expected.items():
            with self.subTest(cohort=name):
                self.assertEqual({r['state_id'] for r in metrics.cohort(rows, name)}, states)
                self.assertEqual(result[name]['num_states'], len(states))
                self.assertEqual(result[name]['num_positions'], positions)
                self.assertEqual(result[name]['num_questions'], questions)
                self.assertEqual(result[name]['survival_counts']['total'], positions)
                self.assertEqual(sum(b['count'] for b in result[name]['calibration']), positions)
        with self.assertRaises(ValueError):
            metrics.cohort(rows, 'E_CONDITIONAL_SURVIVAL')

    def test_new_block_uses_global_survival_when_parent_was_rejected(self):
        r = row('blocked', action='E', length=5, accepted=1, parent_length=3,
                parent_K=1, q=[.8, .6, .4, .3, .2])
        # A stale locally-reset label must not substitute for K_true.
        r['survival_true'] = [1, 0, 0, 1, 1]
        result = metrics.report([r])['E_NEW_BLOCK']
        self.assertAlmostEqual(result['survival_brier'], (.3 ** 2 + .2 ** 2) / 2)
        self.assertIsNone(result['survival_auc'])
        for key in ('K_MAE_question_macro', 'K_bias_question_macro', 'K_bias_state_micro'):
            self.assertIsNone(result[key])
        self.assertEqual((result['num_states'], result['num_positions'], result['num_questions']), (1, 2, 1))

    def test_new_block_keeps_global_index_and_original_probabilities(self):
        r = row('entered', action='E', length=5, accepted=4, parent_length=3,
                parent_K=3, q=[.9, .8, .7, .6, .2])
        result = metrics.report([r])['E_NEW_BLOCK']
        self.assertAlmostEqual(result['survival_brier'], ((1 - .6) ** 2 + .2 ** 2) / 2)
        self.assertEqual(result['survival_auc'], 1.)
        all_result = metrics.report([r])['ALL']
        self.assertAlmostEqual(all_result['K_MAE_question_macro'], 4 - sum(r['q_pred']))

    def test_single_class_empty_and_perfect_survival(self):
        for accepted in (0, 4):
            result = metrics.report([row(accepted=accepted)])['ALL']
            self.assertEqual(result['survival_brier'], 0.)
            self.assertEqual(result['K_MAE_question_macro'], 0.)
            self.assertIsNone(result['survival_auc'])
        perfect = metrics.report([row()])['ALL']
        self.assertEqual(perfect['survival_auc'], 1.)
        for result in metrics.report([]).values():
            self.assertEqual(result['status'], 'empty')
            self.assertEqual((result['num_states'], result['num_positions'], result['num_questions']), (0, 0, 0))
            self.assertIsNone(result['K_MAE_question_macro'])
            self.assertIsNone(result['survival_brier'])
            self.assertIsNone(result['survival_auc'])

    def test_nonmonotonic_q_is_reported_without_projection(self):
        r = row(q=[.2, .8, .3, .9])
        result = metrics.report([r])['ALL']
        self.assertEqual(result['monotonicity_violations'], 2)
        self.assertAlmostEqual(result['survival_brier'], (.8 ** 2 + .2 ** 2 + .3 ** 2 + .9 ** 2) / 4)
        self.assertAlmostEqual(result['K_MAE_question_macro'], .2)
        self.assertEqual(r['q_pred'], [.2, .8, .3, .9])

    def test_reports_are_grouped_by_method_and_seed_not_pooled(self):
        rows = [row(method='A', seed=42), row(method='A', seed=43, q=[0.] * 4),
                row(method='B', seed=42, q=[1.] * 4), row(method='B', seed=43)]
        result = metrics.per_method_report(rows)
        self.assertEqual(set(result), {'A', 'B'})
        self.assertEqual(set(result['A']), {'42', '43'})
        self.assertEqual(result['A']['42']['ALL']['K_MAE_question_macro'], 0.)
        summary = metrics.seed_summary(result, 'A')
        self.assertEqual(summary['seed_count'], 2)
        self.assertEqual(summary['mean'], 1.)
        self.assertEqual(summary['std'], 1.)
        self.assertIsNone(metrics.seed_summary(result, 'missing')['mean'])
        json.dumps(result, allow_nan=False)


class PredictionIntegrityTests(unittest.TestCase):
    def test_duplicate_method_seed_state_is_rejected_even_if_identical(self):
        r = row()
        for function in (metrics.validate_records, metrics.report, metrics.per_method_report):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function([r, deepcopy(r)])

    def test_state_id_is_separate_from_seed_and_method(self):
        rows = [row(seed=42, method='A'), row(seed=43, method='A'),
                row(seed=42, method='B'), row(seed=43, method='B')]
        metrics.validate_records(rows)
        result = metrics.paired_bootstrap(rows, 'A', 'B', samples=20)
        self.assertEqual(result['matched_seed_state_count'], 2)
        self.assertEqual(result['num_states'], 1)
        self.assertEqual(result['num_questions'], 1)
        self.assertEqual(result['delta_MAE'], 0.)

    def test_invalid_lengths_accepted_vectors_and_expected_k_are_rejected(self):
        changes = [dict(proposal_length=0), dict(proposal_length=65), dict(proposal_length=True),
                   dict(K_true=-1), dict(K_true=5), dict(K_true=2.), dict(K_true=True),
                   dict(q_pred=[.5]), dict(q_pred=[-.1] * 4), dict(q_pred=[1.1] * 4),
                   dict(q_pred=[float('nan')] * 4), dict(q_pred=[float('inf')] * 4),
                   dict(K_pred=float('nan')), dict(K_pred=float('inf')), dict(K_pred=2.1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                metrics.validate_records([dict(row(), **change)])

    def test_prediction_k_sum_tolerance_and_extreme_valid_lengths(self):
        metrics.validate_records([dict(row(), K_pred=2.00001), row('short', length=1, accepted=0),
                                  row('long', length=64, accepted=64)])
        with self.assertRaises(ValueError):
            metrics.validate_records([dict(row(), K_pred=2.001)])

    def test_method_report_rejects_unmatched_seed_and_state_coverage(self):
        for rows in ([row(method='A'), row(method='B', seed=43)],
                     [row(method='A'), row(method='B'), row('extra', method='B')]):
            with self.subTest(keys=[(r['method'], r['seed'], r['state_id']) for r in rows]), self.assertRaises(ValueError):
                metrics.per_method_report(rows)


class WholeQuestionBootstrapTests(unittest.TestCase):
    def fixture(self):
        # Question dense has three states; sparse has one. Across seeds the
        # sparse predictions straddle the truth, so ensembling would erase MAE.
        rows = []
        for seed in (42, 43):
            for index in range(3):
                rows.extend([row(f'd{index}', 'dense', 'A', seed, q=[.5] * 4),
                             row(f'd{index}', 'dense', 'B', seed, q=[.25] * 4)])
            rows.extend([row('s', 'sparse', 'A', seed, q=[0.] * 4 if seed == 42 else [1.] * 4),
                         row('s', 'sparse', 'B', seed, q=[.5] * 4)])
        return rows

    def test_averages_seed_errors_then_states_then_questions_not_ensemble(self):
        result = metrics.paired_bootstrap(self.fixture(), 'A', 'B')
        self.assertEqual(result['bootstrap_unit'], 'question')
        self.assertEqual(result['samples'], 2000)
        self.assertEqual(result['num_questions'], 2)
        self.assertEqual(result['num_states'], 4)
        self.assertEqual(result['matched_seed_state_count'], 8)
        self.assertEqual(result['question_units'], [dict(question='dense', num_states=3, mae_a=0., mae_b=1.),
                                                  dict(question='sparse', num_states=1, mae_a=2., mae_b=0.)])
        self.assertEqual(result['mae_a'], 1.)
        self.assertEqual(result['mae_b'], .5)
        self.assertEqual(result['delta_MAE'], .5)
        self.assertEqual(result['ci95'], [-1., 2.])

    def test_every_bootstrap_draw_samples_question_deltas_only(self):
        population_seen = []
        original_choice = random.Random.choice

        def observed_choice(rng, population):
            population_seen.append(tuple(population))
            return original_choice(rng, population)

        with patch.object(metrics.random.Random, 'choice', observed_choice):
            metrics.paired_bootstrap(self.fixture(), 'A', 'B', samples=17)
        self.assertEqual(len(population_seen), 17 * 2)
        self.assertEqual(set(population_seen), {(-1., 2.)})

    def test_reversing_order_and_replicating_seeds_do_not_change_question_uncertainty(self):
        rows = self.fixture()
        first = metrics.paired_bootstrap(rows, 'A', 'B', samples=200, seed=19)
        self.assertEqual(first, metrics.paired_bootstrap(list(reversed(rows)), 'A', 'B', samples=200, seed=19))
        extra = [dict(r, seed=r['seed'] + 100) for r in rows]
        duplicated = metrics.paired_bootstrap(rows + extra, 'A', 'B', samples=200, seed=19)
        for key in ('num_states', 'num_questions', 'mae_a', 'mae_b', 'delta_MAE', 'ci95', 'question_units'):
            self.assertEqual(first[key], duplicated[key])
        self.assertEqual(duplicated['matched_seed_state_count'], 16)

    def test_same_method_and_swapped_direction(self):
        rows = self.fixture()
        same = metrics.paired_bootstrap(rows, 'A', 'A', samples=100)
        self.assertEqual(same['delta_MAE'], 0.)
        self.assertEqual(same['ci95'], [0., 0.])
        ab = metrics.paired_bootstrap(rows, 'A', 'B', samples=2000)
        ba = metrics.paired_bootstrap(rows, 'B', 'A', samples=2000)
        self.assertEqual(ab['delta_MAE'], -ba['delta_MAE'])
        self.assertEqual(ab['ci95'], [-ba['ci95'][1], -ba['ci95'][0]])

    def test_empty_and_single_question(self):
        empty = metrics.paired_bootstrap([], 'A', 'B', samples=20)
        self.assertEqual(empty['status'], 'empty')
        self.assertEqual(empty['num_questions'], 0)
        self.assertIsNone(empty['delta_MAE'])
        self.assertIsNone(empty['ci95'])
        one = [r for r in self.fixture() if r['question_id'] == 'dense']
        result = metrics.paired_bootstrap(one, 'A', 'B', samples=20)
        self.assertEqual(result['num_questions'], 1)
        self.assertEqual(result['ci95'], [-1., -1.])

    def test_seed_pairing_cannot_substitute_one_seed_for_another(self):
        rows = [row(method='A', seed=42), row(method='B', seed=43)]
        with self.assertRaises(ValueError):
            metrics.paired_bootstrap(rows, 'A', 'B', samples=20)

    def test_missing_or_extra_state_or_seed_coverage_is_rejected(self):
        rows = self.fixture()
        changes = [rows[:-1], rows + [row('extra', 'extra', method='A')],
                   [r for r in rows if not (r['method'] == 'B' and r['seed'] == 43)],
                   [r for r in rows if r['method'] != 'B']]
        for index, changed in enumerate(changes):
            with self.subTest(case=index), self.assertRaises(ValueError):
                metrics.paired_bootstrap(changed, 'A', 'B', samples=20)

    def test_real_state_metadata_disagreements_are_rejected(self):
        left = row(action='E', length=6, accepted=4, parent_length=3, parent_K=3)
        changes = [dict(question_id='other'), dict(K_true=3),
                   dict(proposal_length=5, q_pred=[1., 1., 1., 1., 0.]),
                   dict(parent_length=2), dict(parent_K=2), dict(action='R'), dict(split='test')]
        for change in changes:
            right = dict(deepcopy(left), method='B', **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                metrics.paired_bootstrap([left, right], 'A', 'B', samples=20)

    def test_cohort_filter_cannot_hide_mismatched_action_or_parent_metadata(self):
        left = row(action='E', length=6, accepted=4, parent_length=3, parent_K=3)
        for change in (dict(action='R'), dict(parent_K=2)):
            right = dict(deepcopy(left), method='B', **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                metrics.paired_bootstrap([left, right], 'A', 'B', 'E_PARENT_FULL_PREFIX', samples=20)

    def test_cohort_filter_cannot_hide_unmatched_states_outside_selected_cohort(self):
        left = row(action='E', length=6, accepted=4, parent_length=3, parent_K=3)
        right = dict(deepcopy(left), method='B')
        outside = row('unmatched-R', action='R', parent_length=4, parent_K=2)
        with self.assertRaises(ValueError):
            metrics.paired_bootstrap([left, right, outside], 'A', 'B', 'E_PARENT_FULL_PREFIX', samples=20)


class ValidationSelectionAndCompressionTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for seed in (42, 43, 44):
            # Equal ALL/E weighting should favor depth 14 despite worse ALL.
            for method, root_prediction, e_prediction in [('L7', 6., 2.), ('L14', 2., 5.), ('L21', 2., 2.)]:
                rows.extend([predicted('root', 'qroot', method, seed, root_prediction, action='root',
                                       parent_length=None, parent_K=None),
                             predicted('extend', 'qe', method, seed, e_prediction)])
        return rows

    def test_top_two_use_equal_all_e_macro_scores_across_seeds(self):
        result = metrics.select_layers(self.fixture(), [7, 14, 21], {7: 'L7', 14: 'L14', 21: 'L21'})
        self.assertEqual(result['selection_split'], 'val')
        self.assertEqual(result['selected'], [14, 7])
        ranking = {r['depth']: r for r in result['ranking']}
        self.assertEqual(ranking[7]['ALL'], 2.)
        self.assertEqual(ranking[7]['E_PARENT_FULL_PREFIX'], 4.)
        self.assertEqual(ranking[7]['score'], 3.)
        self.assertEqual(ranking[14]['ALL'], 2.5)
        self.assertEqual(ranking[14]['E_PARENT_FULL_PREFIX'], 1.)
        self.assertEqual(ranking[14]['score'], 1.75)

    def test_absent_e_cohort_uses_all_and_ties_break_by_depth(self):
        rows = [row(method=method, seed=seed) for method in ('L7', 'L14') for seed in (42, 43, 44)]
        result = metrics.select_layers(rows, [14, 7], {7: 'L7', 14: 'L14'}, count=1)
        self.assertEqual(result['selected'], [7])
        self.assertIsNone(result['ranking'][0]['E_PARENT_FULL_PREFIX'])
        self.assertEqual(result['ranking'][0]['score'], 0.)

    def test_selection_rejects_explicit_train_test_or_mixed_splits(self):
        rows = self.fixture()
        for changed in ([dict(r, split='test') for r in rows],
                        [dict(r, split='train') for r in rows],
                        [dict(r, split='test') if i == 0 else r for i, r in enumerate(rows)]):
            with self.subTest(splits={r['split'] for r in changed}), self.assertRaises(ValueError):
                metrics.select_layers(changed, [7, 14, 21], {7: 'L7', 14: 'L14', 21: 'L21'})

    def test_missing_layer_or_mismatched_layer_coverage_is_rejected(self):
        rows = self.fixture()
        for changed in ([r for r in rows if r['method'] != 'L21'], rows[:-1]):
            with self.assertRaises(ValueError):
                metrics.select_layers(changed, [7, 14, 21], {7: 'L7', 14: 'L14', 21: 'L21'})

    def test_compression_fraction_and_undefined_zero_denominator(self):
        reports = metrics.per_method_report(gate_rows(raw=5., latent=4.9))
        comparison = metrics.compression_comparison(reports, 'latent128', 'raw')
        for value in comparison.values():
            self.assertAlmostEqual(value['degradation_fraction'], .1)
            self.assertEqual(value['raw']['seed_count'], 3)
        perfect_raw = metrics.per_method_report(gate_rows(raw=6.))
        for value in metrics.compression_comparison(perfect_raw, 'latent128', 'raw').values():
            self.assertIsNone(value['degradation_fraction'])


class FeasibilityGateTests(unittest.TestCase):
    def test_fabricated_valid_pass_requires_every_condition_and_three_seeds(self):
        result = gate(gate_rows())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['selection_split'], 'val')
        self.assertTrue(all(result['conditions'].values()), result['conditions'])
        self.assertAlmostEqual(result['E_improvement_fraction'], (4. - .525) / 4.)
        self.assertAlmostEqual(result['retained_advantage_fraction'], (4. - .525) / (4. - .5))
        self.assertLess(result['native_vs_direct_ALL']['ci95'][1], 0.)
        self.assertLess(result['latent_vs_direct_E']['ci95'][1], 0.)
        json.dumps(result, allow_nan=False)

    def test_pipeline_only_never_becomes_pass_even_with_perfect_gate_evidence(self):
        result = gate(gate_rows(), gate_config(pipeline_check_only=True))
        self.assertEqual(result['status'], 'PIPELINE_ONLY')
        self.assertTrue(all(result['conditions'].values()))

    def test_low_state_or_question_coverage_is_inconclusive(self):
        for rows in (gate_rows(questions=3, states_per_question=1),
                     gate_rows(questions=2, states_per_question=6)):
            with self.subTest(states=len(rows)):
                result = gate(rows)
                self.assertEqual(result['status'], 'INCONCLUSIVE')
                self.assertFalse(result['conditions']['enough_E_full_prefix'])

    def test_coverage_minimum_is_required_for_each_seed(self):
        rows = [r for r in gate_rows() if r['seed'] != 44 or r['question_id'] == 'q0']
        result = gate(rows)
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertFalse(result['conditions']['enough_E_full_prefix'])

    def test_fewer_than_three_seeds_cannot_pass(self):
        for seeds in ((42,), (42, 43)):
            with self.subTest(seeds=seeds):
                result = gate(gate_rows(seeds=seeds))
                self.assertNotEqual(result['status'], 'PASS')
                self.assertFalse(result['conditions']['three_seed_consistency'])

    def test_one_inconsistent_seed_blocks_pass_despite_good_mean(self):
        rows = gate_rows()
        for r in rows:
            if r['method'] == 'latent128' and r['seed'] == 44:
                r['q_pred'] = [.25] * 8
                r['K_pred'] = 2.
        result = gate(rows)
        self.assertNotEqual(result['status'], 'PASS')
        self.assertFalse(result['conditions']['three_seed_consistency'])

    def test_failed_sanity_check_is_invalid(self):
        checks = sanity_checks()
        checks['hidden_logit_alignment'] = False
        result = gate(gate_rows(), checks=checks)
        self.assertEqual(result['status'], 'INVALID')
        self.assertFalse(result['conditions']['required_sanity_checks'])

    def test_bad_compression_or_positive_bias_prevents_pass(self):
        compression = gate(gate_rows(latent=5.3))
        self.assertNotEqual(compression['status'], 'PASS')
        self.assertFalse(compression['conditions']['compression_within_10pct'])
        biased = gate(gate_rows(raw=7.25, latent=7.3, direct=2.))
        self.assertNotEqual(biased['status'], 'PASS')
        self.assertFalse(biased['conditions']['no_catastrophic_positive_bias'])

    def test_native_tie_and_tiny_latent_advantage_are_not_success(self):
        native_tie = gate(gate_rows(raw=2.))
        self.assertNotEqual(native_tie['status'], 'PASS')
        self.assertFalse(native_tie['conditions']['native_beats_direct_ALL_CI'])
        tiny = gate(gate_rows(raw=2.1, latent=2.1))
        self.assertNotEqual(tiny['status'], 'PASS')
        self.assertFalse(tiny['conditions']['latent_improves_E_15pct'])

    def test_gate_rejects_test_rows_and_unmatched_method_coverage(self):
        for rows in ([dict(r, split='test') for r in gate_rows()], gate_rows()[:-1]):
            with self.subTest(case=rows[0]['split']), self.assertRaises(ValueError):
                gate(rows)


if __name__ == '__main__':
    unittest.main()
