import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structured_sparse_collector import (
    MASK_ID, CachedVerifier, ExplicitDrafter, FullContextVerifier, KVExplicitDrafter,
    GraphCollector, annotate_verifier_acceptance,
    collect, commit_one, hash_observation, identity, select_anchors,
)


class FakeEngine:
    def __init__(self, bad=False):
        self.bad = bad

    def observe(self, prefix, native):
        predictions = [i % 10 + 1 for i in range(len(native))]
        if self.bad:
            predictions[0] = 99
        filled = [predictions[i] if x == MASK_ID else x for i, x in enumerate(native)]
        return dict(predictions=predictions, probabilities=[0.2] * len(native),
            filled=filled, filled_probabilities=[0.2] * len(native),
            hidden_states=[[[layer, token % 100] for token in native] for layer in range(5)],
            hidden_layer_indices=[0, 1, 2, 3, 4], topk_token_ids=[[x] for x in predictions],
            topk_logits=[[1.0] for x in predictions], forward_ms=1.5,
            input_state_hash=identity(prefix, native))


def args_for(path):
    return SimpleNamespace(output_dir=Path(path), dataset="gsm8k", shard_rows=4,
        max_proposal_tokens=64, extend_size=8, small_block_size=8,
        max_refinement_steps=3, branch_width=2, min_expand_acceptance_ratio=0.5,
        bad_probe_branches=1, bad_refinement_steps=2, drafter_threshold=0.3)


class StructuredGraphTests(unittest.TestCase):
    def test_verifier_mismatch_is_retained_and_recomputed_label_is_authoritative(self):
        source = dict(state_id="anchor", accepted_len=6, proposal_token_ids_after_fill=[1] * 8)
        checked = annotate_verifier_acceptance(source, 4)
        self.assertEqual(checked["accepted_len"], 4)
        self.assertEqual(checked["backbone_recorded_accepted_len"], 6)
        self.assertEqual(checked["verifier_recomputed_accepted_len"], 4)
        self.assertFalse(checked["verifier_acceptance_matches_backbone"])
        self.assertEqual(source["accepted_len"], 6)

    def test_observation_hash_tracks_state_and_ignores_timing_noise(self):
        observation = FakeEngine().observe([2, 3], [MASK_ID] * 8)
        first = hash_observation([2, 3], [MASK_ID] * 8, observation)
        changed_timing = dict(observation, forward_ms=999.0)
        self.assertEqual(first, hash_observation([2, 3], [MASK_ID] * 8, changed_timing))
        self.assertNotEqual(first, hash_observation([2, 4], [MASK_ID] * 8, observation))

    def run_graph(self, folder, bad=False):
        args = args_for(folder)
        graph = GraphCollector(args, FakeEngine(bad), eos_id=100)
        root = dict(state_id="root", problem_id=0, round_id=4,
                    prefix_token_ids=[2, 3, 4], proposal_token_ids_before_fill=[MASK_ID] * 8,
                    accepted_len=0 if bad else 8)
        graph.run_anchor(root, [i % 10 + 1 for i in range(65)],
                         {n: 2.0 for n in range(8, 65, 8)}, "reference")
        graph.finish()
        return graph

    def test_real_graph_scheduler_and_serialized_raw_states(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = self.run_graph(folder)
            nodes = {n['state_id']: n for n in graph.nodes}
            self.assertEqual(len(nodes), len(graph.nodes))
            self.assertEqual(len(graph.edges), len(nodes) - 1)
            self.assertEqual(max(n['proposal_length'] for n in nodes.values()), 64)
            selected = {}
            for node in nodes.values():
                selected[node['proposal_length']] = selected.get(node['proposal_length'], 0) + int(node['selected_for_expansion'])
                self.assertEqual(node['feature_merge_mode'], 'none')
                self.assertEqual(node['masks_remaining'] + node['committed_tokens'], node['proposal_length'])
                self.assertEqual(node['state_masks_resolved'], node['masks_remaining'] == 0)
                self.assertEqual(node['current_submit_regime'],
                                 'full' if node['submit_accepted_len'] >= node['proposal_length'] else
                                 'near_full' if node['submit_accepted_len'] >= node['proposal_length'] - 2 else
                                 'early_mismatch' if node['submit_accepted_len'] <= 1 else 'mid')
                self.assertEqual(len(node['observation_hash']), 64)
                self.assertEqual(node['verifier_calibration_key'], node['reference_key'])
            self.assertLessEqual(max(selected.values()), 2)
            adjacency = {}
            for e in graph.edges:
                src, dst = nodes[e['src_state_id']], nodes[e['dst_state_id']]
                adjacency.setdefault(e['src_state_id'], set()).add(e['action'])
                self.assertEqual(e['native_unmask_forwards'], 1)
                self.assertAlmostEqual(e['action_cost_ms'], 1.5)
                self.assertAlmostEqual(dst['draft_latency_from_anchor_ms'] - src['draft_latency_from_anchor_ms'], 1.5)
                if e['action'] == 'R':
                    self.assertEqual(src['proposal_length'], dst['proposal_length'])
                    self.assertEqual(dst['refine_steps_since_extend'], src['refine_steps_since_extend'] + 1)
                else:
                    self.assertEqual(dst['proposal_length'] - src['proposal_length'], 8)
                    self.assertEqual(dst['refine_steps_since_extend'], 0)
            self.assertTrue(any(actions == {'R', 'E'} for actions in adjacency.values()))
            # Exercise the real NPZ writer, including mixed proposal lengths.
            raw = Path(folder) / 'raw/gsm8k_structured'
            for shard in raw.glob('*.npz'):
                with np.load(shard) as a:
                    for n in nodes.values():
                        if n['shard'] != shard.name:
                            continue
                        row, length = n['row'], n['proposal_length']
                        native = a['proposal_token_ids_before_fill'][row, :length].tolist()
                        self.assertEqual(a['hidden_states'][row, 0, :length, 1].tolist(),
                                         [x % 100 for x in native])
                        self.assertEqual(n['hidden_state_input_hash'], identity([2, 3, 4], native))
            records = [json.loads(x) for x in (Path(folder)/'nodes.jsonl').read_text().splitlines()]
            self.assertEqual(sum(n['selected_for_expansion'] for n in records), sum(selected.values()))

    def test_bad_children_stored_but_not_extended_deeply(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = self.run_graph(folder, bad=True)
            self.assertEqual(max(n['proposal_length'] for n in graph.nodes), 16)
            self.assertFalse(any(n['selected_for_expansion'] for n in graph.nodes))
            self.assertEqual(sum(e['action'] == 'E' for e in graph.edges), 4)
            probe_nodes = [n for n in graph.nodes if n['proposal_length'] == 16]
            self.assertEqual(len(probe_nodes), 6)  # four E children, one ER/ERR probe

    def test_deterministic_selection(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            x, y = self.run_graph(a), self.run_graph(b)
            self.assertEqual([n['state_id'] for n in x.nodes], [n['state_id'] for n in y.nodes])

    def test_stratified_anchors_not_only_first_round(self):
        rows = [dict(problem_id=7, round_id=i, state_id=str(i), accepted_len=v,
                     proposal_length=8, boundary_index=1) for i, v in enumerate([0, 3, 6, 8])]
        result = select_anchors(rows, 1, 4)
        self.assertEqual({r['accepted_len'] for r in result}, {0, 3, 6, 8})
        self.assertEqual(result[0]['accepted_len'], 8)

    def test_commit_preserves_parent_and_forces_only_when_needed(self):
        native = [5, MASK_ID, MASK_ID]
        updated, chosen = commit_one(native, [9, 8, 7], [0.9, 0.2, 0.1], 8, 0.3)
        self.assertEqual(chosen, [1])
        self.assertEqual(updated, [5, 8, MASK_ID])
        self.assertEqual(native, [5, MASK_ID, MASK_ID])

    def test_half_threshold_keeps_lower_confidence_masks_closed(self):
        native = [MASK_ID, MASK_ID]
        predictions = [5, 6]
        probabilities = [0.4, 0.35]
        _, selected_old = commit_one(native, predictions, probabilities, 8, 0.3)
        updated, selected_new = commit_one(native, predictions, probabilities, 8, 0.5)
        self.assertEqual(selected_old, [0, 1])
        self.assertEqual(selected_new, [0])  # forced highest-confidence position
        self.assertEqual(updated, [5, MASK_ID])

    def test_unmask_frame_uses_absolute_context_position(self):
        native = [MASK_ID] * 8
        updated, chosen = commit_one(native, list(range(1, 9)), [0.6] * 8,
                                     8, 0.5, prefix_length=5)
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(updated[:3], [1, 2, 3])
        self.assertEqual(updated[3:], [MASK_ID] * 5)

    def test_extend_top1_fills_parent_before_new_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = self.run_graph(folder)
            nodes = {n['state_id']: n for n in graph.nodes}
            root = next(n for n in graph.nodes if n['parent_state_id'] is None)
            edge = next(e for e in graph.edges if e['src_state_id'] == root['state_id']
                        and e['action'] == 'E')
            child = nodes[edge['dst_state_id']]
            self.assertEqual(edge['pre_extension_top1_filled_positions'], list(range(8)))
            self.assertEqual(edge['pre_extension_top1_filled_token_ids'], list(range(1, 9)))
            with np.load(Path(folder) / 'raw/gsm8k_structured' / child['shard']) as data:
                native = data['proposal_token_ids_before_fill'][child['row'], :16].tolist()
            self.assertEqual(native[:8], list(range(1, 9)))
            self.assertNotIn(MASK_ID, native[:8])
            self.assertGreater(sum(x == MASK_ID for x in native[8:]), 0)


class PositionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(1, 1)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        length = input_ids.shape[1]
        positions = torch.arange(length)
        logits = torch.full((1, length, 256), -10.0)
        logits[0, positions, positions % 256] = 10
        if 'logits_to_keep' in kwargs:
            logits = logits[:, kwargs['logits_to_keep']]
        hidden = tuple(positions.reshape(1, length, 1).float() + i * 1000 for i in range(5))
        return SimpleNamespace(logits=logits, hidden_states=hidden)


class AlignmentTests(unittest.TestCase):
    def test_exact_positions_across_physical_boundary(self):
        engine = ExplicitDrafter(PositionalModel(), SimpleNamespace(physical_block_size=32, raw_top_k=32))
        for prefix_length in [7, 8, 32, 135]:
            for length in [8, 24, 32, 56, 64]:
                with self.subTest(prefix=prefix_length, length=length):
                    obs = engine.observe([1] * prefix_length, [MASK_ID] * length)
                    self.assertEqual(obs['predictions'], [(i-1) % 256 for i in range(prefix_length, prefix_length + length)])
                    self.assertEqual([v[0] for v in obs['hidden_states'][0]], list(range(prefix_length, prefix_length + length)))
                    self.assertEqual(len(obs['topk_logits']), length)

    def test_stable_prefix_kv_matches_full_observation_and_reuses_prefix(self):
        class CacheAwarePositionalModel(PositionalModel):
            def __init__(self):
                super().__init__()
                self.prefills = 0

            def forward(self, input_ids, past_key_values=None, update_past_key_values=False,
                        logits_to_keep=0, **kwargs):
                offset = 0 if past_key_values is None else past_key_values['length']
                if update_past_key_values:
                    self.prefills += 1
                positions = torch.arange(offset, offset + input_ids.shape[1])
                logits = torch.full((1, len(positions), 256), -10.0)
                logits[0, torch.arange(len(positions)), positions % 256] = 10
                if isinstance(logits_to_keep, torch.Tensor):
                    logits = logits[:, logits_to_keep]
                hidden = tuple(positions.reshape(1, -1, 1).float() + i * 1000
                               for i in range(5))
                cache = {'length': input_ids.shape[1]} if past_key_values is None else past_key_values
                return SimpleNamespace(logits=logits, hidden_states=hidden,
                                       past_key_values=cache)

        model = CacheAwarePositionalModel()
        args = SimpleNamespace(physical_block_size=32, raw_top_k=16)
        full = ExplicitDrafter(model, args)
        cached = KVExplicitDrafter(model, args)
        prefix = [1] * 41
        states = [[MASK_ID] * 8, [7] * 8 + [MASK_ID] * 8,
                  [7] * 24 + [MASK_ID] * 8]
        for native in states:
            expected = full.observe(prefix, native)
            actual = cached.observe(prefix, native)
            for key in ('hidden_states', 'filled'):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(actual['predictions'], [
                (p if p % 32 == 0 else p - 1) % 256
                for p in range(len(prefix), len(prefix) + len(native))])
            self.assertEqual(actual['kv_cached_prefix_len'] % 32, 0)
        cached.observe(prefix, states[-1])
        self.assertTrue(cached.observe(prefix, states[-1])['kv_cache_hit'])
        self.assertGreater(model.prefills, 0)


class MutatingTarget(PositionalModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_commit_hash='test-revision')
        self.dtype = torch.float32
        self.generate_calls = 0
        self.cached_inputs = []

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.generate_calls += 1
        return torch.cat([input_ids, torch.ones((1, max_new_tokens), dtype=torch.long)], dim=1)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None:
            past_key_values = {'length': input_ids.shape[1]}
        else:
            self.cached_inputs.append((past_key_values['length'], input_ids.shape[1]))
            past_key_values['length'] += input_ids.shape[1]
        keep = kwargs.get('logits_to_keep', input_ids.shape[1])
        length = input_ids.shape[1] if not isinstance(keep, int) else min(keep, input_ids.shape[1])
        return SimpleNamespace(past_key_values=past_key_values,
                               logits=torch.ones((1, length, 32)))


class CalibrationTests(unittest.TestCase):
    def test_full_context_verifier_matches_greedy_acceptance_without_kv(self):
        class DirectTarget(MutatingTarget):
            def forward(self, input_ids, use_cache=None, logits_to_keep=None, **kwargs):
                self.cached_inputs.append((use_cache, input_ids.shape[1]))
                logits = torch.full((1, logits_to_keep, 32), -10.0)
                logits[:, :, 1] = 10.0
                logits[:, -1, 2] = 20.0
                return SimpleNamespace(logits=logits)

        model = DirectTarget()
        args = SimpleNamespace(target_model_name='test')
        oracle = FullContextVerifier(model, SimpleNamespace(eos_token_id=31), args)
        _, _, key = oracle.prepare([4, 5])
        accepted, emitted, ids, latency = oracle.score([4, 5], [1, 1, 1], 20)
        self.assertEqual((accepted, emitted, ids), (3, 4, [1, 1, 1, 2]))
        accepted_bad, emitted_bad, ids_bad, _ = oracle.score([4, 5], [1, 7, 1], 20)
        self.assertEqual((accepted_bad, emitted_bad, ids_bad), (1, 2, [1, 1]))
        self.assertEqual(model.cached_inputs, [(False, 5), (False, 5)])
        self.assertGreaterEqual(latency, 0)
        self.assertEqual(len(key), 64)

    def test_cache_isolation_and_reference_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder), target_model_name='test',
                                   max_proposal_tokens=16, extend_size=8, calibration_repeats=2)
            model = MutatingTarget()
            tokenizer = SimpleNamespace(eos_token_id=31, init_kwargs={})
            oracle = CachedVerifier(model, tokenizer, args)
            ref, timing, key = oracle.prepare([1, 2, 3, 4])
            self.assertEqual(len(ref), 17)
            self.assertEqual(set(timing), {8, 16})
            self.assertTrue(all(prefix_length == 3 for prefix_length, _ in model.cached_inputs))
            self.assertEqual({q for _, q in model.cached_inputs}, {9, 17})
            self.assertTrue(all(r['verifier_calibration_key'] == key for r in oracle.records))
            self.assertTrue(all(r['context_hash'] == identity([1, 2, 3, 4]) for r in oracle.records))
            oracle.prepare([1, 2, 3, 4])
            self.assertEqual(model.generate_calls, 1)
            self.assertTrue(all(prefix_length == 3 for prefix_length, _ in model.cached_inputs))


class PackagingTests(unittest.TestCase):
    def test_direct_no_kv_verifier_is_recorded_in_v5_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            args = args_for(Path(folder) / 'output')
            args.max_proposal_tokens = 16
            args.max_unmask_passes = None
            args.remaining_output_budget = None
            args.physical_block_size = 32
            args.calibration_repeats = 2
            args.seed = 42
            args.backbone_zip = Path(folder) / 'unused.zip'
            args.target_model_name = 'fake-target'
            args.target_gpu_memory_gib = 9
            args.num_questions = 1
            args.anchors_per_question = 1
            args.max_rounds_per_question = 0
            args.verifier_mode = 'full_context_no_kv'
            args.drafter_kv_mode = 'stable_block_prefix'
            root = dict(state_id='root', problem_id=0, round_id=0,
                        prefix_token_ids=[2, 3, 4],
                        proposal_token_ids_before_fill=[MASK_ID] * 8,
                        proposal_token_ids_after_fill=[1] * 8, accepted_len=8)
            tokenizer = SimpleNamespace(eos_token_id=31, init_kwargs={})
            with patch('structured_sparse_collector.load_anchors', return_value=[root]), \
                 patch('structured_sparse_collector._load_models',
                       return_value=(tokenizer, MutatingTarget(), None)), \
                 patch('structured_sparse_collector.KVExplicitDrafter', return_value=FakeEngine()):
                collect(args)
            with zipfile.ZipFile(args.output_dir / 'gsm8k_structured_graph.zip') as archive:
                manifest = json.loads(archive.read('graph_manifest.json'))
                nodes = [json.loads(x) for x in archive.read('nodes.jsonl').splitlines()]
                measures = [json.loads(x) for x in archive.read('verifier_calibration.jsonl').splitlines()]
            self.assertEqual(manifest['schema_version'], 'structured_sparse_sre_v5')
            self.assertEqual(manifest['verifier_mode'], 'full_context_no_kv')
            self.assertTrue(all(n['submit_latency_is_node_measurement'] for n in nodes))
            self.assertTrue(all(n['submit_label_source'] == 'direct_full_context_no_kv_greedy'
                                for n in nodes))
            self.assertEqual(len(measures), len(nodes) + 1)  # archived anchor check
            self.assertEqual(len({n['submit_verifier_measurement_id'] for n in nodes}), len(nodes))
            for node in nodes:
                measurement = measures[node['submit_verifier_measurement_id']]
                self.assertEqual(measurement['proposal_hash'], node['submit_verifier_measurement_key'])
                self.assertEqual(measurement['latency_ms'], node['submit_verifier_latency_ms'])

    def test_complete_and_partial_archives(self):
        class BrokenEngine(FakeEngine):
            calls = 0
            def observe(self, prefix, native):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError('simulated interrupted forward')
                return super().observe(prefix, native)
        for broken in [False, True]:
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as folder:
                args = args_for(Path(folder) / 'output')
                args.max_proposal_tokens = 16
                args.max_unmask_passes = None
                args.remaining_output_budget = None
                args.physical_block_size = 32
                args.calibration_repeats = 2
                args.seed = 42
                args.backbone_zip = Path(folder) / 'unused.zip'
                args.target_model_name = 'fake-target'
                args.target_gpu_memory_gib = 9
                args.num_questions = 1
                args.anchors_per_question = 1
                args.max_rounds_per_question = 0
                root = dict(state_id='root', problem_id=0, round_id=0,
                            prefix_token_ids=[2, 3, 4], proposal_token_ids_before_fill=[MASK_ID] * 8,
                            proposal_token_ids_after_fill=[1] * 8, accepted_len=8)
                tokenizer = SimpleNamespace(eos_token_id=31, init_kwargs={})
                with patch('structured_sparse_collector.load_anchors', return_value=[root]), \
                     patch('structured_sparse_collector._load_models', return_value=(tokenizer, MutatingTarget(), None)), \
                     patch('structured_sparse_collector.ExplicitDrafter', return_value=BrokenEngine() if broken else FakeEngine()):
                    if broken:
                        with self.assertRaisesRegex(RuntimeError, 'simulated interrupted'):
                            collect(args)
                    else:
                        collect(args)
                with zipfile.ZipFile(args.output_dir / 'gsm8k_structured_graph.zip') as archive:
                    manifest = json.loads(archive.read('graph_manifest.json'))
                    self.assertEqual(manifest['status'], 'partial' if broken else 'complete')
                    self.assertEqual(manifest['schema_version'], 'structured_sparse_sre_v5')
                    self.assertIn('nodes.jsonl', archive.namelist())
                    self.assertIn('edges.jsonl', archive.namelist())
                    self.assertGreater(manifest['nodes'], 0)


if __name__ == '__main__':
    unittest.main()
