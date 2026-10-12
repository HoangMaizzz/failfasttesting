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
from native_elysia_graph import (NativeElysiaGraphCollector, NativeElysiaRunner,
                                 NativeEosWithoutSnapshot, _validate_v2_native_snapshot)
from structured_sparse_collector import collect, identity
from sparse_extend_world_model_collector import MASK_ID


class FakeFeatureEngine:
    def observe(self, prefix, native):
        predictions = [i + 1 for i in range(len(native))]
        return dict(
            predictions=predictions, probabilities=[0.9] * len(native),
            filled=[predictions[i] if x == MASK_ID else x for i, x in enumerate(native)],
            filled_probabilities=[0.9] * len(native),
            hidden_states=[[[float(layer), float(i)] for i in range(len(native))]
                           for layer in range(5)],
            hidden_layer_indices=list(range(5)),
            topk_token_ids=[[i + 1] for i in range(len(native))],
            topk_logits=[[1.0] for _ in native],
            forward_ms=0.5, input_state_hash=identity(prefix, native),
        )


def snapshot(pass_index):
    native = list(range(1, pass_index + 1)) + [MASK_ID] * (8 - pass_index)
    return dict(
        unmask_forward_index=pass_index,
        proposal_token_ids_before_fill=native,
        proposal_token_ids_after_fill=list(range(1, 9)),
        newly_unmasked_positions=[pass_index - 1],
        masks_remaining=8 - pass_index,
        confidences=[0.9] * 8,
        draft_latency_elapsed_ms=2.0 * pass_index,
        draft_passes_elapsed=pass_index,
        hidden_state_source="native_refinement_forward_output",
        hidden_states=[[[float(layer), float(i)] for i in range(8)] for layer in range(5)],
        hidden_layer_indices=list(range(5)),
        topk_token_ids=[[i + 1] for i in range(8)],
        topk_logits=[[1.0] for _ in range(8)],
    )


def v2_snapshot(pass_index):
    row = snapshot(pass_index)
    before = np.arange(32) >= 3 + pass_index - 1
    after = np.arange(32) >= 3 + pass_index
    region = np.arange(32) < 8
    row.update(
        wm_v2_native_latent_block=np.ones((32, 4)).tolist(),
        wm_v2_proposal_start=3, wm_v2_proposal_length=8,
        wm_v2_physical_block_start=0, wm_v2_prefix_kv_length=0,
        wm_v2_prefix_kv_summary=np.zeros((4, 2, 2, 2, 128)).tolist(),
        wm_v2_prefix_kv_valid=False, wm_v2_native_refine_available=True,
        wm_v2_memory_layer_indices=[0, 7, 15, 23], wm_v2_valid_positions=[True] * 32,
        wm_v2_native_eos_ready=False,
        wm_v2_mask_before_forward=before.tolist(), wm_v2_physical_mask=after.tolist(),
        wm_v2_physical_event=(before & ~after).tolist(),
        wm_v2_proposal_span=((np.arange(32) >= 3) & (np.arange(32) < 11)).tolist(),
        wm_v2_native_active_region=region.tolist(), wm_v2_forward_region=region.tolist(),
        hidden_state_forward_pass=pass_index,
    )
    return row


class FakeNativeRunner:
    def __init__(self):
        self.prompts = []

    def segment(self, prompt):
        self.prompts.append(list(prompt))
        return [snapshot(1), snapshot(2)]


class FakeNativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.call = None

    def get_input_embeddings(self):
        return self.embedding

    def generate_draft_tokens_arbitrary_length(self, inputs, **kwargs):
        self.call = (inputs.tolist(), kwargs)
        return None, None, None, None, None, dict(
            oracle_refinement_snapshots=[snapshot(1), snapshot(2)])


class FakeVerifier:
    def __init__(self, *args):
        self.records = []

    def prepare(self, prefix):
        return None, None, identity(prefix)

    def score(self, prefix, proposal, remaining):
        self.records.append(dict(measurement_id=len(self.records),
                                 latency_ms=1.0, source='fake_direct_verifier'))
        return 0, 1, [9], 1.0


class NativeGraphTests(unittest.TestCase):
    def test_v2_adjacent_snapshots_have_exactly_one_forward_and_stable_prefix(self):
        args = SimpleNamespace(physical_block_size=32, extend_size=8)
        _validate_v2_native_snapshot(v2_snapshot(1), None, 3, args)
        _validate_v2_native_snapshot(v2_snapshot(2), v2_snapshot(1), 3, args)
        altered = v2_snapshot(2)
        altered["wm_v2_prefix_kv_length"] = 32
        with self.assertRaisesRegex(RuntimeError, "prefix KV changed"):
            _validate_v2_native_snapshot(altered, v2_snapshot(1), 3, args)
        altered = v2_snapshot(2)
        altered["wm_v2_mask_before_forward"][5] = False
        with self.assertRaisesRegex(RuntimeError, "full physical mask delta"):
            _validate_v2_native_snapshot(altered, v2_snapshot(1), 3, args)

    def test_v2_runner_rejects_missing_correction_and_extra_bridge_between_r(self):
        args = SimpleNamespace(raw_top_k=1, physical_block_size=32, small_block_size=8,
            extend_size=8, drafter_threshold=0.5, max_refinement_steps=3,
            wm_v2_capture_native_latent=True)
        model = FakeNativeModel()
        with self.assertRaisesRegex(RuntimeError, "missing corrected physical fields"):
            NativeElysiaRunner(model, SimpleNamespace(), args).segment([11, 12, 13])
        second = v2_snapshot(2)
        second["draft_passes_elapsed"] = 3
        second["hidden_state_forward_pass"] = 3
        model.generate_draft_tokens_arbitrary_length = lambda inputs, **kwargs: (
            None, None, None, None, None,
            dict(oracle_refinement_snapshots=[v2_snapshot(1), second]))
        with self.assertRaisesRegex(RuntimeError, "exactly one actual native forward"):
            NativeElysiaRunner(model, SimpleNamespace(), args).segment([11, 12, 13])

    def test_runner_calls_native_generator_and_keeps_sequential_snapshots(self):
        model = FakeNativeModel()
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=0.5,
            max_refinement_steps=3)
        runner = NativeElysiaRunner(model, SimpleNamespace(), args)
        rows = runner.segment([11, 12, 13])
        self.assertEqual(len(rows), 2)
        self.assertEqual(model.call[0], [[11, 12, 13]])
        self.assertEqual(model.call[1]['threshold'], 0.5)
        self.assertEqual(model.call[1]['max_denoising_passes'], 11)
        self.assertEqual(model.call[1]['args'].collector_max_oracle_snapshots, 4)
        self.assertNotIn('use_block_cache', model.call[1])

    def test_first_observable_boundary_may_follow_unlogged_physical_block_pass(self):
        model = FakeNativeModel()
        def with_late_first(inputs, **kwargs):
            return None, None, None, None, None, dict(
                oracle_refinement_snapshots=[snapshot(2)])
        model.generate_draft_tokens_arbitrary_length = with_late_first
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=0.5,
            max_refinement_steps=3)
        rows = NativeElysiaRunner(model, SimpleNamespace(), args).segment([11] * 29)
        self.assertEqual(rows[0]['unmask_forward_index'], 2)

    def test_missing_first_boundary_retries_with_physical_block_bridge_budget(self):
        model = FakeNativeModel()
        limits = []

        def retry_once(inputs, **kwargs):
            limits.append(kwargs['max_denoising_passes'])
            snapshots = [] if len(limits) == 1 else [snapshot(1)]
            return None, None, None, None, None, dict(
                oracle_refinement_snapshots=snapshots,
                oracle_snapshot_attempts=0 if not snapshots else 1,
            )

        model.generate_draft_tokens_arbitrary_length = retry_once
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=0.5,
            max_refinement_steps=3)
        rows = NativeElysiaRunner(model, SimpleNamespace(), args).segment([11] * 29)
        self.assertEqual(len(rows), 1)
        self.assertEqual(limits, [11, 15])

    def test_eos_before_snapshot_returns_actual_terminal_proposal(self):
        model = FakeNativeModel()
        model.generate_draft_tokens_arbitrary_length = lambda inputs, **kwargs: (
            torch.tensor([[11, 12, 13, 42, 151645]], device=inputs.device),
            8, None, 1, [1.0], dict(oracle_refinement_snapshots=[],
                native_termination_reason='eos', oracle_snapshot_attempts=0))
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=0.5,
            max_refinement_steps=3)
        tokenizer = SimpleNamespace(eos_token_id=151645)
        with self.assertRaises(NativeEosWithoutSnapshot) as raised:
            NativeElysiaRunner(model, tokenizer, args).segment([11, 12, 13], 1)
        self.assertEqual(raised.exception.candidate_token_ids, [42, 151645])

    def test_eos_without_snapshot_still_retries_when_earlier_mask_unresolved(self):
        model = FakeNativeModel()
        attempts = []
        def retry_eos(inputs, **kwargs):
            attempts.append(kwargs["max_denoising_passes"])
            rows = [] if len(attempts) == 1 else [snapshot(1)]
            return (torch.tensor([[11, 12, 13, MASK_ID, 151645]], device=inputs.device),
                    8, None, 1, [1.0], dict(oracle_refinement_snapshots=rows,
                                          native_termination_reason="eos"))
        model.generate_draft_tokens_arbitrary_length = retry_eos
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=.5, max_refinement_steps=3)
        rows = NativeElysiaRunner(model, SimpleNamespace(eos_token_id=151645), args).segment([11, 12, 13])
        self.assertEqual(attempts, [11, 15])
        self.assertEqual(len(rows), 1)

    def test_native_forward_can_leave_proposal_mask_unchanged(self):
        model = FakeNativeModel()
        unchanged = dict(snapshot(1), unmask_forward_index=2,
                         draft_latency_elapsed_ms=4.0, draft_passes_elapsed=2,
                         newly_unmasked_positions=[])
        model.generate_draft_tokens_arbitrary_length = lambda inputs, **kwargs: (
            None, None, None, None, None,
            dict(oracle_refinement_snapshots=[snapshot(1), unchanged]))
        args = SimpleNamespace(raw_top_k=32, physical_block_size=32,
            small_block_size=8, extend_size=8, drafter_threshold=0.5,
            max_refinement_steps=3)
        rows = NativeElysiaRunner(model, SimpleNamespace(), args).segment([11] * 29)
        self.assertEqual(len(rows), 2)

    def test_e_fills_parent_top1_then_native_r_keeps_length(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder), dataset='gsm8k', shard_rows=4,
                max_proposal_tokens=16, extend_size=8, small_block_size=8,
                max_refinement_steps=3, branch_width=2,
                min_expand_acceptance_ratio=0.0, bad_probe_branches=0,
                bad_refinement_steps=0, drafter_threshold=0.5)
            runner = FakeNativeRunner()
            graph = NativeElysiaGraphCollector(args, FakeFeatureEngine(), runner,
                                                eos_id=100, verifier=None)
            source = dict(state_id='anchor', problem_id=0, round_id=2,
                prefix_token_ids=[11, 12, 13], accepted_len=0)
            graph.run_anchor(source, list(range(1, 40)), {8: 1.0, 16: 2.0}, 'ref')
            graph.finish()
            nodes = {n['state_id']: n for n in graph.nodes}
            self.assertEqual(len(runner.prompts), 3)
            self.assertEqual(runner.prompts[0], [11, 12, 13])
            self.assertTrue(all(p == [11, 12, 13] + list(range(1, 9))
                                for p in runner.prompts[1:]))
            self.assertEqual(len(nodes), 6)
            for edge in graph.edges:
                src, dst = nodes[edge['src_state_id']], nodes[edge['dst_state_id']]
                self.assertEqual(dst['submit_candidate_source'], 'native_elysia_same_forward_top1')
                if edge['action'] == 'E':
                    self.assertEqual((src['proposal_length'], dst['proposal_length']), (8, 16))
                    self.assertEqual(edge['pre_extension_top1_filled_positions'],
                                     list(range(1, 8)) if src['native_unmask_forward_index'] == 1
                                     else list(range(2, 8)))
                    with np.load(Path(folder) / 'raw/gsm8k_structured' / dst['shard']) as data:
                        native = data['proposal_token_ids_before_fill'][dst['row'], :16].tolist()
                        self.assertIn('native_active_hidden_states', data.files)
                        self.assertEqual(data['native_active_hidden_states'].shape[1], 5)
                        self.assertTrue(data['native_active_topk_valid'][dst['row']].any())
                    self.assertEqual(native[:8], list(range(1, 9)))
                else:
                    self.assertEqual(src['proposal_length'], dst['proposal_length'])
            stored = [json.loads(s) for s in (Path(folder) / 'raw/gsm8k_structured/index.jsonl').read_text().splitlines()]
            self.assertEqual(len(stored), 6)
            self.assertTrue(all(r['submit_candidate_source'] == 'native_elysia_same_forward_top1'
                                for r in stored))

    def test_zero_acceptance_still_extends_through_sixty_four_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder), dataset='gsm8k', shard_rows=8,
                max_proposal_tokens=64, extend_size=8, small_block_size=8,
                max_refinement_steps=3, branch_width=2,
                min_expand_acceptance_ratio=0.5, drafter_threshold=0.5)
            runner = FakeNativeRunner()
            verifier = FakeVerifier()
            graph = NativeElysiaGraphCollector(args, FakeFeatureEngine(), runner,
                                                eos_id=100, verifier=verifier)
            source = dict(state_id='anchor', problem_id=0, round_id=2,
                prefix_token_ids=[11, 12, 13], accepted_len=0)
            graph.run_anchor(source, list(range(1, 100)), {}, 'ref')
            graph.finish()
            self.assertTrue(all(n['submit_accepted_len'] == 0 for n in graph.nodes))
            self.assertEqual({n['proposal_length'] for n in graph.nodes},
                             set(range(8, 65, 8)))
            self.assertEqual(max(n['proposal_length'] for n in graph.nodes), 64)
            self.assertTrue(any(n['selected_for_expansion'] and n['proposal_length'] == 56
                                for n in graph.nodes))

    def test_full_collection_packages_native_schema_and_direct_verifier_records(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder) / 'out', dataset='gsm8k',
                shard_rows=4, max_proposal_tokens=16, extend_size=8,
                small_block_size=8, physical_block_size=32,
                max_refinement_steps=1, max_unmask_passes=None, branch_width=2,
                min_expand_acceptance_ratio=0.0, bad_probe_branches=0,
                bad_refinement_steps=0, drafter_threshold=0.5,
                remaining_output_budget=None, calibration_repeats=2,
                target_gpu_memory_gib=9, seed=42, raw_top_k=32,
                backbone_zip=Path(folder) / 'unused.zip', num_questions=1,
                anchors_per_question=1, max_rounds_per_question=0,
                verifier_mode='full_context_no_kv', drafter_kv_mode='stable_block_prefix',
                unmask_backend='native_elysia', target_model_name='fake',
                dllm_dir=Path(folder))
            source = dict(state_id='anchor', problem_id=0, round_id=2,
                prefix_token_ids=[11, 12, 13],
                proposal_token_ids_after_fill=list(range(1, 9)),
                accepted_len=0)
            with patch('structured_sparse_collector.load_anchors', return_value=[source]), \
                 patch('structured_sparse_collector._load_models',
                       return_value=(SimpleNamespace(eos_token_id=100), object(), FakeNativeModel())), \
                 patch('structured_sparse_collector.KVExplicitDrafter', return_value=FakeFeatureEngine()), \
                 patch('structured_sparse_collector.FullContextVerifier', FakeVerifier), \
                 patch('native_elysia_graph.NativeElysiaRunner', return_value=FakeNativeRunner()):
                collect(args)
            archive = args.output_dir / 'gsm8k_structured_graph.zip'
            with zipfile.ZipFile(archive) as z:
                manifest = json.loads(z.read('graph_manifest.json'))
                rows = [json.loads(x) for x in z.read('nodes.jsonl').splitlines()]
                records = [json.loads(x) for x in z.read('verifier_calibration.jsonl').splitlines()]
                self.assertIsNone(z.testzip())
            self.assertEqual(manifest['schema_version'], 'structured_sparse_native_elysia_v1')
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(records), len(rows) + 1)  # archived anchor recheck
            self.assertTrue(all(r['submit_candidate_source'] == 'native_elysia_same_forward_top1'
                                for r in rows))


if __name__ == '__main__':
    unittest.main()
