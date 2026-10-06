"""Targets cannot enter D forward; hazard semantics and bridge gradients matter."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_latent_models import (AcceptanceHead, NativeVerifier, PairedStudent,
    STUDENT_METHODS, acceptance_loss, objective, outputs, parameter_counts)
from run_paired_native_latent import call_model, QuestionSampler, validation_score


def batch():
    return dict(z_D=torch.randn(2,64,128), c=torch.randn(2,64,20), context=torch.randn(2,8),
        lengths=torch.tensor([8,16]), teacher_hidden=torch.randn(2,64,32),
        candidate_embedding=torch.randn(2,64,64), accepted=torch.tensor([3,16]),
        teacher_valid=torch.arange(64)[None] < torch.tensor([8,16])[:,None])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_d_forward_does_not_read_teacher_candidate_or_acceptance(self):
        torch.manual_seed(42)
        b = batch(); model = PairedStudent('Direct', dropout=0).eval()
        with torch.no_grad():
            a = call_model(model, 'Direct', b)
            changed = dict(b, teacher_hidden=b['teacher_hidden']*100,
                candidate_embedding=b['candidate_embedding']*-999,
                accepted=torch.tensor([0,0]), true_parent_K=torch.tensor([64,64]))
            c = call_model(model, 'Direct', changed)
        torch.testing.assert_close(a['hazard'], c['hazard'], atol=0, rtol=0)

    def test_paired_inference_capacity_and_initial_weights_are_exactly_equal(self):
        native = NativeVerifier('V_joint', dropout=0)
        models = []
        for method in STUDENT_METHODS:
            torch.manual_seed(777)
            models.append(PairedStudent(method, dropout=0, native_head=native.readout))
        for model in models[1:]:
            self.assertEqual(parameter_counts(model)['inference'], parameter_counts(models[0])['inference'])
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        self.assertLess(parameter_counts(models[1])['trainable'], parameter_counts(models[0])['trainable'])

    def test_bridge_acceptance_gradient_reaches_stem_with_frozen_head(self):
        b = batch(); model = PairedStudent('Bridge_behavior', dropout=0)
        cfg = dict(bridge_state_weight=0., bridge_behavior_weight=1., distill_weight=0.)
        pred = call_model(model, model.method, b)
        loss, _ = objective(model.method, pred, b, torch.randn_like(pred['z_V']), cfg)
        loss.backward()
        self.assertGreater(float(model.input[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.readout.parameters()))
        model.train()
        self.assertFalse(model.readout.training)

    def test_state_only_bridge_loss_is_independent_of_labels(self):
        b = batch(); model = PairedStudent('Bridge_state', dropout=0).eval()
        cfg = dict(bridge_state_weight=1., bridge_behavior_weight=1., distill_weight=.3)
        pred = call_model(model, model.method, b); target = torch.randn_like(pred['z_V'])
        a, _ = objective(model.method, pred, b, target, cfg)
        c, _ = objective(model.method, pred, dict(b,accepted=torch.tensor([0,0])), target, cfg)
        torch.testing.assert_close(a,c,atol=0,rtol=0)

    def test_candidate_conditioning_and_native_head_causality(self):
        b = batch(); model = NativeVerifier('V_joint', dropout=0).eval()
        with torch.no_grad():
            a = model(b['teacher_hidden'],b['candidate_embedding'],b['lengths'])['hazard']
            candidate = b['candidate_embedding'].clone(); candidate[:,6:] += 8
            c = model(b['teacher_hidden'],candidate,b['lengths'])['hazard']
        torch.testing.assert_close(a[:,:6], c[:,:6],atol=1e-6,rtol=1e-6)
        self.assertGreater(float((a[:,6:8]-c[:,6:8]).abs().sum()),0)
        hidden = NativeVerifier('V_hidden_only',dropout=0).eval()
        with torch.no_grad():
            x=hidden(b['teacher_hidden'],b['candidate_embedding'],b['lengths'])['hazard']
            y=hidden(b['teacher_hidden'],candidate,b['lengths'])['hazard']
        torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_hazard_survival_expectation_and_censoring(self):
        logits = torch.zeros(1,64,requires_grad=True)
        n=torch.tensor([8]); k=torch.tensor([2])
        result = outputs(logits,n)
        torch.testing.assert_close(result['q'][0,:3],torch.tensor([.5,.25,.125]))
        self.assertEqual(float(result['q'][0,8:].sum()),0)
        loss,parts = acceptance_loss(logits,n,k)
        grad = torch.autograd.grad(parts['nll'],logits,retain_graph=True)[0]
        self.assertLess(float(grad[0,0]),0)
        self.assertGreater(float(grad[0,2]),0)
        self.assertEqual(float(grad[0,3:].abs().sum()),0)
        self.assertTrue(torch.isfinite(loss))

    def test_missing_labels_are_not_negative_targets(self):
        logits=torch.zeros(1,64,requires_grad=True)
        loss,_=acceptance_loss(logits,torch.tensor([8]),torch.tensor([-1]))
        loss.backward()
        self.assertEqual(float(loss),0)
        self.assertEqual(float(logits.grad.abs().sum()),0)

    def test_sampling_and_validation_are_reproducible_question_balanced(self):
        rows={str(i):dict(uid=str(i),question='q'+str(i//2),action='R' if i%2 else 'E') for i in range(8)}
        a=QuestionSampler(list(rows),rows,42);b=QuestionSampler(list(rows),rows,42)
        self.assertEqual(a.sample(20),b.sample(20))
        rec=[dict(question='q1',accepted=0,K_pred=2,action='R',parent_token_changed=True,
            parent_accepted=0,parent_length=8)]*100
        rec += [dict(question='q2',accepted=8,K_pred=8,action='E',parent_token_changed=False,
            parent_accepted=8,parent_length=8)]
        score,cohorts=validation_score(rec)
        self.assertEqual(cohorts['All']['K_question_macro_MAE'],1.)
        self.assertEqual(score,1.)


if __name__ == '__main__':
    unittest.main()
