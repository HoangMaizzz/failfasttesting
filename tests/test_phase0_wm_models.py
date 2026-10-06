"""Causal input, free latent closure, native bookkeeping and gradient tests."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase0_wm_models import (TokenEmbedding, LatentEncoder, VerifierReadout,
    NativeReconstruction, DynamicsPair, PriorDynamics, DirectOutcome,
    ObservationBatch, LatentState, rollout, latent_loss, verifier_loss,
    valid_positions, advance_structure)
from phase0_wm_data import pack_observations, pack_native_targets


def embedding():
    vocabulary = list(range(256)) + [151665]
    return TokenEmbedding(torch.randn(len(vocabulary)+1, 16)*.1, vocabulary, learned=True)


def rows(count=2):
    result=[]
    for i in range(count):
        n=8;c=torch.zeros(n,20);c[:,0]=1;c[:,3:5]=1;c[:,7]=.4;c[:,8]=1;c[:,12:14]=1;c[:,15]=.5
        native=torch.full((n,),151665);stop=torch.arange(100,108)+i
        result.append(dict(uid=str(i),question='q'+str(i),length=n,hidden=torch.randn(n,3,1536).half(),
            ids=torch.stack([native,stop],-1),topk_ids=torch.arange(32).expand(n,-1),
            gaps=-torch.arange(32).float().expand(n,-1),c=c,
            context=torch.tensor([3/1024,n/64,0,0,.5,.5,.125,1]),prefix_ids=torch.tensor([1,2,3]),
            accepted=3,teacher=None,native_target=torch.zeros(n,100),native_mask=torch.ones(n,100,dtype=torch.bool)))
    return result


class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_padding_and_encoder_teacher_independence(self):
        encoder=LatentEncoder(embedding(),32,1,0).eval()
        data=rows(); changed=[dict(r,accepted=8,teacher=torch.rand(8,35),question='different') for r in data]
        a=encoder(pack_observations(data,'cpu'));b=encoder(pack_observations(changed,'cpu'))
        self.assertTrue(torch.equal(a.z,b.z))
        self.assertEqual(tuple(a.z.shape),(2,64,32))
        self.assertEqual(float(a.z[:,8:].abs().sum()),0)

    def test_prefix_order_is_observed(self):
        encoder=LatentEncoder(embedding(),32,1,0).eval();data=rows()
        a=encoder(pack_observations(data,'cpu')).z
        data[0]['prefix_ids']=torch.tensor([3,2,1])
        b=encoder(pack_observations(data,'cpu')).z
        self.assertFalse(torch.allclose(a[0,:8],b[0,:8]))
        self.assertTrue(torch.equal(a[1],b[1]))

    def test_topk_identity_not_only_gaps_is_observed(self):
        encoder=LatentEncoder(embedding(),32,1,0).eval();data=rows()
        a=encoder(pack_observations(data,'cpu')).z
        data[0]['topk_ids']=data[0]['topk_ids']+40
        b=encoder(pack_observations(data,'cpu')).z
        self.assertFalse(torch.allclose(a[0,:8],b[0,:8]))

    def test_readout_interface_is_only_latent_and_length(self):
        readout=VerifierReadout(32,1,0).eval();z=torch.randn(2,64,32);length=torch.tensor([8,16])
        out=readout(z,length)
        self.assertEqual(set(out),{'hazard','tf','probability','margin'})
        self.assertTrue(torch.isfinite(out['hazard']).all())

    def test_independent_actions_free_rollout_gradients(self):
        state=LatentEncoder(embedding(),32,1,0)(pack_observations(rows(),'cpu')).detach()
        pair=DynamicsPair(32,1,0)
        self.assertFalse(set(map(id,pair.refine.parameters())) & set(map(id,pair.extend.parameters())))
        pred=rollout(pair,state,[(0,1,0),(1,0,1)])
        self.assertEqual(pred.lengths.tolist(),[16,24])
        truth=LatentState(torch.randn_like(pred.z),pred.lengths,torch.zeros_like(pred.c),pred.context)
        recon=NativeReconstruction(32,100).eval().requires_grad_(False)
        loss=latent_loss(pred,truth,recon,torch.zeros(2,64,100),valid_positions(pred.lengths)[...,None].expand(-1,-1,100))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in pair.refine.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in pair.extend.parameters()))
        self.assertTrue(all(p.grad is None for p in recon.parameters()))

    def test_R_masks_monotonic_E_materializes_old_proposal(self):
        state=LatentEncoder(embedding(),32,1,0)(pack_observations(rows(),'cpu')).detach()
        state.c[:,0,0]=0;state.c[:,0,1]=1
        model=DynamicsPair(32,1,0)
        r=model(state,torch.zeros(2,dtype=torch.long))
        self.assertTrue((r.c[...,0]<=state.c[...,0]+1e-6).all())
        e=model(state,torch.ones(2,dtype=torch.long))
        self.assertEqual(float(e.c[:,:8,0].abs().sum()),0)
        self.assertEqual(e.lengths.tolist(),[16,16])
        self.assertTrue(torch.equal(e.context[:,2],torch.tensor([8/64,8/64])))
        self.assertTrue(torch.equal(e.context[:,3],torch.zeros(2)))

    def test_max64(self):
        s=LatentState(torch.zeros(1,64,32),torch.tensor([64]),torch.zeros(1,64,20),torch.zeros(1,8))
        with self.assertRaisesRegex(ValueError,'max64'):advance_structure(s,'E')

    def test_no_supervision_missing_teacher(self):
        data=[dict(rows()[0],accepted=None)]
        heads=VerifierReadout(32,1,0)(torch.randn(1,64,32),torch.tensor([8]))
        self.assertEqual(float(verifier_loss(heads,data,'cpu')),0)

    def test_empty_prefix_finite_and_no_truncation(self):
        data=rows();data[0]['prefix_ids']=torch.tensor([],dtype=torch.long)
        out=LatentEncoder(embedding(),32,1,0)(pack_observations(data,'cpu'))
        self.assertTrue(torch.isfinite(out.z).all())
        with self.assertRaisesRegex(ValueError,'no silent truncation'):pack_observations(rows(),'cpu',2)

    def test_direct_requires_only_source_and_actions(self):
        state=LatentEncoder(embedding(),32,1,0)(pack_observations(rows(),'cpu')).detach()
        pred=DirectOutcome(32,0)(state,[(1,0),(0,1)])
        self.assertTrue(torch.isfinite(pred['hazard']).all())

    def test_direct_distinguishes_action_order(self):
        state=LatentEncoder(embedding(),32,1,0)(pack_observations(rows(),'cpu')).detach()
        state=LatentState(*(getattr(state,f)[:1].expand((2,)+getattr(state,f).shape[1:])
                           for f in state.__dataclass_fields__))
        model=DirectOutcome(32,0).eval()
        pred=model(state,[(1,0),(0,1)])
        self.assertFalse(torch.allclose(pred['hazard'][0],pred['hazard'][1]))


if __name__=='__main__':unittest.main()
