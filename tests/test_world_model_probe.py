"""CPU fake-LLM tests; deliberately not presented as a T4/full-model smoke."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
import torch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from world_model_core import pack_observations,WorldModelLearner,ExperienceReplay,prefix_log_distribution
from world_model_probe import ProbeWorldModel
from world_model_environment import native_observation,NativeTrainingEnvironment,MASK_ID
from pretrain_acceptance_world_model import parse_args,ExperienceWriter,explore_questions,package
from test_world_model_pretraining import FakeRunner,FakeVerifier,FakeTokenizer,snap

torch.set_num_threads(1)


def settings(output="unused"):
    return parse_args(["--dllm_dir","unused","--output_dir",str(output),"--model_architecture","token_dual",
        "--hidden_layers","14","28","--latent_dim","16","--raw_top_k","2","--batch_sequences","2",
        "--warmup_updates","0","--horizon_warmup_updates","0","--max_proposal_tokens","24",
        "--max_rounds_per_question","2","--episodes_per_question","2","--package_every_question"])


def obs(uid="a",index=0):
    return native_observation(uid,"q",0,[10,11,12],[],snap(index),None,4,2,settings(),index)


class TeacherVerifier(FakeVerifier):
    def score(self,prefix,proposal,remaining):
        result=super().score(prefix,proposal,remaining)
        self.last_teacher={"margin":[1. if i<result[0] else -1. for i in range(len(proposal))]}
        return result


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.table=torch.randn(MASK_ID+1,4)
        self.model=ProbeWorldModel(4,4,2,dim=16,num_hidden_layers=2,dropout=0).eval()

    def test_no_teacher_or_K_leak(self):
        o=obs(); o.accepted=8; o.teacher_margin=torch.ones(8)
        a=self.model.encoder(pack_observations([o],self.table,"cpu"))
        o.accepted=0; o.teacher_margin=torch.full((8,),-99.)
        b=self.model.encoder(pack_observations([o],self.table,"cpu"))
        self.assertTrue(torch.equal(a.tokens,b.tokens))
        self.assertTrue(torch.equal(self.model.acceptance(a),self.model.acceptance(b)))

    def test_prefix_and_topk_identity_are_observable(self):
        o=obs(); before=self.model.encoder(pack_observations([o],self.table,"cpu")).tokens
        o.prefix_ids=torch.tensor([31,32,33])
        after=self.model.encoder(pack_observations([o],self.table,"cpu")).tokens
        self.assertFalse(torch.equal(before,after))
        o.topk_ids[:]=torch.tensor([17,18])
        newer=self.model.encoder(pack_observations([o],self.table,"cpu")).tokens
        self.assertFalse(torch.equal(after,newer))

    def test_masks_and_prefix_carry_without_oracle(self):
        z=self.model.encoder(pack_observations([obs()],self.table,"cpu"))
        logits=self.model.acceptance(z)
        r=self.model.transition(z,torch.tensor([0]),8)
        e=self.model.transition(z,torch.tensor([1]),8)
        self.assertEqual(e.tokens.shape,(1,16,16))
        self.assertTrue(torch.equal(self.model.acceptance(e)[:,:8],logits))
        self.assertEqual(float(r.mask_probs[0,0]),0)
        self.assertEqual(float(e.mask_probs[0,:8].sum()),0)
        self.assertTrue(torch.equal(self.model.acceptance(r)[:,:1],logits[:,:1]))
        mixed=self.model.encoder(pack_observations([obs(),obs("b")],self.table,"cpu"))
        mixed=self.model.transition(mixed,torch.tensor([0,1]),8)
        r=self.model.transition(mixed.take(torch.tensor([0])),torch.tensor([0]),8)
        self.assertEqual(r.tokens.shape,(1,8,16))
        self.assertTrue(torch.allclose(prefix_log_distribution(self.model.acceptance(r),r.lengths).exp().sum(-1),torch.ones(1)))

    def test_missing_native_extension_snapshot_disables_only_E(self):
        class EUnavailableRunner(FakeRunner):
            def segment(self,prompt,max_snapshots):
                if self.calls:
                    self.calls.append((list(prompt),max_snapshots))
                    raise RuntimeError("Native Elysia generator returned no oracle refinement snapshots")
                return super().segment(prompt,max_snapshots)
        args=settings(); env=NativeTrainingEnvironment(EUnavailableRunner(),FakeVerifier(),0,4,args,lambda *args:None)
        state=env.start("q",0,[10,11,12],65)
        self.assertIn("E",env.actions(state))
        self.assertIsNone(env.step(state,"E",65))
        self.assertNotIn("E",env.actions(state))
        self.assertIn("S",env.actions(state))
        self.assertEqual(env.stats["unavailable_E"],1)

    def test_region_loss_not_dominated_by_old_slots(self):
        errors=torch.cat([torch.zeros(1,56),torch.ones(1,8)],1)
        active=torch.arange(64)[None]>=56
        self.assertEqual(float(self.model.region_mean(errors,active)),1.)

    def test_three_hidden_layers_and_single_forward_teacher_alignment(self):
        args=settings(); args.hidden_layers=[7,14,28]
        row=snap(); row["hidden_layer_indices"]=[7,14,28]
        row["hidden_states"].append(copy.deepcopy(row["hidden_states"][-1]))
        o=native_observation("three","q",0,[10,11,12],[],row,None,4,2,args,0)
        model=ProbeWorldModel(4,4,2,dim=16,num_hidden_layers=3,dropout=0)
        z=model.encoder(pack_observations([o],self.table,"cpu"))
        self.assertEqual(z.tokens.shape,(1,8,16))
        from structured_sparse_collector import FullContextVerifier
        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.emb=torch.nn.Embedding(4,2); self.calls=0
            def get_input_embeddings(self): return self.emb
            def forward(self,**kw):
                self.calls+=1
                self.kw=kw
                return SimpleNamespace(logits=torch.tensor([[[0.,5.,2.,1.],[0.,4.,1.,2.],[0.,0.,0.,3.]]]))
        target=Target()
        verifier=FullContextVerifier(target,SimpleNamespace(eos_token_id=3),SimpleNamespace(capture_verifier_teacher=True))
        result=verifier.score([0],[1,2],65)
        self.assertEqual(target.calls,1)
        self.assertFalse(target.kw["use_cache"])
        self.assertEqual(target.kw["logits_to_keep"],3)
        self.assertEqual(result[:3],(1,2,[1,1]))
        self.assertEqual(verifier.last_teacher["margin"],[3.,-3.])

    def test_horizon_three_uses_predicted_state_and_updates_dynamics(self):
        a,b,c,d=obs("a"),obs("b",1),obs("c",2),obs("d",3)
        for o in (a,b,c,d): o.accepted=7; o.teacher_margin=torch.ones(8)
        replay=ExperienceReplay(); replay.add(a,b,"R"); replay.add(b,c,"R"); replay.add(c,d,"R")
        replay.sample=lambda *args:[([a,b,c,d],["R","R","R"])]
        learner=WorldModelLearner(self.model,self.table,"cpu",warmup_updates=0,horizon_warmup=0)
        seen=[]
        transition=self.model.transition
        def tracked(z,*args):
            result=transition(z,*args); seen.append((z,result)); return result
        self.model.transition=tracked
        previous=copy.deepcopy(self.model.state_dict())
        metrics=learner.update(replay,1,3)
        self.assertEqual(len(seen),3)
        for i in (1,2): self.assertTrue(torch.equal(seen[i][0].tokens,seen[i-1][1].tokens))
        self.assertGreater(metrics["teacher_loss"],0)
        self.assertGreater(metrics["structural_loss"],0)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(any(not torch.equal(value,previous[key]) for key,value in self.model.state_dict().items() if key.startswith("dynamics.")))
        self.assertNotIn("token_table",learner.checkpoint()["model"])
        restored=ProbeWorldModel(**learner.checkpoint()["model_config"])
        restored.load_state_dict(learner.checkpoint()["model"])

    def test_uncapped_questions_exceed_old_caps_and_stop_at_verified_eos(self):
        class ConstantRunner(FakeRunner):
            def segment(self,prompt,max_snapshots):
                rows=super().segment(prompt,max_snapshots)
                for row in rows:
                    row["proposal_token_ids_after_fill"]=[1]*8
                    row["proposal_token_ids_before_fill"][0]=1
                return rows
        class EosVerifier:
            def score(self,prefix,proposal,remaining):
                if len(prefix)-3>=144: return 0,1,[0],1.
                return len(proposal),len(proposal)+1,proposal+[1],1.
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"run"; output.mkdir()
            args=settings(output)
            args.max_rounds_per_question=0; args.max_new_tokens=0
            args.max_context_tokens=4096; args.episodes_per_question=2
            args.num_questions=2; args.validation_questions=1; args.stop_weight=1e9
            learner=WorldModelLearner(self.model,self.table,"cpu",warmup_updates=0,horizon_warmup=0)
            env=NativeTrainingEnvironment(ConstantRunner(),EosVerifier(),0,4,args,None)
            summary=dict(status="running",questions_completed=0)
            explore_questions(args,[dict(question_id=f"q{i}",prompt="test") for i in range(2)],
                FakeTokenizer(),env,learner,ExperienceWriter(output),summary)
            rows=[json.loads(s) for s in (output/"questions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows),4)
            for row in rows:
                self.assertEqual(row["collection_end_reason"],"eos")
                self.assertEqual(len(row["generated_tokens"]),145)
                self.assertEqual(row["generated_tokens"][-1],0)
            rounds=[json.loads(s) for s in (output/"rounds.jsonl").read_text().splitlines()]
            keys=[(r["question"],r["round_id"]) for r in rounds]
            self.assertEqual(len(keys),len(set(keys)))
            self.assertEqual(len(keys),68)

    def test_ten_questions_two_episodes_checkpoint_and_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"run"; output.mkdir()
            args=settings(output); writer=ExperienceWriter(output)
            learner=WorldModelLearner(self.model,self.table,"cpu",warmup_updates=0,horizon_warmup=0)
            env=NativeTrainingEnvironment(FakeRunner(),TeacherVerifier(),0,4,args,None)
            questions=[dict(question_id=f"q{i}",prompt="synthetic unit test, not GSM8K") for i in range(10)]
            summary=dict(status="running",questions_completed=0)
            explore_questions(args,questions,FakeTokenizer(),env,learner,writer,summary)
            self.assertEqual(summary["questions_completed"],10)
            self.assertEqual(summary["episodes_completed"],20)
            self.assertEqual(env.stats["verifier_calls"],40)
            self.assertEqual(summary["evaluation"]["questions"],["q8","q9"])
            self.assertIn("8",summary["evaluation"]["current_by_proposal_length"])
            self.assertTrue(summary["evaluation"]["current_by_question"])
            self.assertTrue(summary["evaluation"]["rollout_by_horizon_and_action"])
            self.assertIsNotNone(summary["evaluation"]["current"]["train_mean_baseline_mae"])
            predictions=[json.loads(x) for x in (output/"online_predictions.jsonl").read_text().splitlines()]
            holdout=[r for r in predictions if r["split"]=="validation"]
            self.assertEqual(len({r["updates"] for r in holdout}),1)
            self.assertTrue(all(r["captured_before_label_and_update"] for r in predictions))
            self.assertGreater(summary["dynamics_updates"],0)
            self.assertGreater(summary["parameter_l2_change"],0)
            teacher=[json.loads(x) for x in (output/"teacher_targets.jsonl").read_text().splitlines()]
            self.assertEqual(len(teacher),40)
            with zipfile.ZipFile(output.with_suffix(".zip")) as z:
                self.assertEqual(json.loads(z.read("summary.json"))["status"],"partial_checkpoint")
            summary["status"]="complete"; package(output,output.with_suffix(".zip"),summary)
            with zipfile.ZipFile(output.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("validation_predictions.jsonl",z.namelist())
                self.assertFalse(any(n.endswith(".safetensors") for n in z.namelist()))
            with np.load(next((output/"experience").glob("*.npz")),allow_pickle=False) as arrays:
                self.assertIn("aligned_topk_token_ids",arrays.files)
                self.assertTrue(arrays["teacher_valid"].any())


if __name__=="__main__": unittest.main()
