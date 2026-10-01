"""Integration/causal-provenance tests on CPU fakes and a tiny real Qwen forward."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_world_model_pretraining import FakeRunner, FakeVerifier, FakeTokenizer, snap
from world_model_core import WorldModelLearner, ExperienceReplay, pack_observations
from world_model_environment import native_observation, MASK_ID
from world_model_twosource import TwoSourceWorldModel, FEATURE_VARIANTS, configure_losses
from world_model_teacher_environment import TeacherVerifier, TeacherTrainingEnvironment
from world_model_training_audit import (detailed_rows, detailed_report, load_saved_replay,
    run_retrained_audit, auc)
from pretrain_acceptance_world_model import parse_args, explore_questions, ExperienceWriter, package
from world_model_hindsight import HindsightLabeler

torch.set_num_threads(1)


def settings(output='unused'):
    return parse_args(['--dllm_dir','unused','--output_dir',str(output),
        '--model_architecture','two_source','--hidden_layers','14','28',
        '--latent_dim','64','--num_questions','4','--validation_questions','2',
        '--max_rounds_per_question','2','--max_proposal_tokens','24','--raw_top_k','2',
        '--batch_sequences','2','--warmup_updates','0','--horizon_warmup_updates','0',
        '--shadow_verify_probability','1','--audit_states_per_question','4',
        '--audit_train_states_per_question','4','--audit_retrain_updates','2',
        '--audit_seeds','7','--audit_variants','full','no_hidden','no_teacher'])


class FakeTeacher(FakeVerifier):
    def score(self,*args):
        result=super().score(*args); length=len(args[1]); accepted=result[0]
        margin=torch.tensor([1. if i<accepted else -1. for i in range(length)])
        features=torch.zeros(length,35); features[:,0]=torch.tanh(margin/5)
        features[:,1]=.4; features[:,2]=(margin>0).float(); features[:,3:]=.2
        self.last_teacher=dict(margin=margin.tolist(),features=features,boundary_hidden=torch.ones(32))
        return result


class TwoSourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.table=torch.randn(MASK_ID+1,4)
        self.args=settings()
        self.model=TwoSourceWorldModel(4,4,2,64,2,dropout=0).eval()

    def observation(self,uid,index=0):
        return native_observation(uid,'q',0,[10,11,12],[],snap(index),None,4,2,self.args,index)

    def test_current_verifier_targets_never_change_student(self):
        o=self.observation('s'); o.verifier_history=torch.zeros(1,48)
        batch=pack_observations([o],self.table,'cpu'); before=self.model.encoder(batch)
        o.accepted=1; o.teacher_margin=torch.randn(8)*100; o.teacher_features=torch.randn(8,35)*100
        after=self.model.encoder(pack_observations([o],self.table,'cpu'))
        self.assertTrue(torch.equal(before.tokens,after.tokens))
        self.assertTrue(torch.equal(self.model.acceptance(before),self.model.acceptance(after)))
        o.verifier_history=torch.randn(1,48)
        memory=self.model.encoder(pack_observations([o],self.table,'cpu'))
        self.assertFalse(torch.equal(before.global_state,memory.global_state))

    def test_memory_zero_padding_is_inert_and_shadow_does_not_advance(self):
        a=self.observation('a'); b=self.observation('b')
        a.verifier_history=torch.randn(1,48); b.verifier_history=torch.randn(8,48)
        one=self.model.encoder(pack_observations([a],self.table,'cpu'))
        mixed=self.model.encoder(pack_observations([a,b],self.table,'cpu'))
        self.assertTrue(torch.allclose(one.tokens[0],mixed.tokens[0],atol=1e-6))
        env=TeacherTrainingEnvironment(FakeRunner(),FakeTeacher(),0,4,self.args,lambda *a:None,token_table=self.table)
        state=env.start('q',0,[10,11,12],65)
        env.shadow(state,65)
        self.assertFalse(state.submitted); self.assertEqual(len(env.verifier_history),0)
        env.submit(state,65)
        self.assertEqual(len(env.verifier_history),1)
        next_state=env.start('q',1,state.prefix+state.emitted,65)
        self.assertEqual(len(next_state.observation.verifier_history),1)
        self.assertEqual(len(state.observation.verifier_history),0)
        env.reset_history(); self.assertEqual(len(env.verifier_history),0)

    def test_horizon_three_reuses_imagined_state_and_trains_teacher(self):
        nodes=[self.observation(str(i),i) for i in range(4)]
        for o in nodes:
            o.accepted=7; o.teacher_margin=torch.ones(8)
            o.teacher_features=torch.zeros(8,35); o.teacher_features[:,2]=1
        replay=ExperienceReplay()
        for a,b in zip(nodes,nodes[1:]): replay.add(a,b,'R')
        replay.sample=lambda *args:[(nodes,['R']*3)]
        learner=WorldModelLearner(self.model,self.table,'cpu',warmup_updates=0,horizon_warmup=0)
        configure_losses(learner,'full')
        seen=[]; original=self.model.transition
        def track(z,*args):
            result=original(z,*args); seen.append((z,result)); return result
        self.model.transition=track
        before=copy.deepcopy(self.model.teacher.state_dict())
        result=learner.update(replay,1,3)
        self.assertTrue(result['teacher_loss']>0)
        self.assertEqual(len(seen),3)
        for i in (1,2): self.assertTrue(torch.equal(seen[i][0].tokens,seen[i-1][1].tokens))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in self.model.teacher.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for p in self.model.teacher_ema.parameters()))

    def test_tiny_real_qwen_shifted_teacher_and_hook_cleanup(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                          num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
        target=Qwen2ForCausalLM(config).eval()
        prefix=[4,5,6]; proposal=[7,8,9]
        verifier=TeacherVerifier(target,SimpleNamespace(eos_token_id=31),SimpleNamespace(capture_verifier_teacher=True))
        result=verifier.score(prefix,proposal,65)
        with torch.no_grad(): truth=target(torch.tensor([prefix+proposal]),output_hidden_states=True,use_cache=False)
        shifted=truth.logits[0,len(prefix)-1:-1]
        agreement=shifted.argmax(-1).eq(torch.tensor(proposal))
        self.assertTrue(torch.equal(verifier.last_teacher['features'][:,2].bool(),agreement))
        self.assertEqual(verifier.last_teacher['features'].shape,(3,35))
        expected=truth.hidden_states[-1][0,len(prefix)-1:len(prefix)+len(proposal)].float() @ verifier.hidden_projection
        self.assertTrue(torch.allclose(verifier.last_teacher['features'][:,3:]*10,expected[:3].clamp(-30,30),atol=1e-6))
        self.assertTrue(torch.allclose(verifier.last_teacher['boundary_hidden']*10,expected[result[0]].clamp(-30,30),atol=1e-6))
        self.assertEqual(len(target.model.norm._forward_hooks),0)
        self.assertEqual(len(target.lm_head._forward_hooks),0)

    def test_shadow_label_survives_censor_and_eos_without_raw_state_advances(self):
        env=TeacherTrainingEnvironment(FakeRunner(),FakeTeacher(),0,4,self.args,lambda *a:None,token_table=self.table)
        state=env.start('q',0,[10,11,12],65)
        labeler=HindsightLabeler(state.prefix)
        labeler.register(state); env.shadow(state,65)
        self.assertEqual(labeler.finish('limit')['exact'],1)
        state=env.start('q',1,[10,11,12],65)
        labeler=HindsightLabeler(state.prefix); labeler.register(state)
        labeler.after_emitted(state.prefix,[1,2,3,4,5,6,7,8])
        self.assertEqual(state.observation.accepted,7)

    def test_fixed_holdout_learning_curve_artifacts_and_matched_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'; output.mkdir(); args=settings(output)
            env=TeacherTrainingEnvironment(FakeRunner(),FakeTeacher(),0,4,args,None,token_table=self.table)
            learner=WorldModelLearner(self.model,self.table,'cpu',warmup_updates=0,horizon_warmup=0)
            configure_losses(learner,'full')
            summary=dict(status='running',questions_completed=0)
            explore_questions(args,[dict(question_id=f'q{i}',prompt='fake integration') for i in range(4)],
                              FakeTokenizer(),env,learner,ExperienceWriter(output),summary)
            rows=[json.loads(s) for s in (output/'online_predictions.jsonl').read_text().splitlines()]
            held=[r for r in rows if r['split']=='validation']
            self.assertEqual({r['updates'] for r in held},{0})
            self.assertEqual([r['train_questions'] for r in summary['learning_curve']],[0,2])
            replay=load_saved_replay(output,'validation',4,3)
            self.assertEqual(set(o.question for o in replay.nodes.values()),{'q2','q3'})
            self.assertTrue(all(o.verifier_history is not None for o in replay.nodes.values()))
            results=run_retrained_audit(args,learner)
            self.assertEqual(results['variants_completed'],3)
            self.assertEqual(results['verifier_calls_for_audit'],0)
            full=results['results'][0]['groups']
            self.assertGreater(full['h0']['n'],0)
            self.assertTrue((output/'factor_audit'/'summary.json').exists())
            with __import__('zipfile').ZipFile(output.with_suffix('.zip')) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('learning_curve.json',archive.namelist())

    def test_local_agreement_does_not_imply_prefix_acceptance(self):
        nodes=[self.observation('s')]; o=nodes[0]; o.accepted=1
        o.teacher_margin=torch.tensor([1.,-1.,1.,1.,1.,1.,1.,1.])
        o.teacher_features=torch.zeros(8,35); o.teacher_features[:,2]=o.teacher_margin>0
        replay=ExperienceReplay(); replay.add_node(o)
        learner=WorldModelLearner(self.model,self.table,'cpu')
        report=detailed_report(detailed_rows(learner,replay))
        self.assertEqual(report['tokens']['h0']['prefix_acceptance']['positive_rate'],1/8)
        self.assertEqual(report['tokens']['h0']['local_teacher_forced_agreement']['positive_rate'],7/8)
        self.assertEqual(auc([0,1],[.1,.9]),1.)
        self.assertEqual(auc([0,1],[.5,.5]),.5)

    def test_all_default_removals_have_valid_real_transition_gradients(self):
        nodes=[self.observation(str(i),i) for i in range(4)]
        for o in nodes:
            o.hidden=torch.cat([o.hidden[:,:1],o.hidden],1)
            o.accepted=7; o.teacher_margin=torch.ones(8)
            o.teacher_features=torch.ones(8,35)*.2; o.teacher_features[:,2]=1
            o.verifier_history=torch.ones(2,48)
        for variant in FEATURE_VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(19)
                model=TwoSourceWorldModel(4,4,2,64,3,dropout=0,variant=variant)
                learner=WorldModelLearner(model,self.table,'cpu',warmup_updates=0,horizon_warmup=0)
                configure_losses(learner,variant)
                replay=ExperienceReplay()
                for a,b in zip(nodes,nodes[1:]): replay.add(a,b,'R')
                replay.sample=lambda *args:[(nodes,['R']*3)]
                result=learner.update(replay,1,3)
                self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
                self.assertEqual(result['sampled_path_depth_counts']['3'],1)


if __name__=='__main__': unittest.main()
