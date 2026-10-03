"""Native late-block FiLM with immutable prefix KV and causal conditioning.

Training replays ONLY the captured last decoder block, not a different full
context denoising state. Every captured proposal is labeled by a fresh no-KV
verifier call. Cached tails stay outside the downloadable result directory.
"""
from __future__ import annotations
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from persistent_world_model_v2 import (BehavioralWorldModelV2, FixedDynamicsV2,
    GatedFiLMV2, pack_v2, film_acceptance_loss, teacher_token_ids)
from run_twosource_grouped_cv import atomic_json, append_jsonl, read_jsonl

ARMS=('late_acceptance','late_kl','head_acceptance','head_kl')


def normal_tensor(x):
    # Native capture runs in inference_mode. Clone outside it before backprop.
    return x.detach().clone() if isinstance(x,torch.Tensor) else x


class ImmutableTailCache:
    def __init__(self, index, kv):self.index,self.kv=index,kv
    def __len__(self):return self.index+1 if self.kv is not None else 0
    def __getitem__(self,index):
        if index!=self.index or self.kv is None:raise IndexError(index)
        return self.kv


class NativeTailHook:
    """Count native forwards, capture exact last-block inputs, optionally inject."""
    def __init__(self,drafter):
        self.drafter=drafter;self.layer=drafter.model.layers[-1]
        self.layer_index=len(drafter.model.layers)-1
        self.adapter=None;self.capture=False;self.mode='late';self.frames={}
        self.pending_z=None;self.latents={};self.counter=0;self.segment_start=0;self.current=None
        self.proposal_start=0;self.proposal_length=8;self.forward_diagnostics=[]
        # Native generation invokes self.forward(...) directly, bypassing the
        # outer module hooks. The inner model is always invoked as a module.
        self.handles=[drafter.model.register_forward_pre_hook(self._forward,with_kwargs=True),
            self.layer.register_forward_pre_hook(self._layer,with_kwargs=True),
            drafter.model.norm.register_forward_hook(self._norm)]
        self.previous_reset=getattr(drafter,'_wm_v2_reset_capture',None)
        drafter._wm_v2_reset_capture=self.reset_attempt
        drafter._persistent_world_film_adapter=None

    def reset_attempt(self):
        self.counter=0;self.frames={};self.current=None

    def begin(self,segment_start,proposal_start,proposal_length,z=None,refine=False):
        self.segment_start=segment_start;self.proposal_start=proposal_start
        self.proposal_length=proposal_length;self.pending_z=z
        if not refine:self.latents={}
        self.forward_diagnostics=[]

    def _forward(self,module,args,kwargs):
        self.counter+=1
        self.input_ids=kwargs.get('input_ids',args[0] if args else None)

    def _layer(self,module,args,kwargs):
        hidden=args[0] if args else kwargs['hidden_states']
        pos=kwargs.get('cache_position')
        if pos is None:
            pos=kwargs.get('position_ids')
            if pos is not None:pos=pos.reshape(-1)
        if pos is None or pos.numel()!=hidden.shape[1]:
            raise RuntimeError('Native tail lacks exact absolute cache positions')
        active=(pos>=self.segment_start)&(pos<self.segment_start+8)
        # Never mutate the immutable prefix KV or a prefill/cache-update forward.
        eligible=bool(active.any()) and not kwargs.get('update_past_key_values',False)
        self.current=dict(positions=pos.detach(),active=active,eligible=eligible)
        if self.capture and eligible:
            kv=kwargs.get('past_key_value')
            pair=kv[self.layer_index] if kv is not None and len(kv)>self.layer_index else None
            frame=dict(hidden=hidden.detach().cpu().clone(),positions=pos.detach().cpu().clone(),
                active=active.detach().cpu().clone(),kwargs={},
                kv=None if pair is None else tuple(t.detach().cpu().clone() for t in pair))
            ids=getattr(self,'input_ids',None)
            frame['input_ids']=ids.detach().cpu().clone() if isinstance(ids,torch.Tensor) else None
            for key in ('attention_mask','position_ids','cache_position','position_embeddings'):
                value=kwargs.get(key)
                frame['kwargs'][key]=(tuple(t.detach().cpu().clone() for t in value)
                    if isinstance(value,tuple) else value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value)
            self.frames[self.counter]=frame
        if self.adapter is not None and eligible and self.mode=='late':
            modified=self._apply(hidden)
            if args:return (modified,*args[1:]),kwargs
            return args,{**kwargs,'hidden_states':modified}

    def _apply(self,hidden):
        if self.counter not in self.latents:
            self.latents[self.counter]=self.pending_z
        latent=self.latents[self.counter]
        if latent is None:return hidden
        pos=self.current['positions']
        relative=((pos-self.proposal_start+1).float()/self.proposal_length)[None]
        active=self.current['active'][None]
        changed=self.adapter(hidden,latent,relative,active,active)
        diagnostic=self.adapter.diagnostics
        self.forward_diagnostics.append(dict(forward=self.counter,
            positions=pos[active[0]].detach().cpu().tolist(),
            gate=diagnostic['gate'][0,active[0],0].cpu().tolist(),
            delta_norm=diagnostic['delta_norm'][0,active[0]].cpu().tolist(),
            relative_delta=diagnostic['relative_delta'][0,active[0]].cpu().tolist()))
        return changed

    def _norm(self,module,args,output):
        if self.current and self.current['eligible']:
            if self.capture and self.counter in self.frames:
                self.frames[self.counter]['head_hidden']=output.detach().cpu().clone()
            if self.adapter is not None and self.mode=='head':return self._apply(output)

    def close(self):
        for handle in self.handles:handle.remove()
        if self.previous_reset is None:delattr(self.drafter,'_wm_v2_reset_capture')
        else:self.drafter._wm_v2_reset_capture=self.previous_reset


def replay_tail(drafter,frame,adapter,z,mode,device):
    """Frozen weights still propagate gradients from output back to FiLM."""
    def move(x):
        if isinstance(x,tuple):return tuple(move(t) for t in x)
        return normal_tensor(x).to(device) if isinstance(x,torch.Tensor) else x
    hidden=move(frame['hidden']);pos=move(frame['positions'])
    active=move(frame['active'])[None]
    relative=((pos-frame['proposal_start']+1).float()/frame['proposal_length'])[None]
    if mode=='late':
        if adapter is not None:hidden=adapter(hidden,z,relative,active,active)
        kwargs={k:move(v) for k,v in frame['kwargs'].items()}
        kwargs.update(past_key_value=ImmutableTailCache(len(drafter.model.layers)-1,move(frame['kv'])),
            update_past_key_values=False,use_cache=True,use_block_cache=False,block_past_key_values=None)
        hidden=drafter.model.layers[-1](hidden,**kwargs)
        if isinstance(hidden,tuple):hidden=hidden[0]
        hidden=drafter.model.norm(hidden)
    else:
        hidden=move(frame['head_hidden'])
        if adapter is not None:hidden=adapter(hidden,z,relative,active,active)
    # Restrict full-vocabulary allocation to the usable captured active rows.
    selected=move(frame['selected']).long()
    logits=drafter.lm_head(hidden[0].index_select(0,selected)).float()
    return logits


def load_fold(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=BehavioralWorldModelV2(**saved['config']).to(device)
    model.load_state_dict(saved['model']);model.freeze_representation()
    dynamics=FixedDynamicsV2().to(device);dynamics.load_state_dict(saved['dynamics']);dynamics.eval().requires_grad_(False)
    return model,dynamics


def make_args(args):
    return SimpleNamespace(target_model_name='Qwen/Qwen2.5-7B-Instruct',
        target_device=args.target_device,drafter_device=args.drafter_device,
        target_gpu_memory_gib=args.target_gpu_memory_gib,target_placement='auto',
        dllm_dir=str(args.dllm_dir),raw_top_k=32,hidden_layers=[7,14,28],
        physical_block_size=32,small_block_size=8,drafter_threshold=.5,
        max_refinement_steps=3,extend_size=8,max_proposal_tokens=64,
        capture_verifier_teacher=True,verifier_teacher_top_k=32,max_context_tokens=4096)


def select_edges(observations,edges,count,seed):
    """Bounded per-question and alternating R/E; selection never sees yield."""
    groups=defaultdict(lambda:defaultdict(list))
    for parent,child,action in edges:
        o=observations[child]
        if action in ('R','E') and o.prefix_ids is not None:
            groups[o.question][action].append((parent,child,action))
    rng=random.Random(seed);selected=[]
    for q in sorted({o.question for o in observations.values()}):
        actions=groups[q]
        roots=[o for o in observations.values() if o.question==q and o.length==8 and float(o.context[3])==0]
        if roots:
            roots.sort(key=lambda o:(o.round_id!=1,o.round_id,o.uid))
            selected.append((None,roots[0].uid,'root'))
        for group in actions.values():
            rng.shuffle(group);group.sort(key=lambda e:observations[e[1]].length)
        for i in range(count-int(bool(roots))):
            candidates=[key for key in ('R','E') if actions[key]]
            if not candidates:break
            action=candidates[i%len(candidates)]
            selected.append(actions[action].pop(0))
    return selected


def capture_frames(args,out,observations,edges,runner,verifier,hook,drafter,temp):
    records=[];paths={};selected=select_edges(observations,edges,args.film_examples_per_question,args.seed)
    hook.capture=True;hook.adapter=None
    for i,(parent,child,action) in enumerate(selected):
        o=observations[child];segment=int(round(float(o.context[2])*64));refine=int(round(float(o.context[3])*3))
        prefix=o.prefix_ids.tolist();prior=o.ids[:segment,1].tolist()
        hook.begin(len(prefix)+len(prior),len(prefix),o.length)
        try:
            snapshots=runner.segment(prefix+prior,max_snapshots=refine+1)
        except Exception as error:
            from native_elysia_graph import NativeEosWithoutSnapshot
            if not isinstance(error,NativeEosWithoutSnapshot):raise
            records.append(dict(child=child,question=o.question,usable=False,reason='native_eos_without_snapshot'));continue
        if len(snapshots)<=refine:
            records.append(dict(child=child,question=o.question,usable=False,reason='native_refine_exhausted'));continue
        snap=snapshots[refine];forward=int(snap['hidden_state_forward_pass'])
        frame=hook.frames.get(forward)
        if frame is None or 'head_hidden' not in frame:
            records.append(dict(child=child,question=o.question,usable=False,reason='snapshot_forward_not_captured'));continue
        proposal=prior+list(snap['proposal_token_ids_after_fill'])
        accepted,_,_,ms=verifier.score(prefix,proposal,65)
        teacher=verifier.last_teacher
        absolute=frame['positions'];relative=absolute-len(prefix)
        clean=(relative>=segment)&(relative<len(proposal))&(relative<=accepted)&frame['active']
        selected_rows=clean.nonzero().flatten()
        if not len(selected_rows):
            records.append(dict(child=child,question=o.question,usable=False,reason='rejection_before_active_segment'));continue
        rp=relative[selected_rows].long()
        if frame['input_ids'] is None or frame['input_ids'].shape[1]!=len(relative):
            raise RuntimeError('Native tail missing exact pre-unmask token IDs')
        mutable=frame['input_ids'][0,selected_rows].eq(151665)
        first_reject=(rp==accepted)&mutable
        # A committed first mismatch cannot be repaired by this native forward.
        # Keep its diagnosis, but do not optimize an impossible correction loss.
        if not bool(((rp<accepted)|first_reject).any()):
            records.append(dict(child=child,question=o.question,usable=False,reason='first_reject_already_committed'));continue
        frame.update(parent=parent,child=child,action=action,question=o.question,
            proposal_start=len(prefix),proposal_length=len(proposal),selected=selected_rows,
            candidate=torch.tensor(proposal)[rp],accepted_region=rp<accepted,first_reject=first_reject,
            support=teacher['topk_ids'][rp].clone(),teacher_logits=teacher['topk_logits'][rp].clone(),
            lse=teacher['logsumexp'][rp].clone(),token_positions=rp,accepted=int(accepted),
            native_proposal=proposal,snapshot=snap)
        # Must recover the same prediction on the EXACT captured native input.
        with torch.no_grad():
            base=replay_tail(drafter,frame,None,None,'late',args.device)
            head=replay_tail(drafter,frame,None,None,'head',args.device)
            frame['base_top1']=base.argmax(-1).cpu()
            parity=bool(torch.equal(base.argmax(-1),head.argmax(-1)))
        if not parity:
            raise RuntimeError(f'Late-tail replay top1 differs from native LM-head at {child}; refusing false FiLM labels')
        file=temp/f'{child}.pt';torch.save(frame,file);paths[child]=file
        records.append(dict(child=child,parent=parent,question=o.question,action=action,
            length=len(proposal),tokens=len(rp),usable=True,tail_replay_top1_parity=parity,
            proposal_matches_saved=proposal==o.ids[:,1].tolist(),fresh_verifier_ms=ms))
        records[-1]['first_reject_mutable']=bool(first_reject.any())
        print(f'[v2-tail] {i+1}/{len(selected)} question={o.question} action={action} tokens={len(rp)}',flush=True)
        del frame,base,head;hook.frames={};gc.collect()
    hook.capture=False
    append_jsonl(out/'film_native_capture_audit.jsonl',records)
    if not paths:raise RuntimeError('No valid native-tail FiLM examples; inspect capture audit')
    return paths


@torch.no_grad()
def conditioning_for_edges(model,dynamics,observations,edges,memory,device):
    result={}
    valid=[e for e in edges if e[2] in ('R','E')]
    for start in range(0,len(valid),8):
        group=valid[start:start+8]
        b=pack_v2([observations[p] for p,_,_ in group],memory,device);state=model.pre(b)
        predicted,_,_=dynamics(state,torch.tensor([int(a=='E') for _,_,a in group],device=device))
        for i,(_,child,_) in enumerate(group):result[child]=predicted.z[i:i+1].cpu().clone()
    actual=defaultdict(list)
    for o in observations.values():
        if o.teacher_is_actual:actual[o.question].append(o)
    # Root forwards cannot be conditioned by the root observation itself.
    for o in observations.values():
        if o.uid in result:continue
        past=[v for v in actual[o.question] if v.round_id<o.round_id]
        if past:
            old=max(past,key=lambda v:v.round_id);b=pack_v2([old],memory,device)
            result[o.uid]=model.post(model.pre(b),b).z.cpu().clone()
        else:result[o.uid]=torch.zeros(1,128)
    return result


def frame_loss(adapter,drafter,frame,z,mode,device,kl_only):
    base=replay_tail(drafter,frame,None,None,mode,device).detach()
    logits=replay_tail(drafter,frame,adapter,z.to(device),mode,device)
    ids=normal_tensor(frame['support']).to(device).long()
    tv=normal_tensor(frame['teacher_logits']).to(device).float()
    target=teacher_token_ids(ids,tv)
    if int(ids.max())>=logits.shape[-1] or int(ids.min())<0:
        raise ValueError('Verifier support ID outside drafter vocabulary')
    candidate=normal_tensor(frame['candidate']).to(device).long()
    selected=normal_tensor(frame['selected']).to(device).long()
    reg=adapter.regularization[0].index_select(0,selected)
    loss=film_acceptance_loss(logits,base,candidate,target,
        normal_tensor(frame['accepted_region']).to(device),normal_tensor(frame['first_reject']).to(device),
        ids,tv,normal_tensor(frame['lse']).to(device),reg,kl_only=kl_only)
    diagnostic=adapter.diagnostics
    details=dict(gate=diagnostic['gate'][0,selected,0].cpu().tolist(),
        delta_norm=diagnostic['delta_norm'][0,selected].cpu().tolist(),
        relative_delta=diagnostic['relative_delta'][0,selected].cpu().tolist(),
        base_top1=base.argmax(-1).cpu().tolist(),conditioned_top1=logits.argmax(-1).cpu().tolist(),
        verifier_top1=target.cpu().tolist(),token_positions=frame['token_positions'].tolist())
    return loss,details


def train_adapter(args,folder,drafter,paths,condition,train_ids,dev_ids,arm):
    from run_persistent_world_model_v2 import Balanced,snapshot
    selected=[]
    for uid,path in paths.items():
        frame=torch.load(path,map_location='cpu',weights_only=False)
        if frame['question'] in train_ids and uid in condition:
            selected.append((uid,frame['action'],frame['proposal_length']))
    sampler=Balanced(selected,lambda row:(row[1],min(row[2],40)),args.seed)
    dev=[]
    for uid,path in paths.items():
        frame=torch.load(path,map_location='cpu',weights_only=False)
        if frame['question'] in dev_ids and uid in condition:dev.append(uid)
    if not dev:raise ValueError('No inner-dev native-tail FiLM examples')
    torch.manual_seed(args.seed+151)
    adapter=GatedFiLMV2(drafter.config.hidden_size).to(args.device)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4)
    mode='head' if arm.startswith('head') else 'late';kl_only=arm.endswith('_kl')
    best=float('inf');best_state=None;stale=0;curve=[]
    for step in range(1,args.film_steps+1):
        uid,_,_=sampler.sample(1)[0];frame=torch.load(paths[uid],map_location='cpu',weights_only=False)
        adapter.train();loss,_=frame_loss(adapter,drafter,frame,condition[uid],mode,args.device,kl_only)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FiLM loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0);optimizer.step()
        if step%args.eval_every and step!=args.film_steps:continue
        adapter.eval();losses=[];rows=[]
        with torch.no_grad():
            for v in dev:
                frame=torch.load(paths[v],map_location='cpu',weights_only=False)
                l,details=frame_loss(adapter,drafter,frame,condition[v],mode,args.device,kl_only)
                losses.append(float(l));rows.append(dict(state_id=v,question=frame['question'],**details))
        score=float(np.mean(losses));improved=score<best
        if improved:best=score;best_state=snapshot(adapter);stale=0
        else:stale+=1
        curve.append(dict(update=step,train_loss=float(loss.detach()),dev_loss=score,selected=improved))
        atomic_json(folder/f'{arm}_curve.json',curve)
        print(f'[v2-film] arm={arm} update={step} dev={score:.4f}',flush=True)
        if stale>=args.patience:break
    adapter.load_state_dict(best_state);adapter.eval()
    torch.save(dict(adapter=best_state,hidden_dim=drafter.config.hidden_size,arm=arm,
        selection='inner_dev_proxy; real-dev guard selected separately',dev_loss=best),folder/f'{arm}_proxy_best.pt')
    return adapter


def token_transition(base,conditioned,verifier,limit):
    result=dict(correct_to_correct=0,correct_to_wrong=0,wrong_to_correct=0,wrong_to_wrong=0,changed=0)
    for a,b,v in zip(base[:limit],conditioned[:limit],verifier[:limit]):
        key=('correct' if a==v else 'wrong')+'_to_'+('correct' if b==v else 'wrong')
        result[key]+=1;result['changed']+=int(a!=b)
    result['net_fixed']=result['wrong_to_correct']-result['correct_to_wrong']
    return result


def paired_summary(rows,seed=42):
    if not rows:return dict(n=0)
    by_q=defaultdict(list)
    for r in rows:by_q[r['question']].append(r['delta_accepted'])
    qmean=np.array([np.mean(v) for v in by_q.values()]);rng=np.random.default_rng(seed)
    ci=np.quantile([rng.choice(qmean,len(qmean),replace=True).mean() for _ in range(1000)],[.025,.975])
    max_length=max(r['length'] for r in rows)
    survival=[]
    for pos in range(1,max_length+1):
        eligible=[r for r in rows if r['length']>=pos]
        survival.append(dict(position=pos,count=len(eligible),
            baseline=float(np.mean([r['base_accepted']>=pos for r in eligible])),
            film=float(np.mean([r['film_accepted']>=pos for r in eligible]))))
    transitions={key:sum(r['token_transitions'][key] for r in rows) for key in rows[0]['token_transitions']}
    return dict(n=len(rows),questions=len(by_q),mean_delta_accepted=float(np.mean([r['delta_accepted'] for r in rows])),
        question_macro_delta_accepted=float(qmean.mean()),question_bootstrap_ci95=ci.tolist(),
        mean_baseline_accepted=float(np.mean([r['base_accepted'] for r in rows])),
        mean_film_accepted=float(np.mean([r['film_accepted'] for r in rows])),
        accepted_prefix_destruction_rate=float(np.mean([r['delta_accepted']<0 for r in rows])),
        full_accept_baseline=float(np.mean([r['base_accepted']==r['length'] for r in rows])),
        full_accept_film=float(np.mean([r['film_accepted']==r['length'] for r in rows])),
        acceptance_fraction_baseline=float(np.mean([r['base_accepted']/r['length'] for r in rows])),
        acceptance_fraction_film=float(np.mean([r['film_accepted']/r['length'] for r in rows])),
        survival=survival,token_transitions=transitions,
        proposal_change_fraction=float(np.mean([r['changed_fraction'] for r in rows])),
        mean_native_replay_baseline_ms=float(np.mean([r['base_drafter_ms'] for r in rows])),
        mean_native_replay_film_ms=float(np.mean([r['film_drafter_ms'] for r in rows])),
        mean_verifier_baseline_ms=float(np.mean([r['base_verifier_ms'] for r in rows])),
        mean_verifier_film_ms=float(np.mean([r['film_verifier_ms'] for r in rows])),
        mean_film_replay_overhead_ms=float(np.mean([r['film_drafter_ms']-r['base_drafter_ms'] for r in rows])))


def paired_native_eval(args,folder,model,dynamics,adapters,observations,memory,qids,environment,verifier,hook):
    """Same saved prefix, seed, threshold and fixed S/R/E schedule; planner OFF.

    Shadow verification labels every observed node but never grounds the next
    state. Native impossible R/E and EOS terminate/exclude that paired path.
    """
    from native_elysia_graph import NativeEosWithoutSnapshot
    from persistent_world_model_v2 import semantic_verifier
    roots={}
    for q in qids:
        options=[o for o in observations.values() if o.question==q and o.length==8 and float(o.context[3])==0]
        if options:
            # Round 1 permits a real previous verifier observation; fallback round0.
            options.sort(key=lambda o:(o.round_id!=1,o.round_id,o.uid));roots[q]=options[0]
    max_tokens=getattr(args,'paired_max_tokens',64)
    schedule=['root']+sum((['R','E'] for _ in range(max_tokens//8-1)),[])+['R']
    all_rows=[];excluded=[]
    for q,root in roots.items():
        prefixes=root.prefix_ids.tolist();trajectories={}
        # Root FiLM has no current-state observation. Only previous actual STOP z.
        past=[o for o in observations.values() if o.question==q and o.round_id<root.round_id and o.teacher_is_actual]
        previous=None
        if past:
            old=max(past,key=lambda o:o.round_id)
            with torch.no_grad():
                b=pack_v2([old],memory,args.device);previous=model.post(model.pre(b),b).z
        for arm in ('off',*adapters):
            torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
            environment.reset_history();hook.capture=False
            hook.adapter=None if arm=='off' else adapters[arm]
            hook.mode='head' if arm.startswith('head') else 'late'
            hook.begin(len(prefixes),len(prefixes),8,previous)
            state=None;results={};executed_trace=[]
            for index,action in enumerate(schedule):
                began=time.perf_counter()
                try:
                    if action=='root':state=environment.start(q,root.round_id,prefixes,65)
                    else:
                        if action not in environment.actions(state):
                            # R has no native forward once all masks are committed.
                            if action=='R':continue
                            break
                        with torch.no_grad():
                            o=state.observation
                            live_memory={o.uid:memory.get(root.uid,torch.empty(0,40))}
                            pre=model.pre(pack_v2([o],live_memory,args.device))
                            predicted,_,_=dynamics(pre,torch.tensor([int(action=='E')],device=args.device))
                        start=len(state.prefix)+(state.observation.length if action=='E' else len(state.prior))
                        length=state.observation.length+(8 if action=='E' else 0)
                        hook.begin(start,len(state.prefix),length,predicted.z,refine=action=='R')
                        child=environment.step(state,action,65)
                        if child is None:
                            if state.extend_exhausted and action=='E':break
                            continue
                        state=child
                except NativeEosWithoutSnapshot:
                    excluded.append(dict(question=q,arm=arm,index=index,reason='native_eos_without_snapshot'));break
                torch.cuda.synchronize(args.drafter_device)
                executed_trace.append(action)
                drafter_ms=(time.perf_counter()-began)*1000
                proposal=state.observation.ids[:,1].tolist()
                accepted,_,_,ms=verifier.score(state.prefix,proposal,65)
                teacher=verifier.last_teacher
                reference=teacher_token_ids(teacher['topk_ids'],teacher['topk_logits']).tolist()
                results[index]=dict(question=q,arm=arm,index=index,action=action,length=len(proposal),
                    accepted=int(accepted),proposal=proposal,reference=reference,verifier_ms=ms,drafter_ms=drafter_ms,
                    film_forward_diagnostics=list(hook.forward_diagnostics),masks_remaining=int(state.observation.scalars[:,0].sum()))
                results[index]['executed_trace']=list(executed_trace)
                results[index]['native_passes']=state.snapshot.get('draft_passes_elapsed')
                # Critically NO submit(), remember() or post() on this shadow label.
                if state.terminal_reason:break
            trajectories[arm]=results
        hook.adapter=None
        base=trajectories['off']
        for arm,results in trajectories.items():
            if arm=='off':continue
            for index,r in results.items():
                b=base.get(index)
                if (b is None or b['length']!=r['length'] or
                    b['executed_trace']!=r['executed_trace'] or b['native_passes']!=r['native_passes']):
                    excluded.append(dict(question=q,arm=arm,index=index,reason='unmatched_native_schedule'));continue
                common=min(b['accepted'],r['accepted'])+int(min(b['accepted'],r['accepted'])<r['length'])
                # After proposals diverge, only reference on their common causal prefix is comparable.
                divergence=next((i for i,(a,c) in enumerate(zip(b['proposal'],r['proposal'])) if a!=c),r['length'])
                limit=min(common,divergence+1)
                row=dict(question=q,arm=arm,index=index,action=r['action'],length=r['length'],
                    base_accepted=b['accepted'],film_accepted=r['accepted'],delta_accepted=r['accepted']-b['accepted'],
                    base_drafter_ms=b['drafter_ms'],film_drafter_ms=r['drafter_ms'],
                    base_verifier_ms=b['verifier_ms'],film_verifier_ms=r['verifier_ms'],
                    changed_fraction=float(np.mean(np.array(b['proposal'])!=np.array(r['proposal']))),
                    common_causal_token_comparison_limit=limit,
                    matched_executed_trace=r['executed_trace'],native_passes=r['native_passes'],
                    token_transitions=token_transition(b['proposal'],r['proposal'],b['reference'],limit),
                    base_proposal=b['proposal'],film_proposal=r['proposal'],base_verifier_top1=b['reference'],
                    film_verifier_top1=r['reference'],film_forward_diagnostics=r['film_forward_diagnostics'])
                all_rows.append(row)
        print(f'[v2-real] question={q} matched_pairs={sum(r["question"]==q for r in all_rows)}',flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    append_jsonl(folder/'real_pairs.jsonl',all_rows);append_jsonl(folder/'real_exclusions.jsonl',excluded)
    groups=defaultdict(list)
    for row in all_rows:
        groups[row['arm']].append(row);groups[row['arm']+'/action='+row['action']].append(row)
        groups[row['arm']+'/L='+str(row['length'])].append(row)
    result={k:paired_summary(v,args.seed) for k,v in groups.items()}
    atomic_json(folder/'real_film_results.json',result)
    return result


def run_film_experiment(args,out,observations,edges,memory):
    from sparse_extend_world_model_collector import _load_models
    from native_elysia_graph import NativeElysiaRunner
    from world_model_teacher_environment import TeacherVerifier,TeacherTrainingEnvironment
    from run_persistent_world_model_v2 import package
    if args.dllm_dir is None:raise ValueError('--dllm_dir required for native FiLM phase')
    native_args=make_args(args)
    tokenizer,target,drafter=_load_models(native_args)
    drafter.eval().requires_grad_(False);target.eval().requires_grad_(False)
    runner=NativeElysiaRunner(drafter,tokenizer,native_args)
    verifier=TeacherVerifier(target,tokenizer,native_args)
    environment=TeacherTrainingEnvironment(runner,verifier,tokenizer.eos_token_id,
        drafter.config.hidden_size,native_args,lambda *_:None,token_table=drafter.get_input_embeddings().weight.detach())
    hook=NativeTailHook(drafter);summary=dict(status='running',folds=[])
    try:
        with tempfile.TemporaryDirectory(prefix='wm_v2_native_tails_') as directory:
            paths=capture_frames(args,out,observations,edges,runner,verifier,hook,drafter,Path(directory))
            for f in range(5):
                folder=out/f'fold_{f}'/'film';folder.mkdir(parents=True,exist_ok=True)
                split=json.loads((folder.parent/'split.json').read_text())
                model,dynamics=load_fold(folder.parent/'full_for_film.pt',args.device)
                condition=conditioning_for_edges(model,dynamics,observations,edges,memory,args.device)
                adapters={}
                for arm in ARMS:
                    adapters[arm]=train_adapter(args,folder,drafter,paths,condition,
                        split['train'],split['dev'],arm)
                    test_rows=[]
                    with torch.no_grad():
                        for uid,path in paths.items():
                            frame=torch.load(path,map_location='cpu',weights_only=False)
                            if frame['question'] not in split['test'] or uid not in condition:continue
                            loss,details=frame_loss(adapters[arm],drafter,frame,condition[uid],
                                'head' if arm.startswith('head') else 'late',args.device,arm.endswith('_kl'))
                            test_rows.append(dict(state_id=uid,question=frame['question'],arm=arm,
                                proxy_loss=float(loss),**details))
                    append_jsonl(folder/f'{arm}_heldout_proxy.jsonl',test_rows)
                # Select between proxy-best and the zero-residual/off checkpoint
                # by INNER-DEV real-verifier acceptance, never by outer test.
                dev_args=SimpleNamespace(**vars(args));dev_args.paired_max_tokens=24
                real_dev=paired_native_eval(dev_args,folder/'real_dev_selection',model,dynamics,adapters,
                    observations,memory,split['dev'][:2],environment,verifier,hook)
                selection={}
                from run_persistent_world_model_v2 import snapshot
                for arm,adapter in adapters.items():
                    gain=real_dev.get(arm,{}).get('question_macro_delta_accepted')
                    use_trained=gain is not None and gain>0
                    chosen=snapshot(adapter) if use_trained else snapshot(GatedFiLMV2(drafter.config.hidden_size))
                    torch.save(dict(adapter=chosen,hidden_dim=drafter.config.hidden_size,arm=arm,
                        selection='inner_dev_real_acceptance_vs_zero_checkpoint',trained_selected=use_trained,
                        dev_gain=gain),folder/f'{arm}_real_best.pt')
                    selection[arm]=dict(trained_selected=use_trained,dev_gain=gain)
                atomic_json(folder/'real_checkpoint_selection.json',selection)
                # Keep raw trained-arm comparison so a guarded zero checkpoint
                # cannot hide whether the architectural intervention worked.
                result=paired_native_eval(args,folder,model,dynamics,adapters,observations,memory,
                    split['test'][:args.real_questions_per_fold],environment,verifier,hook)
                summary['folds'].append(dict(fold=f,real=result,real_dev_selection=selection,
                    checkpoint_selection='inner_dev_real: choose proxy-best vs zero; raw heldout arms also reported'))
                atomic_json(out/'film_summary.json',summary)
                root=json.loads((out/'summary.json').read_text());root['film']=summary;package(out,root)
                del model,dynamics,condition,adapters;gc.collect();torch.cuda.empty_cache()
        summary.update(status='complete',native_tail_examples=len(paths),
            injection='before final decoder block, prefix KV immutable',
            objective='keep accepted + fix first reject + 0.05 KL + 0.01 relative shift',
            checkpoint_selection='inner-dev real-verifier guard between proxy-best and zero residual; outer test never selects',
            fixed_schedule='root8, then R/E alternating to64 or EOS; unavailable R skipped, matched pairs only',
            timing_warning='native replay wall time is NOT production action latency',
            real_verifier='Qwen2.5-7B-Instruct FP16, full context, use_cache=False, memory-sharded 2GPU',
            current_teacher_never_conditions_same_proposal=True)
        pooled=[]
        for f in range(5):pooled+=read_jsonl(out/f'fold_{f}'/'film'/'real_pairs.jsonl')
        summary['pooled_oof_real']={arm:paired_summary([r for r in pooled if r['arm']==arm],args.seed) for arm in ARMS}
        atomic_json(out/'film_summary.json',summary)
        return summary
    finally:
        hook.close();del environment,runner,verifier,target,drafter;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
