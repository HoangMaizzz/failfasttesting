"""D-only state compression, structured R/E simulators, and independent V models.

No LLM is constructed. Token codes are fixed train-vocabulary identity codes,
NOT pretrained semantic embeddings. Their decoder is an explicit student token
head; held-out OOV tokens count as incorrect and coverage is always reported.
The learned LN/linear hidden compressor is trained ONLY for D reconstruction and
then frozen, so dynamics cannot shrink or move its teacher coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from world_model_core import valid_positions, positions, acceptance_nll


@dataclass(frozen=True)
class Layout:
    hidden_dim: int
    layers: int = 3
    code_dim: int = 32

    @property
    def h(self): return self.hidden_dim * self.layers
    @property
    def surface(self): return slice(self.h, self.h + 36)
    @property
    def structure(self): return slice(self.h + 36, self.h + 52)
    @property
    def native(self): return slice(self.h + 52, self.h + 52 + self.code_dim)
    @property
    def stop(self): return slice(self.h + 52 + self.code_dim, self.size)
    @property
    def size(self): return self.h + 52 + self.code_dim * 2


@dataclass
class State:
    x: torch.Tensor                 # B,64,F; ONLY current/imagined D information
    lengths: torch.Tensor           # B
    context: torch.Tensor           # B,8; native observable metadata
    prefix: torch.Tensor            # B,64; fixed token-code prefix summaries

    def to(self, device):
        return State(*(v.to(device) for v in self.__dict__.values()))


class HiddenCompressor(nn.Module):
    def __init__(self, raw_dim=1536, layers=3, dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(raw_dim, elementwise_affine=False)
        self.encoders = nn.ModuleList(nn.Linear(raw_dim, dim) for _ in range(layers))
        self.decoders = nn.ModuleList(nn.Linear(dim, raw_dim) for _ in range(layers))

    def forward(self, raw):
        h = self.norm(raw.float())
        return torch.stack([e(h[:, i]) for i, e in enumerate(self.encoders)], 1)

    def reconstruction_loss(self, raw):
        h = self.norm(raw.float()); z = self(raw)
        reconstruction = torch.stack([d(z[:, i]) for i, d in enumerate(self.decoders)], 1)
        # Reconstruction anchors the representation; no verifier supervision.
        return F.mse_loss(reconstruction, h) + .001 * z.square().mean()


class Preprocessor:
    """Fit exclusively on train questions. All future teachers use frozen maps."""
    def __init__(self, layout, vocabulary, codes, compressor, mean=None, std=None):
        self.layout = layout
        self.vocabulary = torch.as_tensor(vocabulary, dtype=torch.long)
        self.codes = torch.as_tensor(codes, dtype=torch.float32)
        self.compressor = compressor.cpu().eval().requires_grad_(False)
        self.index = {int(t): i for i, t in enumerate(self.vocabulary.tolist())}
        self.mean = torch.zeros(layout.h + 36) if mean is None else mean
        self.std = torch.ones(layout.h + 36) if std is None else std

    def classes(self, ids):
        return torch.tensor([self.index.get(int(t), 0) for t in ids.flatten()], dtype=torch.long).reshape(ids.shape)

    def token_codes(self, ids): return self.codes[self.classes(ids)]

    def save_payload(self):
        return dict(layout=self.layout.__dict__, vocabulary=self.vocabulary, codes=self.codes,
                    compressor={k: v.cpu() for k, v in self.compressor.state_dict().items()},
                    raw_dim=self.compressor.encoders[0].in_features, mean=self.mean, std=self.std)

    @classmethod
    def restore(cls, data):
        layout = Layout(**data['layout'])
        compressor = HiddenCompressor(data['raw_dim'], layout.layers, layout.hidden_dim)
        compressor.load_state_dict(data['compressor'])
        return cls(layout, data['vocabulary'], data['codes'], compressor, data['mean'], data['std'])

    @classmethod
    def fit(cls, states, question_ids, dim, updates, device, seed, progress=None):
        selected = [o for o in states.values() if o.question in set(question_ids)]
        if not selected: raise ValueError('No training questions for preprocessing')
        torch.manual_seed(seed); rng = random.Random(seed)
        layers, raw_dim = selected[0].hidden.shape[1:]
        layout = Layout(dim, layers)
        vocabulary = sorted({int(t) for o in selected for t in o.ids.flatten().tolist()} |
                            {int(t) for o in selected for t in o.topk_ids.flatten().tolist()} | {151665})
        # -1 is UNK, never misreported as an exact reconstruction of an OOV ID.
        vocabulary = [-1] + [t for t in vocabulary if t >= 0]
        generator = torch.Generator().manual_seed(901)
        codes = F.normalize(torch.randn(len(vocabulary), layout.code_dim, generator=generator), dim=-1)
        model = HiddenCompressor(raw_dim, layers, dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        valid = [(o, torch.where(o.scalars[:, 3] > 0)[0].tolist()) for o in selected]
        valid = [(o, pp) for o, pp in valid if pp]
        if not valid: raise ValueError('No valid native hidden rows')
        for step in range(1, updates + 1):
            sample = [rng.choice(valid) for _ in range(128)]
            raw = torch.stack([o.hidden[rng.choice(pp)] for o, pp in sample]).to(device)
            loss = model.reconstruction_loss(raw)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            if progress and (step == updates or step % 100 == 0):
                progress(dict(stage='D_only_compression', step=step, loss=float(loss.detach())))
        result = cls(layout, vocabulary, codes, model)
        # Train-only scaler, sampled by question rather than by long trajectory.
        observations = result.prepare({o.uid: o for o in selected}, device)
        values = []
        groups = {}
        for o in selected: groups.setdefault(o.question, []).append(o)
        for q in sorted(groups):
            for o in rng.sample(groups[q], min(20, len(groups[q]))):
                x = observations[o.uid]['x']; valid_h = o.scalars[:, 3] > 0
                if valid_h.any(): values.append(x[valid_h, :layout.h + 36])
        rows = torch.cat(values)
        result.mean = rows.mean(0); result.std = rows.std(0, unbiased=False).clamp_min(.05)
        return result

    @torch.no_grad()
    def prepare(self, raw_states, device='cpu'):
        result = {}; l = self.layout
        model = self.compressor.to(device)
        for uid, o in raw_states.items():
            if o.length > 64: raise ValueError('Proposal exceeds 64')
            h = model(o.hidden.to(device)).flatten(1).cpu()
            surface = o.surface.float().clone()
            surface[:, :32] = surface[:, :32].clamp(-40, 0) / 10
            surface[:, 32:] = surface[:, 32:].clamp(-4, 4)
            core = (torch.cat([h, surface], -1) - self.mean) / self.std
            core[:, :l.h] *= (o.scalars[:, 3] > 0)[:, None]
            core[:, l.h:l.h + 32] *= (o.scalars[:, 4] > 0)[:, None]
            x = torch.cat([core, o.scalars.float(), self.token_codes(o.ids[:, 0]),
                           self.token_codes(o.ids[:, 1])], -1)
            prefix = self.token_codes(o.prefix_ids)
            p = torch.cat([prefix.mean(0), prefix[-32:].mean(0)]) if len(prefix) else torch.zeros(64)
            result[uid] = dict(x=x, context=o.context.float(), prefix=p,
                length=o.length, token_targets=self.classes(o.ids[:, 1]),
                topk_classes=self.classes(o.topk_ids), topk_gaps=o.gaps.float(),
                raw_stop_ids=o.ids[:, 1].clone(), raw_topk_ids=o.topk_ids.clone(),
                accepted=o.accepted, teacher=o.teacher_features, margin=o.teacher_margin,
                question=o.question, uid=uid)
        model.cpu()
        return result


def pack(rows, device):
    return State(torch.stack([F.pad(r['x'], (0, 0, 0, 64-r['length'])) for r in rows]).to(device),
                 torch.tensor([r['length'] for r in rows], device=device),
                 torch.stack([r['context'] for r in rows]).to(device),
                 torch.stack([r['prefix'] for r in rows]).to(device))


def feature_view(state, layout, representation):
    x = state.x.clone()
    if representation not in ('S', 'H', 'SH', 'SHT'): raise ValueError(representation)
    if representation == 'S': x[..., :layout.h] = 0
    if representation == 'H': x[..., layout.surface] = 0
    if representation != 'SHT': x[..., layout.native] = 0; x[..., layout.stop] = 0
    return x


class TokenDecoder(nn.Module):
    def __init__(self, prep):
        super().__init__(); self.layout = prep.layout
        self.register_buffer('codes', prep.codes.clone())
        self.register_buffer('vocabulary', prep.vocabulary.clone())

    def logits(self, x):
        return F.normalize(x[..., self.layout.stop], dim=-1) @ self.codes.T * 20

    def ids(self, x, k=1):
        return self.vocabulary[self.logits(x).topk(min(k, len(self.vocabulary)), -1).indices]


class NativeTopKDecoder(nn.Module):
    """Frozen D-only readout for native top-k; never a verifier head.

    This is an explicit additional decoding bottleneck. Measure its floor on
    real child states as well as imagined states. Saved gaps only identify a
    CONDITIONAL top-32 distribution, not full-vocabulary probabilities.
    """
    def __init__(self, prep):
        super().__init__()
        self.register_buffer('codes', prep.codes.clone())
        self.register_buffer('vocabulary', prep.vocabulary.clone())
        self.network = nn.Sequential(nn.Linear(prep.layout.size, 128), nn.GELU(),
                                     nn.Linear(128, prep.layout.code_dim))

    def logits(self, x):
        return F.normalize(self.network(x), dim=-1) @ self.codes.T * 20

    def ids(self, x, k=10):
        return self.vocabulary[self.logits(x).topk(min(k, len(self.vocabulary)), -1).indices]


def materialize(state, actions, layout, prior, extend_size=8):
    """Only source + known action. C(X) commits SAME-forward STOP before E."""
    if bool(((actions != 0) & (actions != 1)).any()): raise ValueError('R/E action required')
    lengths = state.lengths + actions * extend_size
    if bool((lengths > 64).any()): raise ValueError('E would exceed 64 tokens')
    x = state.x.clone(); old = valid_positions(state.lengths, 64)
    pos = torch.arange(64, device=x.device)[None]
    new = (pos >= state.lengths[:, None]) & (pos < lengths[:, None])
    context = state.context.clone()
    context[:, 1] = lengths.float() / 64
    context[:, 2] = torch.where(actions.bool(), state.lengths.float()/64, context[:, 2])
    context[:, 3] = torch.where(actions.bool(), 0., context[:, 3]+1/3)
    for i in range(len(x)):
        if actions[i]:
            n = int(state.lengths[i]); x[i, n:n+extend_size] = prior.to(x.device)
            x[i, :n, layout.native] = state.x[i, :n, layout.stop]
            s = x[i, :n, layout.structure]
            s[:, 0] = 0; s[:, 1] = 1; s[:, 2] = 0; s[:, 12] = 0; s[:, 14] = 0
    s = x[..., layout.structure]
    s[..., 5:7] += old[..., None] / 8
    s[..., 9] = pos / lengths[:, None]
    absolute = pos + context[:, 0, None] * 1024
    physical = (context[:, 5] * 64).clamp_min(1)
    small = (context[:, 6] * 64).clamp_min(1)
    s[..., 10] = absolute.remainder(physical[:, None]) / physical[:, None]
    s[..., 11] = absolute.remainder(small[:, None]) / small[:, None]
    s[..., 12] = (pos >= context[:, 2, None]*64).float()
    s[..., 15] = context[:, 4, None]
    x *= valid_positions(lengths, 64)[..., None]
    return State(x, lengths, context, state.prefix.clone()), new


def mean_extension_prior(prepared, paths, layout):
    samples = [prepared[nodes[-1]]['x'][prepared[nodes[0]]['length']:] for nodes, actions in paths if actions == (1,)]
    if not samples: raise ValueError('No train E edges to fit prior')
    return torch.stack(samples).mean(0)


def transition_history(x, source, layout, mean, std, decoder, update):
    """History is transition bookkeeping, NOT immutable cached hidden/logits.

    Full-raw hidden cosine cannot be reconstructed exactly from a lossy hidden
    projection: its frontier change remains a supervised D prediction. For
    copied prefix rows the hidden change is deterministically zero.
    """
    hist=slice(layout.h+32,layout.h+36)
    raw=x[...,hist]*std+mean
    old=valid_positions(source.lengths,64)
    s=x[...,layout.structure]; parent=source.x[...,layout.structure]
    valid=(s[...,3]>0)&(parent[...,3]>0)&old
    changed=torch.zeros_like(s[...,0])
    where=update&old
    with torch.no_grad():
        if where.any():
            changed[where]=(decoder.ids(x[where],1).flatten()!=decoder.ids(source.x[where],1).flatten()).float()
    history=torch.stack([torch.where(old,s[...,7]-parent[...,7],0),changed,
        torch.where(where&valid,raw[...,2].clamp(0,2),0),valid.float()],-1)
    history=(history-mean)/std
    surface=torch.cat([x[...,layout.surface][...,:32],history],-1)
    return torch.cat([x[...,:layout.h],surface,x[...,layout.structure],x[...,layout.native],x[...,layout.stop]],-1)*(old|update)[...,None]


class StructuredDynamics(nn.Module):
    def __init__(self, prep, prior, kind='transformer', representation='SHT', width=128,
                 layers=2, hidden_gate=True, dropout=.05, frontier_gate=True):
        super().__init__(); self.layout = prep.layout; self.kind = kind
        self.representation = representation; self.hidden_gate_enabled = hidden_gate
        self.frontier_gate = frontier_gate
        self.register_buffer('prior', prior.clone())
        self.register_buffer('mask_code', prep.token_codes(torch.tensor([151665]))[0])
        self.register_buffer('history_mean',prep.mean[-4:].clone())
        self.register_buffer('history_std',prep.std[-4:].clone())
        self.decoder = TokenDecoder(prep)
        self.input = nn.Linear(self.layout.size, width)
        self.global_input = nn.Linear(72, width)
        self.action = nn.Embedding(2, width)
        self.new_queries = nn.Parameter(torch.randn(8, width)*.02)
        self.hidden_gate = nn.Linear(self.layout.size, self.layout.layers)
        if kind == 'transformer':
            block = nn.TransformerEncoderLayer(width, 4, width*4, dropout, activation='gelu', batch_first=True, norm_first=True)
            self.network = nn.TransformerEncoder(block, layers, nn.LayerNorm(width), enable_nested_tensor=False)
            self.cross = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        elif kind == 'mlp':
            self.network = nn.Sequential(nn.Linear(width*4, width*2), nn.GELU(), nn.LayerNorm(width*2),
                                         nn.Linear(width*2, width), nn.GELU())
        else: raise ValueError(kind)
        self.output = nn.Linear(width, self.layout.size)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.mask_head = nn.Linear(width, 1); nn.init.zeros_(self.mask_head.weight); nn.init.constant_(self.mask_head.bias, 3)
        self.residual_gate = nn.Linear(width, 1); nn.init.constant_(self.residual_gate.bias, -1)
        self.last_gates = None

    def forward(self, state, actions):
        l = self.layout; base, new = materialize(state, actions, l, self.prior)
        viewed = feature_view(state, l, self.representation)
        gates = self.hidden_gate(viewed).sigmoid() if self.hidden_gate_enabled else torch.ones(*viewed.shape[:2], l.layers, device=viewed.device)
        viewed = torch.cat([(viewed[..., :l.h].reshape(*viewed.shape[:2], l.layers, l.hidden_dim)*gates[..., None]).flatten(2), viewed[..., l.h:]], -1)
        self.last_gates = gates.detach()
        valid = valid_positions(state.lengths, 64)
        source = self.input(viewed) + positions(64, self.input.out_features, viewed.device)[None]
        prefix = state.prefix if self.representation == 'SHT' else torch.zeros_like(state.prefix)
        global_row = self.global_input(torch.cat([state.context, prefix], -1)) + self.action(actions)
        if self.kind == 'transformer':
            encoded = self.network(source + global_row[:, None], src_key_padding_mask=~valid)
            queries = self.new_queries[None].expand(len(viewed), -1, -1) + global_row[:, None]
            future, _ = self.cross(queries, encoded, encoded, key_padding_mask=~valid, need_weights=False)
        else:
            pooled = (source*valid[..., None]).sum(1)/valid.sum(1, keepdim=True)
            left = F.pad(source[:, :-1], (0, 0, 1, 0)); right = F.pad(source[:, 1:], (0, 0, 0, 1))
            encoded = self.network(torch.cat([source, left, right, (pooled+global_row)[:, None].expand_as(source)], -1))
            q = self.new_queries[None].expand(len(viewed), -1, -1)
            future = self.network(torch.cat([q, pooled[:, None].expand_as(q), global_row[:, None].expand_as(q), q*0], -1))
        latent = encoded.clone()
        for i in range(len(viewed)):
            if actions[i]: latent[i, int(state.lengths[i]):int(base.lengths[i])] = future[i]
        delta = self.output(latent)
        pos = torch.arange(64, device=viewed.device)[None]
        active = (pos >= state.context[:, 2, None]*64) & valid
        update = torch.where(actions[:, None].bool(), new, active)
        gate = self.residual_gate(latent).sigmoid() if self.frontier_gate else torch.ones_like(delta[..., :1])
        x = base.x + delta*gate*update[..., None]
        # Known structural metadata is not unconstrained neural output.
        s = base.x[..., l.structure].clone()
        old_mask = state.x[..., l.structure][..., 0].clamp(0, 1)
        predicted_mask = self.mask_head(latent).squeeze(-1).sigmoid()
        remaining = torch.where(actions[:, None].bool(), predicted_mask*new, old_mask*predicted_mask)
        remaining = torch.where(update, remaining, s[..., 0]).clamp(0, 1)
        s[..., 0] = remaining; s[..., 1] = 1-remaining
        s[..., 2] = torch.where(actions[:,None].bool(), (1-remaining)*new, (old_mask-remaining).clamp_min(0))
        s[..., 3:5] = torch.where(update[..., None], torch.ones_like(s[..., 3:5]), s[..., 3:5])
        s[..., 5:7] = torch.where(update[..., None], torch.zeros_like(s[..., 5:7]), s[..., 5:7])
        s[...,7]=torch.where(update,(base.x[...,l.structure][...,7]+delta[...,l.structure][...,7]*gate.squeeze(-1)).clamp(0,1),s[...,7])
        s[...,8]=torch.where(update,torch.ones_like(s[...,8]),s[...,8])
        s[..., 14] = remaining
        immutable_content = valid & ((old_mask <= 0) | actions[:, None].bool())
        stop = torch.where(immutable_content[..., None], state.x[..., l.stop], x[..., l.stop])
        native = remaining[..., None]*self.mask_code + (1-remaining[..., None])*stop
        native = torch.where(immutable_content[..., None], base.x[..., l.native], native)
        x = torch.cat([x[..., :l.h], x[..., l.surface], s, native, stop], -1)
        x = transition_history(x,state,l,self.history_mean,self.history_std,self.decoder,update)
        x = x * valid_positions(base.lengths, 64)[..., None]
        return replace(base, x=x)


class LinearDynamics(nn.Module):
    """Structured Ridge: local residual R, pooled source -> eight E rows."""
    def __init__(self, prep, prior):
        super().__init__(); self.layout = prep.layout
        self.register_buffer('prior', prior); self.decoder = TokenDecoder(prep)
        self.register_buffer('history_mean',prep.mean[-4:].clone())
        self.register_buffer('history_std',prep.std[-4:].clone())
        self.r_weight = None; self.e_weight = None

    def design(self, state):
        x = state.x; valid = valid_positions(state.lengths, 64)
        mean = (x*valid[..., None]).sum(1)/valid.sum(1, keepdim=True)
        left = F.pad(x[:, :-1], (0, 0, 1, 0)); right = F.pad(x[:, 1:], (0, 0, 0, 1))
        context = torch.cat([state.context, state.prefix], -1)
        local = torch.cat([x, left, right, mean[:, None].expand_as(x), context[:, None].expand(-1, 64, -1), x[..., :1]*0+1], -1)
        global_row = torch.cat([mean, x[:, 0], x[torch.arange(len(x)), state.lengths-1], context, torch.ones(len(x), 1, device=x.device)], -1)
        return local, global_row

    @staticmethod
    def ridge(x, y, alpha):
        x = x.double(); y = y.double()
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype); eye[-1, -1] = 0
        return torch.linalg.solve(x.T@x + alpha*eye + 1e-6*torch.eye(len(eye), device=x.device), x.T@y).float()

    def fit(self, prepared, paths, device, seed=42, max_rows=12000, alpha=10):
        rng = random.Random(seed); pairs = [p for p in paths if len(p[1]) == 1]
        rng.shuffle(pairs); rx=[]; ry=[]; ex=[]; ey=[]; nr=0
        for nodes, actions in pairs:
            if len(ex) >= 1800 and nr >= max_rows: break
            parent, child = prepared[nodes[0]], prepared[nodes[1]]
            state = pack([parent], device); a = torch.tensor(actions, device=device)
            base, _ = materialize(state, a, self.layout, self.prior)
            local, glob = self.design(state)
            if actions[0] == 0 and nr < max_rows:
                start = round(float(parent['context'][2])*64); n=parent['length']
                rx.append(local[0, start:n].cpu()); ry.append((child['x'][start:n]-base.x[0, start:n].cpu()))
                nr += n-start
            elif actions[0] == 1 and len(ex) < 1800:
                ex.append(glob[0].cpu()); ey.append(child['x'][parent['length']:].flatten())
        if not rx or not ex: raise ValueError('Ridge needs R and E training edges')
        self.r_weight = self.ridge(torch.cat(rx).to(device), torch.cat(ry).to(device), alpha).cpu()
        self.e_weight = self.ridge(torch.stack(ex).to(device), torch.stack(ey).to(device), alpha).cpu()
        return self

    def forward(self, state, actions):
        l=self.layout; base, new=materialize(state, actions, l, self.prior)
        local, glob=self.design(state); x=base.x.clone()
        r=local@self.r_weight.to(x.device); e=(glob@self.e_weight.to(x.device)).reshape(-1,8,l.size)
        pos=torch.arange(64,device=x.device)[None]; active=(pos>=state.context[:,2,None]*64)&valid_positions(state.lengths,64)
        x += r*active[...,None]*(actions==0)[:,None,None]
        for i in range(len(x)):
            if actions[i]:x[i,int(state.lengths[i]):int(base.lengths[i])]=e[i]
        s=x[...,l.structure].clone(); known=base.x[...,l.structure]
        remaining=x[...,l.structure][...,0].clamp(0,1)
        remaining=torch.where(actions[:,None].bool()&~new,0,torch.minimum(remaining,state.x[...,l.structure][...,0].clamp(0,1)+new))
        s[...,0]=remaining
        for j in (9,10,11,12,13,15):s[...,j]=known[...,j]
        s[...,1]=1-remaining
        update=torch.where(actions[:,None].bool(),new,active)
        s[...,2]=torch.where(actions[:,None].bool(),(1-remaining)*new,(state.x[...,l.structure][...,0]-remaining).clamp_min(0))
        s[...,3:5]=torch.where(update[...,None],1,s[...,3:5])
        s[...,5:7]=torch.where(update[...,None],0,s[...,5:7])
        s[...,7]=x[...,l.structure][...,7].clamp(0,1)
        s[...,8]=torch.where(update,1,s[...,8])
        immutable=valid_positions(state.lengths,64)&((state.x[...,l.structure][...,0]<=0)|actions[:,None].bool())
        stop=torch.where(immutable[...,None],state.x[...,l.stop],x[...,l.stop])
        mask_code=self.decoder.codes[(self.decoder.vocabulary==151665).nonzero()[0,0]]
        native=remaining[...,None]*mask_code+(1-remaining[...,None])*stop
        native=torch.where(immutable[...,None],base.x[...,l.native],native)
        x=torch.cat([x[...,:l.h],x[...,l.surface],s,native,stop],-1)
        x=transition_history(x,state,l,self.history_mean,self.history_std,self.decoder,update)
        x = x * valid_positions(base.lengths,64)[...,None]
        return replace(base,x=x)


def drafter_loss(prediction, target_rows, source, actions, decoder, layout):
    """Targets are D-only; no accepted length or verifier field is accessed."""
    target=pack(target_rows,prediction.x.device);x=prediction.x;y=target.x
    pos=torch.arange(64,device=x.device)[None]
    old=valid_positions(source.lengths,64); new=(pos>=source.lengths[:,None])&valid_positions(target.lengths,64)
    active=(pos>=source.context[:,2,None]*64)&valid_positions(target.lengths,64)
    s=y[...,layout.structure]; fresh_h=active&(s[...,3]>0)&(s[...,5]<1e-5)
    fresh_l=active&(s[...,4]>0)&(s[...,6]<1e-5)
    def masked(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)
    hidden=masked((x[...,:layout.h]-y[...,:layout.h]).square().mean(-1),fresh_h)
    cosine=masked(1-F.cosine_similarity(x[...,:layout.h],y[...,:layout.h],dim=-1),fresh_h)
    surface=masked(F.smooth_l1_loss(x[...,layout.surface],y[...,layout.surface],reduction='none').mean(-1),fresh_l)
    mask=masked((x[...,layout.structure][...,0]-s[...,0]).square(),active)
    confidence=masked((x[...,layout.structure][...,7]-s[...,7]).square(),fresh_l&(s[...,8]>0))
    code=masked((x[...,layout.stop]-y[...,layout.stop]).square().mean(-1),active)
    selected=torch.where(active.flatten())[0]
    # Bounded decoder batch, no enormous B*64*full-vocabulary activations.
    if len(selected)>32:
        selected=selected[torch.randperm(len(selected),device=x.device)[:32]] if x.requires_grad else selected[:32]
    targets=torch.stack([F.pad(r['token_targets'],(0,64-r['length'])) for r in target_rows]).to(x.device)
    token=F.cross_entropy(decoder.logits(x.reshape(-1,layout.size)[selected]),targets.flatten()[selected]) if len(selected) else x.sum()*0
    loss=.25*hidden+.05*cosine+.4*surface+.3*token+.5*mask+code+.3*confidence
    return loss,dict(hidden=float(hidden.detach()),surface=float(surface.detach()),token_ce=float(token.detach()),mask=float(mask.detach()),confidence=float(confidence.detach()))


class VerifierTransformer(nn.Module):
    def __init__(self, layout, representation='SHT', width=128, layers=2, dropout=.05, hidden_gate=True):
        super().__init__();self.layout=layout;self.representation=representation
        self.input=nn.Linear(layout.size,width);self.global_input=nn.Linear(72,width)
        self.gate=nn.Linear(layout.size,layout.layers);self.hidden_gate_enabled=hidden_gate
        block=nn.TransformerEncoderLayer(width,4,width*4,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.network=nn.TransformerEncoder(block,layers,nn.LayerNorm(width),enable_nested_tensor=False)
        self.readout=nn.Linear(width,4);self.last_gates=None

    def forward(self,state):
        x=feature_view(state,self.layout,self.representation)
        g=self.gate(x).sigmoid() if self.hidden_gate_enabled else torch.ones(*x.shape[:2],self.layout.layers,device=x.device)
        x=torch.cat([(x[...,:self.layout.h].reshape(*x.shape[:2],self.layout.layers,self.layout.hidden_dim)*g[...,None]).flatten(2),x[...,self.layout.h:]],-1)
        self.last_gates=g.detach()
        prefix=state.prefix if self.representation=='SHT' else torch.zeros_like(state.prefix)
        h=self.input(x)+positions(64,self.input.out_features,x.device)[None]+self.global_input(torch.cat([state.context,prefix],-1))[:,None]
        # Causal over token slots. Native D hidden may encode suffix: it is available,
        # but causal masking alone does not establish verifier-prefix invariance.
        causal=torch.triu(torch.ones(64,64,dtype=torch.bool,device=x.device),diagonal=1)
        h=self.network(h,mask=causal,src_key_padding_mask=~valid_positions(state.lengths,64))
        out=self.readout(h)
        return dict(hazard=out[...,0],tf=out[...,1],probability=out[...,2].sigmoid(),margin=out[...,3].tanh())


def verifier_loss(heads, rows, device):
    lengths=torch.tensor([r['length'] for r in rows],device=device)
    labels=torch.tensor([-1 if r['accepted'] is None else r['accepted'] for r in rows],device=device)
    loss=acceptance_nll(heads['hazard'],lengths,labels)
    pos=torch.arange(64,device=device)[None]; valid=valid_positions(lengths,64)&(labels>=0)[:,None]
    survival=F.logsigmoid(heads['hazard']).cumsum(-1).exp()
    loss=loss+.5*((survival-(pos<labels[:,None]).float()).square()*valid).sum()/valid.sum().clamp_min(1)
    tf=torch.zeros_like(survival);prob=torch.zeros_like(survival);margin=torch.zeros_like(survival);teacher_valid=torch.zeros_like(valid)
    for i,r in enumerate(rows):
        if r['teacher'] is not None:
            n=r['length']; t=r['teacher'].to(device)
            tf[i,:n]=t[:,2];prob[i,:n]=t[:,1];margin[i,:n]=t[:,0];teacher_valid[i,:n]=True
    if teacher_valid.any():
        loss=loss+.25*F.binary_cross_entropy_with_logits(heads['tf'][teacher_valid],tf[teacher_valid])
        loss=loss+.1*F.mse_loss(heads['probability'][teacher_valid],prob[teacher_valid])
        loss=loss+.1*F.smooth_l1_loss(heads['margin'][teacher_valid],margin[teacher_valid])
    return loss


def tabular_features(state,layout,representation='SHT'):
    x=feature_view(state,layout,representation);valid=valid_positions(state.lengths,64)
    mean=(x*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
    prefix=state.prefix if representation=='SHT' else torch.zeros_like(state.prefix)
    ctx=torch.cat([state.context,prefix],-1)[:,None].expand(-1,64,-1)
    return torch.cat([x,mean[:,None].expand_as(x),ctx],-1).cpu().numpy()


class TabularVerifier:
    """Independent head models. Missing teacher heads are not zero targets."""
    def __init__(self,layout,kind='logistic',representation='SHT',seed=42,max_tokens=50000):
        self.layout=layout;self.kind=kind;self.representation=representation;self.seed=seed;self.max_tokens=max_tokens;self.models={}

    def fit(self,rows):
        from sklearn.linear_model import LogisticRegression,Ridge
        from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
        rows=[r for r in rows if r['accepted'] is not None or r['teacher'] is not None]
        if not any(r['accepted'] is not None for r in rows):raise ValueError('No accepted-length supervision for hazard head')
        rng=np.random.default_rng(self.seed);xx={k:[] for k in ('hazard','tf','probability','margin')};yy={k:[] for k in xx}
        for start in range(0,len(rows),32):
            group=rows[start:start+32];features=tabular_features(pack(group,'cpu'),self.layout,self.representation)
            for i,r in enumerate(group):
                n=r['length'];k=r['accepted']
                if k is not None:
                    m=min(n,k+1);xx['hazard'].append(features[i,:m]);yy['hazard'].append(np.arange(m)<k)
                if r['teacher'] is not None:
                    for name,col in (('tf',2),('probability',1),('margin',0)):
                        xx[name].append(features[i,:n]);yy[name].append(r['teacher'][:,col].numpy())
        for name in xx:
            if not xx[name]:self.models[name]=None;continue
            x=np.concatenate(xx[name]);y=np.concatenate(yy[name]);take=rng.choice(len(x),min(len(x),self.max_tokens),replace=False);x=x[take];y=y[take]
            binary=name in ('hazard','tf')
            if binary and len(np.unique(y))<2:self.models[name]=float(y.mean());continue
            if self.kind=='logistic':model=LogisticRegression(C=.1,max_iter=300) if binary else Ridge(alpha=10)
            else:
                cls=HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
                model=cls(max_iter=160,max_leaf_nodes=15,l2_regularization=10,learning_rate=.05,random_state=self.seed,early_stopping=False)
            model.fit(x,y);self.models[name]=model
        return self

    def __call__(self,state):
        features=tabular_features(state,self.layout,self.representation);shape=features.shape[:2];x=features.reshape(-1,features.shape[-1]);out={}
        for name,model in self.models.items():
            if model is None:
                out[name]=None;continue
            value=np.full(len(x),model) if isinstance(model,float) else (model.predict_proba(x)[:,1] if name in ('hazard','tf') else model.predict(x))
            value=torch.tensor(value.reshape(shape),dtype=torch.float32,device=state.x.device)
            out[name]=torch.logit(value.clamp(1e-6,1-1e-6)) if name in ('hazard','tf') else value.clamp(0,1) if name=='probability' else value.clamp(-1,1)
        return out


class DirectOutcome(nn.Module):
    """Source + known action sequence -> future V heads, with no D supervision."""
    def __init__(self,layout,width=128):
        super().__init__();self.layout=layout
        self.input=nn.Linear(layout.size,width);self.summary=nn.Linear(layout.size+72+6,width)
        self.network=nn.Sequential(nn.Linear(width*2,width*2),nn.GELU(),nn.Linear(width*2,4))

    def forward(self,state,actions):
        valid=valid_positions(state.lengths,64);mean=(state.x*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
        descriptor=F.pad(F.one_hot(actions,num_classes=2).flatten(1).float(),(0,6-actions.shape[1]*2))
        global_row=self.summary(torch.cat([mean,state.context,state.prefix,descriptor],-1))
        h=self.input(state.x)+positions(64,self.input.out_features,state.x.device)[None]
        out=self.network(torch.cat([h,global_row[:,None].expand_as(h)],-1))
        return dict(hazard=out[...,0],tf=out[...,1],probability=out[...,2].sigmoid(),margin=out[...,3].tanh())
