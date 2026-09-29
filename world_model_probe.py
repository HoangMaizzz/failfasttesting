"""Two-stream native/agreement probe. No verifier answers enter the encoder."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from world_model_core import Latent, ActionDynamics, positions, contextualize, transformer, valid_positions


@dataclass
class ProbeLatent(Latent):
    mask_probs: torch.Tensor
    frontier: torch.Tensor
    carry_logits: torch.Tensor | None = None
    carry_valid: torch.Tensor | None = None

    def take(self, indices):
        return ProbeLatent(*(None if x is None else x.index_select(0, indices) for x in (
            self.tokens, self.global_state, self.lengths, self.context,
            self.mask_probs, self.frontier, self.carry_logits, self.carry_valid)))


class ProbeEncoder(nn.Module):
    def __init__(self, hidden_dim, token_dim, top_k, dim, num_hidden_layers, dropout):
        super().__init__()
        self.dim = dim
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim*num_hidden_layers, dim//2)
        self.token_projection = nn.Linear(token_dim, 32)
        self.gaps = nn.Linear(top_k, 32)
        self.fuse = nn.Linear(dim//2+96+32+16+4, dim)
        self.prefix_projection = nn.Linear(token_dim, dim)
        self.prefix_queries = nn.Parameter(torch.randn(8, dim)*.02)
        self.prefix_pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.context = nn.Linear(8, dim)
        self.network = transformer(dim, 2, dropout)

    def candidate_features(self, b):
        return self.token_projection(b['topk_vectors'])

    def forward(self, b):
        # Explicit allowlist: labels/teacher_margin/teacher_valid never read here.
        h = self.hidden_projection(self.hidden_norm(b["hidden"]).flatten(2))
        h = h*b["scalars"][:, :, 3, None]
        tokens = self.token_projection(b["token_vectors"]).flatten(2)
        alternatives = self.candidate_features(b)*b["scalars"][:, :, 4, None]
        x = self.fuse(torch.cat([h,tokens,alternatives,self.gaps(b["gaps"].clamp(-40,0)/10),
                                 b["scalars"],b["history"]],dim=-1))
        prefix = self.prefix_projection(b["prefix_vectors"])
        prefix = prefix+positions(prefix.shape[1], self.dim, prefix.device)
        memory,_ = self.prefix_pool(self.prefix_queries[None].expand(len(x),-1,-1),prefix,prefix,
            key_padding_mask=~valid_positions(b["prefix_lengths"],prefix.shape[1]),need_weights=False)
        valid = valid_positions(b["lengths"],x.shape[1])
        global_state = self.context(b["context"])
        sequence = torch.cat([global_state[:,None],memory,x+positions(x.shape[1],self.dim,x.device)],1)
        padding = torch.cat([torch.zeros(len(x),9,dtype=torch.bool,device=x.device),~valid],1)
        encoded = self.network(sequence,src_key_padding_mask=padding)
        masks = b["scalars"][:,:,0]*valid
        first = torch.where(masks.bool(),torch.arange(x.shape[1],device=x.device),x.shape[1]).min(-1).values
        return ProbeLatent(encoded[:,9:]*valid[:,:,None],encoded[:,0],b["lengths"],b["context"],
                           masks,torch.minimum(first,b["lengths"]))


class ProbeWorldModel(nn.Module):
    schema = "acceptance_probe_dual_v1"

    def __init__(self, hidden_dim, token_dim, top_k=32, dim=128, num_hidden_layers=3, dropout=.1,
                 architecture="token_dual"):
        super().__init__()
        if architecture != "token_dual": raise ValueError("Wrong architecture for ProbeWorldModel")
        if dim%8: raise ValueError("Dual latent dimension must be divisible by 8")
        self.config = dict(hidden_dim=hidden_dim,token_dim=token_dim,top_k=top_k,dim=dim,
                           num_hidden_layers=num_hidden_layers,dropout=dropout,architecture="token_dual")
        self.encoder = ProbeEncoder(hidden_dim,token_dim,top_k,dim,num_hidden_layers,dropout)
        self.dynamics = ActionDynamics(dim,2,dropout)
        self.accept_head = nn.Sequential(nn.Linear(dim+dim//2,dim//2),nn.GELU(),nn.Linear(dim//2,1))
        self.margin_head = nn.Linear(dim//2,1)
        self.mask_head = nn.Linear(dim//2,1)
        self.candidate_head = nn.Linear(dim,32)
        # Frozen projection of frozen pretrained embeddings: cannot collapse by
        # jointly changing the target projection to satisfy reconstruction loss.
        rng = torch.Generator().manual_seed(8721)
        self.register_buffer("candidate_target_projection",torch.randn(token_dim,32,generator=rng)/token_dim**.5)

    def acceptance(self,z):
        agreement = z.tokens[:,:,z.tokens.shape[-1]//2:]
        logits = self.accept_head(torch.cat([agreement,z.global_state[:,None].expand(-1,agreement.shape[1],-1)],-1)).squeeze(-1)
        if z.carry_logits is not None:
            logits = torch.where(z.carry_valid,z.carry_logits,logits)
        return logits

    def transition(self,z,actions,extension_size):
        old_logits = self.acceptance(z)
        prediction = self.dynamics(z,actions,extension_size)
        width = prediction.tokens.shape[1]
        old_masks = F.pad(z.mask_probs,(0,max(0,width-z.mask_probs.shape[1])))[:,:width]
        new_mask_p = torch.sigmoid(self.mask_head(prediction.tokens[:,:,:prediction.tokens.shape[-1]//2]).squeeze(-1))
        positions_ = torch.arange(width,device=actions.device)[None]
        old_region = positions_ < z.lengths[:,None]
        mask = torch.where(actions[:,None].bool(),new_mask_p*~old_region,old_masks*new_mask_p)
        mask = mask*valid_positions(prediction.lengths,width)
        # Conservative R frontier; never use true child masks to choose carry.
        frontier = torch.where(actions.bool(),z.lengths,z.frontier)
        carry_valid = positions_ < frontier[:,None]
        carry = F.pad(old_logits,(0,max(0,width-old_logits.shape[1])))[:,:width]
        return ProbeLatent(prediction.tokens,prediction.global_state,prediction.lengths,prediction.context,
                           mask,frontier,carry,carry_valid)

    @staticmethod
    def region_mean(values,region):
        counts = region.sum(-1)
        per_row = (values*region).sum(-1)/counts.clamp_min(1)
        return (per_row*(counts>0)).sum()/(counts>0).sum().clamp_min(1)

    def teacher_loss(self,z,b):
        pred = torch.tanh(self.margin_head(z.tokens[:,:,z.tokens.shape[-1]//2:]).squeeze(-1))
        target = torch.tanh(b["teacher_margin"].detach()/5)
        loss = F.smooth_l1_loss(pred,target,reduction="none")
        return self.region_mean(loss,b["teacher_valid"])

    def structural_loss(self,z,b):
        valid = valid_positions(z.lengths,z.tokens.shape[1])
        active = valid & (torch.arange(z.tokens.shape[1],device=z.tokens.device)[None]>=z.context[:,2,None]*64)
        mask_error = F.binary_cross_entropy(z.mask_probs.clamp(1e-6,1-1e-6),b["scalars"][:,:,0],reduction="none")
        target = b["token_vectors"][:,:,1].detach() @ self.candidate_target_projection
        error = 1-F.cosine_similarity(self.candidate_head(z.tokens),target,dim=-1)
        return self.region_mean(mask_error+error,active)
