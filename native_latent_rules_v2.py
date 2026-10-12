"""Known native scheduler rules, separate from learned latent dynamics.

Scores here are probabilities of structural commit EVENTS, not dLLM token
probabilities. We enforce native invariants; we cannot reproduce the native
confidence > .5 selection without executing the native vocabulary head.
"""
import torch


BLOCK_SIZE = 32
SUBBLOCK_SIZE = 8


def proposal_span(geometry):
    absolute = geometry[:, 1:2] + torch.arange(BLOCK_SIZE, device=geometry.device)
    return (absolute >= geometry[:, 0:1]) & (absolute < geometry[:, 0:1] + geometry[:, 3:4])


def scheduler_region(mask, geometry, logical_only=False):
    """Next native 8-position region; committed positions stay in its span."""
    eligible = mask >= .5
    if logical_only:
        eligible = eligible & proposal_span(geometry)
    positions = torch.arange(BLOCK_SIZE, device=mask.device)[None]
    first = torch.where(eligible, positions, BLOCK_SIZE).min(-1).values
    start = (first // SUBBLOCK_SIZE) * SUBBLOCK_SIZE
    return (positions >= start[:, None]) & (positions < start[:, None] + SUBBLOCK_SIZE) & (first[:, None] < BLOCK_SIZE)


def next_geometry(geometry, action):
    """E starts the next 8-token segment, using absolute physical boundaries."""
    extended = geometry.clone()
    extended[:, 0] += SUBBLOCK_SIZE
    extended[:, 1] = ((extended[:, 0] + extended[:, 3] - 1) // BLOCK_SIZE) * BLOCK_SIZE
    extended[:, 2] = extended[:, 1]
    extended[:, 4] += 1
    return torch.where(action[:, None].bool(), extended, geometry)


def initial_segment_mask(geometry):
    """Physical input mask of the first exposed unmask (prefix length >32).

    Exact native prefill seeds aligned prefixes. Unaligned prefixes do not
    have a seed; after crossing a physical boundary, cache-update seeds the
    new frame. Earlier proposal rows are already committed by bridge passes.
    """
    start, frame = geometry[:, 0:1], geometry[:, 1:2]
    seeded = ((start.remainder(32) == 0) & (start > 32)) | (frame > start)
    first_mask = torch.where(seeded, torch.maximum(start, frame) + 1, start)
    absolute = frame + torch.arange(BLOCK_SIZE, device=geometry.device)
    return absolute >= first_mask


def align_frame(value, source_geometry, target_geometry):
    """Carry only residual rows with matching absolute positions, not row IDs."""
    positions = target_geometry[:, 1:2] + torch.arange(BLOCK_SIZE, device=value.device)
    source_rows = positions - source_geometry[:, 1:2]
    valid = (source_rows >= 0) & (source_rows < BLOCK_SIZE)
    index = source_rows.clamp(0, BLOCK_SIZE - 1).long()
    aligned = value.gather(1, index[..., None].expand(-1, -1, value.shape[-1]))
    return aligned * valid[..., None], valid


def constrained_commit(scores, before, region):
    """No remask/no out-of-region commit, and at least one commit if eligible.

Uses a straight-through hard decision for training; forward rollout always
uses the exact constrained bits. Argmax chooses a *predicted event*, not a
predicted token or a recreated native confidence.
"""
    eligible = (before >= .5) & region
    soft = scores.sigmoid() * eligible
    hard = (soft > .5) & eligible
    winner = scores.masked_fill(~eligible, -1e9).argmax(-1)
    forced = torch.zeros_like(hard).scatter(1, winner[:, None], True) & eligible
    hard = hard | forced
    event = hard.float() + soft - soft.detach()
    return event, eligible
