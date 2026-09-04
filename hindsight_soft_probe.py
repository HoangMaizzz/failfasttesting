import math


def soft_probe_probability(prefix_length, remaining_masks, proposal_length):
    """Return structural probe probability from post-commit proposal state."""
    k, m, length = int(prefix_length), int(remaining_masks), int(proposal_length)
    if length < 1 or not 0 <= k <= length or not 0 <= m <= length:
        raise ValueError("invalid soft-probe proposal state")
    if k + m > length:
        raise ValueError("prefix and unresolved masks overlap")
    ratio = min(1.0, m / max(1, length - 1))
    immaturity = math.exp(-k / 1.5) * ratio ** 1.5
    return min(0.24, max(0.08, 0.08 + 0.16 * immaturity)), immaturity
