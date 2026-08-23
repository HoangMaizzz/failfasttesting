def prepare_causal_oracle_snapshots(
    frontier_stats,
    factual_proposal,
    physical_draft_passes,
    physical_forward_latency_ms,
):
    stats = frontier_stats or {}
    snapshots = [
        {**snapshot, "candidate_source": "refinement_snapshot"}
        for snapshot in stats.get("oracle_refinement_snapshots") or []
    ]
    if snapshots:
        return snapshots, False

    proposal = [int(token_id) for token_id in factual_proposal or []]
    if not proposal:
        raise RuntimeError(
            "Causal oracle received neither refinement snapshots nor a factual proposal"
        )

    steps = stats.get("steps") or []
    final_step = (
        int(steps[-1].get("step", physical_draft_passes))
        if steps
        else int(physical_draft_passes)
    )
    return [
        {
            "step": final_step,
            "target_len": len(proposal),
            "draft_passes_elapsed": int(physical_draft_passes),
            "draft_latency_elapsed_ms": float(physical_forward_latency_ms),
            "masks_remaining": 0,
            "committed_tokens": len(proposal),
            "filled_tokens": 0,
            "draft_proposal": proposal,
            "candidate_source": "factual_terminal_fallback",
        }
    ], True
