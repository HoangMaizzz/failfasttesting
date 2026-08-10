import argparse
import shutil
from pathlib import Path


FRONTIER_INIT = '''        frontier_stats = {
            "enabled": bool(return_frontier_stats),
            "mode": getattr(args, "frontier_stop_mode", "disabled") if args is not None else "disabled",
            "steps": [],
            "stop_reason": None,
            "final_frontier_score": None,
            "actual_spec_len": None,
        }
        committed_confidences = {}
        frontier_scores = []
        frontier_recent_unmasked = []
        frontier_force_stop = False
'''


FRONTIER_STEP = '''                        if is_drafter and return_frontier_stats and lowconf_threshold is not None and draft_token_end_idx <= x_t.shape[1]:
                            frontier_mode = getattr(args, "frontier_stop_mode", "disabled") if args is not None else "disabled"
                            tau_f = float(lowconf_threshold)
                            target_len = int(spec_len)
                            draft_end_idx = draft_token_start_idx + target_len
                            block_abs_start = x_t.shape[1] - block_size
                            current_step_confidences = {}
                            unmasked_this_step = int(unmask_idx.sum().item())

                            for batch_idx, local_idx in unmask_idx.nonzero(as_tuple=False).tolist():
                                if batch_idx != 0:
                                    continue
                                absolute_pos = block_abs_start + small_block_start_idx + local_idx
                                if draft_token_start_idx <= absolute_pos < draft_end_idx:
                                    committed_confidences[int(absolute_pos)] = float(x1_p[batch_idx, local_idx].float().item())

                            for local_idx in range(x1_p.shape[1]):
                                absolute_pos = block_abs_start + small_block_start_idx + local_idx
                                if draft_token_start_idx <= absolute_pos < draft_end_idx:
                                    current_step_confidences[int(absolute_pos)] = float(max(x1_p[0, local_idx].float().item(), 0.0))

                            confidences = []
                            recoverable = []
                            for absolute_pos in range(draft_token_start_idx, draft_end_idx):
                                if absolute_pos in committed_confidences:
                                    confidences.append(committed_confidences[absolute_pos])
                                    recoverable.append(False)
                                elif x_t[0, absolute_pos].item() != mask_id:
                                    confidences.append(1.0)
                                    recoverable.append(False)
                                elif absolute_pos in current_step_confidences:
                                    confidences.append(current_step_confidences[absolute_pos])
                                    recoverable.append(True)
                                else:
                                    confidences.append(0.0)
                                    recoverable.append(True)

                            frontier_k = 0
                            for confidence in confidences:
                                if confidence >= tau_f:
                                    frontier_k += 1
                                else:
                                    break

                            if frontier_k >= target_len:
                                frontier_score = float(target_len)
                                frontier_confidence = None
                                frontier_recoverable = False
                            else:
                                frontier_confidence = confidences[frontier_k]
                                frontier_recoverable = recoverable[frontier_k]
                                frontier_score = float(frontier_k) + min(1.0, max(frontier_confidence, 0.0) / max(tau_f, 1e-12))

                            previous_score = frontier_scores[-1] if frontier_scores else None
                            frontier_gain = None if previous_score is None else frontier_score - previous_score
                            frontier_scores.append(frontier_score)
                            frontier_recent_unmasked.append(unmasked_this_step)
                            if args is not None:
                                frontier_recent_unmasked = frontier_recent_unmasked[-max(1, int(getattr(args, "frontier_patience", 2))):]

                            masks_remaining = int((x_t[:, draft_token_start_idx:draft_end_idx] == mask_id).sum().item())
                            step_record = {
                                "step": len(frontier_scores),
                                "target_len": target_len,
                                "frontier_k": int(frontier_k),
                                "frontier_score": float(frontier_score),
                                "frontier_gain": None if frontier_gain is None else float(frontier_gain),
                                "frontier_confidence": frontier_confidence,
                                "frontier_recoverable": bool(frontier_recoverable),
                                "unmasked_this_step": unmasked_this_step,
                                "masks_remaining": masks_remaining,
                            }
                            frontier_stats["steps"].append(step_record)
                            frontier_stats["final_frontier_score"] = float(frontier_score)

                            min_steps = int(getattr(args, "frontier_min_steps", 2)) if args is not None else 2
                            patience = int(getattr(args, "frontier_patience", 2)) if args is not None else 2
                            gain_epsilon = float(getattr(args, "frontier_gain_epsilon", 0.0)) if args is not None else 0.0
                            max_unmask = 1
                            cost_token_equiv = float(getattr(args, "frontier_cost_token_equiv", 0.2)) if args is not None else 0.2
                            aggressive_irrecoverable = bool(getattr(args, "frontier_aggressive_irrecoverable", False)) if args is not None else False

                            stop_reason = None
                            if frontier_k >= target_len:
                                stop_reason = "frontier_all_pass"
                            elif aggressive_irrecoverable and frontier_k < target_len and not frontier_recoverable and frontier_confidence is not None and frontier_confidence < tau_f:
                                stop_reason = "frontier_irrecoverable_low_conf"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "mask_efficiency":
                                if len(frontier_recent_unmasked) >= patience and all(x <= max_unmask for x in frontier_recent_unmasked[-patience:]):
                                    stop_reason = "mask_efficiency_stall"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "frontier":
                                if frontier_gain is not None and frontier_gain <= gain_epsilon and unmasked_this_step <= max_unmask:
                                    stop_reason = "frontier_stall"
                            elif len(frontier_scores) >= min_steps and frontier_mode == "cost_aware":
                                if frontier_gain is not None:
                                    if len(frontier_scores) >= 3:
                                        prev_gain = frontier_scores[-2] - frontier_scores[-3]
                                        ratio = max(0.0, min(1.0, frontier_gain / max(prev_gain, 1e-12)))
                                        predicted_gain = frontier_gain * ratio
                                    else:
                                        predicted_gain = frontier_gain
                                    if predicted_gain <= cost_token_equiv:
                                        stop_reason = "cost_aware_low_expected_gain"

                            if stop_reason is not None and frontier_mode not in ("disabled", "none", "off"):
                                frontier_stats["stop_reason"] = stop_reason
                                frontier_force_stop = True
                                if masks_remaining > 0:
                                    fill_start_time = torch.cuda.Event(enable_timing=True)
                                    fill_start_time.record()
                                    fill_logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                                    fill_logits = torch.cat([fill_logits[:, :1, :], fill_logits[:, :-1, :]], dim=1)
                                    fill_end_time = torch.cuda.Event(enable_timing=True)
                                    fill_end_time.record()
                                    torch.cuda.synchronize()
                                    forward_pass_latencies.append(fill_start_time.elapsed_time(fill_end_time))
                                    num_forward_passes += 1

                                    fill_probs = torch.softmax(fill_logits, dim=-1)
                                    draft_mask = (x_t[:, draft_token_start_idx:draft_end_idx] == mask_id)
                                    for rel_pos in draft_mask[0].nonzero(as_tuple=False).flatten().tolist():
                                        absolute_pos = draft_token_start_idx + rel_pos
                                        local_pos = absolute_pos - block_abs_start
                                        if 0 <= local_pos < fill_logits.shape[1]:
                                            token_id = fill_logits[:, local_pos, :].argmax(dim=-1)
                                            token_conf = torch.gather(fill_probs[:, local_pos, :], dim=-1, index=token_id.unsqueeze(-1)).squeeze(-1)
                                            x_t[:, absolute_pos] = token_id
                                            committed_confidences[int(absolute_pos)] = float(token_conf[0].float().item())

                                draft_tokens_unmasked = True
                                frontier_stats["actual_spec_len"] = int(spec_len)
'''


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"Could not find patch location: {label}")
    return text.replace(old, new, 1)


def patch_modeling(path):
    modeling_path = Path(path)
    text = modeling_path.read_text(encoding="utf-8")
    if "frontier_force_stop" in text:
        return False

    text = replace_once(
        text,
        "        last_round_rejected=None,  # rejected tokens from the last round (that might be reusable)\n        **kwargs\n",
        "        last_round_rejected=None,  # rejected tokens from the last round (that might be reusable)\n        return_frontier_stats=False,\n        **kwargs\n",
        "generate_draft_tokens_arbitrary_length signature",
    )
    text = replace_once(
        text,
        "        conf_of_unmasked_tokens = []\n",
        "        conf_of_unmasked_tokens = []\n" + FRONTIER_INIT,
        "frontier state init",
    )
    text = replace_once(
        text,
        "                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]\n                        \n                        # logger.debug",
        "                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]\n" + FRONTIER_STEP + "                        \n                        # logger.debug",
        "frontier observe step",
    )
    text = replace_once(
        text,
        "                            if any([x < lowconf_threshold for x in conf_of_unmasked_tokens]):\n",
        "                            if frontier_force_stop:\n                                logger.debug(f\"{Colors.GREEN}Frontier controller stopped refinement. reason={frontier_stats.get('stop_reason')} spec_len={spec_len}{Colors.RESET}\")\n                                draft_tokens_unmasked = True\n                            elif any([x < lowconf_threshold for x in conf_of_unmasked_tokens]):\n",
        "frontier stop before low confidence extension logic",
    )
    text = replace_once(
        text,
        "        if return_prefill_kvs:\n            return input_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies\n        return input_ids, spec_len, num_forward_passes, forward_pass_latencies\n",
        "        frontier_stats[\"actual_spec_len\"] = int(spec_len)\n        if return_prefill_kvs:\n            if return_frontier_stats:\n                return input_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies, frontier_stats\n            return input_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies\n        if return_frontier_stats:\n            return input_ids, spec_len, num_forward_passes, forward_pass_latencies, frontier_stats\n        return input_ids, spec_len, num_forward_passes, forward_pass_latencies\n",
        "frontier return values",
    )

    backup_path = modeling_path.with_suffix(modeling_path.suffix + ".frontier.bak")
    if not backup_path.exists():
        backup_path.write_text(modeling_path.read_text(encoding="utf-8"), encoding="utf-8")
    modeling_path.write_text(text, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="/content/failfasttesting/Fast_dLLM_v2_1.5B")
    args = parser.parse_args()
    modeling_path = Path(args.model_dir) / "modeling.py"
    bundled_modeling_path = Path(__file__).resolve().with_name("Fast_dLLM_v2_1_5B") / "modeling.py"
    if bundled_modeling_path.exists() and bundled_modeling_path.resolve() != modeling_path.resolve():
        backup_path = modeling_path.with_suffix(modeling_path.suffix + ".frontier.bak")
        if modeling_path.exists() and not backup_path.exists():
            shutil.copy2(modeling_path, backup_path)
        shutil.copy2(bundled_modeling_path, modeling_path)
        print(f"copied_bundled_frontier_modeling: {modeling_path}")
        return
    changed = patch_modeling(modeling_path)
    print(f"{'patched' if changed else 'already_patched'}: {modeling_path}")


if __name__ == "__main__":
    main()
