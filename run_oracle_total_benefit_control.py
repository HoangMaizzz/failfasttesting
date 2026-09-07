#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from oracle_run_guard import guard, preserve

ROOT = Path(__file__).resolve().parent
DISCOVERY = ROOT / "FP16_DISCOVERY"
DEFAULT_SOURCES = []

import search_closed_loop_math50 as search_mod
import run_deterministic_witness as witness_mod


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Search a 50-problem action-faithful trajectory from one broad 150-200 problem discovery pool, "
            "run the real U1 learner, and PASS only when learned CONTINUE decisions "
            "have positive aggregate delta-J benefit. AlwaysSTOP speed is not used."
        )
    )
    p.add_argument("--dllm_dir", default="/kaggle/working/Fast_dLLM_v2_1.5B")
    p.add_argument("--output_dir", default="/kaggle/working/math50_oracle_total_benefit_pc")
    p.add_argument("--target_quantization", default="none")
    p.add_argument("--target_dtype", choices=["auto","fp16","bf16","fp32"], default="fp16")
    p.add_argument("--drafter_dtype", choices=["auto","fp16","bf16","fp32"], default="fp16")
    p.add_argument("--target_device", type=int, default=0)
    p.add_argument("--drafter_device", type=int, default=1)
    p.add_argument('--target_model_name', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--two_gpu', action='store_true')
    p.add_argument('--drafter_threshold', type=float, default=.5)
    p.add_argument('--lowconf_threshold', type=float, default=.7)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay_seeds", nargs="+", type=int, default=[42])
    p.add_argument("--max_candidates", type=int, default=5)
    p.add_argument("--search_restarts", type=int, default=12)
    p.add_argument("--search_iterations", type=int, default=5000)
    p.add_argument("--seed_build_attempts", type=int, default=8,
                   help="Greedy action-faithful 50-problem seeds built from the single broad discovery pool.")
    p.add_argument("--search_seed", type=int, default=20260907)
    p.add_argument("--min_learned_c", type=int, default=5)
    p.add_argument("--offline_min_learned_c", type=int, default=10)
    p.add_argument("--probe_soft_weight", type=float, default=0.5,
                   help="Soft preference for realized probe counts near p=0.08/0.02 expectation; never a hard PASS gate.")
    p.add_argument("--nonprobe_natural_weight", type=float, default=0.25)
    p.add_argument("--benefit_reward", type=float, default=0.01)
    p.add_argument("--learned_c_reward", type=float, default=0.02)
    p.add_argument("--min_total_learned_benefit", type=float, default=1.0,
                   help="PASS requires -sum(delta_J) over actual learned-C rows >= this margin.")
    p.add_argument("--sources", nargs="*", default=[str(x) for x in DEFAULT_SOURCES])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def run(cmd, log_path: Path, cwd=ROOT):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 118)
    print("RUN:", " ".join(map(str, cmd)))
    print("LOG:", log_path)
    print("=" * 118, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def candidate_signature(traj_csv: Path):
    d = pd.read_csv(traj_csv)
    payload = "|".join(f"{int(r.problem_id)}@{str(r.source)}" for _, r in d.iterrows())
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def actual_pass(a, u1_summary):
    min_benefit = abs(float(a.min_total_learned_benefit))
    total_benefit = -float(u1_summary["sum_delta_j_learned_continue"])
    enough_c = bool(u1_summary["learned_continue_non_tie"] >= a.min_learned_c)
    benefit_ok = bool(total_benefit >= min_benefit)
    score_ok = bool(u1_summary["max_continue_score"] > 0.5)
    schedule_ok = bool(u1_summary["schedule_state_mismatch_count"] == 0)
    passed = bool(enough_c and benefit_ok and score_ok and schedule_ok)
    return enough_c, benefit_ok, score_ok, schedule_ok, passed


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(x) for x in a.sources]
    if not source_paths:
        raise ValueError("No discovery sources supplied. Run collect_fp16_nokv_discovery.py first, then pass --sources <discovery_dir>.")
    for p in source_paths:
        if not p.exists():
            raise FileNotFoundError(p)
    if min(a.max_candidates,a.search_restarts,a.search_iterations,a.seed_build_attempts,
           a.min_learned_c,a.offline_min_learned_c) <= 0:
        raise ValueError('Search budgets and learned-C minimum must be positive')
    if not 0 < a.min_total_learned_benefit < float('inf'):
        raise ValueError('Benefit margin must be finite and positive')
    if a.target_quantization not in ('none','int8') or a.target_dtype != 'fp16' or a.drafter_dtype != 'fp16':
        raise ValueError('Use FP16 compute, an FP16 or INT8 target and FP16 drafter')
    if a.two_gpu and a.target_quantization != 'none':
        raise ValueError('Two-GPU FP16 placement cannot be combined with INT8')
    for source in source_paths:
        manifest = source/'DISCOVERY_MANIFEST.json'
        if source.is_dir() and manifest.exists():
            settings = json.loads(manifest.read_text())['setting']
            if settings['target_quantization'] != a.target_quantization:
                raise ValueError('Discovery and witness target quantization must match')
    source_hashes = {}
    for source in source_paths:
        files = [source] if source.is_file() else sorted(source.rglob('benchmark_results.csv'))
        for file in files:
            source_hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    guard(out, {**vars(a), 'source_hashes': source_hashes}, a.resume)
    if a.resume and (out/'FINAL_VERDICT.json').exists():
        print((out/'FINAL_VERDICT.json').read_text(), flush=True)
        return

    pool, boots, seqs = search_mod.load_sources(source_paths, include_screen=False)
    pool_ids = sorted(map(int, pool.keys()))
    print(f"[POOL] {len(pool_ids)} unique MATH IDs, {sum(map(len,pool.values()))} observed physical trace variants")
    if len(pool_ids) < 50:
        raise RuntimeError("Need at least 50 candidate problem IDs")

    attempted_signatures = set()
    trial_rows = []
    live_sources = list(source_paths)

    for trial in range(a.max_candidates):
        pd.DataFrame(trial_rows).to_csv(out/'TRIALS.csv', index=False)
        cand = out / f"candidate_{trial+1:02d}"
        cand.mkdir(parents=True, exist_ok=True)
        search_seed = a.search_seed + trial * 1009
        search_out = cand / "offline_search"

        cmd = [
            sys.executable, "-u", str(ROOT / "search_closed_loop_math50.py"),
            "--sources", *map(str, live_sources),
            "--output", str(search_out),
            "--replay-seeds", *map(str, a.replay_seeds),
            "--min-learned-c", str(a.offline_min_learned_c),
            "--min-learned-benefit", str(abs(a.min_total_learned_benefit)),
            "--restarts", str(a.search_restarts),
            "--iterations", str(a.search_iterations),
            "--seed-build-attempts", str(a.seed_build_attempts),
            "--search-seed", str(search_seed),
            "--probe-soft-weight", str(a.probe_soft_weight),
            "--nonprobe-natural-weight", str(a.nonprobe_natural_weight),
            "--benefit-reward", str(a.benefit_reward),
            "--learned-c-reward", str(a.learned_c_reward),
        ]
        summ_path = search_out / "BEST_SUMMARY.json"
        if not (a.resume and summ_path.exists()):
            run(cmd, out / "logs" / f"search_candidate_{trial+1:02d}.log")
        if not summ_path.exists():
            trial_rows.append({"trial": trial+1, "offline_feasible": False, "reason": "no summary"})
            continue
        offline = json.loads(summ_path.read_text())
        if not offline.get("criteria_pass", False):
            trial_rows.append({
                "trial": trial+1,
                "offline_feasible": False,
                "offline_sum_dJ_learned": offline.get("sum_delta_j_learned_continue"),
                "offline_total_learned_benefit": -float(offline.get("sum_delta_j_learned_continue", 0.0)),
            })
            continue

        sig = candidate_signature(search_out / "BEST_TRAJECTORY_50.csv")
        if sig in attempted_signatures:
            trial_rows.append({"trial": trial+1, "offline_feasible": True, "duplicate": True, "signature": sig})
            continue
        attempted_signatures.add(sig)

        candidate_seed = int(offline.get("replay_seed", a.seed))
        schedule = cand / "DETERMINISTIC_PROBE_SCHEDULE.csv"
        run([
            sys.executable, "-u", str(ROOT / "make_probe_schedule.py"),
            "--decisions", str(search_out / "BEST_DECISIONS_REPLAY.csv"),
            "--trajectory", str(search_out / "BEST_TRAJECTORY_50.csv"),
            "--output", str(schedule),
        ], out / "logs" / f"schedule_candidate_{trial+1:02d}.log")

        actual_dir = cand / "actual_u1"
        run([
            sys.executable, "-u", str(ROOT / "run_deterministic_witness.py"),
            "--dllm_dir", str(a.dllm_dir),
            "--output_dir", str(actual_dir),
            "--trajectory_csv", str(search_out / "BEST_TRAJECTORY_50.csv"),
            "--probe_schedule_csv", str(schedule),
            "--target_quantization", str(a.target_quantization),
            "--target_dtype", str(a.target_dtype),
            "--drafter_dtype", str(a.drafter_dtype),
            "--target_device", str(a.target_device),
            "--drafter_device", str(a.drafter_device),
            '--target_model_name', str(a.target_model_name),
            *(['--two_gpu'] if a.two_gpu else []),
            '--drafter_threshold', str(a.drafter_threshold),
            '--lowconf_threshold', str(a.lowconf_threshold),
            '--max_new_tokens', str(a.max_new_tokens),
            "--seed", str(candidate_seed),
            "--min_learned_c", str(a.min_learned_c),
            "--min_total_learned_benefit", str(abs(a.min_total_learned_benefit)),
        ] + (["--resume"] if a.resume else []), out / "logs" / f"actual_u1_candidate_{trial+1:02d}.log")

        traj = pd.read_csv(search_out / "BEST_TRAJECTORY_50.csv")
        ids = traj.problem_id.astype(int).tolist()
        schedule_df = pd.read_csv(schedule)
        u1_case = actual_dir / "u1_deterministic_oracle_probe"
        u1_summary = witness_mod.summarize_u1(u1_case, ids, len(schedule_df))
        enough_c, benefit_ok, score_ok, schedule_ok, passed = actual_pass(a, u1_summary)
        total_benefit = -float(u1_summary["sum_delta_j_learned_continue"])

        result = {
            "trial": trial+1,
            "signature": sig,
            "replay_seed": candidate_seed,
            "offline": offline,
            "actual_u1": u1_summary,
            "hard_criterion": {
                "learned_continue_non_tie_min": a.min_learned_c,
                "total_learned_benefit_min_ms_per_token": abs(a.min_total_learned_benefit),
                "equivalent_sum_delta_j_max": -abs(a.min_total_learned_benefit),
                "always_stop_speed_required": False,
                "individual_learned_continue_must_all_be_beneficial": False,
                "probe_ratio_hard_required": False,
                "probe_ratio_soft_preference": True,
            },
            "pass_enough_learned_continue": enough_c,
            "pass_positive_total_learned_benefit": benefit_ok,
            "pass_score": score_ok,
            "pass_schedule_state_match": schedule_ok,
            "PASS": passed,
        }
        (cand / "ACTUAL_CANDIDATE_RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
        trial_rows.append({
            "trial": trial+1,
            "signature": sig,
            "offline_feasible": True,
            "offline_sum_dJ_learned": offline.get("sum_delta_j_learned_continue"),
            "offline_total_learned_benefit": -float(offline.get("sum_delta_j_learned_continue", 0.0)),
            "actual_learned_C": u1_summary.get("learned_continue_non_tie"),
            "actual_TP": u1_summary.get("learned_continue_tp"),
            "actual_FP": u1_summary.get("learned_continue_fp"),
            "actual_sum_dJ_learned": u1_summary.get("sum_delta_j_learned_continue"),
            "actual_total_learned_benefit": total_benefit,
            "schedule_mismatch": u1_summary.get("schedule_state_mismatch_count"),
            "actual_probe_ratio": u1_summary.get("realized_probe_ratio_to_nominal_expectation"),
            "actual_probe_total_z": u1_summary.get("probe_total_z"),
            "actual_probe_structural_z": u1_summary.get("probe_structural_z"),
            "actual_probe_floor_z": u1_summary.get("probe_floor_z"),
            "actual_probe_soft_penalty": u1_summary.get("probe_soft_penalty"),
            "PASS": passed,
        })
        pd.DataFrame(trial_rows).to_csv(out / "TRIALS.csv", index=False)

        print("\n[ACTUAL CANDIDATE RESULT]")
        print(json.dumps(result, indent=2, allow_nan=True))

        # Newly observed real U1 trace can be reused as a physical variant on later attempts.
        live_sources.append(u1_case)

        if passed:
            final = out / "FINAL_WITNESS"
            preserve(final)
            shutil.copytree(cand, final)
            final_summary = {
                "PASS": True,
                "selected_trial": trial+1,
                "selected_signature": sig,
                "threshold": 0.5,
                "definition": "delta_J = J_CONTINUE - J_STOP; aggregate benefit = -sum(delta_J_learned_continue)",
                "always_stop_speed_required": False,
                "individual_learned_continue_must_all_be_beneficial": False,
                "precision_setting": f"target_quantization={a.target_quantization}, FP16 compute/drafter; verifier use_cache=False; reusable drafter KVs disabled",
                "probe_ratio_hard_required": False,
                "probe_ratio_soft_preference": True,
                "actual_total_learned_benefit_ms_per_token": total_benefit,
                "actual": result,
            }
            (out / "FINAL_VERDICT.json").write_text(json.dumps(final_summary, indent=2, allow_nan=True), encoding="utf-8")
            txt = (
                "PASS=True\n"
                f"trial={trial+1} signature={sig}\n"
                f"learned_C={u1_summary['learned_continue_non_tie']} TP={u1_summary['learned_continue_tp']} FP={u1_summary['learned_continue_fp']}\n"
                f"sum_dJ_learned={u1_summary['sum_delta_j_learned_continue']:.6f}\n"
                f"total_learned_benefit={total_benefit:.6f} required_min={abs(a.min_total_learned_benefit):.6f}\n"
                f"schedule_mismatch={u1_summary['schedule_state_mismatch_count']} max_continue_score={u1_summary['max_continue_score']:.6f}\n"
                f"probe_ratio={u1_summary['realized_probe_ratio_to_nominal_expectation']:.3f} "
                f"z_total={u1_summary['probe_total_z']:.3f} z_struct={u1_summary['probe_structural_z']:.3f} "
                f"z_floor={u1_summary['probe_floor_z']:.3f} soft_penalty={u1_summary['probe_soft_penalty']:.3f}\n"
                "AlwaysSTOP speedup is NOT required. Probe ratio is a SOFT preference only.\n"
            )
            (out / "FINAL_VERDICT.txt").write_text(txt, encoding="utf-8")
            print("\n" + "#"*118)
            print(txt)
            print("#"*118)
            return

    pd.DataFrame(trial_rows).to_csv(out / "TRIALS.csv", index=False)
    fail = {
        "PASS": False,
        "reason": "No candidate produced positive aggregate ACTUAL learned-C benefit within max_candidates.",
        "max_candidates": a.max_candidates,
        "min_total_learned_benefit": abs(a.min_total_learned_benefit),
        "always_stop_speed_required": False,
        "note": "Offline search alone never causes PASS; the criterion is re-checked on the real U1 run.",
    }
    (out / "FINAL_VERDICT.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
    (out / "FINAL_VERDICT.txt").write_text("PASS=False\n" + fail["reason"] + "\n", encoding="utf-8")
    print("\n[NO VERIFIED WITNESS FOUND]")
    print(json.dumps(fail, indent=2))


if __name__ == "__main__":
    main()
