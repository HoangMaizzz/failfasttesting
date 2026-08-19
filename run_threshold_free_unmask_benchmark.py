import run_adaptive_unmask_only_benchmark as benchmark


DEFAULT_OUTPUT_DIR = "/content/failfasttesting/outputs_threshold_free_unmask_test10"
OLD_OUTPUT_DIR = "/content/failfasttesting/outputs_adaptive_unmask_only_test15"


benchmark.BENCHMARK_VERSION = "threshold_free_unmask_v1"
benchmark.METHOD = {
    "name": "threshold_free_unmask_only",
    "frontier_mode": "cost_aware_v2_refinement_no_threshold",
    "spec_len": 8,
    "incr_len": 8,
    "lowconf_threshold": 0.45,
}


_parse_args = benchmark.parse_args


def parse_args():
    args = _parse_args()
    if args.output_dir == OLD_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR
    return args


benchmark.parse_args = parse_args


if __name__ == "__main__":
    benchmark.main()
