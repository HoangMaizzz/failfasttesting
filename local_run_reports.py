"""Local, atomic diagnostics; called outside generation timing."""
import csv
import json
from pathlib import Path


def save_controller_reports(args):
    if not args.adaptive_td:
        return
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    controller = args.adaptive_td_controller
    snapshot = controller.snapshot()
    if controller.uses_hindsight_delta_j_logistic_f2:
        snapshot["logistic_boundary_checkpoint"] = controller.save_logistic_boundary_checkpoint()
    destination = root / "adaptive_td_runtime_state.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    temporary.replace(destination)
    for name, rows in (
        ("adaptive_td_decisions.csv", getattr(args, "adaptive_decision_rows", [])),
        ("adaptive_full_stream_transitions.csv", controller.full_stream_transitions),
    ):
        if not rows:
            continue
        destination = root / name
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
