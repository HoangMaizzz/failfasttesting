"""Download the local FP16 benchmark assets once for Kaggle/server runs."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)

    models = {
        "target": "Qwen/Qwen2.5-7B-Instruct",
        "drafter": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    }
    for name, model in models.items():
        destination = args.assets / name
        if (destination / "config.json").exists():
            print(f"Already prepared: {destination}", flush=True)
            continue
        print(f"Downloading {model} -> {destination}", flush=True)
        snapshot_download(
            model,
            local_dir=str(destination),
            ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"],
        )

    datasets = {
        "math": ("HuggingFaceH4/MATH-500", None),
        "gsm8k": ("openai/gsm8k", "main"),
        "humaneval": ("openai/openai_humaneval", None),
    }
    for name, (repository, config) in datasets.items():
        destination = args.assets / "datasets" / name
        if destination.exists():
            print(f"Already prepared: {destination}", flush=True)
            continue
        print(f"Downloading dataset {repository} -> {destination}", flush=True)
        data = load_dataset(repository, config, split="test")
        data.save_to_disk(str(destination))

    (args.assets / "sources.json").write_text(
        json.dumps({"models": models, "datasets": datasets}, indent=2),
        encoding="utf-8",
    )
    print(f"Assets prepared at {args.assets.resolve()}", flush=True)


if __name__ == "__main__":
    main()
