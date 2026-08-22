import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="/content/failfasttesting/Fast_dLLM_v2_1.5B",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("Fast_dLLM_v2_1_5B") / "modeling.py"
    target = Path(args.model_dir) / "modeling.py"
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.parent.exists():
        raise FileNotFoundError(target.parent)
    backup = target.with_suffix(target.suffix + ".failfast.bak")
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    print(f"copied_bucket_renewal_modeling: {target}")


if __name__ == "__main__":
    main()
