"""Install and check the instrumented remote-code module beside dLLM weights."""
import hashlib
import json
from pathlib import Path
import shutil


def prepare_native_model_assets(repo_dir: Path, model_dir: Path) -> str:
    """Keep downloaded weights/config, but use this checkout's capture code."""
    repo_dir = Path(repo_dir)
    model_dir = Path(model_dir)
    source = repo_dir / "Fast_dLLM_v2_1_5B" / "modeling.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing instrumented Fast-dLLM source: {source}")
    source_bytes = source.read_bytes()
    for marker in (b"wm_v2_capture_native_latent", b"wm_v2_native_latent_block"):
        if marker not in source_bytes:
            raise ValueError(f"Fast-dLLM source lacks V2 capture support: {source}")
    compile(source_bytes, str(source), "exec")

    required = [model_dir / "config.json", model_dir / "configuration.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Fast-dLLM assets: " + ", ".join(missing))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    reference = config.get("auto_map", {}).get("AutoModelForCausalLM", "")
    if not isinstance(reference, str) or reference.split(".")[0] != "modeling":
        raise ValueError(f"Expected local modeling module in AutoModelForCausalLM, got {reference!r}")

    target = model_dir / "modeling.py"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    digest = hashlib.sha256(source_bytes).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Fast-dLLM capture code differs from pinned checkout: {target}")
    # Validate every custom module referenced by the checkpoint before GPU load.
    for entry in config.get("auto_map", {}).values():
        for ref in entry if isinstance(entry, list) else [entry]:
            if ref is None:
                continue
            if not isinstance(ref, str) or "--" in ref:
                raise ValueError(f"Expected local custom-code reference, got {ref!r}")
            module = model_dir.joinpath(*ref.rsplit(".", 1)[0].split(".")).with_suffix(".py")
            if not module.is_file():
                raise FileNotFoundError(f"Missing custom checkpoint module: {module}")
    return digest
