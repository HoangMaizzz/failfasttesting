"""Explicit Qwen2 layer placement and synchronized two-GPU wall timing."""
import json
from pathlib import Path


def qwen2_device_map(model_path):
    config = json.loads((Path(model_path) / "config.json").read_text())
    if config.get("model_type") != "qwen2" or config.get("tie_word_embeddings", False):
        raise ValueError("Two-GPU delivery requires Qwen2 with untied embeddings.")
    layers = int(config["num_hidden_layers"])
    if layers < 2:
        raise ValueError("At least two decoder layers are required.")
    split = min(layers - 1, max(1, (layers * 2) // 3))
    placement = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    placement.update({f"model.layers.{i}": 0 if i < split else 1 for i in range(layers)})
    return placement


def synchronize_devices(cuda, device, two_gpu=False):
    if two_gpu:
        cuda.synchronize(0)
        cuda.synchronize(1)
    else:
        cuda.synchronize(device)
