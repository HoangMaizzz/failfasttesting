# Structured sparse smoke test: 3 GSM8K questions, K=2, Lmax=64

Select **GPU T4 x2**, Internet on, and add the SpecWorld dataset. This one cell
works in a fresh Save Version run. It installs dependencies, clones the source,
loads the uploaded ZIP or extracted folder, and downloads model weights.

```python
from urllib.request import urlopen

DATASETS = ["gsm8k"]
NUM_QUESTIONS = 3
ANCHORS_PER_QUESTION = 4
MAX_PROPOSAL_TOKENS = 64

url = "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/codex/sparse-extend-world-model/kaggle_structured_sparse_smoke.py"
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_structured_sparse_smoke.py", "exec"))
```

Inspect the runner source in this branch before execution if desired.
To collect both datasets after validating the smoke ZIP, change only
`DATASETS = ["gsm8k", "math"]` and `NUM_QUESTIONS = 100`.

Outputs: `/kaggle/working/structured_sparse_<timestamp>/<dataset>/`.
Each dataset is zipped immediately on completion, before starting the next one.
The fresh run puts source and drafter weights under `/kaggle/temp`, outside
published Output. Existing working-directory drafter weights are reused when
available, and are not deleted. Only download the `*_structured_graph.zip` files.

The v2 dataset has exact one-step R/E edges and Submit labels on every node.
See `STRUCTURED_SPARSE_PROTOCOL.md` for collection policy, calibrated verifier
cost versus measured node cost, and the full-context replay timing limitation.
Old ZIPs are unchanged; v2 features must not be silently mixed with legacy ones.
