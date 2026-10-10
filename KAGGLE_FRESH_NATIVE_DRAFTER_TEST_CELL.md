# Kaggle cell: fresh native dLLM latent-dynamics test

This run generates new GSM8K trajectories with Fast-dLLM instead of consuming
the older Refine/Extend ZIPs. Before each E, it samples 0–3 native R steps;
every observed state also retains its same-forward top-1 STOP fill. It collects
R transitions and E transitions (top-1 fill followed by a fresh native 8-token segment), then trains and tests
the latent model on question-disjoint 70/15/15 splits through horizon 3.

## Kaggle settings

- Turn **Internet on** (GitHub, Hugging Face model, and GSM8K download).
- Select a GPU accelerator. **One T4 is sufficient**; this experiment does not
  load the verifier, so a second T4 is not used.
- No data ZIP, Kaggle Secret, or verifier model is required.
- The launcher discovers tests directly from the checkout so installed
  packages named `tests` cannot shadow the repository's test files.

Paste this as one Python cell and run it:

```python
from urllib.request import urlopen

SOURCE_REF = "codex/factorized-wm-feasibility"
NUM_QUESTIONS = 100
MAX_PROPOSAL_TOKENS = 64
MAX_REFINEMENT_STEPS = 3
THRESHOLD = 0.5
UPDATES = 240
SEEDS = [42, 43]

url = (
    "https://raw.githubusercontent.com/HoangMaizzz/failfasttesting/"
    f"{SOURCE_REF}/kaggle_fresh_native_drafter_test.py"
)
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "kaggle_fresh_native_drafter_test.py", "exec"), globals())
```

For a short first smoke, change `NUM_QUESTIONS` to `3` or `5`; the default is
the requested 100 questions. The runner writes a result ZIP under
`/kaggle/working` and displays a download link. It also stores the native
features/edges, per-question termination reasons, training logs, held-out
metrics by R/E and horizon, and test-time feature-group ablations.

This evaluates prediction of the dLLM's *observable state dynamics* (remaining
masks, confidence, candidate-change events, and learned candidate embeddings).
It does not claim to generate exact next-token IDs or predict verifier
acceptance; no verifier is run.
