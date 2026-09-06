# Stable C-rich50 positive control

Run `run_math_stable_crich50_pipeline.py` with INT8 target, FP16 drafter,
and full-prefix verifier (KV cache disabled). Current U1 remains F2,
raw absolute delta-J weighting, uniform B16/K100 replay, fixed threshold 0.5,
no dynamic threshold and no utility-mass balancing.

The existing dataset is HuggingFaceH4/MATH-500: valid positional IDs are
0 through 499. This integration corrects the bundle's invalid 500..4999
range to 0..499; these are NOT guaranteed unseen problems.

Screening freezes logistic weights with learning rate zero and uses
probability-one probes where the existing controller permits probing.
It is a hindsight collector, not an exact counterfactual oracle for every state.
Screen labels and the offline F2 fit select problems only; the fit is not
loaded into the online controller. Zero learning rate is now explicitly valid;
positive-rate U1 updates are unchanged.

Default bounds: at most 500 screened IDs, initial target 65 candidates,
at most four selection attempts with confirmation seeds 42 and 43.
The final 50-ID sequence uses 20 adaptation problems and 30 evaluation
problems. Learning continues during evaluation. Only passing confirmation
freezes the sequence and enables the final Always-STOP/U1 comparison at seed 44.
Insufficient candidates or failed confirmation stops the pipeline with logs.

This is selected-problem positive-control evidence, not an unbiased MATH
benchmark or held-out-problem generalization. The final seed is held out,
but the selected problems are not. INT8 output equivalence is not guaranteed.

Use `--resume` with the same output directory and experimental settings.
Completed screen batches require SCREEN_COMPLETE.json; partial batch data
is not included in selection. Failed output directories are retained.
An interrupted model process may need to rerun its incomplete batch.
Increasing the cap cannot extend beyond the 500-question dataset.

CPU tests cover zero-LR frozen weights, command settings, dataset bounds,
distinct seeds, ordered completion and incomplete-screen exclusion.
GPU memory usage and runtime have not been verified for this pipeline.
