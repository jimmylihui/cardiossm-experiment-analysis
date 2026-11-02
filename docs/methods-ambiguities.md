# Protocol choices that remain unresolved

The active Methods is the authority. Commented historical settings are not used
to fill missing details. This list describes implementation boundaries; it is
not a proposed manuscript change or evidence that an experiment was run.

| Topic | Unresolved detail | Implementation |
|---|---|---|
| Run boundary | Whether the >30min gap uses window or block completion times | Require --run-gap-basis window_completion or block_completion, with no default. Generate window_run_id and block run_id once; history, state, means, HR, and segment/scale analyses consume these same IDs. A block spanning multiple window-defined runs is rejected. |
| Internal calibration split | Whether 80/20 counts retained blocks or raw windows | Require `--initial-split-unit blocks` or `windows`; record the choice. Block mode keeps blocks disjoint. No silent default. |
| Online optimization | Learning rate, memory/replay counts, sampling convention and whether replay includes validation calibration windows | Require learning rate, sample counts and `--replay-source fit` or `calibration`. Require --online-sampling with_replacement or without_replacement for uniform draws; neither is assumed as a Methods default. |
| FSNet budget and architecture | Equivalent budget for a three-axis sample, exact adaptation of the cited architecture, and the auxiliary-loss exception | Preserve the existing three-axis ResNet18/map-only definition; require an explicit whole-window batch size. Do not call its budget or architecture independently verified against the cited model. No decoder or few-shot module is invented. |
| No-initial-calibration evaluation | Which blocks are scored, distinct from when readings begin | Require `--no-initial-test-scope all` or `final70`; either choice is recorded and is not claimed to recover the original experiment. |
| Fixed folds | Patient assignments, permutation input ordering and RNG implementation for validation seed 100+fold | Keep the private manifest external; validate four folds, 32 test patients, four validation patients per fold, and three training-only supplemental recordings. The unpublished permutation is not recreated by guessing an RNG. |
| Preprocessing/support | Filter, template/fiducials, detailed run-retention procedure and supplemental support provenance | Consume explicit externally prepared arrays. Main pretraining uses retained_mask; supplemental recordings require pretraining_mask and 2339 total windows. Shape/count checks do not validate signal extraction or clinical quality. |
| No-reading evaluation intervals | How to define between-reading segments for pretraining_only and initial_only when no readings are assimilated | Standalone runs store no actual reading positions. Their r_seg is missing until an explicit evaluation-segment grid is supplied; no virtual readings are inferred from old settings. |
| Statistics | The appropriate aggregation for each figure/table, CI contrast, whether tuning is repeated, paired test and full Holm family | CLI requires aggregation, contrast and fixed/reselect tuning. Mean patient r and median patient r are both explicit modes. Median paired deltas differ from differences of medians. Holm accepts externally computed p-values with explicit family labels; it does not choose a paired test. |
| Reading-noise realization | Original seed-to-patient mapping is unpublished | Use a documented SHA-256 seed of patient identity/interval/seed0, independent of model, then Gaussian SD5 draws at readings only. This guarantees sharing, not identity with archived draws. |
| Completion/latency | Bin-closing versus last retained window completion, actual compute latency and native-scale broadcast of a completed block | Preserve conservative max(bin-end, last-window-availability) issuance. Gap clock is a separate required choice above. No real-time latency is asserted; the adapter only aligns block outputs and does not back-date a block estimate to native windows. |

Already fixed without these choices: every same-axis window enters chronological
history; only boundaries in the shared chosen >30min run partition reset the state; inference retains H
and causal-convolution memory across reading updates. Training replay uses its
own expanded history and cannot change inference states. Reading blocks are
predicted before their labels update weights or the offset filter.

All four supervised conditions have explicit runner branches: pretraining_only,
initial_only, periodic_only and initial_periodic. Explanation experiments (SHAP,
full-stage masking) and all original figure-generation workflows are still not
implemented. General-purpose functions must not be described as completed
experiments or numerical reproductions.

Slow scoring reports only MAE, whole-test correlation and between-reading
correlation (with counts), using matched past-only 60min targets/predictions.
It does not emit concordance or drop/rise event columns. Every score/candidate
row declares reference_scale, prediction_scale and run_gap_basis. Compare
rejects mixed/missing scale metadata and slow-target event comparisons. This
does not define a new slow-event experiment.
