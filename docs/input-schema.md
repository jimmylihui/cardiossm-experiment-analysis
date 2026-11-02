# Input schemas

All files are supplied externally and remain local. No examples containing observations or predictions are distributed. Use NumPy NPZ with `allow_pickle=False`; names must be Unicode arrays rather than Python objects.

## Prediction scoring

One NPZ per pseudonymous patient, and one input directory per calibration condition and seed. Array rows must refer to the same retained observations in the same order.

| Key | Shape | Meaning |
|---|---|---|
| `time_s` | T | Strictly increasing completed-block availability times in seconds |
| `reference` | T | Reference MAP in mmHg |
| `test_mask` | T | Boolean test support; calibration and test must be defined externally |
| `reading_mask` | T | Boolean actual scheduled calibration-reading locations |
| `model_names` | M | Unique Unicode model labels |
| `predictions` | T × M | Final assembled MAP predictions in mmHg |

For between-reading correlation, use retained 2min observations and the original segment eligibility. The event definition also assumes 2min blocks: five successive retained steps, at most 12min elapsed. Do not apply that definition unchanged to native 10s observations.

Predictions must be finite on test support. Constant correlations are missing. A patient with insufficient event positives or negatives has missing AUROC/AP, not zero. Report effective patient counts for each endpoint.

## Optional matched offset+HR assembly

With `score --assemble-offset-hr`, provide `neural_predictions` (T × M) instead of `predictions`, plus:

| Key | Shape | Meaning |
|---|---|---|
| `hr_bpm` | T | Monitor HR aligned with each completed block |
| `calibration_mask` | T | Boolean initial-calibration support |

`neural_predictions` contains unsmoothed 2min neural block means generated before the reading update. The assembly command computes past-only 60min means of the neural stream, reference MAP and HR in the same run, with startup min_periods=1 and >30min gap resets. The HR term is block HR minus its past-only mean, with beta=1. It uses no whole-record neural centering. Both network and filter update a reading only after predicting that block.

The `offset_hr` Python API requires an explicit qa and accepts these already computed past-only contributions. The CLI defaults to per-patient/model other-patient test-MAE selection from 0.5/2/8/32. --qa explicitly selects a fixed sensitivity analysis. --assemble-models specifies which networks receive offset and HR; other neural streams remain block network outputs. --candidate-output emits external candidate metrics; compare --tuning reselect repeats selection, excluding every copy of the held-out identity. The generated `OffsetHR` baseline sets the neural stream to zero.

## Output averaging scales

Use the prediction-scoring keys with observations at the native resolution. Supply final assembled predictions at that resolution, including offset and HR if that is the intended model definition. Reference and prediction use the same test observations and bin support. Scales are non-overlapping output averages, measured in seconds; no extra model fitting occurs.

Native-window extraction and quality screening are external preprocessing steps. Valid observations alone cannot establish prospective availability or latency.

## Feature screening

| Key | Shape | Meaning |
|---|---|---|
| `time_s` | T | Completed 2min-block availability time, in seconds |
| `reference` | T | Unaveraged retained block MAP |
| `features` | T × F | Unaveraged retained block features |
| `feature_names` | F | Unicode names; preserve modality labels |

The screening function computes past-only 60min means of MAP and features, selects on the first 30% of blocks, applies a >60min temporal embargo, and evaluates later associations. Missing feature values are handled pairwise. It does not produce a trained BP predictor or establish ECG independence.

## Candidate bootstrap API

`candidate_interval(first_values, first_errors, second_values, second_errors, metric)` accepts arrays with unique patients as rows and qa candidates as columns. Align patient identities across models explicitly before calling. Values contain the metric being reported; errors contain the protocol's tuning score, which may differ from that metric. The bootstrap excludes every occurrence of the held-out identity from tuning. Use patient Fisher-z values with `metric="z_seg"` to obtain cohort r_seg intervals.

## Runner adapter and no-initial-calibration condition

`prepare --models NAME=external-directory ... --tag IP4 --output external-directory`
aligns one condition across model outputs. It requires identical patient, raw
window, completed-block, test/calibration and realized-reading support, consumes
only unsmoothed block means, and never back-dates estimates to native windows.
HRb is copied only when supplied and checked across models. The corresponding
P5_N tag identifies periodic-only runs; PT and I identify standalone conditions.

Optional `reading_values` [T] supplies actual (possibly noisy) measurements,
finite at reading_mask positions. Without calibration, supply per-patient
`calibration_variance` scalars from the external reference analysis: assembly
excludes each target identity, takes the median of the other rows and adds25.
It never estimates R from the target's no-calibration labels. The first reading
block uses the initialized mean0/variance10000 prior before assimilation.

`score --network-only --network-scale block|slow` scores network inputs alone;
slow mode applies matching past-only means to both prediction and reference.
Already assembled inputs use `predictions`; they are not implicitly smoothed.
All derived files must be outside the code repository.

No-initial runner inputs require scalar recording_start_s from the original recording, before retention. The first scheduled time is that value; mapping to completed retained blocks does not restart the interval clock.

## Required shared run metadata and scoring scales

All score, scales and features CLI input records require run_id [T], a
nonnegative monotone integer vector, and scalar run_gap_basis, either
window_completion or block_completion. These must come from the same partition
used by the network. Explicit IDs control all run resets; consumers do not
recompute gaps from completed-block timestamps. Standalone Python utilities
retain a timestamp fallback only for isolated descriptive calls, not runner
output adaptation or CLI scoring.

Training outputs additionally carry window_run_id aligned with raw_t. prepare
checks both IDs across models and verifies each window's run matches its block.
Legacy outputs missing IDs are rejected. No analysis-side fallback is invented.

Score rows and candidate rows include reference_scale/prediction_scale
(block_2min or past_only_60min) and run_gap_basis. Slow network-only scoring emits
only level metrics (MAE/r_test/r_seg, Fisher-z and counts), without concordance,
change-pair or drop/rise event columns. Block mode retains those endpoints.
Compare requires matching scale and run-basis metadata and refuses event metrics
on slow targets. Old metric CSVs must be regenerated with explicit definitions.
