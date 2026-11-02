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

The `offset_hr` Python API accepts these already computed past-only contributions. The CLI uses fixed qa shared across models; other-patient test-MAE selection is available through `select_qa_other_patients`, with `candidate_interval` for bootstrap reselection. The generated `OffsetHR` baseline sets the neural stream to zero.

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
