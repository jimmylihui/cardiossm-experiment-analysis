# CardioSSM experiment analysis

Analysis code and experimental protocol descriptions accompanying **Beyond Calibration: Blood Pressure Tracking With Bed-Based Cardiomechanical Sensing**.

This repository contains reusable analysis functions, a local command-line interface, input schemas, and tests. It contains **no datasets, patient records, saved predictions, model checkpoints, figures, or numerical study results**. CardioSSM was called BCGSSM in earlier experiment code.

## Analyses included

| Analysis | Implementation |
|---|---|
| Whole-test and between-reading correlation | `metrics.py`: patient Pearson r, mean segment Fisher-z, cohort aggregation |
| MAP error and change-direction concordance | `metrics.py`: equal-patient MAE and thresholded direction agreement |
| Event identification between readings | `events.py`: reading-free windows, drop/rise AUROC and average precision |
| Matched offset and HR processing | `calibration.py`: shared assembly for neural outputs and a zero-neural baseline |
| Paired uncertainty and tuning sensitivity | `bootstrap.py`: patient resampling, identity-excluded qa reselection |
| Past-only descriptive analysis | `temporal.py`: completed-observation support, startup and gap resets |
| Feature associations over time | `features.py`: training-only selection, temporal embargo, joint max-null |
| Temporal output scales | `scales.py`: matched non-overlapping output averages for all architectures |

The model label is deliberately separate from the analysis: CardioSSM, BCGnet, BCG-BP-FSNet and other saved prediction streams can use the same functions and support masks.

## Installation

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

No command downloads data, contacts an experiment server, or launches training. Keep inputs in an external directory or in the ignored `private_inputs/` directory. See [input schemas](docs/input-schema.md).

## Usage

Compute patient metrics from externally supplied, already assembled predictions:

```bash
cardiossm-analysis score --input private_inputs/aligned/ --output outputs/patient_metrics.csv
```

Apply exactly the same offset+HR assembly to each supplied neural stream and a generated `OffsetHR` zero-neural baseline, with fixed qa:

```bash
cardiossm-analysis score --input private_inputs/neural/ --assemble-offset-hr --qa 2 \
  --output outputs/matched_patient_metrics.csv
```

Compare paired patients, with correlation as the reporting focus:

```bash
cardiossm-analysis compare --input outputs/matched_patient_metrics.csv \
  --reference OffsetHR --metric r_seg --resamples 10000 --seed 0 \
  --output outputs/correlation_contrasts.csv
```

Use `--metric r_test`, `concordance`, `mae`, `drop_auroc`, `rise_auroc`, `drop_average_precision`, or `rise_average_precision` for the corresponding contrasts. The CLI comparison uses fixed tuning. Outcome-selected qa must instead use the Python `candidate_interval` API to repeat selection inside every bootstrap sample.

```bash
cardiossm-analysis score --input private_inputs/aligned/ --concordance-no-reading \
  --output outputs/reading_free_concordance.csv
cardiossm-analysis scales --input private_inputs/native/ \
  --seconds 10 30 60 120 300 600 1200 1800 3600 --output outputs/scales.csv
cardiossm-analysis features --input private_inputs/features/ \
  --resamples 499 --seed 0 --output outputs/feature_associations.csv
```

The scale command requires native-resolution series. It refuses to generate scales finer than the supplied sampling interval. It computes non-overlapping averages; a centered rolling-average analysis is a separate operation in `temporal.py`.

## Experimental interpretation

Read [protocols](docs/protocols.md) before assembling a comparison. The descriptive past-only averages and the historical retrospective model evaluation are separate protocols. Centered averaging, whole-record centering, and outcome-based qa selection remain explicit in the retrospective assembly.

Report all predefined seeds and calibration intervals. Select neural checkpoints using validation data, preserve the patient split, and pair the same patients across models. A point estimate above a baseline and a paired interval excluding zero are different findings. Averaging patient metrics across seeds is not a prediction ensemble.

This package is a portable analysis refactor. It consumes externally prepared observations or predictions; it does not reproduce private signal extraction, neural training, or every archived experiment invocation.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests generate numerical fixtures in memory. No example patient files or study output is checked in. Binary data and common result formats are excluded by `.gitignore`; `scripts/check_repository.py` also checks the committed file list before publication.

## Repository layout

```text
src/cardiossm_analysis/   reusable calculations and CLI
configs/                 protocol reference values
docs/                    input schemas and experimental definitions
tests/                   generated numerical fixtures and regression checks
scripts/                 repository content check
```
