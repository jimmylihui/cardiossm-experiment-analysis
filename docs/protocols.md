# Experimental protocols

This document specifies calculations, not observed findings. Table and section numbers can change between manuscript versions; names below describe the analysis itself.

## Patient metrics

- **MAE:** compute absolute MAP error on each patient's test blocks, then give patients equal weight in the cohort mean.
- **r_test:** Pearson correlation over a patient's whole test record; aggregate valid patient values with the median.
- **r_seg:** exclude reading blocks, split the between-reading intervals at gaps exceeding 30min, and require at least five blocks, at least 1h span, and reference MAP SD at least 1mmHg. Average eligible segment Fisher-z within each patient. The cohort value is tanh of the median patient mean-z. This is not a pooled correlation.
- **Concordance:** five successive retained 2min steps with at most 12min elapsed; retain actual absolute MAP changes at least 5mmHg. An effectively zero predicted change receives half credit. Require at least ten qualifying pairs per patient; report the median patient concordance. The historical definition permits calibration-reading crossings. Use the explicitly named reading-free sensitivity to exclude them.

Constant or undefined correlations remain missing, with valid-patient and segment counts reported. For paired contrasts, use the common valid patient set. A difference of cohort median correlations differs from a median of patient delta-r values; name the reported quantity.

## Between-reading events

Use the same nominal 10min / maximum 12min endpoints and reject every window containing a calibration reading, including either endpoint. Label MAP decreases of at least 10mmHg and increases of at least 10mmHg separately. Scores are the corresponding signed predicted MAP change. A patient needs at least five positive and five negative windows for AUROC/AP. Aggregate eligible patient metrics with the median. All architectures and the no-BCG baseline use the same windows and eligibility.

## Retrospective offset+HR assembly

The historical comparison uses initial calibration from the first 30% of retained blocks, with initial neural fitting on an internal chronological train/validation split. Periodic error-free readings start at the first test block, then occur at the specified elapsed-hour interval. Each interval must have its own actual reading mask and corresponding neural trajectory.

The assembly implemented here retains these rules:

1. The neural stream is averaged externally using the specified protocol; the historical model comparison used a centered 60min average.
2. Subtract the whole-record neural mean. This is retrospective centering.
3. Initialize offset variance to 10000. Its process variance grows by qa per elapsed hour.
4. Initial-calibration pseudo-observations use smoothed reference MAP, every fifteenth retained calibration block, with variance 4.
5. Periodic readings observe MAP minus the HR contribution; observation variance is estimated from initial-calibration residuals.
6. Record the KF prediction before assimilating a reading at that block. Add the HR contribution with the fixed coefficient already reflected in its units.

The historical neural trajectory at a reading block was generated after its neural update; that order belongs to the external training/inference pipeline, not this scalar KF function. Never interchange these two update orders.

`OffsetHR` sets the neural stream to zero. To compare architecture contributions, match the HR stream, reading positions, quality-screened support, initial calibration, filter parameters, evaluation targets and reporting endpoints.

Original other-patient qa selection minimizes smoothed-reference MAE using other patients' test outcomes. It is retrospective outcome tuning. Keep it distinct from inner-validation selection. Fixed qa shared across models is an available sensitivity. If outcome-selected qa is used, repeat selection within each bootstrap sample and exclude all copies of the held-out patient.

## Past-only descriptive averages

At completed-block availability t, average values in (t−60min,t] within the same run. Use available startup blocks (`min_periods=1`) and restart after gaps >30min. This establishes the support of this averaging operation only; it does not remove future-dependent upstream quality screening or turn separately computed model tables into causal results.

## Temporal feature separation and max-null

For each patient, select the feature with strongest absolute correlation on the first 30% of retained blocks. Eligibility requires at least 60 finite pairs, feature variance, and reference MAP SD at least 1mmHg. Evaluate only blocks more than 60min after the selection boundary.

For null testing, split later blocks at gaps >180s. Keep chunks of at least 61 blocks. Jointly circular-shift every feature by the same randomly chosen amount within each chunk, between 30 and L−30 blocks. Compute the maximum absolute correlation over eligible features for each of 499 draws. Use the plus-one permutation p-value. Report unavailable tests explicitly rather than filling them with a favorable value.

These are descriptive association checks, distinct from prediction validation.

## Seeds, checkpoints, calibration and history

Hold subject folds, encoder, optimizer, calibration budget, update order and data support fixed when attributing differences to architecture or historical input. Select checkpoints using validation data, record all predefined seeds, and report each seed plus a clearly described across-seed summary. An average of patient metrics is not an ensemble of predictions.

Changing calibration duration can change the eligible cohort. State duration in elapsed time, exclusion criteria and the common paired patient set. A controlled history experiment requires freshly matched current-only, bounded-history and full-history training conditions. This repository supplies analysis functions, not those missing training runs.

## Uncertainty

Use paired patient bootstrap, not independent block or overlapping-window resampling. Default intervals are 10000-resample, pointwise percentile 95% CIs conditional on the fitted models and specified tuning procedure. They do not capture the full variability of retraining or correct for searching across many models, intervals and endpoints.
