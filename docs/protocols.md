# Experimental protocols

## Estimator assembly

Two-minute block predictions are issued at completed-block availability. The network term is the within-run past-only 60min average of block outputs, with support (t−3600,t], available startup blocks and one shared run partition. The window-versus-block gap clock is an explicit required choice; downstream consumers never recompute it independently. The HR term is block monitor HR minus its past-only 60min mean, with beta=1mmHg/bpm. No whole-record mean is subtracted from the network term.

At a reading block, the neural network and Kalman filter first predict using their existing states. The reading updates both for later blocks only. The offset variance starts at 10000 and grows by qa per elapsed hour. Initial-calibration pseudo-observations use past-only smoothed MAP every fifteenth retained calibration block with observation variance 4. Periodic readings observe MAP minus the HR contribution; observation variance is the initial-calibration variance of MAP minus its past-only mean minus HR.

The no-BCG `OffsetHR` baseline uses a zero neural contribution with the same HR, calibration and reading masks. Candidate qa values are 0.5, 2, 8 and 32; selection uses other patients' test MAE and excludes the held-out patient's identity. Repeat this selection within each bootstrap resample when using outcome-selected qa. The scoring CLI performs this selection by default; explicit --qa selects a fixed sensitivity. Use --candidate-output plus compare --tuning reselect for within-bootstrap tuning. Aggregation and contrast must also be explicit. Configuration TOMLs are reference values, not run settings.

## Training

Use the explicit condition branches and unresolved choices in [training](training.md) and [protocol ambiguities](methods-ambiguities.md). Do not infer equal raw-window budgets from equal batch numbers across architectures. Select neural checkpoints using validation data and report every predefined seed. The initial calibration is the first 30% of retained blocks; the remaining blocks form test support. The first reading is at the first test block, then every specified elapsed-hour interval, mapped to the next retained block.

## Patient metrics

- **MAE:** mean absolute error within each patient, then equal-patient cohort mean.
- **r_test:** whole-test patient Pearson correlation, then cohort median.
- **r_seg:** exclude reading blocks; split between-reading intervals at gaps >30min; require at least five blocks, 1h span and reference MAP SD ≥1mmHg. Average segment Fisher-z within each patient. The cohort value is tanh of the median patient mean-z.
- **Concordance:** five successive retained 2min steps with elapsed time ≤12min; actual |MAP change| ≥5mmHg; at least ten qualifying pairs per patient. Zero predicted change gets half credit. Report median patient concordance. Reading-crossing windows are included by default, with an explicit reading-free option.

Constant correlations remain missing. Report valid patient/segment counts and use common valid patients for paired contrasts. A difference of cohort medians is distinct from the median patient delta-r.

## Large changes between readings

Use the same nominal 10min / maximum 12min endpoints and reject any window containing a reading, including either endpoint. Label MAP falls ≥10mmHg and rises ≥10mmHg separately; score the signed predicted change. Each patient needs at least five positives and five negatives. Report median patient AUROC/AP on identical eligible support across models.

## Feature associations

Compute past-only 60min means of retained block MAP and features. Select the strongest absolute correlation on the first 30% of blocks, requiring at least 60 finite pairs, feature variance and MAP SD ≥1mmHg. Evaluate blocks more than 60min after the selection boundary.

For the max-null, split later blocks at gaps >180s and keep chunks of at least 61 blocks. Jointly circular-shift all features within each chunk by 30..L−30 blocks. Use the maximum absolute eligible-feature correlation for each of 499 draws and plus-one permutation p-values. These calculations describe associations; the prediction metrics use separately fitted models.

## Scales and uncertainty

Output-scale analysis averages reference and predictions on identical non-overlapping support in each run. Use native observations, not an already smoothed stream, and never generate a scale finer than the input resolution. This does not retrain the encoder or change history length.

Use paired patient bootstrap with 10000 draws and pointwise percentile 95% intervals. Preserve pairing across models and exclude every bootstrap copy of the held-out identity during qa selection. An across-seed mean of patient metrics is not a prediction ensemble. Intervals are conditional on the fitted models and specified tuning procedure.

## Defined capabilities versus unresolved experiments

The CLI prepare adapter validates condition/support alignment before assembly.
Only explicitly selected models receive offset and HR. Network-only slow scoring
uses matched past-only means, whereas output-scale analysis is non-overlapping
averaging. These are distinct analyses. The native broadcast policy for complete
block estimators is unresolved and is not inferred by the adapter.

Mean/median patient aggregation and difference-of-aggregates/median-paired-delta
are separate explicit modes. Holm correction requires supplied p-values and
comparison family labels; the paired test and family are not chosen from old
commented instructions. SHAP, full-stage masking and original figure-generation
workflows remain unimplemented. No study numbers are validated by unit tests.

The shared run_id is also used for between-reading segments, change windows,
feature means and non-overlapping scale bins. Missing run metadata is rejected
by the CLI. Slow scoring emits level metrics only; block-change metrics remain
exclusive to block targets. Scale/run metadata is recorded and must match in
paired comparisons. This prevents combining distinct event or smoothing targets
under identical metric names.
