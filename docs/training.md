# Models and training

## Installation

Install compatible PyTorch and torchvision builds, then:

```bash
python -m pip install -e '.[training]'
```

CardioSSM also requires Mamba2 on a compatible CUDA system:

```bash
python -m pip install 'mamba-ssm==2.3.2.post1' --no-build-isolation
```

Dependency documentation: [Mamba](https://github.com/state-spaces/mamba/tree/v2.3.2.post1), [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Dependencies are imported externally; their source and weights are not bundled.

## Input arrays

Supply one external NPZ per subject, containing already preprocessed 10s windows at 100Hz. No sample files are distributed.

| Key | Shape | Meaning |
|---|---|---|
| `sig` | W × 3 × 1000 | Three-axis BCG windows |
| `fid` | W × 3 × 42 | Aligned engineered features for CardioSSM and BCGnet |
| `map` | W | MAP target in mmHg |
| `t` | W | Strictly increasing window-start times, seconds |
| `availability_s` | W | Strictly increasing completion times, at least t+10s |
| `retained_mask` | W | Externally fixed retained support |
| `reading_indices_4` | R | Four-hour reading positions in retained 2min blocks |
| `reading_indices_8` | R | Eight-hour reading positions |
| `reading_indices_12` | R | Twelve-hour reading positions |
| `reading_indices_24` | R | Twenty-four-hour reading positions |

Each retained 2min block requires at least six windows. A block becomes available after its time bin closes and all its windows complete. Reading indices begin at the first test block. They are optional: the loader otherwise generates elapsed-hour schedules, maps each scheduled time to the next retained block and removes duplicate positions. Supplied indices are checked against that rule. Preprocessing includes the waveform filtering, window standardization and feature extraction described in Methods; this loader consumes those prepared inputs. FSNet consumes the three-axis waveform and does not require `fid`.

Supply a separate JSON mapping fold numbers to disjoint `train`, `val` and `test` subject lists. Subject identifiers and fold assignments remain outside the repository. An outer-test subject occurs in only one fold. All models use the same supplied folds, retained support and reading locations.

## Commands

```bash
cardiossm-train --model cardiossm --input private_inputs/windows \
  --splits private_inputs/folds.json --output outputs/cardiossm-seed0 \
  --seed 0 --device cuda:0 --intervals 4 8 12 24 --lock outputs/worker.lock

cardiossm-train --model bcgnet --input private_inputs/windows \
  --splits private_inputs/folds.json --output outputs/bcgnet-seed0 \
  --seed 0 --device cuda:0 --intervals 4 8 12 24 --lock outputs/worker.lock

cardiossm-train --model fsnet --input private_inputs/windows \
  --splits private_inputs/folds.json --output outputs/fsnet-seed0 \
  --seed 0 --device cuda:0 --intervals 4 8 12 24 --lock outputs/worker.lock
```

## Shared training procedure

- Pretraining uses every fourth training window and every twentieth validation window, three epochs, batch size 256, AdamW weight decay 1e-4, and OneCycle peak learning rate 1e-3. CardioSSM and BCGnet sample individual axis-windows; FSNet samples three-axis windows. Feature/MAP normalization is fitted on training subjects only, and standardized features are clipped to ±10.
- The MAP objective is standardized Huber loss with threshold 1. CardioSSM and BCGnet additionally use their waveform decoders for reconstruction MSE weighted by 0.1. FSNet retains its single-output ResNet18 architecture without an auxiliary decoder.
- Initial calibration is the first 30% of retained blocks. Its first 80% of raw windows train the model; the last 20% select weights. Initial fitting uses learning rate 1e-4, at most 20 epochs, patience five, and keeps the pretrained weights if no epoch improves validation MAE.
- At each reading, the current block is predicted with the existing weights. The reading then labels its windows, and 30 AdamW steps use 64 accumulated reading samples plus 64 initial-calibration replay samples. Updated weights predict only later blocks. Each interval starts from the selected initial-fit weights.
- The output includes raw predictions, block means and past-only 60min block means, with support (t−3600,t], startup min_periods=1 and run resets after gaps >30min. Completed-block availability supplies t. No whole-record neural centering is applied.

The default training seed is zero. Each new invocation requires a fresh output directory. Calibration readings are error-free block MAP values at the supplied locations.

## Architectures

**CardioSSM** uses a shared residual U-block encoder with widths 16/32/64/64/64, yielding a 64-dimensional embedding. Concatenate the 42 features and project to 64 dimensions. Two Mamba2 blocks use state 32, expansion 2, head dimension 32, convolution width 4 and chunk 64. A 64→128→1 head predicts standardized MAP. Historical streams stay within a patient, axis and within-bin window position, and reset at >30min gaps. Sampled CNN targets have live gradients; older cached CNN embeddings are detached and refreshed each epoch/update. Projection and SSM gradients span the valid history. The decoder is used only for pretraining and initial fitting.

**BCGnet** uses the current axis-window, the residual U-block encoder/decoder and the 42-feature MAP head without historical tokens.

**BCG-BP-FSNet** uses a ResNet18 on [batch,3,1,1000] with a single MAP head and per-window/channel standardization. It uses the same supervised stages and update schedule above.

Offset and HR are assembled separately using the analysis package. HR is block monitor HR minus its past-only 60min mean, with beta=1. The scalar Kalman filter predicts the block before assimilating its reading.

## Runtime and checks

Training holds a shared nonblocking file lock and runs one process. Default software checks stop above 16GiB owned process-tree RSS or below 64GiB host available memory; CUDA memory fraction is 0.30. Failure is saved and never automatically retried. CPU checks can override the host-free threshold explicitly.

```bash
cardiossm-train --model bcgnet --input private_inputs/windows \
  --splits private_inputs/folds.json --output outputs/bcgnet-smoke \
  --device cpu --smoke --min-host-free-gib 0
```

Smoke mode uses one fold, one test subject, one pretraining epoch, one fitting epoch and one online optimizer step. Its outputs are labeled as smoke. Tests generate tensors in memory and check reconstruction/backpropagation, supervised fitting, history support, past-only smoothing and prediction-before-update timing. Native Mamba2 forward/backward testing requires CUDA and is explicitly skipped when unavailable.
