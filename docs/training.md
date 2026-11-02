# Models and supervised runners

Install the analysis package and compatible PyTorch/torchvision builds:

```bash
python -m pip install -e '.[training]'
python -m pip install 'mamba-ssm==2.3.2.post1' --no-build-isolation
```

Native CardioSSM training requires compatible Mamba2/CUDA. Dependencies and
weights are not bundled. See [protocol ambiguities](methods-ambiguities.md)
before selecting a study invocation. Configuration TOMLs are reference tables,
not executable run configurations; changing them does not change a run.

## External inputs and support

One external NPZ per subject contains `sig` [W,3,1000], `fid` [W,3,42], `map`
[W], increasing window starts `t` [W], increasing completion times
`availability_s` [W] (at least t+10), and boolean `retained_mask` [W]. Optional
`hr_bpm` [W] is monitor HR and is pooled into HRb for estimator assembly.
FSNet does not need fid and receives three-axis windows. Filtering, waveform
standardization and template/fiducial extraction are external preprocessing;
the loader cannot certify them from array shapes.

Main-cohort pretraining applies retained_mask before stride sampling and
normalization. Each of the three supplemental recordings instead needs an
explicit boolean `pretraining_mask`; their total support is checked against
2339 windows. They are never validation or test patients.

The external split JSON has `folds` mapping fold numbers to disjoint train/val/test
lists, plus `supplemental_subjects`. A non-smoke invocation checks four folds,
32 unique outer-test patients, four validation patients per fold, complete cohort
coverage and training-only supplemental recordings. Fixed identity assignments
and validation-permutation provenance stay private and must be supplied; the
unpublished input order/RNG is not guessed. Smoke inputs may use a reduced
manifest. All models must use the same support, folds and realized readings.

At least six windows are needed per retained two-minute block. Its availability
is max(bin end, all window completion times). A begins readings at the first test
block; B at the original recording start (required scalar recording_start_s). Schedules map every elapsed
4/8/12/24h time to the next completed retained block and deduplicate positions.
Optional reading_indices_N (A) and reading_indices_B_N (B) are checked against
this rule. The default interval is four hours.

## Explicit invocation choices

`cardiossm-train --help` lists the entry point. Input, split and output directories
are explicit; output must be outside the repository, including ignored folders.
No training is triggered by import or by analysis commands. No automatic resume
or retries occur. Do not run a study until the unresolved choices are settled.

Conditions are selected with `--conditions`:

- pretraining_only: no target fitting or updates;
- initial_only: initial fitting, no reading updates;
- periodic_only: pretrained weights, readings from recording start, SD5 noise,
  no initial fitting or replay;
- initial_periodic: initial fitting, error-free readings and calibration replay.

Initial conditions require `--initial-split-unit blocks|windows`. Periodic
conditions require positive `--online-learning-rate` and `--memory-samples`.
Initial+periodic additionally requires `--replay-samples` and
`--replay-source fit|calibration`. Counts denote samples in that architecture:
axis-windows for CardioSSM/BCGnet and whole three-axis windows for FSNet. These
numbers have no inferred Methods defaults. --online-sampling explicitly selects
uniform with_replacement or without_replacement; the latter rejects counts
exceeding the available memory/replay pool. The convention is recorded.

No-initial conditions require `--no-initial-test-scope all|final70`. FSNet requires
an explicit `--fsnet-batch-size` in whole windows. This does not assert equivalence
to the specified 256 axis-windows. Input/architecture/loss exceptions for the
cited FSNet remain unresolved; no new few-shot module or decoder is added.

Known budgets remain: every fourth training window, every twentieth validation
window; three pretraining epochs; AdamW weight decay 1e-4 and OneCycle peak1e-3;
initial fit lr1e-4, at most20 epochs, patience5. Initial fitting compares against
pretrained validation MAE and retains pretrained weights if none improves. The
MAP loss is standardized Huber delta1; CardioSSM/BCGnet add 0.1 reconstruction MSE
during pretraining/initial fit only. Reading updates use30 AdamW steps and no
decoder. Seed defaults to0. `--smoke` reduces budgets, but still requires explicit
unresolved choices and labels the output as smoke.

## Shared run partition

Every condition requires --run-gap-basis window_completion or block_completion.
Methods does not uniquely choose that clock; neither has a default. The loader
creates window_run_id and block run_id once, before training or inference.
History, inference resets, network/reference/HR means, calibration variance,
between-reading segments and scale averages all consume that partition. A block
crossing multiple window-defined runs is rejected rather than silently merged.
Run metadata is saved with predictions; checkpoints must declare the matching
basis. Old outputs lacking run IDs must be regenerated explicitly, not repaired
by independently guessing gaps in the analysis layer.

## CardioSSM history and state

Each patient/axis has one complete chronological stream, independent of block
position. Only a boundary from the explicit shared run partition starts a new stream run.
The context audit checks completeness as well as no future/cross-axis inputs.
The U-block encoder widths16/32/64/64/64 produce64 dimensions; 42 features are
concatenated and projected to64. Two Mamba2 blocks use expansion2, four heads of
32, state32, convolution width4 and dropout0.1. The head is LayerNorm64 followed
by64→128→1, ReLU and dropout0.2.

Training caches detached history CNN embeddings outside sampled targets; live
targets retain gradients, including through projection and temporal states.
Inference instead visits each window once with explicit per-axis H and causal
convolution buffers. These buffers survive weight updates; replay/refresh uses
separate objects and never replays or overwrites inference history. A new
patient/condition/interval gets fresh buffers. The reading block is predicted
before its label enters memory; only later windows use updated weights.

The unfused streaming recurrence operates on native Mamba2 parameters. Training
continues to use its native differentiable scan. Cached inference state is not a
model checkpoint and is not serialized into the public repository.

## Outputs and analysis

External output records contain raw/window, unsmoothed block and past-only
60min network streams, completed timestamps, per-condition masks/reading positions
and realized reading values. Monitor HR is included when supplied. Outputs are
not study results until independently evaluated. B's noise uses an explicitly
documented SHA-256 patient/interval/seed0 convention, shared across model names;
it is not asserted to match archived random draws.

Use `cardiossm-analysis prepare --models NAME=external-directory ... --tag IP4
--output external-aligned-directory` to validate matching support and extract
unsmoothed block inputs. The adapter never double-smooths or invents monitor HR.
For B, assembly uses other-patient calibration variance medians plus25; for A,
its own calibration residual variance is used. See [analysis schema](input-schema.md).

## Validation and runtime limits

Tests generate arrays in memory. CPU parameter fixtures exercise the actual
streaming/index/online-runner paths against independent recurrence equations,
including weight updates, replay isolation, true gaps and unlabelled outcomes.
They do not claim native kernel validation. Native forward/backward and
scan/stream numerical checks are opt-in with `CARDIOSSM_TEST_CUDA=1`, require
Mamba2/CUDA, and are skipped by the CPU-only test invocation.

Training holds a shared nonblocking lock and one worker, with software RSS,
free-memory and CUDA-memory-fraction limits. Failure is recorded externally;
there is no automatic retry. No patient input or study output is bundled.
