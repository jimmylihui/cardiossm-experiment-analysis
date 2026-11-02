"""Choices not fixed by the active Methods must be explicit at invocation."""
from dataclasses import dataclass
import math
from cardiossm_analysis.runs import RUN_GAP_BASES

CONDITIONS = ('pretraining_only', 'initial_only', 'periodic_only', 'initial_periodic')


@dataclass(frozen=True)
class TrainingChoices:
    initial_split_unit: str | None = None
    online_learning_rate: float | None = None
    memory_samples: int | None = None
    replay_samples: int | None = None
    replay_source: str | None = None
    fsnet_batch_size: int | None = None
    no_initial_test_scope: str | None = None
    online_sampling: str | None = None
    run_gap_basis: str | None = None

    def validate(self, architecture, conditions):
        if self.run_gap_basis not in RUN_GAP_BASES:
            raise ValueError('specify the unresolved run gap basis: window_completion or block_completion')
        if not conditions or len(conditions) != len(set(conditions)) or any(c not in CONDITIONS for c in conditions):
            raise ValueError('conditions must be unique explicit supported conditions')
        if any(c in ('initial_only', 'initial_periodic') for c in conditions):
            if self.initial_split_unit not in ('blocks', 'windows'):
                raise ValueError('Methods leaves the internal split unit unresolved; specify blocks or windows')
        if any(c in ('periodic_only', 'initial_periodic') for c in conditions):
            if self.online_sampling not in ('with_replacement','without_replacement'):
                raise ValueError('specify uniform online sampling with_replacement or without_replacement')
            if self.online_learning_rate is None or not math.isfinite(self.online_learning_rate) or self.online_learning_rate <= 0:
                raise ValueError('specify the unresolved online learning rate explicitly')
            if not isinstance(self.memory_samples, int) or self.memory_samples < 1:
                raise ValueError('specify a positive online memory sample count')
        if 'initial_periodic' in conditions:
            if not isinstance(self.replay_samples, int) or self.replay_samples < 1 or self.replay_source not in ('fit', 'calibration'):
                raise ValueError('specify the calibration replay count and source (fit or calibration)')
        if any(c in ('pretraining_only', 'periodic_only') for c in conditions):
            if self.no_initial_test_scope not in ('all', 'final70'):
                raise ValueError('specify the unresolved no-initial-calibration scoring scope')
        if architecture == 'fsnet' and (not isinstance(self.fsnet_batch_size, int) or self.fsnet_batch_size < 1):
            raise ValueError('specify FSNet batch size in three-axis windows; no axis-budget conversion is assumed')
