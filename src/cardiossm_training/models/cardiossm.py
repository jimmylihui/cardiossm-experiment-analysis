"""Shared BCG encoder -> chronological Mamba-2 representations -> one MAP head.
The decoder is auxiliary waveform reconstruction, not a second BP estimator.
All historical/current waveforms use the same trainable encoder.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .bcgnet import BCGnet as U2


class WaveEncoderDecoder(nn.Module):
    """Keep BCGnet's U2 encoder/auxiliary decoder; discard its BP head."""
    def __init__(self):
        super().__init__()
        original = U2(nf=42)
        self.encoder = original.e
        self.decoder = original.d
        self.pool = original.pool
        self.reconstruction = original.out

    def forward(self, x, reconstruct=False):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        embedding = x.mean(-1)
        rec = None
        if reconstruct:
            for i, block in enumerate(self.decoder):
                skip = skips[-2-i]
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
                x = block(torch.cat([x, skip], dim=1))
            rec = self.reconstruction(x)
        return embedding, rec


class CardioSSM(nn.Module):
    architecture_version = 'continuous_axis_mamba2_v2'

    def __init__(self):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as error:
            raise ImportError("CardioSSM requires the external mamba-ssm CUDA dependency; see docs/training.md") from error
        self.wave = WaveEncoderDecoder()
        self.token_projection = nn.Linear(64 + 42, 64)
        self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(2)])
        self.ssm = nn.ModuleList([
            Mamba2(d_model=64, d_state=32, d_conv=4, expand=2,
                   headdim=32, chunk_size=64, use_mem_eff_path=False,
                   ngroups=1, rmsnorm=True, norm_before_gate=False, D_has_hdim=False)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(0.1)
        self.final_norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 1))
        self.prediction_only = False

    def encode_windows(self, x, features, reconstruct=False):
        """Shared raw-window encoder, used identically for past and present."""
        if x.ndim != 3 or x.shape[1:] != (1, 1000):
            raise ValueError('Expected [windows, 1, 1000] BCG inputs.')
        if features.shape != (len(x), 42):
            raise ValueError('Expected 42 aligned features for each BCG window.')
        embedding, reconstruction = self.wave(x, reconstruct=reconstruct)
        token = self.token_projection(torch.cat([embedding, features], dim=-1))
        return token, reconstruction, embedding

    def temporal(self, tokens):
        """Causal sequence; right padding cannot influence earlier outputs."""
        n = tokens.shape[1]
        z = F.pad(tokens, (0, 0, 0, (-n) % 64))
        for norm, block in zip(self.norms, self.ssm):
            z = z + self.dropout(block(norm(z)))
        return self.final_norm(z[:, :n])

    def temporal_step(self, token, states=None):
        """Advance explicit inference state once; never replay it after reading updates."""
        from ..streaming import mamba2_step
        if self.training:
            raise ValueError('streaming inference requires eval mode')
        if states is None:
            states = [None]*len(self.ssm)
        if len(states) != len(self.ssm):
            raise ValueError('one inference state per residual block required')
        z, updated = token, []
        for norm, block, previous in zip(self.norms, self.ssm, states):
            output, current = mamba2_step(block, norm(z), previous)
            z = z + output
            updated.append(current)
        return self.final_norm(z), updated

    def forward(self, waveforms, features, lengths=None, return_all=False,
                reconstruct=None):
        """End-to-end raw sequences, not two separate model predictions.

        waveforms: [batch, time, 1, 1000], features: [batch, time, 42].
        Right-pad only; lengths includes the current (last valid) window.
        The loader must restrict history to the same patient / axis, past
        windows since the last >30-min gap (the full-history training policy). No BP labels enter tokens.
        """
        if waveforms.ndim != 4 or waveforms.shape[2:] != (1, 1000):
            raise ValueError('Expected BCG sequence [batch, time, 1, 1000].')
        batch, time = waveforms.shape[:2]
        if features.shape != (batch, time, 42):
            raise ValueError('Features must be [batch, time, 42].')
        if lengths is None:
            lengths = torch.full((batch,), time, device=waveforms.device, dtype=torch.long)
        lengths = torch.as_tensor(lengths, device=waveforms.device, dtype=torch.long)
        if lengths.shape != (batch,) or (lengths < 1).any() or (lengths > time).any():
            raise ValueError('Each sequence must contain 1..time valid windows.')
        valid = torch.arange(time, device=waveforms.device)[None] < lengths[:, None]
        reconstruct = (self.training and not self.prediction_only) if reconstruct is None else reconstruct
        # Padding must not enter CNN BatchNorm or the reconstruction objective.
        token, rec, emb = self.encode_windows(waveforms[valid], features[valid], reconstruct)
        sequence = token.new_zeros(batch, time, 64)
        sequence[valid] = token
        hidden = self.temporal(sequence)
        if return_all:
            predictions = self.head(hidden) * valid[..., None]
        else:
            predictions = self.head(hidden[torch.arange(batch, device=hidden.device), lengths-1])
        return predictions, rec, dict(valid=valid, embedding=emb, hidden=hidden)
