"""Signal-only inference state, separate from replay/training caches."""
import numpy as np
import torch
from torch.nn import functional as F
from . import data


def mamba2_step(block, token, state=None):
    """One window using the configured Mamba2 parameters and explicit (conv,H).

    This unfused recurrence follows the Methods equations. Training still uses
    the native differentiable scan. No weights, labels or history are stored
    here; callers own the state. Supports this model's full-SSM, one-group setup.
    """
    if block.ngroups != 1 or block.d_ssm != block.d_inner or block.D_has_hdim:
        raise ValueError('unsupported Mamba2 configuration for the Methods recurrence')
    projected = block.in_proj(token)
    gate, xbc, dt = torch.split(projected, [block.d_ssm, block.d_ssm+2*block.d_state, block.nheads], -1)
    if state is None:
        conv = torch.zeros(len(token), xbc.shape[-1], block.d_conv, device=token.device)
        h = torch.zeros(len(token), block.nheads, block.headdim, block.d_state, device=token.device)
    else:
        conv, h = state
    conv = torch.cat((conv[..., 1:], xbc.float().unsqueeze(-1)), -1)
    value = (conv * block.conv1d.weight[:, 0].float()).sum(-1)
    if block.conv1d.bias is not None:
        value = value + block.conv1d.bias.float()
    v, b, c = torch.split(F.silu(value), [block.d_ssm, block.d_state, block.d_state], -1)
    v = v.reshape(len(token), block.nheads, block.headdim)
    delta = F.softplus(dt.float() + block.dt_bias.float())
    decay = torch.exp(-block.A_log.float().exp() * delta)
    h = decay[..., None, None]*h + delta[..., None, None]*v[..., None]*b[:, None, None, :]
    output = (h*c[:, None, None, :]).sum(-1) + block.D.float()[None, :, None]*v
    output = block.norm(output.flatten(1).to(projected.dtype), gate)
    return block.out_proj(output), (conv, h)


class StreamPredictor:
    """Visit every window once; retain H and convolution state across updates.

    Instantiate anew for each patient/condition/interval after loading starting
    weights. Optimizer steps must not replace this object. Replay Engine calls
    neither read nor modify these states. A boundary in the shared explicit partition resets the affected stream.
    """
    def __init__(self, net, pool, norm, device):
        self.net, self.pool, self.norm, self.device = net, pool, norm, device
        self.features = data.nz(pool.F, norm)
        self.streams = {}

    @torch.no_grad()
    def predict(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError('expected a vector of chronological axis-window ids')
        self.net.eval()
        predictions = []
        for start in range(0, len(ids), 512):
            chunk = ids[start:start+512]
            x = torch.as_tensor(self.pool.X[chunk].astype(np.float32), device=self.device)
            f = torch.as_tensor(self.features[chunk], device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                tokens, _, _ = self.net.encode_windows(x, f, reconstruct=False)
                for i, q in enumerate(chunk):
                    data.CHECK_RESOURCE()
                    key = (int(self.pool.pid[q]), int(self.pool.axis[q]))
                    run, position = map(int, self.pool.key[q])
                    previous = self.streams.get(key)
                    if position == run:
                        if previous is not None and position <= previous[1]:
                            raise ValueError('cannot replay or rewind an inference stream')
                        states = None
                    else:
                        if previous is None or previous[0] != run or previous[1]+1 != position:
                            raise ValueError('inference must include every preceding window in order')
                        states = previous[2]
                    hidden, states = self.net.temporal_step(tokens[i:i+1], states)
                    predictions.append(self.net.head(hidden).float().item())
                    self.streams[key] = (run, position, states)
        return np.asarray(predictions)*self.norm[3] + self.norm[2]
