"""Three-axis ResNet18 with a single MAP output."""
from torch import nn


def BCG_BP_FSNet():
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    model.architecture_version = "fsnet_resnet18_map_v1"
    return model


def normalize_windows(x):
    """Per 10s window, per channel: standardize over its 1000 time samples."""
    if x.ndim != 3 or tuple(x.shape[1:]) != (3, 1000):
        raise ValueError("Expected [windows,3,1000]")
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, correction=0).clamp_min(1e-6)
