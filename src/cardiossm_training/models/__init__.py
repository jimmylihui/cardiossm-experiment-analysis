"""Three model definitions; external Mamba/torchvision imports are lazy."""
from .bcgnet import BCGnet
from .cardiossm import CardioSSM
from .fsnet import BCG_BP_FSNet


def build(name):
    if name == "cardiossm":
        return CardioSSM()
    if name == "bcgnet":
        return BCGnet()
    if name == "fsnet":
        return BCG_BP_FSNet()
    raise ValueError("unknown model")
