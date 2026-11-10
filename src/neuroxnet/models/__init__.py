from .neuroxnet import NeuroXNet, build_model, count_parameters
from .baselines import build_baseline

def build(cfg):
    return build_model(cfg) if cfg["model"]["name"] == "neuroxnet" else build_baseline(cfg)
