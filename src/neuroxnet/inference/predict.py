import torch, pandas as pd
from ..models import build
from ..training.train import predict_split

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build(ck["cfg"]).to(device); model.load_state_dict(ck["model"]); model.eval()
    return model, ck["cfg"], ck.get("seed", 42)

def predict_from_checkpoint(ckpt_path, ids, split, out_csv, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, seed = load_model(ckpt_path, device)
    return predict_split(model, cfg, ids, split, seed, device, out_csv)
