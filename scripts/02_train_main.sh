#!/usr/bin/env bash
set -euo pipefail
python scripts/_train_one.py --config configs/neuroxnet.yaml --name neuroxnet --pred-dir internal_test --ckpt-dir checkpoints
mv checkpoints/neuroxnet.pt checkpoints/neuroxnet_final.pt
python - <<'PY'
import sys; sys.path.insert(0,"src"); import torch
from neuroxnet.models import build; from neuroxnet.models.neuroxnet import count_parameters
ck=torch.load("checkpoints/neuroxnet_final.pt",map_location="cpu"); m=build(ck["cfg"]); m.load_state_dict(ck["model"])
print("NeuroXNet parameters (feature extractor):", count_parameters(m, True), " total:", count_parameters(m))
PY
(cd checkpoints && sha256sum neuroxnet_final.pt > SHA256SUMS.txt)
