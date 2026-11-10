#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import sys; sys.path.insert(0,"src")
from neuroxnet.config import load_config; from neuroxnet.training.cross_validate import run_cv
run_cv(load_config("configs/cv5.yaml"), "neuroxnet", seed=42)
PY
(cd checkpoints && sha256sum neuroxnet_final.pt baselines/*.pt cv5/*.pt > SHA256SUMS.txt)
