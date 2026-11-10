#!/usr/bin/env bash
# Eight ablation rows of Table 12 on the SAME 73-study held-out test split as Tables 5-7 (seed 42).
# The full model row of Table 12 is predictions/internal_test/neuroxnet.csv (copied), so it is identical to Tables 5-7 by construction.
set -euo pipefail
for c in backbone_only plus_msam plus_csag plus_fpn plus_label_smoothing det_only focal cb; do
  python scripts/_train_one.py --config configs/ablation/$c.yaml --name $c --pred-dir ablation --ckpt-dir checkpoints/ablation
done
cp predictions/internal_test/neuroxnet.csv predictions/ablation/full_model.csv
python scripts/_param_counts.py
