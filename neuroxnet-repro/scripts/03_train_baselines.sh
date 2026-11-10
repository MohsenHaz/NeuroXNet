#!/usr/bin/env bash
# Seven baselines, identical split (seed 42), preprocessing, schedule and heads. Each writes predictions/internal_test/<name>.csv
set -euo pipefail
for b in resnet50 vgg16 densenet121 effb0 effb4 transunet attunet; do
  python scripts/_train_one.py --config configs/baselines/$b.yaml --name $b --pred-dir internal_test --ckpt-dir checkpoints/baselines
done
(cd checkpoints && sha256sum neuroxnet_final.pt baselines/*.pt > SHA256SUMS.txt)
