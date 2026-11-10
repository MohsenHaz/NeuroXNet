#!/usr/bin/env bash
# Downloads the public CQ500 dataset (491 studies, DICOM) and its published reads.
# Source: http://headctstudy.qure.ai/dataset  (mirror: https://www.kaggle.com/datasets/crawford/qureai-headct)
set -euo pipefail
mkdir -p data/raw/cq500 && cd data/raw/cq500
BASE="http://headctstudy.qure.ai/dataset"
curl -fL -o reads.csv "$BASE/reads.csv" || echo "reads.csv: download manually from $BASE"
for i in $(seq 0 49); do
  f="CQ500CT$i.zip"
  [ -f "$f" ] || curl -fL -o "$f" "https://s3.ap-south-1.amazonaws.com/qure.headct.study/$f" || echo "download $f manually from $BASE"
done
for z in *.zip; do unzip -q -n "$z" -d . ; done
echo "CQ500 downloaded to data/raw/cq500 (one folder per study: CQ500CT<id> CQ500CT<id>/<series>/*.dcm)"
