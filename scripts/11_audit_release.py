#!/usr/bin/env python3
"""Audit real release artifacts; schema-only templates and placeholders do not count."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={
 "CQ500 split":"data/splits/cq500_split_seed42.csv",
 "CV folds":"data/splits/cq500_cv5_folds.csv",
 "derived severity labels":"data/labels/cq500_severity_labels.csv",
 "reader annotations":"data/labels/cq500_reader1_reader2_adjudicated.csv",
 "approved annotation protocol":"data/labels/annotation_protocol.pdf",
 "final checkpoint":"checkpoints/neuroxnet_final.pt",
 "checkpoint checksums":"checkpoints/SHA256SUMS.txt",
 "internal NeuroXNet predictions":"predictions/internal_test/neuroxnet.csv",
 "external predictions":"data/external/external_predictions_deidentified.csv",
 "Grad-CAM ROI IDs":"data/labels/gradcam_roi_subset_ids.csv",
 "Grad-CAM ROI masks":"data/roi_masks",
 "result tables":"results/tables",
 "result figures":"results/figures",
 "training logs":"logs",
}
def populated(path):
 if not path.exists(): return False
 if path.is_dir():
  files=[x for x in path.rglob('*') if x.is_file() and x.name not in {'.gitkeep','README.md'} and not x.name.endswith('.README.md')]
  return bool(files)
 if path.suffix=='.csv':
  try: return sum(1 for _ in path.open(encoding='utf-8'))>1
  except Exception: return False
 if path.name=='annotation_protocol.pdf':
  # Placeholder text is explicitly not the study protocol.
  try:
   from pypdf import PdfReader
   text=' '.join(x.extract_text() or '' for x in PdfReader(str(path)).pages).lower()
   return 'source required' not in text and 'placeholder' not in text
  except Exception: return False
 if path.name=='SHA256SUMS.txt': return 'No trained checkpoints' not in path.read_text(errors='ignore')
 return path.stat().st_size>0
rows=[]
for label,rel in REQUIRED.items():
 ok=populated(ROOT/rel); rows.append({'artifact':label,'path':rel,'present':ok})
 print(f"{'PRESENT' if ok else 'MISSING':7} {label}: {rel}")
missing=[x for x in rows if not x['present']]
report={'present':len(rows)-len(missing),'required':len(rows),'missing':missing}
out=ROOT/'results/checks/release_artifact_audit.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
print(f"\n{report['present']}/{report['required']} required artifact groups present.")
print('Presence only: the audit does not validate scientific correctness, governance, or reproducibility.')
