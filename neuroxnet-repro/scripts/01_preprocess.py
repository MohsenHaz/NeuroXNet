"""Streaming CQ500 preprocessing. Does not retain all volumes in RAM."""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from neuroxnet.config import load_config
from neuroxnet.data.dicom_io import read_series, select_primary_series
from neuroxnet.data.preprocess import brain_window, preprocess_volume
from neuroxnet.data.slice_select import select_window

ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/neuroxnet.yaml'); ap.add_argument('--raw',default='data/raw/cq500'); ap.add_argument('--make-splits',action='store_true'); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
cfg=load_config(a.config); d=cfg['data']; out=pathlib.Path(d['root']); out.mkdir(parents=True,exist_ok=True)
pathlib.Path('logs').mkdir(parents=True,exist_ok=True); pathlib.Path('data/splits').mkdir(parents=True,exist_ok=True)
if a.make_splits:
    import subprocess
    subprocess.run([sys.executable,'scripts/make_frozen_splits.py','--labels',d['label_file'],'--seed',str(a.seed)],check=True)
lab=pd.read_csv(d['label_file'])
if not {'study_id','final_label'}.issubset(lab.columns): raise SystemExit('Populate the source labels CSV first; header-only template cannot preprocess.')
split_path=pathlib.Path(d['split_file'])
if not split_path.exists() or split_path.stat().st_size==0: raise SystemExit(f'Missing frozen split file: {split_path}')
split=pd.read_csv(split_path).set_index('study_id')['split']
# Welford-style aggregate over the actual training subset, one study at a time.
count=0; mean=0.0; M2=0.0
for sid in lab.study_id.astype(str):
    if split.get(sid)!='train': continue
    sdir=select_primary_series(pathlib.Path(a.raw)/sid); vol,thick,_=read_series(sdir)
    if thick>0 and not (d['thickness_range_mm'][0]<=thick<=d['thickness_range_mm'][1]): print(f'warning: {sid} slice thickness {thick} mm outside configured range')
    v=brain_window(vol,d['window_level'],d['window_width'])
    v,_=select_window(v,d['slices']);
    # Match the exact resize/interpolation used by preprocess_volume before statistics.
    from neuroxnet.data.preprocess import resize_inplane
    v=resize_inplane(v,d['height'],d['width']).astype(np.float64).ravel()
    n=v.size; m=float(v.mean()); s=float(((v-m)**2).sum())
    delta=m-mean; total=count+n
    M2 += s + delta*delta*count*n/total; mean += delta*n/total; count=total
if count==0: raise SystemExit('No training studies found in frozen split.')
std=float(np.sqrt(M2/count)); stats={'mean':float(mean),'std':std,'n_voxels':int(count),'split_file':str(split_path),'seed':a.seed}
pathlib.Path('logs/normalisation_constants.json').write_text(json.dumps(stats,indent=2))
cfg['data']['normalisation']={'mean':float(mean),'std':std}
# Second pass: preprocess/write one study at a time; memory use bounded to one volume.
for sid in lab.study_id.astype(str):
    sdir=select_primary_series(pathlib.Path(a.raw)/sid); vol,_,_=read_series(sdir)
    v,mask=preprocess_volume(vol,cfg); np.save(out/f'{sid}.npy',v.astype(np.float16))
print(f'Preprocessed {len(lab)} studies; train-only mean={mean:.6f}, std={std:.6f}')
