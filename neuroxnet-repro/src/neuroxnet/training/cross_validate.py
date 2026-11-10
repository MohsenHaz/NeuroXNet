"""Cross-validation helpers. Manuscript fold sizes/positive counts are explicit."""
import numpy as np, pandas as pd
FOLD_SIZES=[99,98,98,99,97]
FOLD_POS=[39,38,39,39,38]
def make_folds(cfg, seed=42):
    labels=pd.read_csv(cfg['data']['label_file'])
    if not {'study_id','final_label'}.issubset(labels.columns): raise ValueError('labels need study_id,final_label')
    
    if pd.api.types.is_numeric_dtype(labels.final_label):
        positive = labels.final_label.astype(int) > 0
    else:
        positive = labels.final_label.astype(str).str.lower() != 'normal'
    pos=labels.loc[positive,'study_id'].astype(str).to_numpy().copy()
    neg=labels.loc[~positive,'study_id'].astype(str).to_numpy().copy()
    if len(labels)!=491 or len(pos)!=193 or len(neg)!=298: raise ValueError('Expected 491 studies: 193 hydrocephalus-positive and 298 normal')
    rng=np.random.default_rng(seed); rng.shuffle(pos); rng.shuffle(neg); rows=[]; pi=ni=0
    for fold,(size,npos) in enumerate(zip(FOLD_SIZES,FOLD_POS),1):
        nneg=size-npos; rows += [(x,fold) for x in pos[pi:pi+npos]]; pi+=npos
        rows += [(x,fold) for x in neg[ni:ni+nneg]]; ni+=nneg
    return pd.DataFrame(rows,columns=['study_id','fold'])
