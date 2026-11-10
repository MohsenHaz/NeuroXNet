"""Create manuscript-sized patient-level split/fold assignments from the source label table.
Requires true study IDs and final_label. Does not invent IDs. Commit output once reviewed.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

TOTALS={'normal':298,'mild':78,'moderate':69,'severe':46}
LABEL_MAP={0:'normal',1:'mild',2:'moderate',3:'severe','0':'normal','1':'mild','2':'moderate','3':'severe'}
def _class(v):
    x=str(v).strip().lower()
    return LABEL_MAP.get(x, x)
TEST={'normal':44,'mild':12,'moderate':10,'severe':7}
FOLD_SIZES=[99,98,98,99,97]
FOLD_POS=[39,38,39,39,38]

def _allocate(labels, quotas, seed):
    rng=np.random.default_rng(seed); out=[]
    for cls,n in quotas.items():
        ids=np.asarray(labels.loc[labels.final_label.astype(str)==cls,'study_id'].astype(str))
        if len(ids)<n: raise ValueError(f'{cls}: need {n}, found {len(ids)}')
        rng.shuffle(ids); out.extend((sid,cls) for sid in ids[:n])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',default='data/labels/cq500_severity_labels.csv'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',default='data/splits'); a=ap.parse_args()
    lab=pd.read_csv(a.labels)
    if not {'study_id','final_label'}.issubset(lab.columns): raise SystemExit('Need populated labels CSV with study_id,final_label; header-only template is not enough.')
    lab=lab[['study_id','final_label']].drop_duplicates('study_id').copy()
    lab['_class']=lab.final_label.map(_class)
    if len(lab)!=491: raise SystemExit(f'Expected 491 unique studies; found {len(lab)}')
    counts=lab._class.value_counts().to_dict()
    if counts!=TOTALS: raise SystemExit(f'Class totals differ from manuscript protocol: {counts}; expected {TOTALS}')
    # Shuffle each class independently; allocate test counts exactly. Allocate validation from remaining
    # classes by largest-remainder proportional allocation, then train is the remainder.
    rng=np.random.default_rng(a.seed); split=[]
    for cls in TOTALS:
        ids=np.asarray(lab.loc[lab.final_label==cls,'study_id'].astype(str)); rng.shuffle(ids)
        ntest=TEST[cls]; remain=ids[ntest:]
        # proportional target for validation; deterministic largest-remainder rounding to 74 overall
        split.extend((sid,'test') for sid in ids[:ntest])
        # allocate validation proportionally to class remainder
        split.extend((sid,'_remain_'+cls) for sid in remain)
    remain_df=pd.DataFrame(split,columns=['study_id','split'])
    # Determine exact per-class validation quotas using proportional allocation with deterministic tie-break.
    rem_counts={c:TOTALS[c]-TEST[c] for c in TOTALS}; raw={c:74*rem_counts[c]/sum(rem_counts.values()) for c in TOTALS}
    valq={c:int(np.floor(raw[c])) for c in TOTALS}
    for c in sorted(TOTALS,key=lambda x:(raw[x]-valq[x],x),reverse=True)[:74-sum(valq.values())]: valq[c]+=1
    for cls in TOTALS:
        ix=remain_df.index[remain_df.split=='_remain_'+cls].to_numpy(); rng.shuffle(ix)
        val_ix=set(ix[:valq[cls]])
        for i in ix: remain_df.loc[i,'split']='val' if i in val_ix else 'train'
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    final=remain_df[['study_id','split']].sort_values('study_id'); final.to_csv(out/'cq500_split_seed42.csv',index=False)
    # 5 folds with manuscript's exact fold sizes and binary-positive counts. Here positive means final_label != normal.
    pos=lab[lab._class!='normal'].study_id.astype(str).to_numpy(); neg=lab[lab._class=='normal'].study_id.astype(str).to_numpy(); rng.shuffle(pos); rng.shuffle(neg)
    folds=[]; p0=n0=0
    for f,(sz,np_) in enumerate(zip(FOLD_SIZES,FOLD_POS),1):
        nn=sz-np_;
        folds.extend((sid,f) for sid in pos[p0:p0+np_]); p0+=np_
        folds.extend((sid,f) for sid in neg[n0:n0+nn]); n0+=nn
    pd.DataFrame(folds,columns=['study_id','fold']).sort_values('study_id').to_csv(out/'cq500_cv5_folds.csv',index=False)
    print('Wrote exact target counts. Review against the originally frozen ID allocation before treating these as the historical experiment split.')
if __name__=='__main__': main()
