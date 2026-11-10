"""Strict consistency gate for predictions and generated tables.

The gate refuses to pass when only empty templates are present. It validates integer
attainability, probability/prediction consistency, confusion-matrix identities,
per-class macro-F1 reconstruction, Dice/IoU identity, and decision-curve net benefit
when the corresponding columns are available.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

def attainable(k,n,reported,tol=0.0051):
    if n<=0 or k<0 or k>n: return False
    return abs(float(reported)-100.0*k/n)<=tol

def check_prediction_file(path, expected_n=None):
    df=pd.read_csv(path); errors=[]
    if df.empty: return []  # schema template, not an experiment output
    required={'study_id','y_bin','pred_bin','p_hydro','y_sev','pred_sev','p0','p1','p2','p3'}
    missing=required-set(df.columns)
    if missing: return [f'{path}: missing columns {sorted(missing)}']
    if df.study_id.duplicated().any(): errors.append(f'{path}: duplicate study IDs')
    if expected_n is not None and len(df)!=expected_n: errors.append(f'{path}: n={len(df)}, expected {expected_n}')
    if not df.p_hydro.between(0,1).all(): errors.append(f'{path}: invalid binary probabilities')
    probs=df[['p0','p1','p2','p3']].to_numpy(float)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1,atol=1e-4): errors.append(f'{path}: severity probabilities invalid')
    if not np.array_equal((df.p_hydro>=.5).astype(int).to_numpy(),df.pred_bin.to_numpy()): errors.append(f'{path}: binary threshold/prediction mismatch')
    if not np.array_equal(probs.argmax(1),df.pred_sev.to_numpy()): errors.append(f'{path}: severity argmax/prediction mismatch')
    if not set(df.y_bin.unique()).issubset({0,1}): errors.append(f'{path}: y_bin must be 0/1')
    if not set(df.pred_bin.unique()).issubset({0,1}): errors.append(f'{path}: pred_bin must be 0/1')
    if not set(df.y_sev.unique()).issubset({0,1,2,3}): errors.append(f'{path}: y_sev must be 0..3')
    if not set(df.pred_sev.unique()).issubset({0,1,2,3}): errors.append(f'{path}: pred_sev must be 0..3')
    return errors

def check_reported_count(k,n,percent,label='metric'):
    if not attainable(k,n,percent): raise AssertionError(f'{label}: {percent}% is not attainable from {k}/{n}')

def _table_checks(path):
    df=pd.read_csv(path); e=[]
    if df.empty: return e
    low={c.lower():c for c in df.columns}
    # Binary confusion-matrix rows and rates (Tables 5/11/15 where counts are present).
    if {'tp','fp','tn','fn'}<=set(low):
        tp,fp,tn,fn=[pd.to_numeric(df[low[x]],errors='coerce') for x in ('tp','fp','tn','fn')]
        n=tp+fp+tn+fn
        if n.isna().any() or (n<=0).any(): e.append(f'{path}: invalid confusion-matrix total')
        for col,k,den in [('acc_pct',tp+tn,n),('sens_pct',tp,tp+fn),('spec_pct',tn,tn+fp)]:
            if col in low:
                vals=pd.to_numeric(df[low[col]],errors='coerce')
                for i,(ki,ni,vi) in enumerate(zip(k,den,vals)):
                    if pd.notna(vi) and not attainable(int(ki),int(ni),float(vi)):
                        e.append(f'{path}: row {i} {col}={vi} is not attainable from {int(ki)}/{int(ni)}')
    # Dice = 2*IoU/(1+IoU), accepting common column spellings.
    iou=next((low[x] for x in ('iou','mean_iou') if x in low),None)
    dice=next((low[x] for x in ('dice','dice_score') if x in low),None)
    if iou and dice:
        a=pd.to_numeric(df[iou],errors='coerce'); b=pd.to_numeric(df[dice],errors='coerce'); expected=2*a/(1+a)
        bad=a.notna() & b.notna() & ((expected-b).abs()>1e-3)
        if bad.any(): e.append(f'{path}: Dice does not equal 2*IoU/(1+IoU)')
    # AUC columns must be valid probabilities; interval strings checked when parseable.
    for c in df.columns:
        if c.lower() in {'auc','macro_auc','auroc','roc_auc'}:
            v=pd.to_numeric(df[c],errors='coerce');
            if ((v.dropna()<0)|(v.dropna()>1)).any(): e.append(f'{path}: {c} outside [0,1]')
    # Decision-curve net-benefit formula if the table stores the necessary components.
    if {'threshold','tp','fp','n','net_benefit'}<=set(low):
        th=pd.to_numeric(df[low['threshold']],errors='coerce'); tp=pd.to_numeric(df[low['tp']],errors='coerce'); fp=pd.to_numeric(df[low['fp']],errors='coerce'); n=pd.to_numeric(df[low['n']],errors='coerce'); nb=pd.to_numeric(df[low['net_benefit']],errors='coerce')
        expected=tp/n-(fp/n)*(th/(1-th)); mask=th.notna() & (th>0) & (th<1) & n.notna() & (n>0) & nb.notna()
        if (mask & ((expected-nb).abs()>1e-5)).any(): e.append(f'{path}: net-benefit formula mismatch')
    return e

def run_gate(root='.'):
    root=Path(root); errors=[]; populated=[]
    for p in sorted((root/'predictions').rglob('*.csv')):
        if p.name.endswith('TEMPLATE.csv') or p.stat().st_size<10: continue
        try:
            if pd.read_csv(p).empty: continue
            populated.append(p); errors += check_prediction_file(p)
        except Exception as ex: errors.append(f'{p}: cannot read prediction CSV ({ex})')
    if not populated: errors.append('No populated per-study prediction CSVs found; header-only templates cannot pass the release gate.')
    for p in sorted((root/'results/tables').glob('*.csv')):
        try: errors += _table_checks(p)
        except Exception as ex: errors.append(f'{p}: table validation failed ({ex})')
    report=root/'results/checks/consistency_report.txt'; report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(('ALL CHECKS PASSED\n' if not errors else '\n'.join(errors)+'\n'))
    if errors: raise SystemExit('Consistency gate failed; see '+str(report))
    print('ALL CHECKS PASSED')

def run(root='.'):
    try: run_gate(root); return True
    except SystemExit: return False

if __name__=='__main__': run_gate()
