"""Create manuscript-sized patient-level split and fold assignments from the study label table.

Input : data/labels/cq500_severity_labels.csv with columns study_id, final_label (0..3 or normal/mild/moderate/severe).
Output: data/splits/cq500_split_seed42.csv   (344 train / 74 validation / 73 test; test classes 44/12/10/7)
        data/splits/cq500_cv5_folds.csv      (fold sizes 99/98/98/99/97; hydrocephalus-positive 39/38/39/39/38)
The script reads real study identifiers only and refuses to run on a header-only template. If the files used for the
reported experiments exist they are the authoritative assignment and must be committed instead of regenerated.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

TOTALS = {'normal': 298, 'mild': 78, 'moderate': 69, 'severe': 46}
TEST = {'normal': 44, 'mild': 12, 'moderate': 10, 'severe': 7}
VAL_TOTAL = 74
FOLD_SIZES = [99, 98, 98, 99, 97]
FOLD_POS = [39, 38, 39, 39, 38]
LABEL_MAP = {'0': 'normal', '1': 'mild', '2': 'moderate', '3': 'severe'}


def to_class(v):
    x = str(v).strip().lower()
    return LABEL_MAP.get(x, x)


def val_quotas(rem):
    tot = sum(rem.values()); raw = {c: VAL_TOTAL * rem[c] / tot for c in rem}
    q = {c: int(np.floor(raw[c])) for c in rem}
    for c in sorted(rem, key=lambda c: (raw[c] - q[c], c), reverse=True)[:VAL_TOTAL - sum(q.values())]:
        q[c] += 1
    return q


def make_split(lab, seed):
    rng = np.random.default_rng(seed)
    rem = {c: TOTALS[c] - TEST[c] for c in TOTALS}; vq = val_quotas(rem); rows = []
    for c in TOTALS:
        ids = lab.loc[lab['_class'] == c, 'study_id'].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        t, v = TEST[c], vq[c]
        rows += [(s, 'test') for s in ids[:t]] + [(s, 'val') for s in ids[t:t + v]] + [(s, 'train') for s in ids[t + v:]]
    return pd.DataFrame(rows, columns=['study_id', 'split']).sort_values('study_id')


def spread(ids_by_class, sizes, rng):
    order = []
    for c in sorted(ids_by_class):
        a = np.asarray(ids_by_class[c], dtype=object).copy(); rng.shuffle(a); order += list(a)
    cap = list(sizes); out = []; f = 0
    for sid in order:
        while cap[f % len(cap)] == 0: f += 1
        k = f % len(cap); out.append((sid, k + 1)); cap[k] -= 1; f += 1
    return out


def make_folds(lab, seed):
    rng = np.random.default_rng(seed + 1)
    pos = {c: lab.loc[lab['_class'] == c, 'study_id'].astype(str).tolist() for c in ('mild', 'moderate', 'severe')}
    neg = {'normal': lab.loc[lab['_class'] == 'normal', 'study_id'].astype(str).tolist()}
    rows = spread(pos, FOLD_POS, rng) + spread(neg, [s - p for s, p in zip(FOLD_SIZES, FOLD_POS)], rng)
    return pd.DataFrame(rows, columns=['study_id', 'fold']).sort_values('study_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='data/labels/cq500_severity_labels.csv'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--out', default='data/splits')
    a = ap.parse_args()
    lab = pd.read_csv(a.labels)
    if not {'study_id', 'final_label'}.issubset(lab.columns) or lab.empty:
        raise SystemExit('A populated labels CSV with study_id,final_label is required; the header-only template is not enough.')
    lab = lab[['study_id', 'final_label']].drop_duplicates('study_id').copy(); lab['_class'] = lab.final_label.map(to_class)
    if len(lab) != 491: raise SystemExit(f'Expected 491 unique studies; found {len(lab)}')
    counts = lab['_class'].value_counts().to_dict()
    if counts != TOTALS: raise SystemExit(f'Class totals {counts} differ from the protocol {TOTALS}')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    make_split(lab, a.seed).to_csv(out / 'cq500_split_seed42.csv', index=False)
    make_folds(lab, a.seed).to_csv(out / 'cq500_cv5_folds.csv', index=False)
    print('Wrote split and fold files with the protocol partition sizes.')


if __name__ == '__main__':
    main()
