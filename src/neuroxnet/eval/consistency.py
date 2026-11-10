"""Automatic consistency gate. For every table CSV in results/tables it checks that each reported percentage equals
its k/n, that confusion matrices sum to N, that class-wise rows reproduce the macro-F1, and that Dice = 2 IoU/(1+IoU).
Writes results/checks/consistency_report.txt and returns False if any check fails (the build then stops)."""
import math, pathlib, pandas as pd, numpy as np

def _close(a, b, tol=5e-5):
    return abs(a - b) <= tol

class Gate:
    def __init__(self):
        self.lines, self.failed = [], 0
    def check(self, ok, msg):
        self.lines.append(("PASS " if ok else "FAIL ") + msg)
        if not ok: self.failed += 1

def check_ratio_table(gate, df, name, triples):
    """triples: list of (pct_col, k_col, n_col)"""
    for _, r in df.iterrows():
        for pc, kc, nc in triples:
            if pd.isna(r.get(pc)) or pd.isna(r.get(kc)) or pd.isna(r.get(nc)) or r[nc] == 0:
                continue
            gate.check(_close(float(r[pc]), 100 * float(r[kc]) / float(r[nc]), 5e-3),
                       f"{name} [{r.iloc[0]}] {pc}={r[pc]} == {kc}/{nc}={r[kc]}/{r[nc]}={100 * r[kc] / r[nc]:.4f}")

def run(tables_dir="results/tables", out="results/checks/consistency_report.txt"):
    g = Gate(); T = pathlib.Path(tables_dir)
    def load(n):
        p = T / n
        return pd.read_csv(p) if p.exists() else None
    for n in ("table5_binary.csv", "table11_subgroups.csv", "table15_cv.csv"):
        df = load(n)
        if df is None: g.lines.append(f"SKIP {n} missing"); continue
        df["correct"] = df.TP + df.TN; df["N_"] = df.TP + df.FP + df.FN + df.TN; df["Pos"] = df.TP + df.FN; df["Neg"] = df.TN + df.FP
        check_ratio_table(g, df, n, [("acc_pct", "correct", "N_"), ("sens_pct", "TP", "Pos"), ("spec_pct", "TN", "Neg")])
        for _, r in df.iterrows():
            if pd.isna(r.TP): continue
            g.check(_close(r.f1, 2 * r.TP / (2 * r.TP + r.FP + r.FN)), f"{n} [{r.iloc[0]}] F1 reproduces from counts")
    for cmn, pcn, lab in (("table4_internal_cm.csv", "table7_perclass.csv", "internal"), ("table9_external_cm.csv", "table10_external_perclass.csv", "external")):
        cm, pc = load(cmn), load(pcn)
        if cm is None or pc is None: continue
        M = cm.iloc[:4, 1:5].values.astype(int)
        g.check(int(M.sum()) == int(cm.iloc[4, 5]) and (M.sum(1) == cm.iloc[:4, 5].values).all(), f"{cmn} sums to N={M.sum()} and row totals match")
        f1s = []
        for c in range(4):
            tp = M[c, c]; fp = M[:, c].sum() - tp; fn = M[c].sum() - tp
            prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0; f1s.append(f1)
            r = pc.iloc[c]
            g.check(_close(r.prec, prec) and _close(r.rec, rec) and _close(r.f1, f1), f"{pcn} class {c} P/R/F1 reproduce from {cmn}")
        g.check(_close(float(pc.iloc[4].f1), float(np.mean(f1s))), f"{pcn} macro-F1 {pc.iloc[4].f1} == mean of class F1 {np.mean(f1s):.4f}")
    gc = load("table13_gradcam.csv")
    if gc is not None:
        for _, r in gc.iterrows():
            g.check(_close(r.dice, 2 * r.iou / (1 + r.iou), 5e-4), f"table13 [{r.cls}] Dice = 2 IoU/(1+IoU)")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    status = "ALL CHECKS PASSED" if g.failed == 0 else f"{g.failed} CHECK(S) FAILED"
    open(out, "w").write("\n".join(g.lines + ["", status]) + "\n")
    print(status, "->", out)
    return g.failed == 0

if __name__ == "__main__":
    import sys; sys.exit(0 if run() else 1)
