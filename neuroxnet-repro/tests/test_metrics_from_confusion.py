import sys, numpy as np; sys.path.insert(0, "src")
from neuroxnet.eval.metrics import binary_metrics_from_counts, multiclass_metrics
from neuroxnet.eval.mcnemar import mcnemar_exact
from neuroxnet.eval.decision_curve import net_benefit
def test_binary_from_counts():
    m = binary_metrics_from_counts(dict(TP=28, FP=1, FN=1, TN=43))
    assert abs(m["acc"] - 71 / 73) < 1e-12 and abs(m["sens"] - 28 / 29) < 1e-12 and abs(m["spec"] - 43 / 44) < 1e-12 and abs(m["f1"] - 56 / 58) < 1e-12
def test_multiclass_identities():
    rng = np.random.default_rng(0); y = rng.integers(0, 4, 200); pred = y.copy(); flip = rng.random(200) < 0.2; pred[flip] = rng.integers(0, 4, flip.sum())
    P = rng.dirichlet(np.ones(4), 200); m = multiclass_metrics(y, pred, P)
    assert m["cm"].sum() == 200 and abs(m["macro_f1"] - np.mean([r["f1"] for r in m["per_class"]])) < 1e-12
    assert all(abs(r["TP"] + r["FN"] - r["N"]) == 0 for r in m["per_class"])
def test_mcnemar_and_decision_curve():
    y = np.array([1] * 29 + [0] * 44); a = y.copy(); b = y.copy(); b[0] = 0; b[29] = 1
    r = mcnemar_exact(y, a, b); assert r["discordant"] == 2 and r["p_value"] == 0.5
    nb, nb_all, _ = net_benefit(y, y.astype(float), np.array([0.5, 29 / 73]))
    assert abs(nb[0] - 29 / 73) < 1e-12 and abs(nb_all[1]) < 1e-12
