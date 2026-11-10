import sys; sys.path.insert(0, "src")
from neuroxnet.eval.wilson import wilson
def test_known_values():
    # values independently verified (NeuroXNet manuscript, Table 5 row): 71/73, 28/29, 43/44
    for k, n, lo, hi in [(71, 73, 0.905, 0.992), (28, 29, 0.828, 0.994), (43, 44, 0.882, 0.996), (0, 10, 0.0, 0.278), (10, 10, 0.722, 1.0)]:
        a, b = wilson(k, n); assert abs(a - lo) < 0.0015 and abs(b - hi) < 0.0015, (k, n, a, b)
def test_bounds():
    for k in range(0, 21):
        lo, hi = wilson(k, 20); assert 0 <= lo <= k / 20 <= hi <= 1
