import sys, numpy as np; sys.path.insert(0, "src")
from neuroxnet.data.slice_select import select_window
def test_even_odd_and_short():
    for S in (128, 129, 200, 201, 127, 40, 1):
        v = np.arange(S, dtype=float)[:, None, None] * np.ones((1, 4, 4)); w, m = select_window(v, 128)
        assert w.shape == (128, 4, 4) and m.shape == (128,)
        if S >= 128:
            start = S // 2 - 64; assert w[0, 0, 0] == start and w[-1, 0, 0] == start + 127 and m.all()
        else:
            d = 128 - S; before = d // 2
            assert m.sum() == S and (~m[:before]).all() and m[before:before + S].all() and (w[:before] == 0).all()
