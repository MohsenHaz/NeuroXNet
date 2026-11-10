"""Arithmetic checks of the split/fold generator on label vectors built inside the test (no study data is shipped)."""
import sys; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
import make_frozen_splits as g


def _labels():
    y = np.array([0] * 298 + [1] * 78 + [2] * 69 + [3] * 46)
    lab = pd.DataFrame({"study_id": [f"T{i:03d}" for i in range(491)], "final_label": y})
    lab["_class"] = lab.final_label.map(g.to_class); return lab


def test_split_sizes_and_test_counts():
    lab = _labels(); s = g.make_split(lab, 42); m = s.merge(lab, on="study_id")
    assert s.split.value_counts().to_dict() == {"train": 344, "val": 74, "test": 73}
    assert m[m.split == "test"]._class.value_counts().to_dict() == g.TEST
    assert s.study_id.is_unique and set(s.study_id) == set(lab.study_id)


def test_fold_sizes_and_positives():
    lab = _labels(); f = g.make_folds(lab, 42); m = f.merge(lab, on="study_id")
    assert f.fold.value_counts().sort_index().tolist() == g.FOLD_SIZES
    assert m[m._class != "normal"].fold.value_counts().sort_index().tolist() == g.FOLD_POS
    assert f.study_id.is_unique and len(f) == 491


def test_string_and_int_labels_equivalent():
    lab = _labels(); lab2 = lab.copy(); lab2["final_label"] = lab2._class; lab2["_class"] = lab2.final_label.map(g.to_class)
    assert g.make_split(lab, 7).equals(g.make_split(lab2, 7))
