"""The gate must accept sensitivity/specificity computed from their own denominators and reject impossible values."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from neuroxnet.eval import consistency as c


def _write(tmp_path, sens):
    d = tmp_path / "results" / "tables"; d.mkdir(parents=True)
    pd.DataFrame([{"Method": "m", "TP": 28, "FP": 1, "FN": 1, "TN": 43, "acc_pct": 97.26, "sens_pct": sens, "spec_pct": 97.73}]).to_csv(d / "t5.csv", index=False)
    return d / "t5.csv"


def test_valid_sensitivity_specificity_pass(tmp_path):
    assert c._table_checks(_write(tmp_path, 96.55)) == []


def test_impossible_sensitivity_fails(tmp_path):
    assert c._table_checks(_write(tmp_path, 96.0))
