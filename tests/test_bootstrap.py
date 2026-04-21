import numpy as np
import pandas as pd

from vrp.util.bootstrap import (
    bootstrap_paths,
    bootstrap_metrics,
    var_and_es,
    confidence_intervals,
)


def test_bootstrap_paths_length_matches_input():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-02", periods=500)
    ret = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    paths = bootstrap_paths(ret, block_size=40, n_paths=50, seed=1)
    assert len(paths) == 50
    for p in paths:
        assert len(p) == len(ret)


def test_bootstrap_metrics_columns():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 500))
    df = bootstrap_metrics(ret, block_size=40, n_paths=20, seed=1)
    for c in ("ann_return", "ann_vol", "sharpe", "max_drawdown"):
        assert c in df.columns
    assert len(df) == 20


def test_var_and_es():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 10_000))
    v, es = var_and_es(ret, alpha=0.05)
    assert -0.02 < v < -0.01
    assert es < v


def test_confidence_intervals():
    df = pd.DataFrame({"x": np.linspace(-1, 1, 101)})
    ci = confidence_intervals(df, alphas=(0.05, 0.95))
    assert abs(ci.loc["x", "p05"] - (-0.9)) < 0.02
    assert abs(ci.loc["x", "p95"] - 0.9) < 0.02
