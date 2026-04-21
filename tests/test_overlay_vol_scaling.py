import numpy as np
import pandas as pd

from vrp.overlays.vol_scaling import target_vol_scale


def test_scale_down_when_realized_exceeds_target():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-02", periods=500)
    ret = pd.Series(rng.normal(0, 0.02, 500), index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20, leverage_cap=1.0)
    realized_vol_late = scaled.iloc[-100:].std() * np.sqrt(252)
    assert realized_vol_late < 0.15


def test_leverage_cap_applied_when_realized_below_target():
    idx = pd.bdate_range("2020-01-02", periods=500)
    ret = pd.Series(0.0001, index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20, leverage_cap=1.0)
    assert (scaled == ret).all()


def test_scale_preserves_zero_days():
    idx = pd.bdate_range("2020-01-02", periods=100)
    ret = pd.Series([0.01] * 30 + [0.0] * 40 + [0.01] * 30, index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20, leverage_cap=1.0)
    assert (scaled.iloc[30:70] == 0.0).all()
