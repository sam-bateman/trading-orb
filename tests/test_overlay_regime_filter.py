import pandas as pd

from vrp.overlays.regime_filter import vix_regime_mask, apply_mask


def _synth_data(n=100, vix_level=15.0, front=15.0, second=16.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(vix_level, index=idx),
            pd.Series(front, index=idx),
            pd.Series(second, index=idx))


def test_mask_all_active_in_quiet_regime():
    vix, front, second = _synth_data(vix_level=15.0)
    mask = vix_regime_mask(vix, front, second)
    assert mask.all()


def test_mask_turns_off_when_vix_spikes():
    vix, front, second = _synth_data(n=50, vix_level=15.0)
    vix.iloc[20:30] = 35.0
    mask = vix_regime_mask(vix, front, second)
    assert not mask.iloc[20:30].any()
    assert not mask.iloc[30:36].any()
    assert mask.iloc[37:].all()


def test_mask_turns_off_on_backwardation():
    vix, front, second = _synth_data(n=50)
    front.iloc[10:20] = 20.0
    second.iloc[10:20] = 17.0
    mask = vix_regime_mask(vix, front, second)
    assert not mask.iloc[10:20].any()


def test_apply_mask_zeros_daily_returns():
    idx = pd.bdate_range("2020-01-02", periods=20)
    daily_return = pd.Series(0.01, index=idx)
    mask = pd.Series([True] * 10 + [False] * 10, index=idx)
    out = apply_mask(daily_return, mask)
    assert (out.iloc[:10] == 0.01).all()
    assert (out.iloc[10:] == 0.0).all()
