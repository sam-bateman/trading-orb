import pandas as pd

from vrp.overlays.tail_hedge import add_tail_hedge
from vrp.strategies.strategy_b import run_strategy_b


def _synth_spx_vix(n=252, spx_level=100.0, vix_level=20.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(spx_level, index=idx),
            pd.Series(vix_level, index=idx))


def test_hedge_reduces_net_on_flat_underlying():
    spx, vix = _synth_spx_vix(252)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert hedged["net_daily_return"].sum() < strat["daily_return"].sum()


def test_hedge_reduces_drawdown_on_crash():
    n = 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[n // 2:] = 80.0
    vix = pd.Series(25.0, index=idx)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    strat_eq = (1 + strat["daily_return"]).cumprod()
    net_eq = (1 + hedged["net_daily_return"]).cumprod()
    low_day = strat_eq.idxmin()
    assert net_eq.loc[low_day] > strat_eq.loc[low_day]


def test_hedge_structure_keys():
    spx, vix = _synth_spx_vix(120)
    strat = run_strategy_b(spx, vix, target_delta=-0.30)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert set(hedged.keys()) >= {"net_daily_return", "hedge_daily_return",
                                    "hedge_legs"}
    assert len(hedged["net_daily_return"]) == len(strat["daily_return"])
