import pytest

from vrp.util.bs import bs_price, bs_delta, strike_from_delta


def test_bs_price_at_the_money_call_put_parity():
    S, K, T, sigma = 100.0, 100.0, 0.25, 0.20
    call = bs_price(S, K, T, sigma, r=0.0, option_type="call")
    put = bs_price(S, K, T, sigma, r=0.0, option_type="put")
    assert abs(call - put) < 1e-9


def test_bs_price_deep_itm_call_floor():
    px = bs_price(200.0, 100.0, 0.01, 0.20, r=0.0, option_type="call")
    assert abs(px - 100.0) < 0.5


def test_bs_delta_signs_and_put_call_parity():
    call_d = bs_delta(100.0, 100.0, 0.25, 0.20, r=0.0, option_type="call")
    put_d = bs_delta(100.0, 100.0, 0.25, 0.20, r=0.0, option_type="put")
    assert 0 < call_d < 1
    assert -1 < put_d < 0
    # Put-call parity on delta (dividend-free): call_delta - put_delta = 1.
    # NOT call_delta + put_delta ≈ 0 — that only holds for d1 = 0
    # (ATM-forward, not ATM-spot).
    assert abs(call_d - put_d - 1.0) < 1e-9


def test_strike_from_delta_round_trip():
    target_delta = -0.30
    S, T, sigma = 100.0, 30 / 365.0, 0.20
    K = strike_from_delta(S, T, sigma, r=0.0, option_type="put",
                          target_delta=target_delta)
    recovered = bs_delta(S, K, T, sigma, r=0.0, option_type="put")
    assert abs(recovered - target_delta) < 1e-4


def test_strike_from_delta_out_of_range():
    with pytest.raises(ValueError):
        strike_from_delta(100, 0.25, 0.20, r=0, option_type="put", target_delta=-1.5)
