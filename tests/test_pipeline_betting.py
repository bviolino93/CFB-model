import numpy as np
import pandas as pd

from pipeline import betting as B
from pipeline.backtest import max_drawdown


def test_devig_and_breakeven():
    assert np.isclose(B.american_to_prob(-110), 110 / 210)
    assert np.isclose(B.american_to_prob(150), 0.4)
    a, b = B.devig(-110, -110)
    assert np.isclose(a, 0.5) and np.isclose(b, 0.5)
    a, b = B.devig(-150, 130)
    assert np.isclose(a + b, 1) and a > 0.5


def test_ev_and_kelly():
    assert np.isclose(B.expected_value(110 / 210, -110), 0)      # break-even
    assert B.expected_value(0.55, -110) > 0
    assert B.kelly_fraction(0.50, -110) == 0                       # no edge, no bet
    full = (0.91 * 0.55 - 0.45) / 0.91
    assert np.isclose(B.kelly_fraction(0.55, -110, fraction=0.25, cap=1), full / 4, atol=1e-3)
    assert B.kelly_fraction(0.99, 100) == 0.05                     # capped


def test_pick_bets_needs_edge():
    g = pd.DataFrame({"home_spread_odds": [-110, -110], "away_spread_odds": [-110, -110]})
    p = pd.DataFrame({"p_platt": [0.53, 0.56]})
    out = B.pick_bets(g, p, "spread")
    assert list(out.index) == [1] and out.side.iloc[0] == "home"


def test_max_drawdown():
    assert np.isclose(max_drawdown(pd.Series([1100, 880, 990]), 1000), -0.2)
