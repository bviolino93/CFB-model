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


def test_clv_sign():
    from pipeline.backtest import clv_points
    from pipeline.model import Spec
    g = pd.DataFrame({"open_spread": [-3.0, -3.0], "close_spread": [-5.0, -5.0],
                      "open_total": [50.0, 50.0], "close_total": [52.0, 52.0]})
    sp = Spec(market="spread", line="open")
    assert list(clv_points(g, pd.Series(["home", "away"]), sp)) == [2.0, -2.0]
    tt = Spec(market="total", line="open")
    assert list(clv_points(g, pd.Series(["over", "under"]), tt)) == [2.0, -2.0]
    assert list(clv_points(g, pd.Series(["home", "away"]), Spec(market="spread"))) == [0, 0]


def test_prior_weight_fades_by_week_5():
    from pipeline.features import prior_weight
    d = pd.DataFrame({"week": [1, 2, 3, 4, 5, 9, 1],
                      "season_type": ["regular"] * 6 + ["postseason"]})
    assert prior_weight(d).tolist() == [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0]


def test_injury_feed_only_counts_news_known_before_kickoff():
    from pipeline.features import add_injury_feed
    g = pd.DataFrame({"game_id": [1], "home_team": ["Ohio State"],
                      "away_team": ["Northwestern"],
                      "kickoff": [pd.Timestamp("2026-11-14T17:00Z")]})
    rows = [
        {"team": "Ohio State", "points": 3, "note": "J. Sayin starting QB out",
         "until": "2026-11-14", "added_at": "2026-11-14T07:45:22-05:00"},
        {"team": "Northwestern", "points": 2, "note": "LB out",
         "until": "2026-11-13", "added_at": "2026-11-01T08:00:00-05:00"},   # expired
        {"team": "Northwestern", "points": 5, "note": "QB out",
         "until": "2026-11-21", "added_at": "2026-11-14T18:00:00-05:00"},   # after kickoff
    ]
    out = add_injury_feed(g, rows).iloc[0]
    assert out.inj_h_pts == 3 and out.inj_a_pts == 0
    assert out.inj_diff == -3 and out.inj_qb_diff == -1
