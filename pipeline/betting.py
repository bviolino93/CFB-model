"""Stage 4: betting math.

  american_to_prob(-110)          -> 0.5238  (break-even, vig included)
  devig(-110, -110)               -> (0.5, 0.5)  fair probabilities
  expected_value(0.55, -110)      -> +0.05 per $1 staked
  kelly_fraction(0.55, -110)      -> share of bankroll at full Kelly

A bet is flagged only when the model's probability beats the de-vigged
(fair) probability by at least MIN_EDGE *and* the bet has positive EV at
the price actually being paid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def american_to_prob(odds):
    """Implied probability of American odds (includes the book's vig)."""
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def american_to_decimal(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, 1 + 100.0 / -o, 1 + o / 100.0)


def devig(odds_a, odds_b):
    """Two-way market -> fair probabilities (multiplicative / proportional
    method: scale both implied probabilities so they add to 1)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    tot = pa + pb
    return pa / tot, pb / tot


def expected_value(p_win, odds, p_push=0.0):
    """EV per $1 staked. A push returns the stake."""
    b = american_to_decimal(odds) - 1
    p_lose = 1 - p_win - p_push
    return p_win * b - p_lose


def kelly_fraction(p_win, odds, fraction=C.KELLY_FRACTION, cap=0.05):
    """Fractional Kelly stake as a share of bankroll (0 when EV <= 0),
    capped at `cap` (5%) so one wrong probability can't sink the roll."""
    b = american_to_decimal(odds) - 1
    f = (b * p_win - (1 - p_win)) / b
    return np.clip(f * fraction, 0, cap)


def pick_bets(games: pd.DataFrame, probs: pd.DataFrame, market: str,
              prob_col="p_platt", min_edge=C.MIN_EDGE, price=None) -> pd.DataFrame:
    """One row per game where either side clears the edge bar.

    games: rows from the model table; probs: Stage 3 output (same index)
    with the home-cover / over probability in `prob_col`.
    price: odds you actually bet at (e.g. -105 at 734). Default: the
    market's median price for that side, or -110 when missing.
    """
    if market == "spread":
        oa, ob = games.home_spread_odds, games.away_spread_odds
        side_a, side_b = "home", "away"
    else:
        oa, ob = games.over_odds, games.under_odds
        side_a, side_b = "over", "under"
    oa = oa.fillna(C.DEFAULT_ODDS)
    ob = ob.fillna(C.DEFAULT_ODDS)
    fair_a, fair_b = devig(oa, ob)
    p_a = probs[prob_col].to_numpy()
    p_b = 1 - p_a
    pay_a = np.full(len(games), price, float) if price else oa.to_numpy(float)
    pay_b = np.full(len(games), price, float) if price else ob.to_numpy(float)

    edge_a, edge_b = p_a - fair_a, p_b - fair_b
    take_a = edge_a >= edge_b
    out = pd.DataFrame({
        "side": np.where(take_a, side_a, side_b),
        "p_model": np.where(take_a, p_a, p_b),
        "p_fair": np.where(take_a, fair_a, fair_b),
        "odds": np.where(take_a, pay_a, pay_b),
    }, index=games.index)
    out["edge"] = out.p_model - out.p_fair
    out["ev"] = expected_value(out.p_model, out.odds)
    out["kelly"] = kelly_fraction(out.p_model, out.odds)
    flagged = (out.edge >= min_edge) & (out.ev > 0)
    return out[flagged]
