import pandas as pd

import cfb_streak


def _games():
    # Miami: three straight overs (vs our recorded close, then pred_total
    # fallback); Duke: over, over, then under.
    rows = []
    for i, (date, opp, pts, pred) in enumerate([
            ("2026-10-10T19:00:00Z", "Duke", 70, 50.0),
            ("2026-10-17T19:00:00Z", "Clemson", 60, 50.0),
            ("2026-10-31T19:00:00Z", "Duke", 66, 50.0)]):
        rows.append({"game_id": i + 1, "season": 2026, "start_date": date, "home_team": "Miami",
                     "away_team": opp, "margin": 3.0, "points": pts, "pred_total": pred,
                     "close_total": float("nan")})
    rows.append({"game_id": 9, "season": 2026, "start_date": "2026-11-01T19:00:00Z",
                 "home_team": "Duke", "away_team": "Wake Forest", "margin": 1.0, "points": 40,
                 "pred_total": 50.0, "close_total": float("nan")})
    return pd.DataFrame(rows)


def test_over_streaks_use_recorded_closes_then_model_totals():
    closes = {"Duke Blue Devils|Miami Hurricanes|2026-10-31":
              {"commence": "2026-10-31T19:00:00Z", "total": 70.5}}
    st = cfb_streak.streaks(_games(), {}, 2026)
    assert st["Miami"] == (3, 0)
    assert st["Duke"][0] == 0
    # the recorded close (70.5) makes the 66-point game an under: run broken
    st = cfb_streak.streaks(_games(), closes, 2026)
    assert st["Miami"] == (0, 0)
    assert cfb_streak.signal({"Miami": (3, 0)}, "Miami", "Duke").startswith(
        "Miami has gone over in 3 straight games")
    assert cfb_streak.signal({"Miami": (2, 0)}, "Miami", "Duke") is None


def test_streak_bets_skip_the_handicapper_review():
    import cfb_handicap
    picks = pd.DataFrame([{"away": "A", "home": "B", "commence": "2026-11-14T19:00:00Z",
                           "market": "totals", "pick": "Under", "point": 55.5, "price": -110,
                           "book": "FanDuel", "stake": 5.0, "win_chance": 0.545, "ev": 0.04,
                           "why": "Over streak: ...", "sources": "streak", "game_id": 1,
                           "injuries": [], "situation": []}])

    class Boom:
        def __getattr__(self, k):
            raise AssertionError("no API call for a streak bet")

    out = cfb_handicap.review(picks, {}, [], pd.Timestamp("2026-11-10", tz="UTC"), client=Boom())
    assert len(out) == 1 and out.stake.iloc[0] == 5.0
