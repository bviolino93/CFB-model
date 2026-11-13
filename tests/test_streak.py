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


def test_streak_moves_the_fair_total():
    import selector as sel
    a = sel.Adjustment("Miami Hurricanes", "totals", -2.0, "streak")
    assert a.active


def test_view_brief_starts_from_the_stats_projection():
    import cfb_handicap
    g = pd.DataFrame([{"game_id": 7, "season": 2026, "home_team": "Toledo", "away_team": "Massachusetts",
                       "start_date": "2026-11-12T00:30:00Z", "margin": float("nan"),
                       "pred_margin": 13.6, "pred_total": 54.5}])
    v = {"game_id": 7, "away": "UMass Minutemen", "home": "Toledo Rockets",
         "commence": "2026-11-12T00:30:00Z", "offers": {}}
    brief = cfb_handicap._view_brief(v, g, {})
    assert "Stats model projection (objective starting point): Toledo 34, Massachusetts 20" in brief


def test_view_brief_flags_old_qb_absence():
    import pandas as pd, cfb_handicap, cfb_news
    g = pd.DataFrame([{"game_id": 1, "season": 2026, "home_team": "UCLA", "away_team": "Illinois",
                       "start_date": "2026-11-14T02:00:00Z", "margin": None, "points": None,
                       "pred_margin": 14.0, "pred_total": 48.0}])
    v = {"game_id": 1, "away": "Illinois", "home": "UCLA", "commence": "2026-11-14T02:00:00Z", "offers": {}}
    e = {"teams": [{"team": "UCLA", "regular_qb": "Nico Iamaleava", "expected_qb": "Joey Backup",
                    "qb_status": "out", "players": []}]}
    news = {cfb_news.game_key("UCLA", "Illinois", v["commence"]): e}
    st = {"UCLA": ["N.Iamaleava"] * 5 + ["J.Backup"] * 3}
    b = cfb_handicap._view_brief(v, g, news, None, st)
    assert "not new" in b and "last 3 games" in b
    st = {"UCLA": ["N.Iamaleava"] * 8}
    assert "NEW: the backup's first start" in cfb_handicap._view_brief(v, g, news, None, st)
