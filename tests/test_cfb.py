import pandas as pd

import cfb_picks
from selector import Fair


def _sched():
    return pd.DataFrame([
        {"game_id": 1, "home_team": "Miami", "away_team": "Florida State",
         "start_date": "2026-10-31T19:30:00.000Z"},
        {"game_id": 2, "home_team": "Miami (OH)", "away_team": "Ohio",
         "start_date": "2026-10-31T19:30:00.000Z"},
        {"game_id": 3, "home_team": "San José State", "away_team": "Hawai'i",
         "start_date": "2026-11-01T03:00:00.000Z"},
        {"game_id": 4, "home_team": "App State", "away_team": "UL Monroe",
         "start_date": "2026-10-31T17:00:00.000Z"},
    ])


def test_odds_names_match_schedule_schools():
    fair = {
        "a": Fair("Miami Hurricanes", "Florida State Seminoles", "2026-10-31T19:30:00Z"),
        "b": Fair("Miami (OH) RedHawks", "Ohio Bobcats", "2026-10-31T19:30:00Z"),
        "c": Fair("San Jose State Spartans", "Hawaii Rainbow Warriors", "2026-11-01T03:00:00Z"),
        "d": Fair("Appalachian State Mountaineers", "Louisiana Monroe Warhawks",
                  "2026-10-31T17:00:00Z"),
        "e": Fair("Nowhere Tech Owls", "Elsewhere A&M Ducks", "2026-10-31T17:00:00Z"),
    }
    assert cfb_picks.match_games(fair, _sched()) == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_week_runs_to_sunday_after_saturday():
    end = cfb_picks.week_end(pd.Timestamp("2026-10-25T22:00Z"))      # Sunday evening
    assert end == pd.Timestamp("2026-11-01T12:00Z")
    assert cfb_picks.week_end(pd.Timestamp("2026-10-31T15:00Z")) == end   # Saturday


def test_grade_by_game_id():
    g = pd.DataFrame([{"game_id": 7, "margin": 10.0, "home_points": 31.0,
                       "away_points": 21.0}])
    bets = [{"game_id": 7, "home": "H", "away": "A", "pick": "H", "point": -7.5,
             "market": "spreads", "price": -110, "stake": 11.0, "result": None},
            {"game_id": 7, "home": "H", "away": "A", "pick": "Over", "point": 55.5,
             "market": "totals", "price": -110, "stake": 11.0, "result": None}]
    out = cfb_picks.grade(bets, g)
    assert [b["result"] for b in out] == ["win", "loss"]
    assert out[0]["profit"] == 10.0 and out[0]["final"] == "21-31"


def test_total_rules_pick_the_right_pattern():
    sat_noon = "2026-10-31T16:00:00Z"
    sat_night = "2026-11-01T00:00:00Z"      # 8pm ET
    assert cfb_picks.total_rule(47.5, 24.5, sat_noon, False, 2.0)[:2] == ("Over", 0.60)
    assert cfb_picks.total_rule(52.5, 21.0, sat_noon, False, 0.0)[:2] == ("Over", 0.56)
    # our possession total disagrees, or is missing: no over
    assert cfb_picks.total_rule(52.5, 21.0, sat_noon, False, -1.0) is None
    assert cfb_picks.total_rule(52.5, 21.0, sat_noon, False) is None
    assert cfb_picks.total_rule(63.5, 3.0, sat_night, True)[:2] == ("Under", 0.545)
    assert cfb_picks.total_rule(63.5, 3.0, sat_noon, True)[:2] == ("Under", 0.535)
    assert cfb_picks.total_rule(52.5, 7.0, sat_noon, True) is None


def test_thin_model_spreads_dropped():
    rows = pd.DataFrame([
        {"home": "A", "market": "spreads", "sources": "combined"},
        {"home": "B", "market": "spreads", "sources": "combined"},
        {"home": "C", "market": "spreads", "sources": "market,combined"},
        {"home": "D", "market": "totals", "sources": "trend"}])
    ctx = {"A": {"spread": (0.6, {})}, "B": {"spread": (-1.4, {})},
           "C": {"spread": (0.2, {})}, "D": {}}
    out = cfb_picks.drop_thin_spreads(rows, ctx)
    assert list(out.home) == ["B", "C", "D"]


def test_news_check_reads_claude_json_and_shows_injuries():
    from types import SimpleNamespace as NS
    import cfb_news
    key = cfb_news.game_key("Texas", "Oklahoma", "2026-10-31T19:30:00Z")
    reply = ("Found reports.\n```json\n" + __import__("json").dumps({"games": [{
        "key": key, "teams": [
            {"team": "Texas", "regular_qb": "Arch Manning", "expected_qb": "Matthew Caldwell",
             "qb_status": "out", "players": [
                 {"name": "Arch Manning", "pos": "QB", "status": "out", "note": "ankle"},
                 {"name": "Ryan Wingo", "pos": "WR", "status": "questionable", "note": "hamstring"}]},
            {"team": "Oklahoma", "regular_qb": "John Mateer", "expected_qb": "John Mateer",
             "qb_status": "starting", "players": []}]}]}) + "\n```")
    calls = []

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    return NS(stop_reason="end_turn", content=[NS(type="text", text=reply)])

    games = [{"home": "Texas", "away": "Oklahoma", "kickoff": "2026-10-31T19:30:00Z"}]
    now = pd.Timestamp("2026-10-29T12:00Z")
    news = cfb_news.fetch(games, {}, now, client=Client)
    assert len(calls) == 1 and calls[0]["tools"][0]["type"] == "web_search_20260209"
    # reused within the day: no second lookup
    cfb_news.fetch(games, news, now + pd.Timedelta(hours=5), client=Client)
    assert len(calls) == 1
    rows = cfb_news.injuries(news[key], "Texas")
    assert rows[0] == {"team": "Texas", "name": "Arch Manning", "pos": "QB", "status": "Out",
                       "pts": None, "note": "Caldwell starting"}
    assert [r["name"] for r in rows] == ["Arch Manning", "Ryan Wingo"]
    assert rows[1]["pts"] == -0.2       # a questionable starter's first miss (educated guess)
    assert cfb_news.injuries(news[key], "Oklahoma") == []
    picks = pd.DataFrame([{"game_id": 5, "commence": "2026-10-31T19:30:00Z"}])
    g = pd.DataFrame([{"game_id": 5, "home_team": "Texas", "away_team": "Oklahoma"}])
    assert len(cfb_picks.attach_news(picks, g, news).injuries[0]) == 2


def test_only_a_backup_qbs_first_start_moves_the_line():
    import cfb_news
    from selector import Fair
    kick = "2026-10-31T19:30:00Z"
    e = {"checked": kick, "teams": [
        {"team": "Michigan", "regular_qb": "Bryce Underwood", "expected_qb": "Tommy Carr",
         "qb_status": "out", "players": []},
        {"team": "UCLA", "regular_qb": "Nico Iamaleava", "expected_qb": "Ty Dieffenbach",
         "qb_status": "out", "players": []}]}
    # Carr has never started: his first start costs Michigan a point
    assert cfb_news.qb_change(e, "Michigan", ["Bryce Underwood"] * 7) == (-1.0, 0)
    # Dieffenbach has started UCLA's last 2 (names as the 2026 play-by-play
    # writes them): priced in
    ucla = ["Shotgun #9 N.Iamaleava"] * 5 + ["Shotgun #14 T.Dieffenbach",
                                              ":25) Shotgun #14 T.Dieffenbach"]
    assert cfb_news.qb_change(e, "UCLA", ucla) == (0.0, 2)
    row = cfb_news.injuries(e, "UCLA", ucla)[0]
    assert row["pts"] is None and row["note"] == "Dieffenbach 3rd start, priced in"
    assert cfb_news.injuries(e, "Michigan", ["Bryce Underwood"] * 7)[0]["pts"] == -1.0
    # no news, or the regular starting: nothing
    assert cfb_news.qb_change({}, "Michigan", ["Bryce Underwood"]) == (0.0, 0)

    upcoming = pd.DataFrame([{"game_id": 5, "home_team": "Rutgers",
                              "away_team": "Michigan"}]).set_index("game_id")
    fair = {"e": Fair("Rutgers Scarlet Knights", "Michigan Wolverines", kick)}
    news = {cfb_news.game_key("Rutgers", "Michigan", kick): e}
    done = set()
    adj = cfb_picks.qb_adjustments(fair, {"e": 5}, upcoming, news,
                                   {"Michigan": ["Bryce Underwood"] * 7}, done)
    # Michigan is away: the home margin goes UP a point, the total down one
    assert [(a.market, a.points) for a in adj] == [("spreads", 1.0), ("totals", -1.0)]
    assert cfb_picks.qb_adjustments(fair, {"e": 5}, upcoming, news, {}, done) == []


def test_other_starters_count_once_and_are_capped():
    import cfb_news
    p = lambda st, **k: {"name": "X", "pos": "WR", "status": st, **k}
    assert cfb_news.player_pts(p("out", starter=True, games_missed=0)) == -0.5
    assert cfb_news.player_pts(p("questionable", starter=True, games_missed=0)) == -0.2
    assert cfb_news.player_pts(p("out", starter=True, games_missed=2)) == 0.0   # in the ratings
    assert cfb_news.player_pts(p("out", starter=False, games_missed=0)) == 0.0
    assert cfb_news.player_pts(p("season")) == 0.0          # old news, no games_missed
    assert cfb_news.player_pts({**p("out"), "pos": "QB"}) == 0.0
    e = {"teams": [{"team": "Iowa State", "players": [p("out", games_missed=0)] * 5}]}
    assert cfb_news.others_pts(e, "Iowa State") == -1.5
