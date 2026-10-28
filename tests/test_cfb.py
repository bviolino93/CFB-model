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
    # our possession count 3+ under a sub-60 total: the count's own under
    assert cfb_picks.total_rule(52.5, 7.0, sat_noon, True, -3.5)[:2] == ("Under", 0.535)
    assert cfb_picks.total_rule(52.5, 7.0, sat_noon, True, 4.0) is None


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


def test_missing_starters_move_the_total_by_side_of_the_ball():
    import cfb_news
    out = lambda pos: {"name": pos, "pos": pos, "status": "out", "starter": True, "games_missed": 0}
    e = {"teams": [{"team": "Texas", "players": [out("RB"), out("WR"), out("DL")]}]}
    assert cfb_news.total_pts(e, "Texas") == -0.5     # two offense down, one defense up
    e = {"teams": [{"team": "Texas", "players": [out("WR")] * 5}]}
    assert cfb_news.total_pts(e, "Texas") == -1.5


def test_last_line_before_kickoff_is_kept_as_the_close():
    import ledger
    f = Fair("Miami Hurricanes", "Florida State Seminoles", "2026-10-31T19:30:00Z")
    f.market_margin, f.market_total = 7.0, 51.5
    c = ledger.record_lines({}, {"a": f}, "2026-10-29T19:00Z")
    k = "Florida State Seminoles|Miami Hurricanes|2026-10-31"
    assert c[k]["spread"] == -7.0 and c[k]["total"] == 51.5
    f.market_margin = 8.5
    c = ledger.record_lines(c, {"a": f}, "2026-10-31T15:00Z")
    assert c[k]["spread"] == -8.5
    f.market_margin = 20.0                     # live line after kickoff
    c = ledger.record_lines(c, {"a": f}, "2026-10-31T20:00Z")
    assert c[k]["spread"] == -8.5


def test_ledger_keeps_the_alerted_line():
    from datetime import datetime, timezone
    import ledger
    bet = {"away": "A", "home": "H", "commence": "2026-10-31T19:30:00Z",
           "market": "totals", "pick": "Over", "point": 50.5, "price": -110}
    t = datetime(2026, 10, 29, tzinfo=timezone.utc)
    h = ledger.merge([], [bet], t, "r1")
    h = ledger.merge(h, [{**bet, "point": 52.0, "price": -105}], t, "r2")
    assert (h[0]["point"], h[0]["first_point"], h[0]["first_price"]) == (52.0, 50.5, -110)


def test_one_game_finds_schools():
    import cfb_game
    rows = pd.DataFrame([
        {"event_id": "a", "home_team": "Miami (OH) RedHawks", "away_team": "Ohio Bobcats",
         "commence_time": "2026-10-31T16:00:00Z"},
        {"event_id": "b", "home_team": "Miami Hurricanes", "away_team": "Florida State Seminoles",
         "commence_time": "2026-10-31T19:30:00Z"},
        {"event_id": "c", "home_team": "Ohio State Buckeyes", "away_team": "Penn State Nittany Lions",
         "commence_time": "2026-10-31T20:00:00Z"}])
    names = {"Miami (OH) RedHawks": "Miami (OH)", "Miami Hurricanes": "Miami",
             "Ohio Bobcats": "Ohio", "Ohio State Buckeyes": "Ohio State"}
    assert cfb_game.find_event(rows, "Miami", names) == "b"
    assert cfb_game.find_event(rows, "Ohio State", names) == "c"
    assert cfb_game.find_event(rows, "Ohio", names) == "a"
    assert cfb_game.find_event(rows, "Seminoles at Hurricanes", names) == "b"
    assert cfb_game.find_event(rows, "Penn State", names) == "c"
    assert cfb_game.find_event(rows, "Alabama", names) is None


def test_one_game_report_and_checks_file(tmp_path):
    import json
    import cfb_game
    d = {"league": "cfb", "found": True, "query": "Ohio State",
         "checked": "2026-10-27T03:00:00+00:00", "title": "Penn State @ Ohio State",
         "kickoff": "2026-10-31T20:00:00Z",
         "lines": [{"market": "Total", "sharp": "45.5", "model": "45.5"}],
         "offers": [{"market": "totals", "side": "Under 45.5", "price": -110,
                     "book": "FanDuel", "win": 0.5, "ev": -0.045, "bet": False}],
         "verdicts": [{"market": "Total", "bet": False, "head": "Pass",
                       "text": "None of the totals patterns fits this game."}]}
    md = cfb_game.to_md(d)
    assert "| Under 45.5 | -110 | FanDuel | 50.0% | -4.5% |" in md
    assert "**Total: Pass.**" in md
    p = tmp_path / "checks.json"
    cfb_game.save_check(str(p), d)
    cfb_game.save_check(str(p), {**d, "checked": "2026-10-27T04:00:00+00:00"})
    c = json.loads(p.read_text())["checks"]
    assert len(c) == 1 and c[0]["checked"].startswith("2026-10-27T04")


def _dog_rows():
    rows = []
    for i, (price, ev, fair) in enumerate([(150, 0.05, -3.0), (180, 0.08, -4.0),
                                           (450, 0.20, -10.0), (130, 0.02, -2.0),
                                           (120, 0.04, -2.5), (110, 0.035, -1.5)]):
        rows.append({"event_id": f"e{i}", "home": f"H{i}", "away": "A",
                     "market": "h2h", "pick": f"H{i}", "price": price, "ev": ev,
                     "fair_market": fair, "kelly": 0.2, "book": "draftkings",
                     "point": None})
    rows.append({**rows[0], "book": "pinnacle", "ev": 0.5})          # not my book
    rows.append({**rows[1], "pick": "A", "price": 105, "ev": 0.3})   # favourite
    return pd.DataFrame(rows)


def test_underdogs_best_few_short_dogs_at_my_books():
    import selector as sel
    d = cfb_picks.underdogs(_dog_rows(), ["draftkings"], 1000, sel.Policy(min_ev=0.03))
    assert list(d.pick) == ["H1", "H0", "H4"]
    assert (d.stake == 10.0).all()


def test_underdog_reason_and_grade():
    import ledger
    p = pd.DataFrame([{"home": "Purdue Boilermakers", "away": "Iowa Hawkeyes",
                       "pick": "Purdue Boilermakers", "market": "h2h", "point": None,
                       "price": 170, "book": "draftkings", "win_chance": 0.38,
                       "fair_market": -4.0, "sources": "market", "ev_market": 0.03,
                       "p_push": 0.0}]).itertuples().__next__()
    row = pd.Series({"home_team": "Purdue", "away_team": "Iowa"})
    why = cfb_picks.explain_dog(p, {"row": row}, None)
    assert why.startswith("DraftKings pays +170 on Purdue, which needs a 37%")
    assert "sharp books' moneylines make Purdue worth about +162" in why
    b = {"market": "h2h", "pick": "H", "home": "H", "away": "A", "point": None}
    assert ledger.grade_one(b, 21, 17) == "win" and ledger.grade_one(b, 17, 21) == "loss"


def test_close_keeps_the_moneyline_chance():
    import ledger
    f = Fair("Miami Hurricanes", "Florida State Seminoles", "2026-10-31T19:30:00Z")
    f.market_margin, f.market_total, f.market_p_home = 7.0, 51.5, 0.7
    c = ledger.record_lines({}, {"a": f}, "2026-10-29T19:00Z")
    assert c["Florida State Seminoles|Miami Hurricanes|2026-10-31"]["p_home"] == 0.7
