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


def test_bets_need_the_model_on_their_side(monkeypatch):
    monkeypatch.setattr(cfb_picks, "BACKTESTED_ONLY", False)
    rows = pd.DataFrame([
        # model leans home 1.6: home side backed, away side not
        {"home": "A", "pick": "A", "market": "spreads", "fair": 8.6, "fair_market": 7.0, "sources": "combined"},
        {"home": "A", "pick": "B", "market": "spreads", "fair": 8.6, "fair_market": 7.0, "sources": "combined"},
        # a 1.2-point spread lean is under the 1.5 spreads need
        {"home": "H", "pick": "H", "market": "spreads", "fair": 4.2, "fair_market": 3.0, "sources": "combined"},
        # a book off the consensus with only 0.4 of model lean: a price bet, dropped
        {"home": "C", "pick": "C", "market": "spreads", "fair": -8.6, "fair_market": -9.0, "sources": "market,combined"},
        # under backed by 1.2 points; over not
        {"home": "D", "pick": "Under", "market": "totals", "fair": 50.3, "fair_market": 51.5, "sources": "combined"},
        {"home": "D", "pick": "Over", "market": "totals", "fair": 50.3, "fair_market": 51.5, "sources": "combined"},
        # a totals pattern its way counts even under a point
        {"home": "E", "pick": "Under", "market": "totals", "fair": 49.6, "fair_market": 50.0, "sources": "trend"},
        # moneyline dog: the spread lean counts
        {"home": "F", "pick": "G", "market": "h2h", "fair": -5.5, "fair_market": -4.0, "sources": "combined"}])
    out = cfb_picks.model_backed(rows)
    assert list(zip(out.home, out.pick)) == [("A", "A"), ("D", "Under"), ("E", "Under"), ("F", "G")]
    # backtested-only mode: just the totals pattern
    monkeypatch.setattr(cfb_picks, "BACKTESTED_ONLY", True)
    out = cfb_picks.model_backed(rows)
    assert list(zip(out.home, out.pick)) == [("E", "Under")]


def test_news_check_reads_claude_json_and_shows_injuries(monkeypatch):
    from types import SimpleNamespace as NS
    import cfb_news
    monkeypatch.setattr(cfb_news, "ALREADY_SPENT", {})
    key = cfb_news.game_key("Texas", "Oklahoma", "2026-10-31T19:30:00Z")
    reply = ("Found reports.\n```json\n" + __import__("json").dumps({"games": [{
        "key": key, "teams": [
            {"team": "Texas", "regular_qb": "Arch Manning", "expected_qb": "Matthew Caldwell",
             "qb_status": "out", "players": [
                 {"name": "Arch Manning", "pos": "QB", "status": "out", "note": "ankle"},
                 {"name": "Ryan Wingo", "pos": "WR", "status": "questionable", "note": "hamstring",
                  "starter": True, "games_missed": 0}]},
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


def test_news_lookups_stay_near_kickoff_and_capped():
    import cfb_news
    from types import SimpleNamespace as NS
    calls = []

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    usage = NS(input_tokens=100_000, output_tokens=2_000, cache_read_input_tokens=0,
                               cache_creation_input_tokens=0, server_tool_use=NS(web_search_requests=3))
                    return NS(stop_reason="end_turn", content=[NS(type="text", text="none")], usage=usage)

    cfb_news._spent.update(dict.fromkeys(cfb_news._spent, 0))
    now = pd.Timestamp("2026-11-29T12:00Z")
    # October 2026 is paused (already over budget before the guard)
    assert cfb_news.fetch([{"home": "A", "away": "B", "kickoff": "2026-10-31T19:30:00Z"}], {},
                          pd.Timestamp("2026-10-29T12:00Z"), client=Client)["_budget"]["usd"] == 49.0
    assert calls == []
    far = [{"home": "A", "away": "B", "kickoff": "2026-12-05T19:30:00Z"}]   # a week out
    cfb_news.fetch(far, {}, now, client=Client)
    assert calls == []
    soon = [{"home": f"H{i}", "away": f"A{i}", "kickoff": f"2026-11-30T{i:02d}:30:00Z"} for i in range(20)]
    cfb_news.fetch(soon, {}, now, client=Client)
    assert cfb_news._spent["games"] == cfb_news.MAX_LOOKUPS          # soonest 15, then stop
    assert calls[0]["model"] == "claude-sonnet-5-5" and calls[0]["output_config"]["effort"] == "low"
    n = len(calls)
    cfb_news.fetch(soon, {}, now, client=Client)                     # cap is per run
    assert len(calls) == n
    assert round(cfb_news.cost(), 2) == round(n * (0.2 + 0.02 + 0.03), 2)
    cfb_news._spent.update(dict.fromkeys(cfb_news._spent, 0))
    # the month's running total stops lookups at the budget, and resets next month
    full = {"_budget": {"month": "2026-11", "usd": cfb_news.MONTHLY_BUDGET}}
    n = len(calls)
    out = cfb_news.fetch(soon[:4], full, now, client=Client)
    assert len(calls) == n and out["_budget"]["usd"] == cfb_news.MONTHLY_BUDGET
    out = cfb_news.fetch(soon[:4], full, pd.Timestamp("2026-12-01T00:00Z"), client=Client)
    assert len(calls) == n + 1 and out["_budget"] == {"month": "2026-12", "usd": 0.25}
    cfb_news._spent.update(dict.fromkeys(cfb_news._spent, 0))


def test_other_starters_count_once_and_are_capped():
    import cfb_news
    p = lambda st, **k: {"name": "X", "pos": "WR", "status": st, **k}
    assert cfb_news.player_pts(p("out", starter=True, games_missed=0)) == -0.5
    assert cfb_news.player_pts(p("questionable", starter=True, games_missed=0)) == -0.2
    assert cfb_news.player_pts(p("out", starter=True, games_missed=2)) == 0.0   # in the ratings
    assert cfb_news.player_pts(p("out", starter=False, games_missed=0)) == 0.0
    assert cfb_news.player_pts(p("season")) == 0.0          # old news, no games_missed
    assert cfb_news.player_pts({**p("out"), "pos": "QB"}) == 0.0
    # week-old "out, no timetable" list entries with nothing on starter or
    # games missed (Kent State, Oct 2026) are already in the line
    assert cfb_news.player_pts(p("out", starter=None, games_missed=None)) == 0.0
    assert cfb_news.player_pts(p("out", starter=True, games_missed=None)) == 0.0
    assert cfb_news.player_pts({**p("out", starter=True, games_missed=0), "pos": "K"}) == 0.0
    e = {"teams": [{"team": "Iowa State", "players": [p("out", starter=True, games_missed=0)] * 5}]}
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
    assert (h[0]["point"], h[0]["price"], h[0]["now_point"]) == (50.5, -110, 52.0)
    h = ledger.merge(h, [], t, "r3")      # dropped before kickoff: still counts
    assert len(h) == 1 and h[0]["on_board"] is False and h[0]["point"] == 50.5


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
                     "fair_market": fair, "fair": fair + 1.5, "sources": "combined",
                     "kelly": 0.2, "book": "draftkings", "point": None})
    rows.append({**rows[0], "book": "pinnacle", "ev": 0.5})          # not my book
    rows.append({**rows[1], "event_id": "e9", "pick": "H9", "home": "H9",
                 "fair": rows[1]["fair_market"], "sources": "market", "ev": 0.4})  # price only
    rows.append({**rows[1], "pick": "A", "price": 105, "ev": 0.3})   # favourite
    return pd.DataFrame(rows)


def test_underdogs_best_few_short_dogs_at_my_books(monkeypatch):
    monkeypatch.setattr(cfb_picks, "BACKTESTED_ONLY", False)
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


def test_bets_the_schedule_hasnt_scored_grade_from_the_scores_feed():
    import cfb_picks
    bet = {"home": "Western Kentucky Hilltoppers", "away": "Delaware Blue Hens",
           "commence": "2026-10-27T23:00:00Z", "market": "spreads",
           "pick": "Delaware Blue Hens", "point": 1.5, "price": -120, "stake": 12.0}
    now = pd.Timestamp("2026-10-29T02:00Z")
    assert cfb_picks.needs_scores([bet], now)
    assert not cfb_picks.needs_scores([bet], pd.Timestamp("2026-10-28T01:00Z"))   # still on
    scores = [{"home": "Western Kentucky Hilltoppers", "away": "Delaware Blue Hens",
               "commence": "2026-10-27T23:00:00Z", "home_score": 24.0, "away_score": 23.0}]
    out = cfb_picks.grade_scores([dict(bet)], scores)[0]
    assert out["result"] == "win" and out["final"] == "23-24" and out["profit"] == 10.0
    other = [{**scores[0], "commence": "2026-09-12T23:00:00Z"}]           # same teams, other date
    assert not cfb_picks.grade_scores([dict(bet)], other)[0].get("result")


def test_dropped_bet_gets_a_skip_alert():
    from datetime import datetime, timezone
    import ledger
    bet = {"away": "Boston College Eagles", "home": "Duke Blue Devils",
           "commence": "2026-10-31T16:00:00Z", "market": "h2h",
           "pick": "Boston College Eagles", "point": None, "price": 245,
           "book": "FanDuel", "stake": 9.61}
    t = datetime(2026, 10, 29, tzinfo=timezone.utc)
    h = ledger.merge([], [bet], t, "r1")
    was = ledger.on_board(h)
    h = ledger.merge(h, [], t, "r2")
    gone = ledger.dropped(was, h, t)
    names = {"Boston College Eagles": "Boston College", "Duke Blue Devils": "Duke"}
    note = cfb_picks.render_dropped(gone, names)
    assert "~~Boston College to win (+245) at FanDuel~~ · Boston College @ Duke" in note


def test_red_flag_drops_bets_on_the_other_side():
    import cfb_news
    e = {"situation": [
        {"factor": "coaching", "favors": "Iowa State", "red_flag": True, "note": "OSU coach fired Tuesday"},
        {"factor": "weather", "favors": "under", "red_flag": False, "note": "15 mph wind"}]}
    assert cfb_news.red_flag_against(e, "spreads", "Oklahoma State Cowboys",
                                     "Oklahoma State", "Iowa State") == "OSU coach fired Tuesday"
    assert cfb_news.red_flag_against(e, "spreads", "Iowa State Cyclones",
                                     "Iowa State", "Oklahoma State") == ""
    assert cfb_news.red_flag_against(e, "totals", "Over", "", "") == ""   # not a red flag
    e["situation"][1]["red_flag"] = True
    assert cfb_news.red_flag_against(e, "totals", "Over", "", "") == "15 mph wind"
    assert cfb_news.red_flag_against(e, "totals", "Under", "", "") == ""
    picks = pd.DataFrame([
        {"game_id": 1, "home": "Iowa State Cyclones", "pick": "Oklahoma State Cowboys",
         "market": "spreads", "point": -9.5, "commence": "2026-10-31T23:00:00Z"},
        {"game_id": 1, "home": "Iowa State Cyclones", "pick": "Under",
         "market": "totals", "point": 58.5, "commence": "2026-10-31T23:00:00Z"}])
    g = pd.DataFrame([{"game_id": 1, "home_team": "Iowa State", "away_team": "Oklahoma State"}])
    news = {cfb_news.game_key("Iowa State", "Oklahoma State", "2026-10-31T23:00:00Z"): e}
    out = cfb_picks.drop_red_flags(picks, g, news)
    assert list(out.pick) == ["Under"]


def test_handicapper_passes_halves_and_explains(monkeypatch):
    from types import SimpleNamespace as NS
    import cfb_handicap
    monkeypatch.setattr(cfb_handicap, "MATCHUP_ALL", False)
    replies = iter(['```json\n{"decision": "pass", "reason": "Coach fired Tuesday."}\n```',
                    '```json\n{"decision": "half", "reason": "Wind cuts against the over."}\n```',
                    '```json\n{"decision": "bet", "reason": "Their run game should control it."}\n```'])

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    return NS(content=[NS(type="text", text=next(replies))], stop_reason="end_turn",
                              usage=NS(input_tokens=1000, output_tokens=100))

    base = {"away": "A", "home": "B", "commence": "2026-11-07T19:00:00Z", "price": -110,
            "book": "FanDuel", "win_chance": 0.56, "ev": 0.06, "why": "Model likes it.",
            "game_id": 1, "injuries": [], "situation": []}
    picks = pd.DataFrame([{**base, "market": "spreads", "pick": "A", "point": -3.0, "stake": 10.0},
                          {**base, "market": "totals", "pick": "Over", "point": 50.5, "stake": 8.0},
                          {**base, "game_id": 2, "market": "spreads", "pick": "C", "point": 4.0, "stake": 6.0}])
    news = {}
    out = cfb_handicap.review(picks, news, [], pd.Timestamp("2026-11-03", tz="UTC"), client=Client())
    assert list(out.pick) == ["Over", "C"]
    assert list(out.stake) == [4.0, 6.0]
    assert out.why.iloc[1].endswith("Handicapper: Their run game should control it.")
    assert len(news["_reviews"]) == 3 and news["_budget"]["usd"] > 0
    # same bets again: answered from the cache, no new calls
    again = cfb_handicap.review(picks, news, [], pd.Timestamp("2026-11-03", tz="UTC"), client=Client())
    assert list(again.pick) == ["Over", "C"]


def test_handicapper_leans_default_to_pass_and_bet_small(monkeypatch):
    from types import SimpleNamespace as NS
    import cfb_handicap
    monkeypatch.setattr(cfb_handicap, "MIN_LEAN", 0.5)
    replies = iter(['```json\n{"decision": "bet", "reason": "They have rolled similar teams by 20."}\n```',
                    '```json\n{"decision": "pass", "reason": "Nothing beyond the number."}\n```',
                    'no json at all'])

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    assert kw["system"] == cfb_handicap.LEAN_SYSTEM
                    return NS(content=[NS(type="text", text=next(replies))], stop_reason="end_turn",
                              usage=NS(input_tokens=1000, output_tokens=100))

    now = pd.Timestamp("2026-11-05", tz="UTC")
    rows = []
    for i, (lean, ev) in enumerate([(1.2, 0.01), (0.9, 0.0), (0.7, -0.01), (0.3, 0.02)]):
        rows.append({"event_id": f"e{i}", "market": "spreads", "pick": f"T{i}", "point": -3.0,
                     "price": -110, "book": "DraftKings", "ev": ev, "p_win": 0.52, "p_push": 0.0,
                     "commence": "2026-11-07T19:00:00Z", "away": f"T{i}", "home": f"U{i}", "lean": lean})
    rows.append({**rows[0], "event_id": "e9", "book": "Caesars"})          # not my book
    priced = pd.DataFrame(rows)
    ids = {f"e{i}": i + 1 for i in range(10)}
    picks = pd.DataFrame([{"event_id": "e2", "market": "totals"}])
    cand = cfb_handicap.lean_candidates(priced, lambda p: p.lean, ids, ["DraftKings", "FanDuel"],
                                        picks, now, 1000)
    assert list(cand.pick) == ["T0", "T1", "T2"]          # 0.3 lean and Caesars left out
    assert (cand.stake == 5.0).all()
    cand = cand.assign(injuries=[[]] * 3, situation=[[]] * 3)
    out = cfb_handicap.review(cand, {}, [], now, client=Client(), leans=True)
    assert list(out.pick) == ["T0"]                       # pass and unparseable both drop
    assert out.why.iloc[0].startswith("Handicapper's pick: They have rolled")


def test_pattern_bets_need_this_seasons_case():
    from types import SimpleNamespace as NS
    import cfb_handicap
    seen = []
    replies = iter(['```json\n{"decision": "pass", "reason": "Both offenses are scoring 40."}\n```',
                    'garbled'])

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    seen.append(kw["system"])
                    return NS(content=[NS(type="text", text=next(replies))], stop_reason="end_turn",
                              usage=NS(input_tokens=1000, output_tokens=100))

    base = {"away": "A", "home": "B", "commence": "2026-11-07T19:00:00Z", "price": -110,
            "book": "FanDuel", "win_chance": 0.55, "ev": 0.05, "game_id": 1,
            "injuries": [], "situation": [], "market": "totals", "stake": 10.0,
            "why": "A conference game at night with a total of 60+: the under has hit about 55% in games like this."}
    picks = pd.DataFrame([{**base, "pick": "Under", "point": 60.5},
                          {**base, "game_id": 2, "pick": "Under", "point": 61.5}])
    out = cfb_handicap.review(picks, {}, [], pd.Timestamp("2026-11-03", tz="UTC"), client=Client())
    assert len(out) == 0                                  # pass, and garbled counts as pass
    assert seen == [cfb_handicap.PATTERN_SYSTEM] * 2
