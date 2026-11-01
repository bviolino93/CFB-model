"""
The handicapper: a last look at each college bet the way a sharp human
bettor would, with everything the model has (Brian, 2026-11-01: "think like
a human bettor that has all the data you have").

For each bet the picks run is about to make, Claude gets the model's number
and lean, the price, the line the bet was first alerted at, the injury and
situation news already looked up (cfb_news), and what this kind of bet did
in the backtest. It answers bet, half (bet half the stake) or pass, with
the football reason. No web searches: it reuses the news, so a review is
about a cent. Reviews are kept in news.json ("_reviews") so a bet is
reviewed again only when its line, price or news changes. Spend counts
toward cfb_news's monthly stop.

Unbacktested: nobody has shown judgment calls like this beat the close, so
the weekly closing-line review is the judge. HANDICAP = False turns it off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import pandas as pd

import cfb_news

HANDICAP = True

SYSTEM = """You are a sharp, disciplined college football bettor making the \
final call on bets a model has flagged. You see the model's numbers, the \
price, the news a researcher found (injuries, coaching, motivation, travel, \
weather) and how this kind of bet has done historically.

Analyze the matchup itself: how each offense's strengths (passing, \
rushing, efficiency, pace) line up against the other defense, recent form \
and scoring, injuries and the situation. Think like a professional: the betting market is efficient, so the model's \
edge is small and real only when nothing it can't see undermines it. Pass \
when there is a concrete reason the number is stale or wrong: news the model \
doesn't price (a key player out for the first time on the side we bet, a \
coaching or discipline story, a motivation mismatch, bad weather for an \
over), a line that has already moved past our number, or a spot that has \
historically burned this kind of bet. Bet half when the case is real but \
something cuts against it. Otherwise bet: don't pass on vague feelings, and \
don't invent facts that aren't in what you're given.

Bets that rest on a historical pattern (the totals patterns) get extra \
scrutiny: the pattern describes past seasons, and you can see this one. \
Pass a pattern bet when this season's evidence clearly contradicts it, for \
example an under on two teams whose games have been going well over, \
whose offenses are scoring at will, or where our own count sits above the \
line; or an over on two teams whose games keep coming in low against \
weak opposition. Say what in this season's results made you pass.

Answer with one ```json block and nothing after it:
{"decision": "bet|half|pass", "reason": "<one or two sentences on the football case, plain English, no jargon>"}"""

TRACK = {
    "spreads": "Model spreads with a 1.5+ point lean won 56% against opening lines in "
               "2017-21 and 55% in 2022-25; against closing lines the edge mostly fades, "
               "so betting early matters.",
    "scoring": "Recent scoring runs: in 2017-25, betting WITH a two-team run of games going "
               "6+ points over or under our totals lost (the market overreacts); fading it won "
               "about 55%, but not in 2014-16, so treat it as a mild lean, not a reason by itself. "
               "Spread form (beating our margins) predicted nothing against the close.",
    "totals": "The totals patterns (big favorite with a low total -> over; totals of 60+ -> "
              "under) won about 55-56% against closing totals since 2015 (the big-favorite "
              "over ran 53% in 2017-21, its weakest stretch; 24+ favorites with totals "
              "under 50 hit 62-65% in every era). Our own totals model alone is "
              "about break-even.",
}


LEAN_SYSTEM = """You are a sharp college football handicapper. The model only \
leans to this side (not enough for a bet on its numbers alone), so this is your \
call on the football: recent form and who it came against, common opponents, \
how each team has scored against expectations lately, the matchup, injuries and the situation (coaching, motivation, travel, weather). \
Bet only when the football case is clearly strong and nothing in the news cuts \
against it; otherwise pass. Most of these should be passes. Don't invent facts \
that aren't in what you're given.

Answer with one ```json block and nothing after it:
{"decision": "bet|pass", "reason": "<one or two sentences on the football case, plain English>"}"""

PATTERN_SYSTEM = """You are a sharp college football handicapper making the \
final call on a bet. Brian wants every bet to rest on analysis of this game, \
not on old patterns or a model number alone. Break down the matchup: how \
each offense (passing, rushing, efficiency, pace) lines up against the other \
defense, both teams' season scoring and points allowed, recent form and who \
it came against, injuries and the situation (weather, motivation, travel). \
Bet only when that analysis makes a clear case for this side; pass when it \
is unclear or points the other way. Don't invent facts that aren't in what \
you're given.

Answer with one ```json block and nothing after it:
{"decision": "bet|pass", "reason": "<two or three sentences on the matchup case, plain English>"}"""

# Brian (2026-11-01): "let's start doing that type of analysis". Every
# model bet, not only the totals patterns, must have the matchup case.
MATCHUP_ALL = True


def is_pattern(why: str) -> bool:
    """A totals-pattern bet (cfb_picks.TOTAL_RULES reasons)."""
    return "in games like this" in (why or "")


LEAN_STAKE = 0.005          # of bankroll ($5 on $1,000) for a handicapper's own pick
MAX_LEANS = 10              # games reviewed per run beyond the bets, biggest lean first
MIN_LEAN = 0.0              # any game the model has a side on (Brian: analyze every game)


def _results(g: pd.DataFrame | None, school: str, before) -> list[tuple]:
    """This season's finished games for `school`, oldest first: (opponent,
    margin, home/away, date, points for, points against, total vs our
    pre-game total)."""
    if g is None or not len(g):
        return []
    x = g[((g.home_team == school) | (g.away_team == school)) & g.margin.notna()].copy()
    x["when"] = pd.to_datetime(x.start_date, utc=True)
    x = x[x.when < pd.Timestamp(before)]
    if not len(x):
        return []
    x = x[x.season == x.season.max()].sort_values("when")
    out = []
    for r in x.itertuples():
        home = r.home_team == school
        opp = r.away_team if home else r.home_team
        m = r.margin if home else -r.margin
        pf, pa = (r.home_points, r.away_points) if home else (r.away_points, r.home_points)
        vs = float(r.points - r.pred_total) if pd.notna(getattr(r, "pred_total", None)) else None
        out.append((opp, float(m), "home" if home else "away", f"{r.when:%b %-d}", pf, pa, vs))
    return out


def form_text(g, school: str, before) -> str:
    res = _results(g, school, before)
    if not res:
        return f"{school}: no results on file this season."
    w = sum(r[1] > 0 for r in res)
    pf = sum(r[4] for r in res) / len(res)
    pa = sum(r[5] for r in res) / len(res)
    last = "; ".join(f"{d} {'vs' if h == 'home' else 'at'} {o} {'W' if m > 0 else 'L'} {f:g}-{a:g}"
                     for o, m, h, d, f, a, _ in res[-6:])
    vs = [r[6] for r in res[-3:] if r[6] is not None]
    trend = ""
    if len(vs) >= 2:
        avg = sum(vs) / len(vs)
        trend = (f" Its last {len(vs)} games went {abs(avg):.1f} points a game "
                 f"{'over' if avg > 0 else 'under'} our pre-game totals.")
    return (f"{school} is {w}-{len(res) - w} this season, scoring {pf:.1f} and allowing {pa:.1f} "
            f"a game. Latest: {last}.{trend}")


def common_text(g, a: str, b: str, before) -> str:
    ra = {r[0]: r[1] for r in _results(g, a, before)}
    rb = {r[0]: r[1] for r in _results(g, b, before)}
    both = sorted(set(ra) & set(rb))
    if not both:
        return "No common opponents this season."
    return "Common opponents: " + "; ".join(f"{o}: {a} {ra[o]:+g}, {b} {rb[o]:+g}" for o in both) + "."


STATS = [("off_epa", "offense EPA/play", True), ("off_sr", "offense success rate", True),
         ("off_pass_epa", "passing EPA/play", True), ("off_rush_epa", "rushing EPA/play", True),
         ("off_ppd", "points per drive", True), ("def_epa", "defense EPA/play allowed", False),
         ("def_pass_epa", "pass defense EPA allowed", False), ("def_rush_epa", "run defense EPA allowed", False),
         ("def_ppd", "points per drive allowed", False), ("off_drv", "drives a game (pace)", True)]


def _team_table(g: pd.DataFrame, season) -> pd.DataFrame:
    """Each team's latest rating this season (entering its most recent game
    on file), from the h_/a_ columns, one row per team."""
    x = g[g.season == season]
    parts = []
    for side in ("h", "a"):
        cols = {f"{side}_{k}": k for k, _, _ in STATS if f"{side}_{k}" in x.columns}
        t = x[["start_date", f"{'home' if side == 'h' else 'away'}_team", *cols]].rename(
            columns={**cols, f"{'home' if side == 'h' else 'away'}_team": "team"})
        parts.append(t)
    t = pd.concat(parts).dropna(subset=["off_epa"]).sort_values("start_date")
    return t.groupby("team").last()


def matchup_text(g, game_id) -> str:
    """The two teams' efficiency profiles going into this game, each stat
    with its rank among this season's teams (1 = best)."""
    if g is None or game_id is None:
        return ""
    row = g[g.game_id == game_id]
    if not len(row) or "h_off_epa" not in g.columns:
        return ""
    r = row.iloc[0]
    try:
        table = _team_table(g, r.season)
    except Exception:
        return ""
    n = len(table)
    out = []
    for side, team in (("a", r.away_team), ("h", r.home_team)):
        bits = []
        for k, label, high in STATS:
            v = r.get(f"{side}_{k}")
            if v is None or pd.isna(v) or k not in table:
                continue
            rank = int((table[k] > v).sum() + 1) if high else int((table[k] < v).sum() + 1)
            bits.append(f"{label} {v:.2f} (#{rank} of {n})")
        if bits:
            out.append(f"{team} profile: " + "; ".join(bits) + ".")
    return " ".join(out)


def _brief(p, injuries: list, situation: list, first: dict | None, g=None,
           schools: tuple | None = None) -> str:
    side = f"{p.pick} {p.point:+g}" if p.market == "spreads" else f"{p.pick} {p.point:g}"
    lines = [f"Game: {p.away} at {p.home}, kickoff {pd.Timestamp(p.commence):%a %b %-d %H:%M UTC}",
             f"Bet: {side} ({p.market}) at {int(p.price):+d}, {p.book}",
             f"Model: win chance {p.win_chance:.1%}, expected value {p.ev:+.1%}",
             f"Model's reasoning: {p.why}",
             f"History: {TRACK.get(p.market, '')}",
             f"Scoring trends: {TRACK['scoring']}"]
    if first:
        lines.append(f"First alerted at {first.get('point')} ({first.get('price')}); "
                     f"the line now is {p.point} ({int(p.price)}).")
    if schools:
        away, home = schools
        lines += [form_text(g, away, p.commence), form_text(g, home, p.commence),
                  common_text(g, away, home, p.commence)]
        prof = matchup_text(g, getattr(p, "game_id", None))
        if prof:
            lines.append("Matchup (season efficiency, opponent-adjusted inputs to our model): " + prof)
    lines.append("Injuries: " + (json.dumps(injuries) if injuries else "none reported"))
    lines.append("Situation: " + (json.dumps(situation) if situation else "nothing reported"))
    return "\n".join(lines)


def _parse(text: str, default: str = "bet") -> dict:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    try:
        d = json.loads(m[-1]) if m else {}
    except json.JSONDecodeError:
        d = {}
    dec = str(d.get("decision", "")).lower()
    return {"decision": dec if dec in ("bet", "half", "pass") else default,
            "reason": str(d.get("reason", "")).strip()}


def _ask(client, brief: str, system: str = SYSTEM) -> dict:
    resp = client.beta.messages.create(
        model=cfb_news.MODEL, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": brief}],
        output_config={"effort": cfb_news.EFFORT},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    cfb_news._tally(getattr(resp, "usage", None))
    default = "pass" if system in (LEAN_SYSTEM, PATTERN_SYSTEM) else "bet"
    if resp.stop_reason == "refusal":
        return {"decision": default, "reason": ""}
    return _parse("".join(b.text for b in resp.content if b.type == "text"), default)


def review(picks: pd.DataFrame, news: dict, history: list | None = None,
           now=None, client=None, g: pd.DataFrame | None = None,
           leans: bool = False) -> pd.DataFrame:
    """picks: the run's bets with injuries/situation attached. Drops passes,
    halves 'half' stakes, and adds the reason to `why`. With leans=True the
    rows are model leans, not bets: only a 'bet' keeps one, as the
    handicapper's own pick. Never fails the run."""
    if not HANDICAP or not len(picks):
        return picks.iloc[:0] if leans else picks
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    month = f"{now:%Y-%m}"
    budget = news.get("_budget", {})
    spent = max(budget.get("usd", 0.0) if budget.get("month") == month else 0.0,
                cfb_news.ALREADY_SPENT.get(month, 0.0))
    start = cfb_news.cost()
    first = {}
    for b in history or []:
        first[(b.get("game_id"), b.get("market"), b.get("pick"))] = {
            "point": b.get("first_point", b.get("point")), "price": b.get("first_price", b.get("price"))}
    done = news.setdefault("_reviews", {})
    keep, stakes, whys = [], [], []
    for p in picks.itertuples():
        inj = list(getattr(p, "injuries", []) or [])
        sit = list(getattr(p, "situation", []) or [])
        sch = None
        if g is not None and getattr(p, "game_id", None) is not None:
            row = g[g.game_id == p.game_id]
            if len(row):
                sch = (row.iloc[0].away_team, row.iloc[0].home_team)
        brief = _brief(p, inj, sit, first.get((getattr(p, "game_id", None), p.market, p.pick)), g, sch)
        pattern = not leans and (MATCHUP_ALL or is_pattern(p.why))
        if leans:
            brief = "LEAN ONLY (model is short of a bet).\n" + brief
        elif pattern:
            brief = "MODEL BET: bet only if the matchup analysis supports it.\n" + brief
        key = hashlib.sha1(brief.encode()).hexdigest()[:16]
        got = done.get(key)
        if got is None:
            if spent + cfb_news.cost() - start >= cfb_news.MONTHLY_BUDGET:
                print("handicap: monthly budget reached; bets go out unreviewed")
                got = None
            else:
                try:
                    if client is None:
                        if not os.environ.get("ANTHROPIC_API_KEY"):
                            print("handicap: no ANTHROPIC_API_KEY, skipping")
                            return picks.iloc[:0] if leans else picks
                        import anthropic
                        client = anthropic.Anthropic()
                    got = _ask(client, brief, LEAN_SYSTEM if leans else
                               PATTERN_SYSTEM if pattern else SYSTEM)
                    done[key] = {**got, "at": now.isoformat()}
                except Exception as e:          # a review is a check, never a reason to fail
                    print(f"handicap: review failed ({type(e).__name__}: {e})")
                    got = None
        dec = (got or {}).get("decision", "pass" if leans else "bet")
        if pattern and got is not None and dec == "half":
            dec = "pass"
        reason = (got or {}).get("reason", "")
        print(f"handicap{' (lean)' if leans else ''}: {p.pick} {p.point} -> {dec}: {reason}")
        if leans:
            keep.append(dec == "bet")
            stakes.append(p.stake)
            whys.append(f"Handicapper's pick: {reason} (The model leans this way too, "
                        f"by less than a bet needs; small stake.)")
            continue
        keep.append(dec != "pass")
        stakes.append(round(p.stake * (0.5 if dec == "half" else 1.0), 2))
        whys.append(f"{p.why} Handicapper: {reason}" if reason else p.why)
    # keep the review cache to the last 300
    if len(done) > 300:
        for k in sorted(done, key=lambda k: done[k].get("at", ""))[:len(done) - 300]:
            done.pop(k)
    news["_budget"] = {"month": month, "usd": round(spent + cfb_news.cost() - start, 4)}
    out = picks.assign(stake=stakes, why=whys)
    return out[keep]


def lean_candidates(priced: pd.DataFrame, lean_pts, ids: dict, my_books, picks: pd.DataFrame,
                    now, bankroll: float, hours: float = 72) -> pd.DataFrame:
    """Games the model leans on but doesn't bet: the best price at my books
    for the side it leans to, MIN_LEAN+ points, not already a pick, kicking
    off within `hours`. The biggest MAX_LEANS leans, one per game."""
    if priced is None or priced.empty:
        return priced.iloc[:0] if priced is not None else pd.DataFrame()
    p = priced[priced.book.isin(my_books) & priced.market.isin(["spreads", "totals"])].copy()
    p["lean"] = lean_pts(p)
    kick = pd.to_datetime(p.commence, utc=True)
    now = pd.Timestamp(now)
    p = p[(kick > now) & (kick <= now + pd.Timedelta(hours=hours)) & (p.lean >= MIN_LEAN) & (p.lean > 0)
          & (p.ev >= -0.03)]
    if len(picks):
        taken = set(zip(picks.event_id, picks.market))
        p = p[[k not in taken for k in zip(p.event_id, p.market)]]
    if p.empty:
        return p
    p = (p.sort_values("ev", ascending=False).drop_duplicates(["event_id", "market"])
          .sort_values("lean", ascending=False).drop_duplicates("event_id").head(MAX_LEANS))
    p["game_id"] = [ids.get(e) for e in p.event_id]
    p = p[p.game_id.notna()]
    p["stake"] = round(bankroll * LEAN_STAKE, 2)
    p["win_chance"] = p.p_win / (1 - p.p_push)
    p["why"] = [f"The model leans {r.pick} by {r.lean:.1f} points (a bet needs "
                f"{1.5 if r.market == 'spreads' else 1.0:g})." for r in p.itertuples()]
    return p
