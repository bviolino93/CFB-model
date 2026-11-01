"""
College football picks: the NFL model's pick engine (bviolino93/NFL-model picks.py)
pointed at college games. Runs on a schedule from .github/workflows/picks.yml,
separately from the Streamlit app (app.py).

    ODDS_API_KEY=... python cfb_picks.py --out site/index.html

Same steps as the NFL picks:
  1. every book's spreads and totals from The Odds API (us + eu)
  2. sharp-weighted, de-vigged fair lines (selector.consensus), with the
     result distributions built from college games (bigger margins, fewer
     key numbers than the NFL)
  3. cfb_model's prediction of result-minus-line moves the fair line:
     spreads; totals move on TOTAL_RULES (patterns that held in every era,
     checked against our own possessions x points-per-possession total)
  4. every DraftKings/FanDuel offer priced; bets that clear the EV bar are
     sized (quarter Kelly) and explained in football terms

The bet history, alerts and dashboard work like the NFL model's
(bviolino93/NFL-model), published to GitHub Pages from the picks branch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import cfb_data
import cfb_model
import cfb_news
import ledger
import selector as sel

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
SCORES_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/scores"
DASHBOARD_URL = "https://bviolino93.github.io/CFB-model/"
MY_BOOKS = "draftkings,fanduel"
BOOK_NAME = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

# The model's lean is shrunk already; the caps stop a data glitch (a team
# with two games of play-by-play) from making a huge edge.
SPREAD_CAP = 3.0
TOTAL_CAP = 3.0
# The efficiency model's totals don't beat the line (model overs went 47-60%
# by season), so it only informs. Totals are bet on the patterns below.
MODEL_MARKETS = {"spread"}

# Totals patterns from totals_research.py: each held in 2012-16, 2017-21
# and 2022-25, against opening and closing totals. (bet side, rule, chance
# the bet side wins, plain-English reason). The chances are a little under
# what history shows so stated edges stay conservative. The first rule that
# matches a game applies. own_gap is our possession total minus the market
# (cfb_model.possession_total): the over patterns only count when our own
# count agrees, since without it they hit ~52%, a loser at -110. For the
# 60+ under it made no difference against closing totals, so it's ignored.
TOTAL_RULES = [
    ("Over", lambda t, sp, night, conf, gap: sp >= 24 and t < 50 and gap >= 0, 0.60,
     "a {fav}-point favorite with a total under 50, and our possession count "
     "agrees: since 2015 the over has hit about 63% in games like this (135 "
     "games). Big favorites keep scoring and low totals leave no room for it"),
    ("Over", lambda t, sp, night, conf, gap: sp >= 21 and t < 55 and gap >= 0, 0.56,
     "a {fav}-point favorite with a total under 55, and our possession count "
     "agrees: since 2015 the over has hit about 56-58% in games like this "
     "(500 games). Mismatches score more than low totals allow for"),
    ("Under", lambda t, sp, night, conf, gap: t >= 60 and night and conf, 0.545,
     "a conference game at night with a total of 60+: since 2017 the under has "
     "hit about 55% in games like this (450 games). Books shade big totals up "
     "for public over money"),
    ("Under", lambda t, sp, night, conf, gap: t >= 60, 0.535,
     "a total of 60+: since 2017 the under has hit about 54-55% (1,800 games). "
     "Books shade big totals up for public over money"),
    # Our own count on its own (Brian's call, 2026-10-26). Only unders: when
    # it sits 3+ under a total below 60 the under hit 53.5% against openers
    # in 2017-25 (387 games, 45-65% by season) but 50% against closing
    # totals; overs off the count alone ran 48-50%. At 53.5% it clears the
    # 3% edge bar only at about -105 or better, so it bets rarely.
    ("Under", lambda t, sp, night, conf, gap: gap <= -3, 0.535,
     "our possession count puts this game {gap:.0f} points under the total: "
     "when our count sits that far under, the under has hit about 53-54% "
     "against early lines since 2017 (387 games), less against closing lines, "
     "so only a good price gets it a bet"),
]
# College picks go through at a 3% edge (NFL: 2%).
MIN_EV = 0.03
# Injury news is looked up for the picks and for games with an offer within
# this much of MIN_EV (~15-25 games a week, ~15 cents each), not all ~60.
NEWS_NEAR_MISS = 0.02
# Every bet needs the model on its side (Brian, 2026-10-31: "bets that will
# win, not just better price"): the model's own adjustments (ratings, injury
# news) must move the fair line at least this many points toward the bet,
# or a validated totals pattern must point its way. A book being off the
# sharp consensus no longer carries a bet on its own. In the 2017-25
# backtest, spread leans under 1 point covered 50%.
MIN_LEAN = 1.0
# Spreads need more (2026-11-01 re-run of cfb_model.backtest, scored vs the
# opening line): 1+ point leans won 56.4% in 2017-21 but 51.2% in 2022-25;
# 1.5+ won 56.4% and 55.2% (n=367, 181). Against the close both fade, so
# the edge is in betting early lines; picks.yml runs from Sunday night.
MIN_SPREAD_LEAN = 1.5

# Half stakes until the picks show they beat closing lines (Brian,
# 2026-10-31, after a 2-6 start in college): every stake, dogs included.
STAKE_SCALE = 0.5

# Totals-patterns-only mode (tried 2026-10-31, Brian wants spreads too:
# "I want spread and totals"). Off: spreads and totals with MIN_LEAN.
BACKTESTED_ONLY = False
# Underdog moneylines: off since 2026-10-31 (0-2, no backtested edge;
# Brian asked for spreads and totals). Set True to bet them again.
BET_DOGS = False

# Underdog moneylines (Brian, 2026-10-28). moneyline_research.py, every
# book's closing moneyline 2012-25: dogs past +400 lose 25-45% (the
# consensus overrates them), the model's lean barely moves a moneyline
# and its dog calls don't win. The best price on the board for a +300-or-
# shorter dog, when it beat the de-vigged consensus, returned +7% in
# 2012-21 but lost in 2022-25. So: priced off the moneyline market, moved
# by the same spread adjustments as the spreads, bet only when DraftKings
# or FanDuel pay more than that is worth. Small, separate stakes.
DOG_MAX_PRICE = 300           # +300 or shorter
DOG_MAX_BETS = 3              # the best few each week
DOG_MAX_STAKE = 0.01          # of bankroll per dog ($10 on $1,000)

# Odds API name -> schedule name where the school part differs.
ALIASES = {
    "appalachian state": "app state", "hawaii": "hawaii", "umass": "massachusetts",
    "southern mississippi": "southern miss", "louisiana monroe": "ul monroe",
    "ul monroe": "ul monroe", "sam houston state": "sam houston",
    "connecticut": "uconn", "fiu": "florida international",
    "north carolina state": "nc state", "san jose state": "san jose state",
    "texas san antonio": "utsa", "central florida": "ucf",
    "louisiana lafayette": "louisiana", "miami fl": "miami",
    "mississippi": "ole miss", "brigham young": "byu",
    "southern methodist": "smu", "texas christian": "tcu",
    "louisiana state": "lsu", "southern california": "usc",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"['’.]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _matches(odds_name: str, school: str) -> int:
    """Length of the school name if the Odds API name is that school plus
    a nickname ("Miami (OH) RedHawks" -> "Miami (OH)"), else 0."""
    o, sc = _norm(odds_name), _norm(school)
    for a, b in ALIASES.items():
        if o.startswith(a + " ") and b == sc:
            return len(sc) + 1
    return len(sc) if o == sc or o.startswith(sc + " ") else 0


def match_games(fair: dict, sched: pd.DataFrame) -> dict:
    """Odds event id -> schedule game_id. A game matches when both teams'
    names do, or one does and it's that team's only game within a day."""
    s = sched.assign(day=pd.to_datetime(sched.start_date, utc=True))
    out = {}
    for ev, f in fair.items():
        kick = pd.Timestamp(f.commence)
        near = s[(s.day - kick).abs() <= pd.Timedelta(hours=36)]
        best, score = None, 0
        for r in near.itertuples():
            h, a = _matches(f.home_team, r.home_team), _matches(f.away_team, r.away_team)
            if not (h or a):
                # neutral-site games are sometimes listed the other way round
                h, a = _matches(f.home_team, r.away_team), _matches(f.away_team, r.home_team)
                if h and a:
                    h = a = 0       # flipped home/away: skip rather than mis-sign
            sc = (h > 0) + (a > 0) + 0.001 * (h + a)
            if sc > score:
                best, score = r.game_id, sc
        if best is not None and score >= 1:
            out[ev] = int(best)
    return out


def outcome_models(g: pd.DataFrame):
    d = g.dropna(subset=["close_spread", "close_total", "margin"])
    d = d[d.season >= 2012]
    return (sel.OutcomeModel(-d.close_spread.to_numpy(float), d.margin.to_numpy(float)),
            sel.OutcomeModel(d.close_total.to_numpy(float), d.points.to_numpy(float)))


def fetch_rows(key: str, regions: str = "us,eu") -> pd.DataFrame:
    r = requests.get(ODDS_URL, params={
        "apiKey": key, "regions": regions, "markets": "spreads,totals,h2h",
        "oddsFormat": "american"}, timeout=30)
    r.raise_for_status()
    print(f"odds: {len(r.json())} college events, credits left "
          f"{r.headers.get('x-requests-remaining')}")
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([x for ev in r.json() for x in _rows(ev, snap)])


def week_end(now) -> pd.Timestamp:
    """The next Sunday 12:00 UTC, after Saturday's late games: a run from
    Sunday evening on looks at the coming Saturday."""
    now = pd.Timestamp(now).tz_convert("UTC")
    end = now.normalize() + pd.Timedelta(days=(6 - now.weekday()) % 7, hours=12)
    return end if end > now else end + pd.Timedelta(days=7)


def _rows(event, snap):
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                yield {
                    "snapshot_utc": snap,
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": bk.get("key"),
                    "market": mk.get("key"),
                    "outcome": oc.get("name"),
                    # Props put the player in `description`.
                    "player": oc.get("description", ""),
                    "point": oc.get("point"),
                    "price": oc.get("price"),
                    "last_update": mk.get("last_update", bk.get("last_update")),
                }


# ----------------------------------------------------------------------
# Plain-English reasons
# ----------------------------------------------------------------------
def _tier(p: float) -> str:
    if p >= 0.9:
        return "top 10%"
    if p >= 0.75:
        return "top quarter"
    if p <= 0.1:
        return "bottom 10%"
    if p <= 0.25:
        return "bottom quarter"
    return "middle of the pack"


class Ranks:
    """Where each team's opponent-adjusted unit sits among the teams
    playing this week (1 = best; defenses rank low EPA allowed as best)."""

    def __init__(self, rows: pd.DataFrame):
        t = []
        for side, team in (("h", "home_team"), ("a", "away_team")):
            c = {f"{u}_{s}": rows[f"{side}_{u}_{s}"].values for u in ("off", "def")
                 for s in ("epa", "pass_epa", "rush_epa", "plays")}
            t.append(pd.DataFrame({"team": rows[team].values, **c}))
        t = pd.concat(t).drop_duplicates("team").set_index("team")
        self.p = t.rank(pct=True)
        for c in t.columns:
            if c.startswith("def_"):
                self.p[c] = 1 - self.p[c] + 1 / len(t)

    def tier(self, team, col) -> str | None:
        try:
            v = self.p.loc[team, col]
        except KeyError:
            return None
        return None if pd.isna(v) else _tier(v)


UNIT = {"epa": ("offense", "defense", "per play"),
        "pass_epa": ("passing game", "pass defense", "per dropback"),
        "rush_epa": ("running game", "run defense", "per carry")}


def _matchup(me, opp, stat, ranks):
    """'Oregon's passing game (top 10%) against Purdue's pass defense
    (bottom quarter)', from whichever side of the ball favours `me` more."""
    o, d, _ = UNIT[stat]
    p = lambda t, c: ranks.p[c].get(t, 0.5)
    if p(me, "off_" + stat) - p(opp, "def_" + stat) >= p(me, "def_" + stat) - p(opp, "off_" + stat):
        return (f"{me}'s {o} ({ranks.tier(me, 'off_' + stat)}) against {opp}'s {d} "
                f"({ranks.tier(opp, 'def_' + stat)})")
    return (f"{me}'s {d} ({ranks.tier(me, 'def_' + stat)}) against {opp}'s {o} "
            f"({ranks.tier(opp, 'off_' + stat)})")


def _units(home, away, stat, ranks, over: bool) -> str:
    """The two units (of the four offenses and defenses) that most push a
    game over (or under): 'Oregon's passing game (top 10%) and Purdue's
    pass defense (bottom quarter)'."""
    o, d, _ = UNIT[stat]
    p = lambda t, c: ranks.p[c].get(t, 0.5)
    # how much each unit pushes scoring up: good offenses, bad defenses
    cand = [(p(t, "off_" + stat), f"{t}'s {o} ({ranks.tier(t, 'off_' + stat)})") for t in (home, away)]
    cand += [(1 - p(t, "def_" + stat), f"{t}'s {d} ({ranks.tier(t, 'def_' + stat)})") for t in (home, away)]
    cand = sorted((c for c in cand if (c[0] >= 0.75 if over else c[0] <= 0.25)),
                  reverse=over)
    return " and ".join(t for _, t in cand[:2])


def _fav(home, away, m):
    return f"{home if m >= 0 else away} by {abs(m):.1f}"


def explain(p, row: pd.Series, parts: dict, ranks: Ranks, adj: float,
            fair_market: float, call_out: bool = True) -> str:
    """The model's case for a bet in football terms: what it expects, then
    each opinion behind it (cfb_model.LineModel) that pushes our way, with
    the matchup that drives the efficiency numbers, and the biggest one
    that pushes the other way. parts: opinion -> (push on the home margin
    or total, the opinion itself in points)."""
    home, away = row.home_team, row.away_team
    spread = p.market == "spreads"
    if spread:
        sign = 1 if p.pick == p.home else -1
        me, opp = (home, away) if sign > 0 else (away, home)
        mine = sign * (fair_market + adj)
        line = float(p.point)
        room = mine + line
        call = (f"The model has {me} winning by {mine:.1f}" if mine >= 0
                else f"The model has {me} losing by only {-mine:.1f}")
        call += (f", and they're getting {line:g} ({room:.1f} points of room)."
                 if line > 0 else f", {room:.1f} more than the {-line:g} they're giving.")
    else:
        over = str(p.pick).lower() != "under"
        sign = 1 if over else -1
        mine = fair_market + adj
        call = (f"The model puts this game at {mine:.1f} points, "
                f"{abs(mine - float(p.point)):.1f} {'over' if over else 'under'} "
                f"the {float(p.point):g} total.")

    def say(k, view, toward):
        if spread:
            t, u = (me, opp) if toward else (opp, me)
            if k == "eff":
                return (f"play-by-play efficiency, adjusted for opponents, has it "
                        f"{_fav(home, away, view)}: {_matchup(t, u, 'epa', ranks)}")
            if k == "elo":
                return f"Elo has it {_fav(home, away, view)}"
            # scores: fitted with a negative weight. When the line has moved
            # far from a team's results, the market has usually been right
            # and then some, so it pushes further the market's way.
            return (f"the market ({_fav(home, away, fair_market)}) is far from what "
                    f"the scores so far say ({_fav(home, away, view)}); when the "
                    f"market strays that far from the scoreboard it has usually "
                    f"been right, and then some")
        up = over == toward
        if k == "eff":
            # no number: the efficiency fit runs high on low totals and
            # low on high ones; its direction is what the backtest trusts
            units = _units(home, away, "epa", ranks, up)
            return (f"play-by-play efficiency points to {'more' if up else 'less'} "
                    f"scoring than the total" + (f": {units}" if units else ""))
        return f"scoring so far, adjusted for opponents, puts it at {view:.1f}"

    def agrees(k, push, view):
        # An opinion with a negative fitted weight pushes AWAY from what it
        # says; only "scores" has a sentence for that, so the others are
        # cited only when they say what they push.
        return k == "scores" if spread and (push > 0) != (view > fair_market) \
            else True

    pro = sorted(((sign * v, k, view) for k, (v, view) in parts.items()
                  if sign * v >= 0.2 and agrees(k, v, view)), reverse=True)
    con = sorted(((-sign * v, k, view) for k, (v, view) in parts.items()
                  if -sign * v >= 0.4 and agrees(k, v, view)), reverse=True)
    why = [call] if call_out else []
    if pro:
        why.append("Why: " + "; ".join(say(k, view, True) for _, k, view in pro) + ".")
    if con:
        _, k, view = con[0]
        why.append("Working against it: " + say(k, view, False) + ".")
    return " ".join(w[0].upper() + w[1:] for w in why)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def total_rule(total: float, abs_spread: float, kickoff, conference: bool,
               own_gap: float = np.nan):
    """The first TOTAL_RULES row matching a game, or None. A missing
    possession total (own_gap NaN) fails the comparisons, so the over
    patterns don't fire without it."""
    night = pd.Timestamp(kickoff).tz_convert("America/New_York").hour >= 19
    for side, rule, p, why in TOTAL_RULES:
        if rule(total, abs_spread, night, bool(conference), own_gap):
            return side, p, why
    return None


def own_total_text(r, home: str, away: str) -> str:
    """Our possession-built total in words, or "" without the data."""
    if pd.isna(r.get("poss_total")):
        return ""
    return (f"Our own count: about {r.poss:.0f} possessions a side, {home} "
            f"scoring {r.ppd_home:.1f} points a possession and {away} "
            f"{r.ppd_away:.1f}, which builds to about {r.poss_total:.0f}.")


def rule_shift(tm, total: float, side: str, p: float) -> float:
    """Smallest move of the fair total (in quarter points, toward the bet
    side) at which the bet side wins with chance p, pushes aside, at the
    market total itself."""
    sign = 1 if side == "Over" else -1
    for k in range(0, 41):
        mu = total + sign * 0.25 * k
        p_over, p_push = tm.probs(mu, total)
        win = p_over if side == "Over" else 1 - p_over - p_push
        if win / max(1e-9, 1 - p_push) >= p:
            return mu - total
    return sign * 10.0


def lean(priced: pd.DataFrame) -> pd.Series:
    """Points the model's adjustments move the fair line toward each
    offer's side (fair minus the sharp consensus, signed to the side)."""
    toward = np.where(priced.market == "totals",
                      np.where(priced.pick.astype(str).str.lower() == "under", -1, 1),
                      np.where(priced.pick == priced.home, 1, -1))
    return (priced.fair - priced.fair_market) * toward


def model_backed(priced: pd.DataFrame, min_lean: float = MIN_LEAN) -> pd.DataFrame:
    """Offers the model itself argues for: MIN_LEAN points of lean their
    way, or a totals pattern pushing their way."""
    if priced.empty:
        return priced
    ln = lean(priced)
    trend = ((priced.market == "totals") & (ln > 0)
             & priced.sources.astype(str).str.split(",").map(lambda s: "trend" in s))
    if BACKTESTED_ONLY:
        return priced[trend]
    need = np.where(priced.market == "spreads", max(min_lean, MIN_SPREAD_LEAN), min_lean)
    return priced[(ln >= need - 1e-9) | trend]


def team_starters(g: pd.DataFrame, season: int) -> dict:
    """School -> its starting QBs this season, oldest first."""
    st = cfb_data.load_starters(season)
    if st.empty:
        return {}
    when = g.set_index("game_id").start_date
    st = st.assign(when=st.game_id.map(when)).dropna(subset=["when"]).sort_values("when")
    return st.groupby("pos_team").starter.apply(list).to_dict()


def qb_adjustments(fair: dict, ids: dict, upcoming: pd.DataFrame, news: dict,
                   starters: dict, done: set) -> list:
    """Fair-line nudges from the injury news (cfb_news): a backup QB's first
    start takes NEW_QB_PTS off that team's margin and off the total; other
    starters missing their first game take OTHER_* off the margin. Games in
    `done` are skipped (and added to it), so a game is adjusted once however
    many times the news is checked."""
    adj = []
    for ev, gid in ids.items():
        if gid not in upcoming.index or ev in done:
            continue
        u, f = upcoming.loc[gid], fair[ev]
        e = news.get(cfb_news.game_key(u.home_team, u.away_team, f.commence))
        if not e:
            continue
        done.add(ev)
        for school, sign in ((u.home_team, 1), (u.away_team, -1)):
            pts, _ = cfb_news.qb_change(e, school, starters.get(school, []))
            if pts:
                print(f"news: {school} starting a backup QB for the first time: "
                      f"{pts:+.1f} on their margin and the total")
                note = f"{school}'s backup QB makes his first start"
                adj += [sel.Adjustment(f.home_team, "spreads", sign * pts, "injury", note),
                        sel.Adjustment(f.home_team, "totals", pts, "injury", note)]
            other = cfb_news.others_pts(e, school)
            if other:
                print(f"news: {school} missing starters for the first time: {other:+.1f}")
                adj.append(sel.Adjustment(f.home_team, "spreads", sign * other, "injury",
                                          f"{school} is missing starters ({other:+.1f})"))
            tot = cfb_news.total_pts(e, school)
            if tot:
                print(f"news: {school}'s missing starters move the total {tot:+.1f}")
                adj.append(sel.Adjustment(f.home_team, "totals", tot, "injury",
                                          f"{school}'s missing starters move the total "
                                          f"{tot:+.1f}"))
    return adj


def dog_offers(priced: pd.DataFrame, my_books, min_ev: float,
               backed: bool = True) -> pd.DataFrame:
    """Underdog moneylines at my books: a dog at the sharp books (the
    market spread has the other team favoured), +100 to DOG_MAX_PRICE,
    clearing min_ev, and (backed) with the model leaning MIN_LEAN points
    their way."""
    if priced.empty:
        return priced
    d = priced[(priced.market == "h2h") & priced.book.isin(my_books)
               & (priced.price >= 100) & (priced.price <= DOG_MAX_PRICE)
               & (priced.ev >= min_ev)]
    d = d[(d.pick == d.home) != (d.fair_market >= 0)]
    return model_backed(d) if backed else d


def underdogs(priced, my_books, bankroll, pol) -> pd.DataFrame:
    """The best DOG_MAX_BETS underdogs, quarter Kelly capped at
    DOG_MAX_STAKE of the bankroll."""
    d = dog_offers(priced, my_books, pol.min_ev).copy()
    if d.empty:
        return d
    d = (d.sort_values("ev", ascending=False)
          .drop_duplicates(["event_id", "pick"]).head(DOG_MAX_BETS))
    d["stake"] = (np.minimum(d.kelly * pol.kelly_mult, DOG_MAX_STAKE)
                  * bankroll).round(2)
    return d.reset_index(drop=True)


def explain_dog(p, c: dict, ranks) -> str:
    """An underdog moneyline: what the price needs, what we make it, and
    the football behind it when the model moved the line its way."""
    row = c.get("row")
    names = ({p.home: row.home_team, p.away: row.away_team}
             if row is not None else {})
    me = names.get(p.pick, p.pick)
    book = BOOK_NAME.get(p.book, p.book)
    need = sel.american_to_prob(p.price)
    why = [f"{book} pays {int(p.price):+d} on {me}, which needs a {need:.0%} "
           f"win chance; we make it {p.win_chance:.0%}."]
    sign = 1 if p.pick == p.home else -1
    srcs = str(p.sources).split(",")
    if "spread" in c and "combined" in srcs and sign * c["spread"][0] > 0:
        pts, parts = c["spread"]
        mine = sign * (p.fair_market + pts)
        dog_by = abs(p.fair_market)
        why.append(f"The market has them as {dog_by:.1f}-point underdogs; "
                   + (f"the model has {me} winning by {mine:.1f}."
                      if mine >= 0 else
                      f"the model has them losing by only {-mine:.1f}."))
        rest = explain(p._replace(market="spreads", point=dog_by), row, parts,
                       ranks, pts, p.fair_market, call_out=False)
        if rest:
            why.append(rest)
    if "market" in srcs:
        q = (p.ev_market + 1 - p.p_push) / (sel.payout(p.price) + 1)
        q /= max(1e-9, 1 - p.p_push)
        fair_odds = 100 * (1 - q) / q if q < 0.5 else -100 * q / (1 - q)
        why.append(f"{'Also, the' if len(why) > 1 else 'The'} sharp books' "
                   f"moneylines make {me} worth about {fair_odds:+.0f}, so "
                   f"{book} is paying more than the market thinks they're worth.")
    return " ".join(why)


def board(priced: pd.DataFrame, my_books, now, hours: float = 24) -> list:
    """Every game kicking off in the next `hours`: for the spread and the
    total, the side the model leans to, by how many points, and the best
    price for it at my books. Bets or not; for Brian's look at the slate."""
    if priced is None or priced.empty:
        return []
    p = priced[priced.book.isin(my_books) & priced.market.isin(["spreads", "totals"])].copy()
    kick = pd.to_datetime(p.commence, utc=True)
    p = p[(kick > now) & (kick <= now + pd.Timedelta(hours=hours))]
    if p.empty:
        return []
    p["lean"] = lean(p)
    p = p.sort_values("ev", ascending=False).drop_duplicates(["event_id", "market", "pick"])
    out = []
    for (eid, mk), s in p.groupby(["event_id", "market"]):
        best = s.sort_values(["lean", "ev"], ascending=False).iloc[0]
        out.append({"commence": str(best.commence), "away": best.get("away"), "home": best.home,
                    "market": mk, "pick": best.pick, "point": float(best.point),
                    "price": int(best.price), "book": best.book,
                    "lean": round(float(best.lean), 2), "ev": round(float(best.ev), 4),
                    "win_chance": round(float(best.p_win / max(1e-9, 1 - best.p_push)), 4),
                    "sources": str(best.sources)})
    return sorted(out, key=lambda r: -r["lean"])


def write_board(priced, my_books, now, path):
    rows = board(priced, my_books, now)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"updated": now.isoformat(), "games": rows}, fh, indent=1, default=str)
    print(f"board: {len(rows)} spread/total reads in the next 24h")
    for r in rows[:40]:
        print(f"  {r['away']} @ {r['home']}: {r['pick']} {r['point']:g} ({r['price']}) "
              f"lean {r['lean']:+.1f} ev {r['ev']:+.1%}")


def build_picks(rows: pd.DataFrame, my_books, bankroll=1000.0, min_ev=MIN_EV,
                now=None, g: pd.DataFrame | None = None, news: dict | None = None,
                out: dict | None = None):
    """news: the injury news cache (cfb_news), or None to skip the check.
    It's updated in place with whatever this run looks up. out, if given,
    gets every priced offer ("priced") and the model's working ("ctx"),
    for the one-game check (cfb_game.py)."""
    out = {} if out is None else out
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    kick = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    rows = rows[(kick > now) & (kick < week_end(now))]
    g = cfb_model.load_features() if g is None else g
    mm, tm = outcome_models(g)
    fair = sel.consensus(rows, mm, tm)
    models = cfb_model.fit_models(g[g.margin.notna()])
    ids = match_games(fair, g[g.season == cfb_data.current_season(now)])
    print(f"matched {len(ids)} of {len(fair)} college games to the schedule")
    upcoming = g[g.game_id.isin(ids.values())].set_index("game_id")
    ranks = Ranks(upcoming) if len(upcoming) else None
    adj, ctx = [], {}
    for ev, gid in ids.items():
        f, r = fair[ev], upcoming.loc[gid]
        c = ctx.setdefault(f.home_team, {"row": r})
        for kind, mkt, cap in (("spread", f.market_margin, SPREAD_CAP),
                               ("total", f.market_total, TOTAL_CAP)):
            m = models[kind]
            if np.isnan(mkt) or r[m.feats].isna().any():
                continue
            frame = r[m.feats].to_frame().T.astype(float)
            pts = float(np.clip(m.predict(frame, np.array([mkt]))[0], -cap, cap))
            c[kind] = (pts, m.parts(r, mkt))
            market = "spreads" if kind == "spread" else "totals"
            src = "combined" if kind in MODEL_MARKETS else "combined_info"
            if abs(pts) >= 0.05:
                adj.append(sel.Adjustment(f.home_team, market, pts, src))
        if not (np.isnan(f.market_total) or np.isnan(f.market_margin)):
            hit = total_rule(f.market_total, abs(f.market_margin), f.commence,
                             r.get("conference_game", False),
                             r.get("poss_total", np.nan) - f.market_total)
            if hit:
                side, p, why = hit
                fav = round(abs(f.market_margin) * 2) / 2
                gap = r.get("poss_total", np.nan) - f.market_total
                c["trend"] = why.format(fav=f"{fav:g}", gap=abs(gap) if gap == gap else 0)
                adj.append(sel.Adjustment(f.home_team, "totals",
                                          rule_shift(tm, f.market_total, side, p), "trend"))
    sel.apply_adjustments(fair, adj)
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    out.update(priced=priced, ctx=ctx, ranks=ranks)
    if priced.empty:
        return priced, fair, ids
    offers = priced[priced.market.isin(["spreads", "totals"])
                    & priced.book.isin(my_books)]
    picks = sel.select(model_backed(offers), bankroll, pol)
    # Look up the news for the games picked. A backup QB's first start
    # moves that game's fair line, so price and pick again; repeat for any
    # new games that brings in.
    starters = team_starters(g, cfb_data.current_season(now)) if news is not None else {}
    done = set()
    for _ in range(4):
        # the picks plus near misses: any offer within NEWS_NEAR_MISS of the
        # edge bar, so injury news can also turn a pass into a bet
        # (before the model-backed filter: the news can add the lean)
        near = pd.concat([offers[offers.ev >= min_ev - NEWS_NEAR_MISS],
                          dog_offers(priced, my_books, min_ev - NEWS_NEAR_MISS,
                                     backed=False)])
        cand = pd.concat([picks.reindex(columns=["event_id", "commence"]),
                          near[["event_id", "commence"]]]).drop_duplicates("event_id")
        if news is None or not len(cand):
            break
        games = [{"home": upcoming.loc[ids[e], "home_team"], "away": upcoming.loc[ids[e], "away_team"],
                  "kickoff": c} for e, c in cand.itertuples(index=False) if ids.get(e) in upcoming.index]
        news.update(cfb_news.fetch(games, news, now))
        qa = qb_adjustments(fair, {e: ids[e] for e in set(cand.event_id) if e in ids},
                            upcoming, news, starters, done)
        if not qa:
            break
        sel.apply_adjustments(fair, qa)
        priced = sel.price_offers(rows, fair, mm, tm, pol)
        offers = priced[priced.market.isin(["spreads", "totals"])
                        & priced.book.isin(my_books)]
        picks = sel.select(model_backed(offers), bankroll, pol)
    out["priced"] = priced
    dogs = underdogs(priced, my_books, bankroll, pol) if BET_DOGS and not BACKTESTED_ONLY else priced.iloc[:0]
    # One bet per team per game: a moneyline on a team the spreads already
    # back is the same bet twice (UAB +7.5 and UAB +250 both lost 10-30).
    if len(dogs) and len(picks):
        backed = set(zip(picks["event_id"], picks["pick"]))
        dogs = dogs[[k not in backed for k in zip(dogs["event_id"], dogs["pick"])]]
    if len(dogs):
        picks = pd.concat([picks, dogs], ignore_index=True)
        # a moneyline has no point; None, not NaN, so picks.json stays JSON
        picks["point"] = picks.point.astype(object).where(picks.market != "h2h", None)
    if len(picks):
        picks["stake"] = (picks["stake"] * STAKE_SCALE).round(2)
        picks["win_chance"] = picks.p_win / (1 - picks.p_push)
        why = []
        for p in picks.itertuples():
            c = ctx.get(p.home, {})
            if p.market == "h2h":
                why.append(explain_dog(p, c, ranks))
                continue
            kind = "spread" if p.market == "spreads" else "total"
            srcs = str(p.sources).split(",")
            if kind == "total" and "trend" in srcs and "trend" in c:
                t = c["trend"]
                own = own_total_text(c["row"], c["row"].home_team, c["row"].away_team)
                why.append(f"{t[0].upper()}{t[1:]}." + (f" {own}" if own else ""))
            elif kind in c and "combined" in srcs:
                pts, parts = c[kind]
                why.append(explain(p, c["row"], parts, ranks, pts, p.fair_market))
            elif "injury" in srcs:
                why.append("")
            else:
                fv = (f"put the total at {p.fair_market:.1f}" if p.market == "totals"
                      else f"make {p.home if p.fair_market >= 0 else p.away} a "
                           f"{abs(p.fair_market):.1f}-point favorite")
                why.append(f"{BOOK_NAME.get(p.book, p.book)} hasn't caught up "
                           f"with the sharp books, which {fv}.")
        # the injury news that moved a line gets its own sentence
        inj = {}
        for f in fair.values():
            for a in f.adjustments:
                if a.source == "injury":
                    inj.setdefault((a.home_team, a.market), []).append(a.note)
        def news_text(p):
            mk = "spreads" if p.market == "h2h" else p.market
            notes = inj.get((p.home, mk), [])
            if not notes or "injury" not in str(p.sources).split(","):
                return ""
            return ("News: " + "; ".join(notes) + ". A backup QB's first start is "
                    "priced a point worse (history: lines overrate that team by about "
                    "a point); other starters missing their first game, half a point "
                    "each, and on totals a missing offensive starter lowers the total "
                    "half a point, a missing defender raises it (educated guesses).")
        why = [" ".join(x for x in (w, news_text(p)) if x)
               for w, p in zip(why, picks.itertuples())]
        picks["why"] = why
        picks["game_id"] = [ids.get(e) for e in picks.event_id]
        picks["book"] = picks.book.map(lambda b: BOOK_NAME.get(b, b))
    return picks, fair, ids


def grade(bets: list[dict], g: pd.DataFrame) -> list[dict]:
    done = g.dropna(subset=["margin"]).set_index("game_id")
    for b in bets:
        if b.get("result") or b.get("game_id") not in done.index:
            continue
        r = done.loc[b["game_id"]]
        b["result"] = ledger.grade_one(b, r.home_points, r.away_points)
        b["final"] = f"{int(r.away_points)}-{int(r.home_points)}"
        b["profit"] = round({"win": ledger.payout(b["stake"], b["price"]),
                             "loss": -b["stake"], "push": 0.0}[b["result"]], 2)
    return bets


def fetch_scores(key: str) -> list[dict]:
    """Finished games from the last 3 days (the Odds API scores feed, 2
    credits). The free schedule data can take days to post weeknight scores."""
    r = requests.get(SCORES_URL, params={"apiKey": key, "daysFrom": 3}, timeout=30)
    r.raise_for_status()
    out = []
    for ev in r.json():
        pts = {s["name"]: s["score"] for s in ev.get("scores") or []}
        if ev.get("completed") and ev["home_team"] in pts and ev["away_team"] in pts:
            out.append({"home": ev["home_team"], "away": ev["away_team"],
                        "commence": ev["commence_time"],
                        "home_score": float(pts[ev["home_team"]]),
                        "away_score": float(pts[ev["away_team"]])})
    return out


def needs_scores(bets: list[dict], now) -> bool:
    """Any bet still ungraded 4+ hours after kickoff."""
    late = pd.Timestamp(now) - pd.Timedelta(hours=4)
    return any(not b.get("result") and not b.get("pulled")
               and pd.Timestamp(b["commence"]) < late for b in bets)


def grade_scores(bets: list[dict], scores: list[dict]) -> list[dict]:
    """Grade what the schedule data hasn't from the scores feed. Bets keep
    the Odds API's team names, so they match the feed directly."""
    for b in bets:
        if b.get("result") or b.get("pulled"):
            continue
        for sc in scores:
            if (sc["home"] == b["home"] and sc["away"] == b["away"]
                    and abs(pd.Timestamp(sc["commence"]) - pd.Timestamp(b["commence"]))
                    <= pd.Timedelta(hours=24)):
                b["result"] = ledger.grade_one(b, sc["home_score"], sc["away_score"])
                b["final"] = f"{int(sc['away_score'])}-{int(sc['home_score'])}"
                b["profit"] = round({"win": ledger.payout(b["stake"], b["price"]),
                                     "loss": -b["stake"], "push": 0.0}[b["result"]], 2)
                break
    return bets


def school_names(fair: dict, ids: dict, g: pd.DataFrame, history) -> dict:
    """Odds API name -> school name, so the page says "Ohio State", not
    "Buckeyes"."""
    s = g.set_index("game_id")
    out = {}
    for ev, gid in ids.items():
        out[fair[ev].home_team] = s.loc[gid, "home_team"]
        out[fair[ev].away_team] = s.loc[gid, "away_team"]
    for b in history:
        if b.get("game_id") in s.index:
            out.setdefault(b["home"], s.loc[b["game_id"], "home_team"])
            out.setdefault(b["away"], s.loc[b["game_id"], "away_team"])
    return out


COLS = ["away", "home", "commence", "market", "pick", "point", "price",
        "book", "stake", "win_chance", "ev", "why", "game_id", "injuries"]


def attach_news(picks: pd.DataFrame, g: pd.DataFrame, news: dict,
                starters: dict | None = None) -> pd.DataFrame:
    """Each pick's injury news (cfb_news) as an "injuries" list, both
    teams, by school name, with the points a backup QB's first start
    took off."""
    starters = starters or {}
    s = g.set_index("game_id")
    out = []
    for p in picks.itertuples():
        if p.game_id not in s.index:
            out.append([])
            continue
        r = s.loc[p.game_id]
        e = news.get(cfb_news.game_key(r.home_team, r.away_team, p.commence), {})
        out.append(cfb_news.injuries(e, r.away_team, starters.get(r.away_team))
                   + cfb_news.injuries(e, r.home_team, starters.get(r.home_team)))
    return picks.assign(injuries=out)


def render_html(picks, fair, made, bankroll, history, names) -> str:
    data = {
        "title": "Fieldline College", "names": names,
        "nav": {"href": "https://bviolino93.github.io/NFL-model/", "text": "NFL picks →"},
        "updated": made, "bankroll": bankroll,
        "picks": picks[COLS].to_dict("records") if len(picks) else [],
        "fair": [{"away": f.away_team, "home": f.home_team,
                  "market_margin": f.market_margin, "margin": f.margin,
                  "market_total": f.market_total, "total": f.total}
                 for f in fair.values()
                 if not (np.isnan(f.market_margin) or np.isnan(f.market_total))],
        "history": {"summary": ledger.summary(history), "bets": history},
        "paper": None,
    }
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dashboard.html")) as fh:
        return fh.read().replace("/*DATA*/", blob)


def render_alert(bets: list[dict], names: dict) -> str:
    n = lambda t: names.get(t, t)
    out = [f"**{len(bets)} new college bet{'s' if len(bets) != 1 else ''}**\n"]
    for b in bets:
        kick = pd.to_datetime(b["commence"], utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        pick = n(b["pick"]) if b["market"] != "totals" else b["pick"]
        pt = ("to win" if b["market"] == "h2h" else f"{b['point']:+g}"
              if b["market"] == "spreads" else f"{b['point']:g}")
        out.append(f"- **{pick} {pt} ({int(b['price']):+d})** at {b['book']}, "
                   f"${b['stake']:,.2f} ({b['win_chance'] * 100:.0f}% to win) · "
                   f"{n(b['away'])} @ {n(b['home'])}, {kick}")
    out.append(f"\nDashboard: {DASHBOARD_URL}")
    return "\n".join(out) + "\n"


def render_dropped(bets: list[dict], names: dict) -> str:
    """Alerted bets a later run no longer makes: skip them if not bet yet.
    They stay in the record at the alerted price."""
    n = lambda t: names.get(t, t)
    out = [f"**Skip {'these' if len(bets) != 1 else 'this'} if you haven't bet "
           f"{'them' if len(bets) != 1 else 'it'}: {len(bets)} college "
           f"bet{'s' if len(bets) != 1 else ''} dropped off the board**\n"]
    for b in bets:
        kick = pd.to_datetime(b["commence"], utc=True).tz_convert(
            "America/New_York").strftime("%a %-I:%M %p ET")
        pick = n(b["pick"]) if b["market"] != "totals" else b["pick"]
        pt = ("to win" if b["market"] == "h2h" else f"{b['point']:+g}"
              if b["market"] == "spreads" else f"{b['point']:g}")
        out.append(f"- ~~{pick} {pt} ({int(b['price']):+d}) at {b['book']}~~ · "
                   f"{n(b['away'])} @ {n(b['home'])}, {kick}")
    out.append("\nThe line moved or the model no longer backs it. If you already "
               "bet it, it still counts in the record at the price you got alerted.")
    out.append(f"\nDashboard: {DASHBOARD_URL}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--json", default="site/picks.json")
    ap.add_argument("--history", default="site/history.json")
    ap.add_argument("--alert", default="alert.md",
                    help="appended to when this run finds new bets")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--books", default=os.environ.get("MY_BOOKS") or MY_BOOKS)
    ap.add_argument("--news", default="site/news.json",
                    help="injury news cache (cfb_news), kept on the picks branch")
    ap.add_argument("--closes", default="site/closes.json",
                    help="last consensus line seen before each game (for CLV)")
    ap.add_argument("--lines-csv", default="")
    ap.add_argument("--board", default="site/board.json",
                    help="every game in the next 24h: the model's side and lean")
    a = ap.parse_args()
    key = None
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip('"\'')
        if "=" in key:
            key = key.split("=", 1)[1].strip().strip('"\'')
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = fetch_rows(key)
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    now = datetime.now(timezone.utc)
    g = cfb_model.load_features()
    try:
        with open(a.news) as fh:
            news = json.load(fh)
    except (OSError, ValueError):
        news = {}
    info = {}
    picks, fair, ids = build_picks(rows, books, a.bankroll, now=now, g=g, news=news, out=info)
    write_board(info.get("priced"), books, now, a.board)
    try:
        with open(a.closes) as fh:
            closes = json.load(fh)
    except (OSError, ValueError):
        closes = {}
    os.makedirs(os.path.dirname(a.closes) or ".", exist_ok=True)
    with open(a.closes, "w") as fh:
        json.dump(ledger.record_lines(closes, fair, now), fh, indent=1)
    os.makedirs(os.path.dirname(a.news) or ".", exist_ok=True)
    with open(a.news, "w") as fh:
        json.dump(news, fh, indent=1)
    starters = team_starters(g, cfb_data.current_season(now))
    picks = attach_news(picks, g, news, starters) if len(picks) else picks.assign(injuries=[])
    made = now.strftime("%a %b %-d, %H:%M UTC")
    for path in (a.out, a.json, a.history):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = picks[COLS].to_dict("records") if len(picks) else []
    out = json.loads(json.dumps(out, default=str))
    history = ledger.load(a.history)
    seen = {ledger._key(b) for b in history}
    was = ledger.on_board(history)
    history = ledger.merge(history, out, now, now.isoformat())
    gone = ledger.dropped(was, history, now)
    history = grade(history, g)
    if key and needs_scores(history, now):
        try:
            history = grade_scores(history, fetch_scores(key))
        except requests.RequestException as e:      # grading can wait for the next run
            print(f"scores: lookup failed ({e})")
    ledger.save(a.history, history)
    names = school_names(fair, ids, g, history)
    with open(a.out, "w") as fh:
        fh.write(render_html(picks, fair, made, a.bankroll, history, names))
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    fresh = [p for p in out if ledger._key(p) not in seen]
    for bets, render in ((fresh, render_alert), (gone, render_dropped)):
        if bets:
            gap = "\n" if os.path.exists(a.alert) else ""   # after the NFL alert
            with open(a.alert, "a") as fh:
                fh.write(gap + render(bets, names))
    print(f"{len(out)} college bets, {len(fresh)} new, {len(gone)} dropped")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")


if __name__ == "__main__":
    main()
