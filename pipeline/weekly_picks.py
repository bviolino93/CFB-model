"""Weekly pick generator: ranked card for one slate.

    python -m pipeline.weekly_picks                         # today's slate
    python -m pipeline.weekly_picks --date 2026-11-14 --lines my_lines.csv
    python -m pipeline.weekly_picks --injuries sheet_rows.csv --bankroll 1000

Steps
  1. Fair lines: the Ridge + XGBoost score models (opponent-adjusted
     play-by-play ratings, trained on every past season with lines) give a
     fair spread and fair total for every FBS-vs-FBS game on the date.
     No Platt filter: every game gets a number.
  2. Injuries: active pre-kickoff rows from the app's injuries_auto.json
     (and optionally the app's injury sheet, exported as CSV) move the fair
     line. Each row says "team is N points worse" (app scale: starting QB
     out 5, doubtful 2.5, star non-QB 1.5, OL/DL cluster 2, one starter
     0.75). Totals drop by 0.6 x the points, like the app.
  3. Market: current lines come from --lines (CSV), or The Odds API when
     ODDS_API_KEY is set. Prices: 734's usual -105 unless the CSV gives one.
  4. Remove the slate-wide lean (median model-minus-market gap, as the app
     does), then rank every game by point edge. Tier 1: >= 3.0 pts on spreads or
     >= 4.0 on totals (top 5). Tier 2: >= 2.0 pts (top 5). Each play gets a
     one-sentence reason, a size (1 unit flat, with quarter Kelly for
     reference) and the worst number at 734 that still clears Tier 1.
     Gaps over 7 points are listed under "check the news first", not bet.

Lines CSV columns: away_team, home_team, home_spread (home side, -7 = home
favoured), total; optional spread_odds, total_odds (American, default -105).

Honest caveat (from pipeline/backtest.py): in 2018-2025, games where this
model disagreed with the closing line by 3+ points covered about 49-53%.
The card ranks disagreements; it does not prove an edge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from .betting import kelly_fraction
from .features import add_injury_feed, stage2
from .ingest import cfb_data, stage1
from .model import Spec, fit

INJ_AUTO_URL = ("https://raw.githubusercontent.com/bviolino93/CFB-model/"
                "injury-feed/injuries_auto.json")
INJ_TOTAL_SHARE = 0.6           # same as app.py
TIER1 = {"spread": 3.0, "total": 4.0}
TIER2 = 2.0
TOP_N = 5
# Gaps this big usually mean the model is missing news (QB, injuries,
# coaching). In 2018-2025 the model's 8+ point spread gaps covered 47%,
# and the app already passes 7+ point gaps. They're listed, not bet.
MAX_EDGE = 7.0
# Remove the slate-wide lean before ranking (the app does the same on
# slates of 10+ games): e.g. if the model is 3 points over the market on
# the median total, that's a level bias, not 50 Over bets.
CENTER_MIN_GAMES = 10
BOOK_ODDS = C.YOUR_BOOK_ODDS    # -105 at 734

ALIASES = {
    "umass": "massachusetts", "pitt": "pittsburgh", "miami fl": "miami",
    "miami (fl)": "miami", "miami ohio": "miami (oh)", "miami (ohio)": "miami (oh)",
    "ole miss": "ole miss", "mississippi": "ole miss", "ulm": "ul monroe",
    "louisiana-monroe": "ul monroe", "louisiana monroe": "ul monroe",
    "ul lafayette": "louisiana", "louisiana-lafayette": "louisiana",
    "appalachian state": "app state", "app st": "app state",
    "southern mississippi": "southern miss", "fiu": "florida international",
    "fau": "florida atlantic", "usf": "south florida", "central florida": "ucf",
    "connecticut": "uconn", "brigham young": "byu", "southern methodist": "smu",
    "texas christian": "tcu", "louisiana state": "lsu", "north carolina state": "nc state",
    "unc": "north carolina", "ut san antonio": "utsa", "texas-san antonio": "utsa",
    "texas el paso": "utep", "jmu": "james madison", "sam houston state": "sam houston",
    "hawaii": "hawaii", "san jose state": "san jose state", "texas am": "texas a&m",
    "ecu": "east carolina", "wku": "western kentucky", "niu": "northern illinois",
    "wmu": "western michigan", "cmu": "central michigan", "emu": "eastern michigan",
    "bgsu": "bowling green", "mtsu": "middle tennessee", "middle tennessee state": "middle tennessee",
    "kennesaw st": "kennesaw state", "jax state": "jacksonville state",
    "nmsu": "new mexico state", "sdsu": "san diego state", "csu": "colorado state",
}


def team_key(t) -> str:
    s = unicodedata.normalize("NFKD", str(t or "").strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[‘’'`.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r" st$", " state", s)
    return ALIASES.get(s, s)


def match_team(name: str, keys: dict) -> str | None:
    """Map a book's team name ("Ohio State Buckeyes", "Miami (FL)") to a
    schedule name. Exact key first, then the longest schedule name the
    book's name starts with (so "Miami (OH) RedHawks" -> "Miami (OH)")."""
    k = team_key(name)
    if k in keys:
        return keys[k]
    best = None
    for sk, sname in keys.items():
        if k.startswith(sk + " ") and (best is None or len(sk) > len(best[0])):
            best = (sk, sname)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_injury_rows(extra_csv: str | None = None) -> list[dict]:
    rows = []
    try:
        with urllib.request.urlopen(INJ_AUTO_URL, timeout=15) as r:
            rows += (json.loads(r.read()) or {}).get("rows") or []
    except Exception as e:
        print(f"  injury feed unavailable ({e})")
    if extra_csv and os.path.exists(extra_csv):
        rows += pd.read_csv(extra_csv).to_dict("records")
    # newest entry wins when the same team and note appear twice
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["k"] = df.team.map(team_key) + "|" + df.note.astype(str).str[:40].str.lower()
    df = df.sort_values("added_at").drop_duplicates("k", keep="last")
    return df.drop(columns="k").to_dict("records")


def load_lines(path: str | None, slate: pd.DataFrame) -> pd.DataFrame:
    """Current lines keyed by game_id: home_spread, total, spread_odds, total_odds."""
    keys = {team_key(t): t for t in pd.concat([slate.home_team, slate.away_team])}
    if path:
        raw = pd.read_csv(path)
    elif os.environ.get("ODDS_API_KEY"):
        raw = _odds_api_lines(os.environ["ODDS_API_KEY"])
    else:
        raise SystemExit("No lines: pass --lines file.csv or set ODDS_API_KEY")
    out = []
    by_pair = {(r.home_team, r.away_team): r.game_id for r in slate.itertuples()}
    for r in raw.to_dict("records"):
        h, a = match_team(r.get("home_team"), keys), match_team(r.get("away_team"), keys)
        gid = by_pair.get((h, a))
        flip = False
        if gid is None and (a, h) in by_pair:       # book listed home/away the other way
            gid, flip = by_pair[(a, h)], True
        if gid is None:
            continue
        sp = pd.to_numeric(r.get("home_spread"), errors="coerce")
        out.append({"game_id": gid,
                    "mkt_spread": -sp if flip else sp,
                    "mkt_total": pd.to_numeric(r.get("total"), errors="coerce"),
                    "spread_odds": pd.to_numeric(r.get("spread_odds"), errors="coerce"),
                    "total_odds": pd.to_numeric(r.get("total_odds"), errors="coerce"),
                    "book": r.get("source_book") or r.get("book") or ""})
    df = pd.DataFrame(out).drop_duplicates("game_id")
    df["spread_odds"] = df.spread_odds.fillna(BOOK_ODDS)
    df["total_odds"] = df.total_odds.fillna(BOOK_ODDS)
    return df


def _odds_api_lines(key: str) -> pd.DataFrame:
    url = ("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
           f"?apiKey={key}&regions=us&markets=spreads,totals&oddsFormat=american")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    rows = []
    for g in data:
        sp, tot = [], []
        for b in g.get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    if m["key"] == "spreads" and o["name"] == g["home_team"]:
                        sp.append(o.get("point"))
                    if m["key"] == "totals" and o["name"] == "Over":
                        tot.append(o.get("point"))
        rows.append({"home_team": g["home_team"], "away_team": g["away_team"],
                     "home_spread": np.median(sp) if sp else np.nan,
                     "total": np.median(tot) if tot else np.nan,
                     "source_book": "Odds API median"})
    return pd.DataFrame(rows)


def center_slate(slate: pd.DataFrame):
    """Subtract the median model-minus-market gap on spreads and totals
    (slates of CENTER_MIN_GAMES+ games). Injury moves stay in, since they're
    applied game by game on top of the centred number."""
    notes = []
    has = slate.mkt_spread.notna()
    if has.sum() >= CENTER_MIN_GAMES:
        base = slate.fair_spread + slate.inj_h_pts - slate.inj_a_pts      # pre-injury
        shift = float(np.median(base[has] - slate.mkt_spread[has]))
        slate["fair_spread"] = slate.fair_spread - shift
        notes.append(f"spreads {shift:+.1f} home")
    has = slate.mkt_total.notna()
    if has.sum() >= CENTER_MIN_GAMES:
        shift = float(np.median(slate.fair_total0[has] - slate.mkt_total[has]))
        slate["fair_total"] = slate.fair_total - shift
        notes.append(f"totals {shift:+.1f}")
    return slate, ", ".join(notes)


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------
def _ranks(tg: pd.DataFrame, season: int, week: int, season_type: str) -> pd.DataFrame:
    """Adjusted EPA ranks (1 = best) among FBS teams with ratings this week."""
    r = tg[(tg.season == season) & (tg.week == week) & (tg.season_type == season_type)
           & (tg.tier == "fbs")].drop_duplicates("team").set_index("team")
    return pd.DataFrame({"off": r.rO_epa.rank(ascending=False),
                         "def": r.rA_epa.rank(ascending=True), "n": len(r)})


def _ord(x) -> str:
    return f"#{int(x)}" if pd.notna(x) else "#?"


def build_card(slate: pd.DataFrame, ranks: pd.DataFrame, sig_spread: float,
               sig_total: float, bankroll: float) -> pd.DataFrame:
    plays = []
    for r in slate.itertuples():
        rh, ra = ranks.reindex([r.home_team]).iloc[0], ranks.reindex([r.away_team]).iloc[0]
        inj = {r.home_team: (r.inj_h_pts, r.inj_h_note), r.away_team: (r.inj_a_pts, r.inj_a_note)}
        inj_txt = "; ".join(f"{t} {n} ({-p:+g} pts)" for t, (p, n) in inj.items() if p)
        if pd.notna(r.mkt_spread):
            edge = r.mkt_spread - r.fair_spread          # > 0: home is getting too much
            home = edge > 0
            team, opp = (r.home_team, r.away_team) if home else (r.away_team, r.home_team)
            line = r.mkt_spread if home else -r.mkt_spread
            fair = r.fair_spread if home else -r.fair_spread
            rt, ro = (rh, ra) if home else (ra, rh)
            why = (f"Model has {team} {fair:+.1f} vs {line:+.1f} on the board: {team} ranks "
                   f"{_ord(rt.off)} in adjusted offense EPA and {_ord(rt['def'])} on defense, "
                   f"{opp} {_ord(ro.off)} and {_ord(ro['def'])}")
            plays.append(dict(game_id=r.game_id, kickoff=r.kickoff, market="spread",
                              pick=f"{team} {line:+.1f}", edge=abs(edge),
                              ok_to=f"{team} {fair + TIER1['spread']:+.1f}",
                              odds=r.spread_odds, sigma=sig_spread, why=why, inj=inj_txt,
                              fair=f"{team} {fair:+.1f}", game=f"{r.away_team} at {r.home_team}"))
        if pd.notna(r.mkt_total):
            edge = r.fair_total - r.mkt_total
            side = "Over" if edge > 0 else "Under"
            ok = r.fair_total - TIER1["total"] if edge > 0 else r.fair_total + TIER1["total"]
            why = (f"Model total {r.fair_total:.1f} vs {r.mkt_total:g}: offenses rank "
                   f"{_ord(rh.off)} ({r.home_team}) and {_ord(ra.off)} ({r.away_team}) in "
                   f"adjusted EPA, defenses {_ord(rh['def'])} and {_ord(ra['def'])}")
            plays.append(dict(game_id=r.game_id, kickoff=r.kickoff, market="total",
                              pick=f"{side} {r.mkt_total:g}", edge=abs(edge),
                              ok_to=f"{side} {ok:.1f}", odds=r.total_odds, sigma=sig_total,
                              why=why, inj=inj_txt, fair=f"{r.fair_total:.1f}",
                              game=f"{r.away_team} at {r.home_team}"))
    card = pd.DataFrame(plays)
    if card.empty:
        return card
    card["tier"] = np.where(card.edge > MAX_EDGE, 3,
                            np.where(card.edge >= card.market.map(TIER1), 1,
                                     np.where(card.edge >= TIER2, 2, 0)))
    card["p_cover"] = norm.cdf(card.edge / card.sigma)
    card["qk_pct"] = kelly_fraction(card.p_cover, card.odds)
    card["qk_$"] = card.qk_pct * bankroll
    card["units"] = np.where(card.tier == 1, 1.0, 0.0)
    card.loc[card.inj != "", "why"] = card.why + "; injuries: " + card.inj
    return card.sort_values(["tier", "edge"], ascending=[True, False]).reset_index(drop=True)


def render(card: pd.DataFrame, date: str, n_games: int, n_lines: int, n_inj: int) -> str:
    out = [f"# Weekly card: {date}", "",
           f"{n_games} FBS games, {n_lines} with lines, {n_inj} injury entries applied. "
           f"Prices at {BOOK_ODDS} (734).", ""]
    for tier, title, note in (
            (1, "Tier 1: Best bets", "spreads >= 3.0 pts, totals >= 4.0 pts"),
            (2, "Tier 2: Leans / watchlist",
             "2.0+ pts; bet only with a better number or late injury confirmation"),
            (3, "Check the news first", f"gaps over {MAX_EDGE:g} pts; the model is "
             "probably missing something, so these are not bets")):
        t = card[card.tier == tier].head(TOP_N)
        out += [f"## {title} ({note})", ""]
        if t.empty:
            out += ["None this week.", ""]
            continue
        for i, r in enumerate(t.to_dict("records"), 1):
            if tier == 1:
                tail = f"1u (quarter Kelly ${r['qk_$']:,.0f}) · OK at 734 to {r['ok_to']}"
            elif tier == 2:
                tail = f"watch · bet at {r['ok_to']} or better"
            else:
                tail = "no bet"
            out.append(f"{i}. **{r['pick']}** · {r['edge']:.1f} pts · {tail}  \n"
                       f"   _{r['game']}._ {r['why']}.")
        out.append("")
    out += ["---",
            "Fair lines are the play-by-play model's own numbers. In 2018-2025 its "
            "disagreements with the closing line covered about 49-53%, so treat edges as "
            "a ranking, not proof. Quarter Kelly uses a normal error of ~16.6 pts and "
            "overstates confidence; flat 1u is the default."]
    return "\n".join(out)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"),
                    help="slate date (Eastern), default today")
    ap.add_argument("--lines", help="CSV of current lines (see module doc)")
    ap.add_argument("--injuries", help="optional CSV export of the app's injury sheet")
    ap.add_argument("--bankroll", type=float, default=C.START_BANKROLL)
    args = ap.parse_args(argv)

    day = pd.Timestamp(args.date)
    season = cfb_data.current_season(day.tz_localize("America/New_York"))
    seasons = sorted(set(C.ALL_SEASONS) | {season - 1, season})
    games, pbp, teams, venues, extras = stage1(seasons, verbose=False)
    print(f"Building ratings through {args.date} ...", flush=True)
    df, tg, _ = stage2(games, pbp, teams, venues, verbose=False, extras=extras)

    train = [s for s in seasons if 2018 <= s < season]
    sp = fit(df, train, Spec(market="spread"), verbose=False)
    tt = fit(df, train, Spec(market="total"), verbose=False)

    et = df.kickoff.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    slate = df[(et == args.date) & ~df.non_fbs_game].copy()
    if slate.empty:
        raise SystemExit(f"No FBS-vs-FBS games on {args.date}")
    slate["fair_margin"] = 0.5 * (sp.ridge.predict(slate[sp.features])
                                  + sp.xgb.predict(slate[sp.features]))
    slate["fair_total0"] = 0.5 * (tt.ridge.predict(slate[tt.features])
                                  + tt.xgb.predict(slate[tt.features]))

    rows = load_injury_rows(args.injuries)
    slate = add_injury_feed(slate, rows)
    notes = pd.DataFrame(rows)
    for side, col in (("home", "inj_h_note"), ("away", "inj_a_note")):
        if notes.empty:
            slate[col] = ""
            continue
        keyed = notes.assign(k=notes.team.map(team_key)).groupby("k").note.agg(
            lambda x: "; ".join(map(str, x)))
        slate[col] = slate[f"{side}_team"].map(team_key).map(keyed).fillna("")
    slate.loc[slate.inj_h_pts == 0, "inj_h_note"] = ""
    slate.loc[slate.inj_a_pts == 0, "inj_a_note"] = ""
    # A team N points worse: home margin falls by N (home) or rises by N (away).
    slate["fair_margin_adj"] = slate.fair_margin - slate.inj_h_pts + slate.inj_a_pts
    slate["fair_spread"] = -slate.fair_margin_adj
    slate["fair_total"] = slate.fair_total0 - INJ_TOTAL_SHARE * (slate.inj_h_pts + slate.inj_a_pts)

    lines = load_lines(args.lines, slate)
    slate = slate.merge(lines, on="game_id", how="left")
    slate, shifts = center_slate(slate)
    wk = slate.iloc[0]
    ranks = _ranks(tg, season, int(wk.week), wk.season_type)
    card = build_card(slate, ranks, sp.sigma, tt.sigma, args.bankroll)
    n_inj = int(((slate.inj_h_pts != 0) | (slate.inj_a_pts != 0)).sum())
    text = render(card, args.date, len(slate), int(slate.mkt_spread.notna().sum()), n_inj)
    if shifts:
        text = text.replace("Prices at", f"Slate lean removed ({shifts}). Prices at", 1)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUTPUT_DIR, f"weekly_card_{args.date}.md"), "w") as f:
        f.write(text)
    keep = ["game_id", "kickoff", "away_team", "home_team", "fair_margin", "inj_h_pts",
            "inj_a_pts", "fair_spread", "mkt_spread", "fair_total0", "fair_total",
            "mkt_total", "book"]
    slate[keep].to_csv(os.path.join(C.OUTPUT_DIR, f"weekly_fair_lines_{args.date}.csv"),
                       index=False)
    if not card.empty:
        card.to_csv(os.path.join(C.OUTPUT_DIR, f"weekly_plays_{args.date}.csv"), index=False)
    print(text)
    return card, slate


if __name__ == "__main__":
    main(sys.argv[1:])
