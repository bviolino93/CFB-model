"""
One college game through the model, on demand: the sharp line and the
model's line, the best DraftKings/FanDuel price for each side, and whether
any of it is a bet.

    ODDS_API_KEY=... python cfb_game.py Ohio State
    python cfb_game.py "Miami Florida State" --lines-csv lines.csv

Same pipeline as cfb_picks.py (build_picks, injury news included), so the
numbers match what a picks run would say about this game; it just reports
every side, not only the ones that clear the edge bar. Runs from the
"Check one game" workflow, which also shows the result on the Fieldline
page (share/index.html, from checks.json on the checks branch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import cfb_data
import cfb_model
import cfb_picks as cp
import selector as sel

MARKETS = ("spreads", "totals")
CHECKS_KEPT = 12


def _words(q: str) -> list[str]:
    return [w for w in cp._norm(q).replace("@", " ").replace("/", " ")
            .replace(",", " ").split() if w not in {"at", "vs", "v"}]


def find_event(rows: pd.DataFrame, query: str, names: dict | None = None):
    """event_id of the soonest game whose two team names (Odds API name,
    plus the school name when known) hold every word of the query in
    order of team ("Ohio State", "miami florida state"), or None. Exact
    school names win, so "Miami" is the Hurricanes, not Miami (OH)."""
    words = _words(query)
    if not words or rows.empty:
        return None
    names = names or {}
    games = rows.drop_duplicates("event_id").sort_values("commence_time")
    text = lambda t: cp._norm(f"{t} {names.get(t, '')}").split()
    q = " ".join(words)
    best, score = None, 0
    for g in games.itertuples():
        h, a = text(g.home_team), text(g.away_team)
        if not all(w in h or w in a for w in words):
            continue
        sc = 1 + sum(q == cp._norm(names.get(t, "")) or q == cp._norm(t)
                     for t in (g.home_team, g.away_team))
        if sc > score:
            best, score = g.event_id, sc
    return best


def _fav(home: str, away: str, margin: float) -> str:
    team = home if margin >= 0 else away
    return f"{team} by {abs(margin):.1f}"


def report(rows: pd.DataFrame, query: str, books, bankroll: float = 1000.0,
           min_ev: float = cp.MIN_EV, now=None, g: pd.DataFrame | None = None,
           news: dict | None = None) -> dict:
    """The check as a dict (what the page shows); to_md() makes the issue
    comment and run summary from it."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    base = {"league": "cfb", "query": query, "checked": now.isoformat()}
    kick = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce") \
        if len(rows) else pd.Series(dtype="datetime64[ns, UTC]")
    week = rows[(kick > now) & (kick < cp.week_end(now))] if len(rows) else rows
    g = cfb_model.load_features() if g is None else g
    sched = g[g.season == cfb_data.current_season(now)]
    schools = _school_names(rows, sched)
    ev = find_event(week, query, schools)
    if ev is None:
        later = find_event(rows, query, schools)
        why = ("That game has kicked off or isn't this week; the model only "
               "prices games before kickoff in the current week."
               if later is not None else
               f"No college game this week matches \"{query}\". Try the "
               "school's name, like Ohio State or Miami.")
        return {**base, "found": False, "title": "No game found", "message": why}
    out = {}
    picks, fair, ids = cp.build_picks(rows[rows.event_id == ev], books, bankroll,
                                      min_ev, now=now, g=g, news=news, out=out)
    f = fair.get(ev)
    if f is None:
        return {**base, "found": False, "title": "No game found",
                "message": "Not enough books have this game up to price it."}
    names = cp.school_names(fair, ids, g, [])
    n = lambda t: names.get(t, t)
    home, away = n(f.home_team), n(f.away_team)
    d = {**base, "found": True, "title": f"{away} @ {home}", "away": away,
         "home": home, "kickoff": str(f.commence), "lines": [], "offers": [],
         "verdicts": []}
    if ev not in ids:
        d["note"] = ("This game isn't in the schedule data, so only the sharp "
                     "books' line is used; the model adds nothing to it.")
    if not np.isnan(f.market_margin):
        d["lines"].append({"market": "Spread",
                           "sharp": _fav(home, away, f.market_margin),
                           "model": _fav(home, away, f.margin)})
    if not np.isnan(f.market_total):
        d["lines"].append({"market": "Total", "sharp": f"{f.market_total:.1f}",
                           "model": f"{f.total:.1f}"})
    priced = out.get("priced", pd.DataFrame())
    if priced.empty:
        d["message"] = (f"Not enough books have lines up to price it "
                        f"(need {sel.Policy().min_books}).")
        return d
    label = lambda r: (f"{n(r.pick)} {r.point:+g}" if r.market == "spreads"
                       else f"{r.pick} {r.point:g}")
    mine = priced[priced.market.isin(MARKETS) & priced.book.isin(books)]
    best = mine.sort_values("ev", ascending=False).drop_duplicates(["market", "pick"])
    bet_keys = {(b.market, b.pick) for b in picks.itertuples()} if len(picks) else set()
    for m in MARKETS:
        for r in best[best.market == m].itertuples():
            d["offers"].append({
                "market": m, "side": label(r), "price": int(r.price),
                "book": cp.BOOK_NAME.get(r.book, r.book),
                "win": round(r.p_win / (1 - r.p_push) if r.p_push < 1 else 0, 4),
                "ev": round(float(r.ev), 4), "bet": (r.market, r.pick) in bet_keys})
    ctx = out.get("ctx", {}).get(f.home_team, {})
    for m in MARKETS:
        side = best[best.market == m]
        name = "Spread" if m == "spreads" else "Total"
        if side.empty:
            continue
        top = side.iloc[0]
        b = picks[picks.market == m] if len(picks) else picks
        if len(b):
            p = b.iloc[0]
            d["verdicts"].append({
                "market": name, "bet": True,
                "head": f"Bet {label(p)} ({int(p.price):+d}) at {p.book}, ${p.stake:,.2f}",
                "text": str(p.why or "")})
            continue
        head = ("Pass" if top.ev < min_ev / 2
                else f"Lean {label(top)}, but under the bar")
        text = (f"The best side is {label(top)} at {top.ev * 100:+.1f}% "
                f"expected value; a bet needs {min_ev * 100:.0f}%.")
        if m == "spreads" and "spread" in ctx:
            lean = ctx["spread"][0]
            if abs(lean) >= 0.05:
                toward = home if lean > 0 else away
                text += (f" The model leans {toward} by {abs(lean):.1f} against "
                         f"the sharp line")
                text += ("; a model-only spread needs a 1-point lean." if
                         abs(lean) < cp.MIN_SPREAD_LEAN else ".")
        if m == "totals":
            if "trend" in ctx:
                t = ctx["trend"]
                text += f" Totals pattern: {t}."
            else:
                text += " None of the totals patterns fits this game."
            if "row" in ctx:
                own = cp.own_total_text(ctx["row"], home, away)
                text += f" {own}" if own else ""
        d["verdicts"].append({"market": name, "bet": False, "head": head,
                              "text": text})
    return d


def _school_names(rows: pd.DataFrame, sched: pd.DataFrame) -> dict:
    """Odds API name -> schedule school name, where one matches."""
    if rows.empty:
        return {}
    schools = set(sched.home_team) | set(sched.away_team)
    out = {}
    for t in set(rows.home_team) | set(rows.away_team):
        hits = [(cp._matches(t, s), s) for s in schools]
        hit = max(hits, default=(0, None))
        if hit[0]:
            out[t] = hit[1]
    return out


def to_md(d: dict) -> str:
    out = [f"# {d['title']}\n"]
    made = pd.Timestamp(d["checked"]).strftime("%a %b %-d, %H:%M UTC")
    if d.get("kickoff"):
        kick = pd.to_datetime(d["kickoff"], utc=True).tz_convert(
            "America/New_York").strftime("%a %b %-d, %-I:%M %p ET")
        out.append(f"_College · {kick} · checked {made}_\n")
    if not d.get("found"):
        return "\n".join(out + [d.get("message", ""), ""])
    if d.get("lines"):
        out.append("## What the model makes it\n")
        out.append("| | Sharp books | Model |\n|---|---|---|")
        out += [f"| {x['market']} | {x['sharp']} | {x['model']} |" for x in d["lines"]]
        out.append("")
    if d.get("note"):
        out.append(d["note"] + "\n")
    if d.get("message"):
        out.append(d["message"] + "\n")
        return "\n".join(out)
    out.append("## Best price at your books\n")
    if not d["offers"]:
        out.append("None of your books has this game up yet.\n")
    else:
        out.append("| Side | Price | Book | Win chance | EV |\n|---|---|---|---|---|")
        for o in sorted(d["offers"], key=lambda o: (o["market"], -o["ev"])):
            out.append(f"| {o['side']}{' ✅' if o['bet'] else ''} | {o['price']:+d} | "
                       f"{o['book']} | {o['win'] * 100:.1f}% | {o['ev'] * 100:+.1f}% |")
        out.append("")
    out.append("## Verdict\n")
    for v in d["verdicts"]:
        out.append(f"**{v['market']}: {v['head']}.** {v['text']}".strip() + "\n")
    out.append("---\nWin chances are model estimates, not guarantees. The model "
               "hasn't yet shown an edge over closing lines. 21+ where legal; "
               "1-800-GAMBLER.\n")
    return "\n".join(out)


def save_check(path: str, d: dict, keep: int = CHECKS_KEPT) -> None:
    """Newest first, one entry per game (a re-check replaces the old one)."""
    try:
        with open(path) as fh:
            checks = json.load(fh).get("checks", [])
    except (OSError, ValueError):
        checks = []
    same = lambda c: c.get("title") == d["title"] and c.get("found") and d.get("found")
    checks = [d] + [c for c in checks if not same(c)]
    with open(path, "w") as fh:
        json.dump({"updated": d["checked"], "checks": checks[:keep]}, fh,
                  indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="+", help="school name(s), e.g. Ohio State")
    ap.add_argument("--md", default="game.md")
    ap.add_argument("--checks", default="",
                    help="add the result to this checks.json (the page reads it)")
    ap.add_argument("--run-url", default="", help="link back to the run")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--books", default=os.environ.get("MY_BOOKS") or cp.MY_BOOKS)
    ap.add_argument("--news", default="",
                    help="injury news cache (cfb_news); looked up as picks runs do")
    ap.add_argument("--lines-csv", default="")
    a = ap.parse_args()
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip('"\'')
        if "=" in key:
            key = key.split("=", 1)[1].strip().strip('"\'')
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = cp.fetch_rows(key)
    news = None
    if a.news:
        try:
            with open(a.news) as fh:
                news = json.load(fh)
        except (OSError, ValueError):
            news = {}
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    d = report(rows, " ".join(a.game), books, a.bankroll, news=news)
    if a.run_url:
        d["run"] = a.run_url
    md = to_md(d)
    with open(a.md, "w") as fh:
        fh.write(md)
    if a.checks:
        save_check(a.checks, d)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
