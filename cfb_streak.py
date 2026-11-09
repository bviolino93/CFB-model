"""Over streaks: the one totals trend that held up in every era.

Brian (2026-11-09): "there has to be trends". A scan of about 75 situations
against real closing lines (2014-19 to find, 2021-25 to check) found one
that held in every stretch: when either team has gone OVER the closing
total in each of its last 3 games, the under hit

    53.8% in 2014-16, 54.8% in 2017-19, 55.7% in 2021-23, 53.5% in 2024-25

(about 1,275 games, roughly 9 a week; break-even at -110 is 52.4%). The
market overreacts to hot scoring runs. Scripts: project files
cfb/sim-research/trends.py and tot.py.

Free data has no 2026 lines, so a game's closing total comes from our own
closes.json (every game since 2026-10-27). For older games it falls back to
our model's pre-game total (pred_total), which tested weaker (51-52% in
2021-25), so the fallback only fills in until every team's last 3 games are
on record (from about 2026-11-21).
"""
from __future__ import annotations

import pandas as pd

STREAK_LEN = 3          # games in a row over the total
STREAK_P = 0.545        # under hit rate, 2014-25 against closing totals
STREAK_STAKE = 0.005    # of bankroll: $5 flat on $1,000


def closing_totals(g: pd.DataFrame, closes: dict | None) -> pd.Series:
    """game_id -> the closing total we recorded (closes.json), for this
    season's finished games."""
    from cfb_picks import _matches     # cfb_picks imports this module
    if not closes or g is None or not len(g):
        return pd.Series(dtype=float)
    s = g[g.margin.notna()].assign(day=pd.to_datetime(g.start_date, utc=True))
    out = {}
    for k, c in closes.items():
        if "total" not in c or k.count("|") != 2:
            continue
        away, home, _ = k.split("|")
        kick = pd.Timestamp(c.get("commence") or k.split("|")[2])
        kick = kick.tz_localize("UTC") if kick.tzinfo is None else kick.tz_convert("UTC")
        near = s[(s.day - kick).abs() <= pd.Timedelta(hours=36)]
        for r in near.itertuples():
            if _matches(home, r.home_team) and _matches(away, r.away_team):
                out[int(r.game_id)] = float(c["total"])
                break
    return pd.Series(out, dtype=float)


def streaks(g: pd.DataFrame, closes: dict | None, season: int) -> dict:
    """school -> (games in a row over the total, entering its next game,
    and how many of those used a recorded closing line)."""
    if g is None or not len(g):
        return {}
    x = g[(g.season == season) & g.margin.notna()].copy()
    if not len(x):
        return {}
    ct = closing_totals(x, closes)
    x["line"] = x.game_id.map(ct)
    x["real"] = x.line.notna()
    if "close_total" in x:
        x["line"] = x.line.fillna(x.close_total)
    if "pred_total" in x:
        x["line"] = x.line.fillna(x.pred_total)
    x["when"] = pd.to_datetime(x.start_date, utc=True)
    x = x.sort_values("when")
    out = {}
    for side in ("home_team", "away_team"):
        for team, rows in x.groupby(side):
            out.setdefault(team, []).extend(rows[["when", "points", "line", "real"]].itertuples(index=False))
    res = {}
    for team, games in out.items():
        games = sorted(games, key=lambda r: r.when)
        run, real = 0, 0
        for r in reversed(games):
            if pd.isna(r.line) or r.points == r.line or r.points < r.line:
                break
            run += 1
            real += bool(r.real)
        res[team] = (run, real)
    return res


def signal(st: dict, home: str, away: str) -> str | None:
    """The reason text when either team is on a run of STREAK_LEN overs,
    else None."""
    hot = [(t, st.get(t, (0, 0))[0]) for t in (away, home)]
    hot = [(t, n) for t, n in hot if n >= STREAK_LEN]
    if not hot:
        return None
    who = " and ".join(f"{t} has gone over in {n} straight games" for t, n in hot)
    return (f"{who}. When either team is on a run of 3+ overs, the under has hit "
            f"about 54-56% against closing totals in every stretch since 2014 "
            f"(1,275 games): the market overreacts to hot scoring runs")


def describe(st: dict, team: str) -> str:
    """Plain-English over/under run for the handicapper's brief."""
    n = st.get(team, (0, 0))[0]
    return f"{team}: {n} straight over{'s' if n != 1 else ''}" if n else f"{team}: no current run of overs"
