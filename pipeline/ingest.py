"""Stage 1: data ingestion and cleaning.

Sources (all free, no key needed):
  * Schedules, scores and betting lines: the cfbfastR data files on GitHub,
    read through the repo's own cfb_data.py (the same feed the live app uses).
    Lines are the median across books of the closing (and opening) spread,
    total, spread price and moneyline.
  * Play-by-play: cfbfastR season files (~120 MB each). Each season is
    collapsed to one row per offense per game and cached, then the raw file
    is deleted.
  * Optional: with CFBD_API_KEY set, team and venue coordinates come from
    the CollegeFootballData.com API for travel distance. Without a key,
    travel is left blank (or read from pipeline/data/locations.csv).

Cleaning:
  * Every game gets a tier for each team (fbs / fcs / other). Games with no
    FBS team are dropped; FBS-vs-FCS games are kept but flagged
    (`non_fbs_game`), and all non-FBS teams share one pooled rating in
    Stage 2, so a 63-3 win over an FCS team can't inflate a rating.
  * Garbage-time plays are removed from the play-by-play before any stat is
    computed (score gap > 38 in Q2, > 28 in Q3, > 22 in Q4).
  * Margins are capped at +/-28 (`margin_capped`) for point-based features.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

# The repo root holds cfb_data.py; make it importable however we're run.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import cfb_data  # noqa: E402

PBP_COLS = ["game_id", "pos_team", "def_pos_team", "home", "period",
            "pos_score_diff_start", "EPA", "success", "rush", "pass",
            "yards_gained", "sack_vec", "int", "fumble_vec",
            "pass_breakup_player_name"]


# ---------------------------------------------------------------------------
# Games, scores and lines
# ---------------------------------------------------------------------------
def _tier(div: pd.Series) -> pd.Series:
    d = div.astype(str).str.lower()
    return np.where(d == "fbs", "fbs", np.where(d == "fcs", "fcs", "other"))


def _side_medians(lines: pd.DataFrame, sched: pd.DataFrame, market: str,
                  value: str) -> pd.DataFrame:
    """Median across books of `value` (lines or odds) for the home and away
    side of `market`. The lines file labels teams by abbreviation in older
    seasons, so map each label to the team id it appears with in >=80% of
    its games (same trick as cfb_data._home_rows)."""
    g = sched.set_index("game_id")[["home_id", "away_id"]]
    m = lines[lines.market_type == market].join(g, on="game_id")
    m = m.dropna(subset=["home_id"])
    co = pd.concat([m[["abbr", "game_id", "home_id"]].rename(columns={"home_id": "tid"}),
                    m[["abbr", "game_id", "away_id"]].rename(columns={"away_id": "tid"})])
    co = co.drop_duplicates()
    cnt = co.groupby(["abbr", "tid"]).size().rename("n").reset_index()
    tot = co.groupby("abbr").game_id.nunique()
    best = cnt.sort_values("n").groupby("abbr").tail(1).set_index("abbr")
    best = best[best.n >= 0.8 * tot.reindex(best.index)]
    m["tid"] = m.abbr.map(best.tid)
    home = m[m.tid == m.home_id].groupby("game_id")[value].median()
    away = m[m.tid == m.away_id].groupby("game_id")[value].median()
    return pd.DataFrame({"home": home, "away": away})


def load_games(seasons=None) -> pd.DataFrame:
    """One row per game with at least one FBS team.

    Key columns: game_id, season, week, season_type, kickoff (UTC),
    home_team/away_team (+ ids, conference, tier), neutral_site,
    home_points/away_points, margin (home - away), margin_capped, points,
    close_spread/open_spread (home side, -7 = home favoured by 7),
    close_total/open_total, home_spread_odds/away_spread_odds,
    over_odds/under_odds, home_ml/away_ml, non_fbs_game.
    """
    seasons = list(seasons or C.ALL_SEASONS)
    g = cfb_data.load_games(seasons)
    g["kickoff"] = pd.to_datetime(g.start_date, utc=True)
    g["home_tier"] = _tier(g.home_division)
    g["away_tier"] = _tier(g.away_division)
    g["non_fbs_game"] = (g.home_tier != "fbs") | (g.away_tier != "fbs")
    g["margin_capped"] = g.margin.clip(-C.MOV_CAP, C.MOV_CAP)

    # Prices: spread odds, total odds and moneylines (median across books).
    lines = pd.read_csv(cfb_data._fetch(cfb_data.LINES_URL, "lines.csv.gz", 24 * 7))
    lines = lines[lines.season.isin(seasons)]
    sp = _side_medians(lines, g, "spread", "odds")
    ml = _side_medians(lines, g, "money_line", "odds")
    tot = lines[lines.market_type == "total"]
    side = tot.abbr.astype(str).str.lower()
    g = g.set_index("game_id")
    g["home_spread_odds"], g["away_spread_odds"] = sp.home, sp.away
    g["home_ml"], g["away_ml"] = ml.home, ml.away
    g["over_odds"] = tot[side == "over"].groupby("game_id").odds.median()
    g["under_odds"] = tot[side == "under"].groupby("game_id").odds.median()
    g = g.reset_index()

    keep = ["game_id", "season", "week", "season_type", "kickoff", "completed",
            "neutral_site", "conference_game", "venue_id", "venue",
            "home_id", "home_team", "home_conference", "home_tier",
            "away_id", "away_team", "away_conference", "away_tier",
            "home_points", "away_points", "margin", "margin_capped", "points",
            "close_spread", "open_spread", "close_total", "open_total",
            "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
            "home_ml", "away_ml", "non_fbs_game"]
    return g[keep].sort_values("kickoff").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Play-by-play -> per-offense game stats
# ---------------------------------------------------------------------------
def garbage_mask(period: pd.Series, gap: pd.Series) -> pd.Series:
    """True for garbage-time plays (score gap too big for the quarter)."""
    lim = period.map(C.GARBAGE_GAP).fillna(np.inf)
    lim = lim.where(period <= 4, np.inf)          # never in overtime
    return gap.abs() > lim


def collapse_pbp(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, offense) stats from scrimmage plays outside garbage time.

    epa        EPA per play
    sr         success rate
    isoppp     explosiveness: mean EPA on successful plays only (IsoPPP)
    ypp        yards per play
    havoc      share of plays with a TFL, sack, interception, forced fumble
               or pass breakup (this is havoc *suffered* by the offense;
               the defense's havoc rate is its opponents' value)
    """
    d = raw[(raw.rush == 1) | (raw["pass"] == 1)].dropna(subset=["EPA"])
    d = d[~garbage_mask(d.period, d.pos_score_diff_start.fillna(0))]
    tfl = (d.rush == 1) & (d.yards_gained < 0)
    hav = (tfl | (d.sack_vec == 1) | (d["int"] == 1) | (d.fumble_vec == 1)
           | d.pass_breakup_player_name.notna())
    d = d.assign(havoc=hav.astype(float),
                 succ_epa=d.EPA.where(d.success == 1))
    a = d.groupby(["game_id", "pos_team"]).agg(
        plays=("EPA", "size"), epa=("EPA", "mean"), sr=("success", "mean"),
        isoppp=("succ_epa", "mean"), ypp=("yards_gained", "mean"),
        havoc=("havoc", "mean")).reset_index()
    return a.rename(columns={"pos_team": "team"})


def load_pbp_stats(seasons=None, verbose=True) -> pd.DataFrame:
    """Per-offense game stats for every season, cached one file per season."""
    seasons = [y for y in (seasons or C.ALL_SEASONS) if y >= cfb_data.PBP_FIRST]
    os.makedirs(C.PIPE_CACHE, exist_ok=True)
    cur = cfb_data.current_season()
    out = []
    for y in seasons:
        small = os.path.join(C.PIPE_CACHE, f"pbp_team_{y}.parquet")
        if not os.path.exists(small) or y >= cur:
            if verbose:
                print(f"  downloading play-by-play {y} (~120 MB) ...", flush=True)
            raw_path = cfb_data._fetch(cfb_data.PBP_URL.format(season=y),
                                       f"pipe_pbp_{y}.parquet", 0)
            raw = pd.read_parquet(raw_path, columns=PBP_COLS)
            collapse_pbp(raw).assign(season=y).to_parquet(small)
            os.remove(raw_path)
        out.append(pd.read_parquet(small))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Locations for travel distance (optional)
# ---------------------------------------------------------------------------
def _cfbd(path: str):
    req = urllib.request.Request(f"{C.CFBD_BASE}{path}", headers={
        "Authorization": f"Bearer {C.CFBD_API_KEY}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def load_locations() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(teams, venues): team home coordinates and venue coordinates.

    Order: cached CSVs -> CFBD API (needs CFBD_API_KEY) -> None.
    teams: team, lat, lon      venues: venue_id, lat, lon
    """
    os.makedirs(C.PIPE_CACHE, exist_ok=True)
    tp = os.path.join(C.PIPE_CACHE, "team_locations.csv")
    vp = os.path.join(C.PIPE_CACHE, "venue_locations.csv")
    manual = os.path.join(os.path.dirname(__file__), "data", "locations.csv")
    if os.path.exists(tp) and os.path.exists(vp):
        return pd.read_csv(tp), pd.read_csv(vp)
    if C.CFBD_API_KEY:
        try:
            teams = pd.DataFrame([
                {"team": t.get("school"),
                 "lat": (t.get("location") or {}).get("latitude"),
                 "lon": (t.get("location") or {}).get("longitude")}
                for t in _cfbd("/teams")]).dropna()
            venues = pd.DataFrame([
                {"venue_id": v.get("id"),
                 "lat": v.get("latitude") or (v.get("location") or {}).get("x"),
                 "lon": v.get("longitude") or (v.get("location") or {}).get("y")}
                for v in _cfbd("/venues")]).dropna()
            teams.to_csv(tp, index=False)
            venues.to_csv(vp, index=False)
            return teams, venues
        except Exception as e:  # network blocked, bad key, ...
            print(f"  CFBD location lookup failed ({e}); travel left blank")
    if os.path.exists(manual):
        return pd.read_csv(manual), None
    return None, None


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
def stage1(seasons=None, verbose=True):
    """Run Stage 1: returns (games, pbp_stats, teams_loc, venues_loc)."""
    seasons = list(seasons or C.ALL_SEASONS)
    if verbose:
        print(f"Stage 1: loading games and lines for {seasons[0]}-{seasons[-1]}")
    games = load_games(seasons)
    if verbose:
        print(f"  {len(games):,} games with an FBS team "
              f"({int(games.non_fbs_game.sum()):,} vs FCS/lower, flagged)")
        print("Stage 1: play-by-play (garbage time removed)")
    pbp = load_pbp_stats(seasons, verbose)
    teams, venues = load_locations()
    if verbose:
        print(f"  {len(pbp):,} offense-game rows; travel data: "
              f"{'yes' if teams is not None else 'no (set CFBD_API_KEY)'}")
    return games, pbp, teams, venues


if __name__ == "__main__":
    g, p, _, _ = stage1()
    print(g.tail().to_string())
    print(p.tail().to_string())
