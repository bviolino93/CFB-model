"""Stage 2: feature engineering with strict anti-leakage.

Rule: every feature for game N is built only from games that kicked off
before game N. Two mechanisms enforce it:

  1. Opponent-adjusted ratings are refit once per week, on games that
     kicked off before that week's first kickoff (`ratings_cutoff`).
  2. Rolling EWMAs are shifted one game, so game N's value never enters
     its own feature.

`audit_leakage()` checks both on the finished table and raises if any
feature could have seen the game it describes.

How opponent adjustment works
-----------------------------
For each metric (EPA/play, success rate, IsoPPP, yards/play, havoc, points)
we fit, every week, a ridge regression on all earlier team-games:

    value(offense i vs defense j) = mu + O_i + A_j + hfa * home

O_i is how good offense i is, A_j is how much defense j allows (higher =
worse defense; for havoc, higher = more havoc forced), hfa is home field.
Last season's games are included at a fading weight so week 1 has ratings.
All non-FBS teams share one pooled rating.

Each past game is then re-expressed relative to its opponent:

    adj_off     = value - mu - A_opponent - hfa * home      (offense above avg)
    adj_allowed = allowed - mu - O_opponent + hfa * home    (allowed above avg)

and the 4- and 8-game EWMAs of those adjusted numbers are the trend
features. The as-of-week ridge ratings themselves (season-to-date,
schedule-adjusted) are also kept as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from . import config as C
from .ingest import haversine_miles

METRICS = ["epa", "sr", "isoppp", "ypp", "havoc", "pts"]
NON_FBS = "__NON_FBS__"


# ---------------------------------------------------------------------------
# 1. One row per team per game
# ---------------------------------------------------------------------------
def team_games(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Long table: two rows per game (one per team) with what the team's
    offense did (off_*) and what its defense allowed (allowed_*)."""
    # Unplayed games stay in (with blank outcomes) so they get features too.
    g = games.copy()
    neutral = g.neutral_site.astype(bool)
    base = ["game_id", "season", "season_type", "week", "kickoff",
            "venue_id", "neutral_site"]
    home = g[base].assign(
        team=g.home_team, opp=g.away_team, tier=g.home_tier, opp_tier=g.away_tier,
        h=np.where(neutral, 0, 1), pf=g.home_points, pa=g.away_points)
    away = g[base].assign(
        team=g.away_team, opp=g.home_team, tier=g.away_tier, opp_tier=g.home_tier,
        h=np.where(neutral, 0, -1), pf=g.away_points, pa=g.home_points)
    tg = pd.concat([home, away], ignore_index=True)

    # Points with the margin capped: a 70-0 win counts as 28-0.
    tg["off_pts"] = np.minimum(tg.pf, tg.pa + C.MOV_CAP)
    tg["allowed_pts"] = np.minimum(tg.pa, tg.pf + C.MOV_CAP)

    stats = pbp[["game_id", "team", "epa", "sr", "isoppp", "ypp", "havoc"]]
    off = stats.rename(columns={m: f"off_{m}" for m in METRICS if m != "pts"})
    alw = stats.rename(columns={"team": "opp",
                                **{m: f"allowed_{m}" for m in METRICS if m != "pts"}})
    tg = tg.merge(off, on=["game_id", "team"], how="left")
    tg = tg.merge(alw, on=["game_id", "opp"], how="left")

    pool = C.POOL_NON_FBS
    tg["rid"] = np.where(pool & (tg.tier != "fbs"), NON_FBS, tg.team)
    tg["opp_rid"] = np.where(pool & (tg.opp_tier != "fbs"), NON_FBS, tg.opp)
    return tg.sort_values(["kickoff", "game_id", "h"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Weekly opponent-adjusted ratings (ridge), fit only on earlier games
# ---------------------------------------------------------------------------
def week_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    """Each (season, season_type, week) and its first kickoff. Ratings used
    for any game in that week are fit strictly before this time."""
    wk = (games.groupby(["season", "season_type", "week"]).kickoff.min()
          .rename("cutoff").reset_index().sort_values("cutoff"))
    wk["weeks_in"] = wk.groupby("season").cumcount()
    return wk.reset_index(drop=True)


def _fit_ridge(rows: pd.DataFrame, value: pd.Series, w: np.ndarray):
    """value = mu + O[rid] + A[opp_rid] + hfa*h. Returns (mu, hfa, O, A)."""
    teams = pd.Index(sorted(set(rows.rid) | set(rows.opp_rid)))
    n, k = len(rows), len(teams)
    oi = teams.get_indexer(rows.rid)
    ai = teams.get_indexer(rows.opp_rid)
    r = np.arange(n)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (r, oi)), shape=(n, k)),
        sparse.csr_matrix((np.ones(n), (r, ai)), shape=(n, k)),
        sparse.csr_matrix(rows.h.to_numpy(float).reshape(-1, 1)),
    ]).tocsr()
    m = Ridge(alpha=C.RIDGE_ALPHA, fit_intercept=True)
    m.fit(X, value.to_numpy(float), sample_weight=w)
    coef = m.coef_
    return (float(m.intercept_), float(coef[-1]),
            pd.Series(coef[:k], teams), pd.Series(coef[k:2 * k], teams))


def weekly_ratings(tg: pd.DataFrame, cutoffs: pd.DataFrame, verbose=True):
    """For every week cutoff and metric: mu, hfa and each rating id's O and A,
    fit on team-games with kickoff < cutoff (this season, plus last season
    at a fading weight). Returns (ratings_long, league) DataFrames and a
    list of (cutoff, latest_kickoff_used) pairs for the leakage audit."""
    rat, lg, used = [], [], []
    for i, c in cutoffs.iterrows():
        past = tg[(tg.kickoff < c.cutoff)
                  & tg.season.isin([c.season, c.season - 1])]
        if past.empty:
            continue
        prior_w = C.PRIOR_SEASON_WEIGHT * np.exp(-c.weeks_in / C.PRIOR_SEASON_DECAY_WEEKS)
        w_all = np.where(past.season == c.season, 1.0, prior_w)
        used.append((c.cutoff, past.kickoff.max()))
        for met in METRICS:
            y = past[f"off_{met}"]
            ok = y.notna().to_numpy()
            if ok.sum() < 50:
                continue
            mu, hfa, O, A = _fit_ridge(past[ok], y[ok], w_all[ok])
            key = dict(season=c.season, season_type=c.season_type, week=c.week)
            lg.append({**key, "metric": met, "mu": mu, "hfa": hfa})
            rat.append(pd.DataFrame({**key, "metric": met, "rid": O.index,
                                     "O": O.values, "A": A.reindex(O.index).values}))
        if verbose and i % 40 == 0:
            print(f"    ratings through {c.season} {c.season_type} wk {c.week}", flush=True)
    return pd.concat(rat, ignore_index=True), pd.DataFrame(lg), used


def _attach_ratings(tg: pd.DataFrame, ratings: pd.DataFrame, league: pd.DataFrame):
    """Add, for every team-game, the as-of-week mu/hfa and the team's and
    opponent's O and A for each metric (missing team -> 0 = average)."""
    key = ["season", "season_type", "week"]
    out = tg.copy()
    for met in METRICS:
        lgm = league[league.metric == met].drop(columns="metric").rename(
            columns={"mu": f"mu_{met}", "hfa": f"hfa_{met}"})
        out = out.merge(lgm, on=key, how="left")
        r = ratings[ratings.metric == met].drop(columns="metric")
        mine = r.rename(columns={"rid": "rid", "O": f"rO_{met}", "A": f"rA_{met}"})
        theirs = r.rename(columns={"rid": "opp_rid", "O": f"oppO_{met}", "A": f"oppA_{met}"})
        out = out.merge(mine, on=key + ["rid"], how="left")
        out = out.merge(theirs, on=key + ["opp_rid"], how="left")
        has = out[f"mu_{met}"].notna()
        for col in (f"rO_{met}", f"rA_{met}", f"oppO_{met}", f"oppA_{met}"):
            out.loc[has, col] = out.loc[has, col].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# 3. Opponent-adjusted game values and their EWMAs (shifted one game)
# ---------------------------------------------------------------------------
def add_adjusted_and_ewma(tg: pd.DataFrame) -> pd.DataFrame:
    out = tg.copy()
    for met in METRICS:
        mu, hfa = out[f"mu_{met}"], out[f"hfa_{met}"]
        out[f"adj_off_{met}"] = out[f"off_{met}"] - mu - out[f"oppA_{met}"] - hfa * out.h
        out[f"adj_allowed_{met}"] = (out[f"allowed_{met}"] - mu - out[f"oppO_{met}"]
                                     + hfa * out.h)
    out = out.sort_values(["team", "kickoff"]).reset_index(drop=True)
    grp = out.groupby("team", sort=False)
    adj_cols = [f"adj_{s}_{m}" for m in METRICS for s in ("off", "allowed")]
    prev = grp[adj_cols].shift(1)          # <- game N never sees itself
    prev["team"] = out.team
    for span in C.EWMA_SPANS:
        ew = (prev.groupby("team", sort=False)[adj_cols]
              .transform(lambda s: s.ewm(span=span, adjust=True, ignore_na=True).mean()))
        for col in adj_cols:
            out[f"ewm{span}_{col[4:]}"] = ew[col]
    out["last_kickoff"] = grp.kickoff.shift(1)
    out["games_before"] = grp.cumcount()
    out["games_before_season"] = out.groupby(["team", "season"]).cumcount()
    rest = (out.kickoff - out.last_kickoff).dt.total_seconds() / 86400
    out["opener"] = (out.games_before_season == 0).astype(int)
    out["rest_days"] = rest.where(out.opener == 0).clip(upper=C.REST_CAP_DAYS).fillna(
        C.REST_CAP_DAYS)
    return out


def add_travel(tg: pd.DataFrame, games: pd.DataFrame, teams_loc, venues_loc):
    """Miles from each team's campus to the game site (venue coordinates when
    known, else the home team's campus). Blank when no location data."""
    tg = tg.copy()
    tg["travel_miles"] = np.nan
    if teams_loc is None:
        return tg
    tl = teams_loc.drop_duplicates("team").set_index("team")
    site_lat = games.set_index("game_id").home_team.map(tl.lat)
    site_lon = games.set_index("game_id").home_team.map(tl.lon)
    if venues_loc is not None:
        vl = venues_loc.drop_duplicates("venue_id").set_index("venue_id")
        vid = games.set_index("game_id").venue_id
        site_lat = vid.map(vl.lat).fillna(site_lat)
        site_lon = vid.map(vl.lon).fillna(site_lon)
    tg["travel_miles"] = haversine_miles(
        tg.team.map(tl.lat), tg.team.map(tl.lon),
        tg.game_id.map(site_lat), tg.game_id.map(site_lon))
    return tg


# ---------------------------------------------------------------------------
# 4. Game-level model table
# ---------------------------------------------------------------------------
TEAM_FEATS = (
    [f"ewm{s}_{side}_{m}" for s in C.EWMA_SPANS for side in ("off", "allowed")
     for m in METRICS]
    + [f"r{x}_{m}" for x in ("O", "A") for m in METRICS]
    + ["rest_days", "opener", "travel_miles", "games_before_season"])


def game_table(games: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home (h_) and away (a_) features, matchup diffs,
    situational features, the market lines and the targets."""
    side = tg[["game_id", "team", "last_kickoff"] + TEAM_FEATS]
    h = side.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home_team"})
    a = side.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away_team"})
    df = games.merge(h, on=["game_id", "home_team"], how="left")
    df = df.merge(a, on=["game_id", "away_team"], how="left")
    cut = week_cutoffs(games)
    df = df.merge(cut[["season", "season_type", "week", "cutoff"]]
                  .rename(columns={"cutoff": "ratings_cutoff"}),
                  on=["season", "season_type", "week"], how="left")

    # Matchup features: expected home offense vs away defense, and vice versa.
    for m in METRICS:
        sgn = -1.0 if m == "havoc" else 1.0   # havoc suffered is bad for offense
        for s in C.EWMA_SPANS:
            hx = df[f"h_ewm{s}_off_{m}"] + df[f"a_ewm{s}_allowed_{m}"]
            ax = df[f"a_ewm{s}_off_{m}"] + df[f"h_ewm{s}_allowed_{m}"]
            df[f"diff_ewm{s}_{m}"] = sgn * (hx - ax)
            df[f"sum_ewm{s}_{m}"] = hx + ax
        hx = df[f"h_rO_{m}"] + df[f"a_rA_{m}"]
        ax = df[f"a_rO_{m}"] + df[f"h_rA_{m}"]
        df[f"diff_rating_{m}"] = sgn * (hx - ax)
        df[f"sum_rating_{m}"] = hx + ax

    df["home_field"] = (~df.neutral_site.astype(bool)).astype(int)
    df["rest_diff"] = df.h_rest_days - df.a_rest_days
    df["travel_diff"] = df.a_travel_miles - df.h_travel_miles

    # Targets (NaN for unplayed games). close_spread is from the home side,
    # so home covers when margin + close_spread > 0.
    df["home_cover"] = np.where(df.margin.isna() | df.close_spread.isna(), np.nan,
                                (df.margin + df.close_spread > 0).astype(float))
    df["spread_push"] = (df.margin + df.close_spread == 0).astype(int)
    df["over_hit"] = np.where(df.points.isna() | df.close_total.isna(), np.nan,
                              (df.points > df.close_total).astype(float))
    return df.sort_values("kickoff").reset_index(drop=True)


def model_features(df: pd.DataFrame) -> list[str]:
    """Columns Stage 3 should feed the model (no outcomes, no market)."""
    cols = [c for c in df.columns if c.startswith(("diff_", "sum_"))]
    cols += [f"{p}_{f}" for p in ("h", "a") for f in TEAM_FEATS]
    cols += ["home_field", "rest_diff", "travel_diff"]
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# 5. Leakage audit
# ---------------------------------------------------------------------------
def audit_leakage(df: pd.DataFrame, ratings_used) -> dict:
    """Raise if any feature could have seen its own game or a later one."""
    problems = []
    for p in ("h", "a"):
        bad = df[f"{p}_last_kickoff"].notna() & (df[f"{p}_last_kickoff"] >= df.kickoff)
        if bad.any():
            problems.append(f"{bad.sum()} games where {p}_ EWMA used a game at/after kickoff")
    bad = df.ratings_cutoff.notna() & (df.ratings_cutoff > df.kickoff)
    if bad.any():
        problems.append(f"{bad.sum()} games whose rating cutoff is after kickoff")
    late = [(c, k) for c, k in ratings_used if k >= c]
    if late:
        problems.append(f"{len(late)} weekly ridge fits trained on games at/after cutoff")
    if problems:
        raise AssertionError("LEAKAGE: " + "; ".join(problems))
    return {"games_checked": len(df), "ridge_fits_checked": len(ratings_used),
            "status": "no leakage found"}


# ---------------------------------------------------------------------------
def stage2(games, pbp, teams_loc=None, venues_loc=None, verbose=True):
    """Run Stage 2: returns (model_table, team_games, audit)."""
    if verbose:
        print("Stage 2: team-game table")
    tg = team_games(games, pbp)
    cut = week_cutoffs(games)
    if verbose:
        print(f"  {len(tg):,} team-games; fitting {len(cut)} weekly ridge ratings x "
              f"{len(METRICS)} metrics")
    ratings, league, used = weekly_ratings(tg, cut, verbose)
    tg = _attach_ratings(tg, ratings, league)
    tg = add_adjusted_and_ewma(tg)
    tg = add_travel(tg, games, teams_loc, venues_loc)
    df = game_table(games, tg)
    audit = audit_leakage(df, used)
    if verbose:
        print(f"  leakage audit: {audit['status']} "
              f"({audit['games_checked']:,} games, {audit['ridge_fits_checked']} fits)")
    return df, tg, audit
