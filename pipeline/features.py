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

import re

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
    """Pre-kickoff feature columns (no outcomes, no market lines)."""
    cols = [c for c in df.columns if c.startswith(("diff_", "sum_", "pri_", "qb_", "inj_"))]
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
def stage2(games, pbp, teams_loc=None, venues_loc=None, verbose=True, extras=None,
           injury_rows=None):
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
    df = add_signals(df, tg, extras, injury_rows)
    audit = audit_leakage(df, used)
    if verbose:
        print(f"  leakage audit: {audit['status']} "
              f"({audit['games_checked']:,} games, {audit['ridge_fits_checked']} fits)")
    return df, tg, audit


# ---------------------------------------------------------------------------
# 6. Preseason priors for weeks 1-4 (fade linearly to zero by week 5)
# ---------------------------------------------------------------------------
PRIOR_FULL_WEEKS = 5     # weight 1.0 in week 1, 0.75, 0.5, 0.25, 0 from week 5


def prior_weight(df: pd.DataFrame) -> pd.Series:
    w = ((PRIOR_FULL_WEEKS - df.week) / (PRIOR_FULL_WEEKS - 1)).clip(0, 1)
    return w.where(df.season_type == "regular", 0.0).astype(float)


def returning_production(players: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Share of last season's production by players on this season's roster.

    ret_pass  dropbacks      ret_rush  rushing yards     ret_rec  receiving yards
    ret_def   havoc credits (sacks, INTs, PBUs, forced fumbles), by name
    ret_off   0.5 pass + 0.25 rush + 0.25 rec
    ret_qb    1 if last season's top passer is on the roster
    Transfers in are not counted (same as CFBD's returning production).
    """
    out = []
    for s in sorted(rosters.season.unique()):
        prev = players[players.season == s - 1]
        ros = rosters[rosters.season == s]
        if prev.empty or ros.empty:
            continue
        ids = set(zip(ros.team, ros.athlete_id))
        names = set(zip(ros.team, ros.name))
        off = prev[prev.pid.notna()].copy()
        off["back"] = [(t, int(p)) in ids for t, p in zip(off.team, off.pid)]
        de = prev[prev.name.notna()].copy()
        de["back"] = [(t, n) in names for t, n in zip(de.team, de.name)]
        agg = {}
        for col, key in (("dropbacks", "ret_pass"), ("rush_yds", "ret_rush"),
                         ("rec_yds", "ret_rec")):
            v = off[col].clip(lower=0)
            tot = v.groupby(off.team).sum()
            agg[key] = (v * off.back).groupby(off.team).sum() / tot.where(tot > 0)
        tot = de.havoc.groupby(de.team).sum()
        agg["ret_def"] = (de.havoc * de.back).groupby(de.team).sum() / tot.where(tot > 0)
        top = off.sort_values("dropbacks").groupby("team").tail(1).set_index("team")
        agg["ret_qb"] = top.back.astype(float)
        r = pd.DataFrame(agg)
        r["ret_off"] = 0.5 * r.ret_pass + 0.25 * r.ret_rush + 0.25 * r.ret_rec
        out.append(r.rename_axis("team").reset_index().assign(season=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["team", "season", "ret_off", "ret_def", "ret_qb"])


def prior_ratings(tg: pd.DataFrame) -> pd.DataFrame:
    """Each team's final schedule-adjusted points ratings from last season
    (the as-of ratings before its last game), keyed to the next season."""
    last = tg[tg.rO_pts.notna()].sort_values("kickoff").groupby(["team", "season"]).tail(1)
    return pd.DataFrame({"team": last.team, "season": last.season + 1,
                         "prior_off": last.rO_pts.values, "prior_def": last.rA_pts.values})


def add_preseason_priors(df: pd.DataFrame, tg: pd.DataFrame, players=None,
                         rosters=None, talent=None) -> pd.DataFrame:
    """pri_* features: preseason information times the week-1-to-5 fade
    weight `pri_w`. Everything here is known before the season starts
    (caveat: the free roster files are one snapshot per season, so a
    mid-season transfer could appear a few weeks early)."""
    df = df.copy()
    w = prior_weight(df)
    df["pri_w"] = w
    pr = prior_ratings(tg)
    parts = [pr]
    if players is not None and rosters is not None:
        parts.append(returning_production(players, rosters))
    if talent is not None:
        t = talent.copy()
        t["talent"] = t.groupby("season").talent.transform(lambda x: (x - x.mean()) / x.std())
        parts.append(t[["team", "season", "talent"]])
    info = parts[0]
    for p in parts[1:]:
        info = info.merge(p, on=["team", "season"], how="outer")
    for side in ("home", "away"):
        x = info.add_prefix(f"{side[0]}p_").rename(
            columns={f"{side[0]}p_team": f"{side}_team", f"{side[0]}p_season": "season"})
        df = df.merge(x, on=[f"{side}_team", "season"], how="left")
    # Matchup priors (spread side) and their totals counterparts.
    h_net = df.hp_prior_off - df.hp_prior_def
    a_net = df.ap_prior_off - df.ap_prior_def
    df["pri_rating_diff"] = w * (h_net - a_net)
    df["pri_rating_sum"] = w * (df.hp_prior_off + df.ap_prior_def
                                + df.ap_prior_off + df.hp_prior_def)
    for k in ("ret_off", "ret_def", "ret_qb", "talent"):
        if f"hp_{k}" in df:
            df[f"pri_{k}_diff"] = w * (df[f"hp_{k}"] - df[f"ap_{k}"])
            df[f"pri_{k}_sum"] = w * (df[f"hp_{k}"] + df[f"ap_{k}"])
    # Last year's rating matters less when little production returns.
    if "hp_ret_off" in df:
        df["pri_rating_x_ret"] = w * (h_net * df.hp_ret_off.fillna(0.5)
                                      - a_net * df.ap_ret_off.fillna(0.5))
    return df


# ---------------------------------------------------------------------------
# 7. Injury / QB signals
# ---------------------------------------------------------------------------
def add_qb_change(df: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Historical stand-in for QB injury news, since there is no historical
    college injury report: did a team start a different QB than in its
    previous game, and how many starts does this QB have for the team?

    qb_change   1 if this game's starter differs from the last game's
                (week 1: last season's most frequent starter)
    qb_starts   starts for this team in the current + previous season, cap 10

    Caveat: this uses who actually started. That's usually public before
    kickoff, but not always (game-time decisions), so treat it as an upper
    bound on what pregame injury news is worth.
    """
    df = df.copy()
    st = starters.drop(columns=["season"], errors="ignore").merge(
        df[["game_id", "kickoff", "season"]], on="game_id", how="inner")
    st = st.sort_values(["pos_team", "kickoff"]).reset_index(drop=True)
    g = st.groupby("pos_team", sort=False)
    prev = g.starter.shift(1)
    prev_season = g.season.shift(1)
    # week 1: compare with last season's most frequent starter
    mode = (st.groupby(["pos_team", "season"]).starter
            .agg(lambda x: x.mode().iloc[0]).rename("mode").reset_index())
    mode["season"] += 1
    st = st.merge(mode, on=["pos_team", "season"], how="left")
    new_season = prev_season.values != st.season.values
    ref = np.where(new_season, st["mode"], prev)
    st["qb_change"] = np.where(pd.isna(ref), np.nan, (st.starter != ref).astype(float))
    # starts by this QB for this team in this and the previous season
    st["qb_starts"] = 0.0
    for (team, qb), grp in st.groupby(["pos_team", "starter"]):
        k = grp.kickoff.values
        ss = grp.season.values
        cnt = [(np.sum((k < k[i]) & (ss >= ss[i] - 1))) for i in range(len(k))]
        st.loc[grp.index, "qb_starts"] = np.minimum(cnt, 10)
    for side in ("home", "away"):
        x = st[["game_id", "pos_team", "qb_change", "qb_starts"]].rename(columns={
            "pos_team": f"{side}_team", "qb_change": f"{side[0]}_qb_change",
            "qb_starts": f"{side[0]}_qb_starts"})
        df = df.merge(x, on=["game_id", f"{side}_team"], how="left")
    df["qb_change_diff"] = df.a_qb_change - df.h_qb_change       # + favours home
    df["qb_starts_diff"] = df.h_qb_starts - df.a_qb_starts
    df["qb_change_sum"] = df.a_qb_change + df.h_qb_change
    return df


def _team_key(t) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(t or "").strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\u2018\u2019'`.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r" st$", " state", s)


def add_injury_feed(df: pd.DataFrame, rows) -> pd.DataFrame:
    """Injury impact from the app's injury format, as model features.

    `rows`: list of dicts or a DataFrame with the app's columns
    (team, points, note, until, added_at): "team is N points WORSE until
    `until`", the same rows the app reads from the sheet and from
    injuries_auto.json on the injury-feed branch. An entry counts for a game
    only if it was added before kickoff and is still in force that day.

    inj_h_pts / inj_a_pts   points off each team
    inj_diff                away points - home points (+ favours home), spreads
    inj_sum                 total points off both teams, totals
    inj_qb_diff             1 per side whose entry mentions a QB (+ favours home)
    """
    df = df.copy()
    inj = pd.DataFrame(rows)
    for c in ("inj_h_pts", "inj_a_pts", "inj_diff", "inj_sum", "inj_qb_diff"):
        df[c] = 0.0
    if inj.empty:
        return df
    inj["key"] = inj.team.map(_team_key)
    inj["points"] = pd.to_numeric(inj.points, errors="coerce").fillna(0)
    inj["until"] = pd.to_datetime(inj.until.astype(str).str[:10], errors="coerce")
    inj["added"] = pd.to_datetime(inj.added_at, errors="coerce", utc=True)
    inj["qb"] = inj.note.astype(str).str.contains(r"\bQB\b|quarterback", case=False)
    day = df.kickoff.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    for side, p in (("home", "h"), ("away", "a")):
        keys = df[f"{side}_team"].map(_team_key)
        pts, qb = np.zeros(len(df)), np.zeros(len(df))
        for i, (k, d, ko) in enumerate(zip(keys, day, df.kickoff)):
            m = inj[(inj.key == k) & ((inj.until >= d) | inj.until.isna())
                    & ((inj.added < ko) | inj.added.isna())]
            pts[i], qb[i] = m.points.sum(), float(m.qb.any())
        df[f"inj_{p}_pts"], df[f"_qb_{p}"] = pts, qb
    df["inj_diff"] = df.inj_a_pts - df.inj_h_pts
    df["inj_sum"] = df.inj_a_pts + df.inj_h_pts
    df["inj_qb_diff"] = df._qb_a - df._qb_h
    return df.drop(columns=["_qb_h", "_qb_a"])


# ---------------------------------------------------------------------------
# 8. Market features (for the market-residual model)
# ---------------------------------------------------------------------------
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """mkt_* columns. line_move = close - open (home side for spreads, so a
    negative move means money came in on the home team)."""
    df = df.copy()
    df["mkt_has_open_spread"] = df.open_spread.notna().astype(int)
    df["mkt_open_spread"] = df.open_spread.fillna(df.close_spread)
    df["mkt_close_spread"] = df.close_spread
    df["mkt_spread_move"] = (df.close_spread - df.open_spread).fillna(0)
    df["mkt_has_open_total"] = df.open_total.notna().astype(int)
    df["mkt_open_total"] = df.open_total.fillna(df.close_total)
    df["mkt_close_total"] = df.close_total
    df["mkt_total_move"] = (df.close_total - df.open_total).fillna(0)
    return df


def add_signals(df, tg, extras, injury_rows=None):
    """Priors, QB change, injury feed and market features in one call."""
    ex = extras or {}
    df = add_preseason_priors(df, tg, ex.get("players"), ex.get("rosters"), ex.get("talent"))
    if ex.get("starters") is not None:
        df = add_qb_change(df, ex["starters"])
    df = add_injury_feed(df, injury_rows or [])
    return add_market_features(df)
