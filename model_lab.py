"""
Saturday Edge — Model Lab
=========================

A separate research app. It does not touch the betting app.

One question: do your fundamentals (SP+, SRS, talent, returning production)
add ANY out-of-sample predictive value on top of the market spread?

Method
------
The market spread is the strongest single predictor of margin that exists.
So we make it the baseline, not the opponent:

    Baseline : margin ~ spread
    Candidate: margin ~ spread + fundamentals

We fit on past seasons and test on the NEXT season (walk-forward), never
random folds — random folds leak the future into the past and make noise
look like skill.

If the fundamentals earn no weight out of sample, you have your answer
cheaply, and no amount of extra modelling will fix it.
"""

import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

BASE_URL = "https://api.collegefootballdata.com"

st.set_page_config(page_title="Model Lab", page_icon="🔬", layout="wide")

try:
    API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error("Missing CFBD_API_KEY in Streamlit Secrets.")
    st.stop()


# ---------------------------------------------------------------- data layer

def _get(path, params=None):
    r = requests.get(
        BASE_URL + path,
        headers={"Authorization": f"Bearer {API_KEY}"},
        params=params or {},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _num(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def games(year):
    return _get("/games", {"year": int(year), "seasonType": "regular"})


@st.cache_data(ttl=86400, show_spinner=False)
def lines(year):
    return _get("/lines", {"year": int(year), "seasonType": "regular"})


@st.cache_data(ttl=86400, show_spinner=False)
def ratings(year):
    """SP+, SRS, talent and returning production keyed by team."""
    out = {}
    for key, path, params in [
        ("sp", "/ratings/sp", {"year": year}),
        ("srs", "/ratings/srs", {"year": year}),
        ("talent", "/talent", {"year": year}),
        ("returning", "/player/returning", {"year": year}),
    ]:
        try:
            out[key] = _get(path, params) or []
        except Exception:
            out[key] = []
    return out


def _team_key(row):
    return str(row.get("team") or row.get("school") or "").strip().lower()


def feature_frame(year):
    """One row per team with the fundamental inputs, z-scored within season."""
    r = ratings(year)
    rows = {}

    for x in r["sp"]:
        k = _team_key(x)
        if not k:
            continue
        rows.setdefault(k, {})["sp"] = _num(x.get("rating"))

    for x in r["srs"]:
        k = _team_key(x)
        if not k:
            continue
        rows.setdefault(k, {})["srs"] = _num(x.get("rating"))

    for x in r["talent"]:
        k = _team_key(x)
        if not k:
            continue
        rows.setdefault(k, {})["talent"] = _num(x.get("talent"))

    for x in r["returning"]:
        k = _team_key(x)
        if not k:
            continue
        rows.setdefault(k, {})["returning"] = _num(
            x.get("totalPPA") if x.get("totalPPA") is not None else x.get("percentPPA")
        )

    df = pd.DataFrame(
        [{"team": k, **v} for k, v in rows.items()]
    )
    if df.empty:
        return df

    # Z-score within the season so units are comparable across years.
    for c in ["sp", "srs", "talent", "returning"]:
        if c not in df.columns:
            df[c] = np.nan
        s = pd.to_numeric(df[c], errors="coerce")
        mu, sd = s.mean(), s.std(ddof=0)
        df[c + "_z"] = 0.0 if (not sd or math.isnan(sd) or sd == 0) else (s - mu) / sd
        df[c + "_z"] = df[c + "_z"].fillna(0.0)

    return df.set_index("team")


def closing_spreads(year):
    """
    Median spread across providers per game. CFBD does not reliably expose a
    timestamped closing line, so this is the consensus quoted line — an
    imperfect but usable proxy. See the caveat shown in the app.
    """
    out = {}
    for row in lines(year) or []:
        gid = row.get("id")
        if gid is None:
            continue
        vals = []
        for ln in (row.get("lines") or []):
            s = _num(ln.get("spread"))
            if s is not None:
                vals.append(s)
        if vals:
            out[int(gid)] = float(np.median(vals))
    return out


def build_season(year):
    """Assemble one modelling row per completed game with a market spread."""
    feats = feature_frame(year)
    spreads = closing_spreads(year)
    if feats.empty or not spreads:
        return pd.DataFrame()

    rows = []
    for g in games(year) or []:
        if g.get("completed") is not True:
            continue
        hp, ap = _num(g.get("homePoints")), _num(g.get("awayPoints"))
        if hp is None or ap is None:
            continue
        gid = g.get("id")
        if gid is None or int(gid) not in spreads:
            continue

        home = str(g.get("homeTeam") or "").strip().lower()
        away = str(g.get("awayTeam") or "").strip().lower()
        if home not in feats.index or away not in feats.index:
            continue

        h, a = feats.loc[home], feats.loc[away]
        # CFBD spread convention is home-negative when the home team is favoured.
        market = spreads[int(gid)]

        rows.append({
            "season": year,
            "week": g.get("week"),
            "game_id": int(gid),
            "home": home,
            "away": away,
            "margin": hp - ap,                      # actual home margin
            "market_home_margin": -market,          # market's implied home margin
            "neutral": 1.0 if g.get("neutralSite") else 0.0,
            "sp_diff": float(h["sp_z"] - a["sp_z"]),
            "srs_diff": float(h["srs_z"] - a["srs_z"]),
            "talent_diff": float(h["talent_z"] - a["talent_z"]),
            "returning_diff": float(h["returning_z"] - a["returning_z"]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation

BASE_FEATURES = ["market_home_margin"]
FUND_FEATURES = ["sp_diff", "srs_diff", "talent_diff", "returning_diff", "neutral"]


def ats_hit_rate(pred_margin, actual_margin, market_margin):
    """
    Would betting the side this model prefers have covered?
    Pushes are excluded. 52.4% is break-even at -110.
    """
    pick_home = pred_margin > market_margin
    covered_home = actual_margin > market_margin
    push = np.isclose(actual_margin, market_margin)
    live = ~push
    if live.sum() == 0:
        return np.nan, 0
    wins = (pick_home[live] == covered_home[live]).sum()
    return wins / live.sum(), int(live.sum())


def walk_forward(df, seasons):
    """Train on all prior seasons, test on the next. Never random folds."""
    results = []
    for i in range(1, len(seasons)):
        test_year = seasons[i]
        train_years = seasons[:i]
        tr = df[df["season"].isin(train_years)]
        te = df[df["season"] == test_year]
        if len(tr) < 200 or len(te) < 50:
            continue

        y_tr, y_te = tr["margin"].to_numpy(), te["margin"].to_numpy()
        mkt_te = te["market_home_margin"].to_numpy()

        # Baseline: market only.
        b = Ridge(alpha=1.0).fit(tr[BASE_FEATURES], y_tr)
        p_base = b.predict(te[BASE_FEATURES])

        # Candidate: market + fundamentals.
        cols = BASE_FEATURES + FUND_FEATURES
        c = Ridge(alpha=1.0).fit(tr[cols], y_tr)
        p_cand = c.predict(te[cols])

        base_hit, n_live = ats_hit_rate(p_base, y_te, mkt_te)
        cand_hit, _ = ats_hit_rate(p_cand, y_te, mkt_te)

        results.append({
            "Test season": test_year,
            "Games": len(te),
            "Market MAE": mean_absolute_error(y_te, p_base),
            "Model MAE": mean_absolute_error(y_te, p_cand),
            "Model ATS%": cand_hit,
            "Bets graded": n_live,
            **{f"w_{k}": v for k, v in zip(cols, c.coef_)},
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------- app

st.title("Model Lab")
st.caption(
    "Does adding fundamentals to the market spread improve out-of-sample "
    "prediction? Fit on past seasons, tested on the next one."
)

col1, col2 = st.columns(2)
with col1:
    first = st.number_input("First season", 2015, 2030, 2019, step=1)
with col2:
    last = st.number_input("Last season", 2015, 2030, 2025, step=1)

if st.button("Run walk-forward test", type="primary", use_container_width=True):
    seasons = list(range(int(first), int(last) + 1))
    frames = []
    prog = st.progress(0.0, text="Loading seasons…")
    for i, y in enumerate(seasons):
        try:
            frames.append(build_season(y))
        except Exception as e:
            st.warning(f"{y}: could not load ({e})")
        prog.progress((i + 1) / len(seasons), text=f"Loaded {y}")
    prog.empty()

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        st.error("No usable seasons loaded. Check the API key and season range.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    st.success(f"{len(df):,} completed games with market spreads.")

    res = walk_forward(df, seasons)
    if res.empty:
        st.error("Not enough data to walk forward. Widen the season range.")
        st.stop()

    st.subheader("Out-of-sample results")
    show = res[["Test season", "Games", "Market MAE", "Model MAE", "Model ATS%", "Bets graded"]].copy()
    show["Market MAE"] = show["Market MAE"].map("{:.2f}".format)
    show["Model MAE"] = show["Model MAE"].map("{:.2f}".format)
    show["Model ATS%"] = show["Model ATS%"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}")
    st.dataframe(show, use_container_width=True, hide_index=True)

    # --- verdict -----------------------------------------------------------
    mae_gain = (res["Market MAE"] - res["Model MAE"]).mean()
    ats = res["Model ATS%"].dropna()
    ats_mean = ats.mean() if len(ats) else float("nan")
    total_bets = int(res["Bets graded"].sum())

    st.subheader("Verdict")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg MAE improvement", f"{mae_gain:+.3f} pts",
              help="Points of error removed vs the market alone. Positive is better.")
    c2.metric("ATS win rate", "—" if math.isnan(ats_mean) else f"{ats_mean:.1%}",
              help="Break-even at -110 is 52.4%.")
    c3.metric("Games tested", f"{total_bets:,}")

    if math.isnan(ats_mean):
        st.warning("Could not compute an ATS rate.")
    elif ats_mean >= 0.534 and mae_gain > 0.05:
        st.success(
            "The fundamentals add out-of-sample value and clear break-even with "
            "margin. This is worth pursuing — but confirm it holds on a season "
            "you have never looked at before trusting it."
        )
    elif ats_mean >= 0.524:
        st.info(
            "Marginally above break-even. At this sample size that is not "
            "distinguishable from noise. Treat it as unproven, not as an edge."
        )
    else:
        st.error(
            "The fundamentals do not beat the market spread out of sample. "
            "This is the most common result and it is genuinely useful: it "
            "says the edge is not in public power ratings, so a more complex "
            "model built on the same inputs will not help."
        )

    st.subheader("Fitted weights by season")
    st.caption(
        "How much weight each input earned. A market weight near 1.0 is expected. "
        "Fundamental weights near 0, or flipping sign between seasons, mean the "
        "model is fitting noise rather than signal."
    )
    wcols = [c for c in res.columns if c.startswith("w_")]
    w = res[["Test season"] + wcols].copy()
    w.columns = ["Test season"] + [c[2:] for c in wcols]
    st.dataframe(w.round(3), use_container_width=True, hide_index=True)

    with st.expander("Important caveats", expanded=False):
        st.markdown(
            """
- **The spread used here is a consensus of quoted lines, not a timestamped
  closing line.** Real closing lines are sharper, so this test is if anything
  *generous* to the model.
- **Beating this test is necessary, not sufficient.** You still have to beat
  the price you can actually get, after vig, at the moment you bet.
- **Every input is public.** If fundamentals do add value, expect it to be
  small and concentrated in low-liquidity games, not marquee matchups.
- **Do not tune settings until this reports success.** Re-running with
  different ranges until it passes is how you fool yourself.
            """
        )
