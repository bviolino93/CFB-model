
import streamlit as st
import pandas as pd
import base64
import html
import streamlit.components.v1 as components
from datetime import date

# ===== Embedded CFB v0.2.0 model engine =====

import math
import requests
from statistics import NormalDist, mean, pstdev

BASE_URL = "https://api.collegefootballdata.com"
MODEL_VERSION = "0.2.1-MATCHUP-FIX"

DEFAULT_HFA = 2.5

# Distribution widths are intentionally wider early in the season.
BASE_MARGIN_SD = 15.8
BASE_TOTAL_SD = 12.8

def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}

def cfbd_get(path, api_key, params=None):
    r = requests.get(
        BASE_URL + path,
        headers=_headers(api_key),
        params=params or {},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def fetch_games(api_key, year):
    return cfbd_get("/games", api_key, {"year": year, "seasonType": "regular"})

def fetch_sp(api_key, year):
    return cfbd_get("/ratings/sp", api_key, {"year": year})

def fetch_srs(api_key, year):
    return cfbd_get("/ratings/srs", api_key, {"year": year})

def fetch_ppa(api_key, year):
    return cfbd_get(
        "/ppa/teams",
        api_key,
        {
            "year": year,
            "excludeGarbageTime": "true",
            "classification": "fbs",
        },
    )

def fetch_advanced(api_key, year):
    return cfbd_get(
        "/stats/season/advanced",
        api_key,
        {
            "year": year,
            "excludeGarbageTime": "true",
            "classification": "fbs",
        },
    )

def fetch_talent(api_key, year):
    return cfbd_get("/talent", api_key, {"year": year})

def fetch_returning(api_key, year):
    return cfbd_get("/player/returning", api_key, {"year": year})

def _safe_fetch(func, *args):
    try:
        return func(*args)
    except Exception:
        return []

def _num(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _team_map(rows):
    return {str(r.get("team")): r for r in rows or [] if r.get("team")}

def _values(mapping, getter, reverse=False):
    vals = []
    for row in mapping.values():
        v = getter(row)
        if v is not None:
            vals.append(-v if reverse else v)
    if not vals:
        return 0.0, 1.0
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 1.0
    return mu, sd if sd > 1e-9 else 1.0

def _z(value, mu, sd, cap=3.0):
    if value is None:
        return 0.0
    z = (value - mu) / sd
    return max(-cap, min(cap, z))

def _blend(cur, prev, current_weight):
    if cur is None and prev is None:
        return None
    if cur is None:
        return prev
    if prev is None:
        return cur
    return current_weight * cur + (1.0 - current_weight) * prev

def _current_weight_for_week(week):
    try:
        w = int(week)
    except Exception:
        w = 1
    if w <= 1:
        return 0.15
    if w == 2:
        return 0.35
    if w == 3:
        return 0.50
    if w == 4:
        return 0.65
    if w == 5:
        return 0.75
    if w == 6:
        return 0.85
    return 0.90

def _sp_fields(row):
    if not row:
        return {
            "rating": None, "offense": None, "defense": None,
            "special": None, "pace": None, "success_off": None,
            "explosive_off": None, "pass_off": None, "rush_off": None,
            "success_def": None, "explosive_def": None,
            "pass_def": None, "rush_def": None, "havoc_def": None,
        }
    off = row.get("offense") or {}
    deff = row.get("defense") or {}
    st = row.get("specialTeams") or {}
    havoc = deff.get("havoc") or {}
    return {
        "rating": _num(row.get("rating")),
        "offense": _num(off.get("rating")),
        "defense": _num(deff.get("rating")),
        "special": _num(st.get("rating")),
        "pace": _num(off.get("pace")),
        "success_off": _num(off.get("success")),
        "explosive_off": _num(off.get("explosiveness")),
        "pass_off": _num(off.get("passing")),
        "rush_off": _num(off.get("rushing")),
        "success_def": _num(deff.get("success")),
        "explosive_def": _num(deff.get("explosiveness")),
        "pass_def": _num(deff.get("passing")),
        "rush_def": _num(deff.get("rushing")),
        "havoc_def": _num(havoc.get("total")),
    }

def _ppa_fields(row):
    if not row:
        return {"off": None, "off_pass": None, "off_rush": None,
                "def": None, "def_pass": None, "def_rush": None}
    off = row.get("offense") or {}
    deff = row.get("defense") or {}
    return {
        "off": _num(off.get("overall")),
        "off_pass": _num(off.get("passing")),
        "off_rush": _num(off.get("rushing")),
        "def": _num(deff.get("overall")),
        "def_pass": _num(deff.get("passing")),
        "def_rush": _num(deff.get("rushing")),
    }

def _adv_fields(row):
    if not row:
        return {
            "off_success": None, "off_expl": None, "off_ppa": None,
            "off_pass_ppa": None, "off_rush_ppa": None,
            "off_ppo": None, "off_plays": None, "off_drives": None,
            "def_success": None, "def_expl": None, "def_ppa": None,
            "def_pass_ppa": None, "def_rush_ppa": None,
            "def_ppo": None, "def_havoc": None,
        }
    off = row.get("offense") or {}
    deff = row.get("defense") or {}
    off_pass = off.get("passingPlays") or {}
    off_rush = off.get("rushingPlays") or {}
    def_pass = deff.get("passingPlays") or {}
    def_rush = deff.get("rushingPlays") or {}
    def_havoc = deff.get("havoc") or {}
    return {
        "off_success": _num(off.get("successRate")),
        "off_expl": _num(off.get("explosiveness")),
        "off_ppa": _num(off.get("ppa")),
        "off_pass_ppa": _num(off_pass.get("ppa")),
        "off_rush_ppa": _num(off_rush.get("ppa")),
        "off_ppo": _num(off.get("pointsPerOpportunity")),
        "off_plays": _num(off.get("plays")),
        "off_drives": _num(off.get("drives")),
        "def_success": _num(deff.get("successRate")),
        "def_expl": _num(deff.get("explosiveness")),
        "def_ppa": _num(deff.get("ppa")),
        "def_pass_ppa": _num(def_pass.get("ppa")),
        "def_rush_ppa": _num(def_rush.get("ppa")),
        "def_ppo": _num(deff.get("pointsPerOpportunity")),
        "def_havoc": _num(def_havoc.get("total")),
    }

def load_model_data(api_key, year):
    """
    Bulk-loads the model inputs. Every endpoint has a graceful fallback so
    a missing/free-tier field does not crash the app.
    """
    cur_sp = _team_map(_safe_fetch(fetch_sp, api_key, year))
    prev_sp = _team_map(_safe_fetch(fetch_sp, api_key, year - 1))
    cur_srs = _team_map(_safe_fetch(fetch_srs, api_key, year))
    prev_srs = _team_map(_safe_fetch(fetch_srs, api_key, year - 1))
    cur_ppa = _team_map(_safe_fetch(fetch_ppa, api_key, year))
    prev_ppa = _team_map(_safe_fetch(fetch_ppa, api_key, year - 1))
    cur_adv = _team_map(_safe_fetch(fetch_advanced, api_key, year))
    prev_adv = _team_map(_safe_fetch(fetch_advanced, api_key, year - 1))
    talent = _team_map(_safe_fetch(fetch_talent, api_key, year))
    returning = _team_map(_safe_fetch(fetch_returning, api_key, year))

    # Pre-compute distribution stats used to place unlike metrics on common scales.
    srs_vals = [_num(x.get("rating")) for x in cur_srs.values()]
    srs_vals = [x for x in srs_vals if x is not None]
    if not srs_vals:
        srs_vals = [_num(x.get("rating")) for x in prev_srs.values()]
        srs_vals = [x for x in srs_vals if x is not None]
    srs_mu = mean(srs_vals) if srs_vals else 0.0
    srs_sd = pstdev(srs_vals) if len(srs_vals) > 1 else 1.0
    if srs_sd <= 1e-9:
        srs_sd = 1.0

    talent_vals = [_num(x.get("talent")) for x in talent.values()]
    talent_vals = [x for x in talent_vals if x is not None]
    talent_mu = mean(talent_vals) if talent_vals else 0.0
    talent_sd = pstdev(talent_vals) if len(talent_vals) > 1 else 1.0
    if talent_sd <= 1e-9:
        talent_sd = 1.0

    ret_vals = [_num(x.get("percentPPA")) for x in returning.values()]
    ret_vals = [x for x in ret_vals if x is not None]
    ret_mu = mean(ret_vals) if ret_vals else 0.0
    ret_sd = pstdev(ret_vals) if len(ret_vals) > 1 else 1.0
    if ret_sd <= 1e-9:
        ret_sd = 1.0

    return {
        "year": year,
        "sp_current": cur_sp,
        "sp_previous": prev_sp,
        "srs_current": cur_srs,
        "srs_previous": prev_srs,
        "ppa_current": cur_ppa,
        "ppa_previous": prev_ppa,
        "adv_current": cur_adv,
        "adv_previous": prev_adv,
        "talent": talent,
        "returning": returning,
        "stats": {
            "srs_mu": srs_mu, "srs_sd": srs_sd,
            "talent_mu": talent_mu, "talent_sd": talent_sd,
            "ret_mu": ret_mu, "ret_sd": ret_sd,
        },
    }

# Kept for backwards compatibility with earlier app versions.
def load_rating_maps(api_key, year):
    d = load_model_data(api_key, year)
    return d["sp_current"], d["sp_previous"]

def _team_snapshot(team, data, week):
    cw = _current_weight_for_week(week)

    cur_sp = _sp_fields(data["sp_current"].get(team))
    prev_sp = _sp_fields(data["sp_previous"].get(team))
    sp = {}
    for k in cur_sp:
        sp[k] = cur_sp[k] if cur_sp[k] is not None else prev_sp[k]

    cur_srs_row = data["srs_current"].get(team) or {}
    prev_srs_row = data["srs_previous"].get(team) or {}
    srs = _blend(_num(cur_srs_row.get("rating")), _num(prev_srs_row.get("rating")), cw)

    cur_ppa = _ppa_fields(data["ppa_current"].get(team))
    prev_ppa = _ppa_fields(data["ppa_previous"].get(team))
    ppa = {k: _blend(cur_ppa[k], prev_ppa[k], cw) for k in cur_ppa}

    cur_adv = _adv_fields(data["adv_current"].get(team))
    prev_adv = _adv_fields(data["adv_previous"].get(team))
    adv = {k: _blend(cur_adv[k], prev_adv[k], cw) for k in cur_adv}

    talent = _num((data["talent"].get(team) or {}).get("talent"))
    ret_row = data["returning"].get(team) or {}
    returning = _num(ret_row.get("percentPPA"))
    returning_pass = _num(ret_row.get("percentPassingPPA"))
    returning_usage = _num(ret_row.get("usage"))

    stats = data["stats"]
    talent_z = _z(talent, stats["talent_mu"], stats["talent_sd"])
    returning_z = _z(returning, stats["ret_mu"], stats["ret_sd"])

    completeness_items = [
        sp["rating"], sp["offense"], sp["defense"], srs,
        ppa["off"], ppa["def"], adv["off_success"], adv["def_success"],
        talent, returning
    ]
    completeness = sum(x is not None for x in completeness_items) / len(completeness_items)

    source = "Current-year SP+" if team in data["sp_current"] else (
        "Prior-year SP+ fallback" if team in data["sp_previous"] else "Average fallback"
    )

    return {
        "team": team,
        "rating": sp["rating"] or 0.0,
        "offense": sp["offense"] or 0.0,
        "defense": sp["defense"] or 0.0,
        "special": sp["special"] or 0.0,
        "pace": sp["pace"] or 0.0,
        "source": source,
        "sp": sp,
        "srs": srs,
        "ppa": ppa,
        "adv": adv,
        "talent": talent,
        "talent_z": talent_z,
        "returning": returning,
        "returning_pass": returning_pass,
        "returning_usage": returning_usage,
        "returning_z": returning_z,
        "current_data_weight": cw,
        "completeness": completeness,
    }

def _team_base_power(team, data, week):
    t = _team_snapshot(team, data, week)

    sp_rating = t["sp"]["rating"] if t["sp"]["rating"] is not None else 0.0
    srs_z = _z(t["srs"], data["stats"]["srs_mu"], data["stats"]["srs_sd"])
    # Convert standardized secondary ratings into modest point adjustments.
    srs_adj = 1.50 * srs_z
    talent_adj = 0.65 * t["talent_z"]
    return_adj = 0.65 * t["returning_z"]

    # SP+ remains the anchor. Secondary metrics can move a team only a few points.
    power = sp_rating + srs_adj + talent_adj + return_adj

    t["components"] = {
        "sp_anchor": sp_rating,
        "srs_adjustment": srs_adj,
        "talent_adjustment": talent_adj,
        "returning_adjustment": return_adj,
        "base_power": power,
    }
    return power, t

def _matchup_adjustment(off, deff):
    """
    Small game-specific adjustment from offense-vs-defense efficiency.
    Positive means advantage to the offense.
    """
    vals = []

    # PPA matchups. Defensive PPA allowed: lower is better, so offense minus defense.
    if off["ppa"]["off_pass"] is not None and deff["ppa"]["def_pass"] is not None:
        vals.append((off["ppa"]["off_pass"] - deff["ppa"]["def_pass"]) * 2.0)
    if off["ppa"]["off_rush"] is not None and deff["ppa"]["def_rush"] is not None:
        vals.append((off["ppa"]["off_rush"] - deff["ppa"]["def_rush"]) * 1.5)

    # Success/explosiveness are matchup checks rather than major standalone ratings.
    if off["adv"]["off_success"] is not None and deff["adv"]["def_success"] is not None:
        vals.append((off["adv"]["off_success"] - deff["adv"]["def_success"]) * 8.0)
    if off["adv"]["off_expl"] is not None and deff["adv"]["def_expl"] is not None:
        vals.append((off["adv"]["off_expl"] - deff["adv"]["def_expl"]) * 0.25)

    if not vals:
        return 0.0
    adj = sum(vals) / len(vals)
    return max(-2.0, min(2.0, adj))

def _pace_adjustment(a, h):
    """
    Estimate game pace from plays per drive when available.
    Intentionally capped because early-season samples are noisy.
    """
    paces = []
    for t in (a, h):
        plays = t["adv"].get("off_plays")
        drives = t["adv"].get("off_drives")
        if plays is not None and drives and drives > 0:
            paces.append(plays / drives)
    if not paces:
        # SP+ pace can be missing/zero in preseason; don't force a fake adjustment.
        raw = [x["sp"]["pace"] for x in (a, h) if x["sp"]["pace"] not in (None, 0)]
        if not raw:
            return 0.0
        avg = sum(raw) / len(raw)
        # SP pace is centered near 0 in some versions.
        return max(-2.0, min(2.0, avg * 0.08))

    avg_ppd = sum(paces) / len(paces)
    # Around 6 plays/drive is ordinary. This is deliberately conservative.
    return max(-2.5, min(2.5, (avg_ppd - 6.0) * 2.0))

def _total_from_sp(a, h):
    """
    SP+ offense and defense components are already expressed on a scoring-like
    scale. Cross the offense with the opponent defense rather than summing raw
    component ratings.
    """
    ao, ad = a["sp"]["offense"], a["sp"]["defense"]
    ho, hd = h["sp"]["offense"], h["sp"]["defense"]
    if None not in (ao, ad, ho, hd):
        away_pts = (ao + hd) / 2.0
        home_pts = (ho + ad) / 2.0
        return away_pts + home_pts

    # Safe fallback only when component ratings are missing.
    return 53.0

def _total_efficiency_adjustment(a, h):
    vals = []
    # High offensive PPA and permissive defensive PPA push totals upward.
    for off, opp in ((a, h), (h, a)):
        if off["ppa"]["off"] is not None and opp["ppa"]["def"] is not None:
            vals.append((off["ppa"]["off"] + opp["ppa"]["def"]) * 1.25)
        if off["adv"]["off_ppo"] is not None and opp["adv"]["def_ppo"] is not None:
            vals.append((off["adv"]["off_ppo"] + opp["adv"]["def_ppo"] - 8.0) * 0.35)
    if not vals:
        return 0.0
    return max(-4.0, min(4.0, sum(vals) / len(vals)))

def _uncertainty(week, away, home):
    try:
        w = int(week)
    except Exception:
        w = 1

    margin_sd = BASE_MARGIN_SD - min(2.0, max(0, w - 1) * 0.30)
    total_sd = BASE_TOTAL_SD - min(1.2, max(0, w - 1) * 0.18)

    completeness = (away["completeness"] + home["completeness"]) / 2
    if completeness < 0.70:
        margin_sd += 1.0
        total_sd += 0.8
    elif completeness < 0.85:
        margin_sd += 0.5
        total_sd += 0.4

    margin_sd = max(13.2, min(18.0, margin_sd))
    total_sd = max(11.2, min(15.0, total_sd))

    # Confidence is about data reliability, not probability of the pick winning.
    base_conf = 70 + min(8, max(0, w - 1))
    conf = int(round(base_conf + (completeness - 0.80) * 12))
    conf = max(62, min(84, conf))
    return margin_sd, total_sd, conf, completeness

def project_game(game, data_or_current, previous_map=None, hfa=DEFAULT_HFA):
    """
    v0.2 accepts the full model-data dictionary.
    For compatibility, if old SP maps are supplied it creates a SP-only shell.
    """
    if isinstance(data_or_current, dict) and "sp_current" in data_or_current:
        data = data_or_current
    else:
        data = {
            "year": game.get("season"),
            "sp_current": data_or_current or {},
            "sp_previous": previous_map or {},
            "srs_current": {}, "srs_previous": {},
            "ppa_current": {}, "ppa_previous": {},
            "adv_current": {}, "adv_previous": {},
            "talent": {}, "returning": {},
            "stats": {
                "srs_mu": 0.0, "srs_sd": 1.0,
                "talent_mu": 0.0, "talent_sd": 1.0,
                "ret_mu": 0.0, "ret_sd": 1.0,
            },
        }

    away, home = game["awayTeam"], game["homeTeam"]
    week = game.get("week", 1)
    neutral = bool(game.get("neutralSite"))
    applied_hfa = 0.0 if neutral else float(hfa)

    away_power, ar = _team_base_power(away, data, week)
    home_power, hr = _team_base_power(home, data, week)

    away_match = _matchup_adjustment(ar, hr)
    home_match = _matchup_adjustment(hr, ar)

    # Matchup differential moves the margin, capped by the individual matchup caps.
    matchup_margin_adj = home_match - away_match

    base_margin = home_power - away_power
    home_margin = base_margin + matchup_margin_adj + applied_hfa

    raw_sp_total = _total_from_sp(ar, hr)
    efficiency_adj = _total_efficiency_adjustment(ar, hr)
    pace_adj = _pace_adjustment(ar, hr)

    # Keep the total independent of the market.
    total = raw_sp_total + efficiency_adj + pace_adj
    total = max(34.0, min(82.0, total))

    # Reconcile the scoring split to the independently estimated margin.
    home_score = max(7.0, (total + home_margin) / 2.0)
    away_score = max(7.0, (total - home_margin) / 2.0)
    total = home_score + away_score
    home_margin = home_score - away_score

    margin_sd, total_sd, confidence, completeness = _uncertainty(week, ar, hr)
    home_wp = 1.0 - NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)

    components = {
        "base_power_margin": base_margin,
        "matchup_margin_adjustment": matchup_margin_adj,
        "hfa_adjustment": applied_hfa,
        "sp_total_base": raw_sp_total,
        "efficiency_total_adjustment": efficiency_adj,
        "pace_total_adjustment": pace_adj,
    }

    return {
        "away": away,
        "home": home,
        "away_rating": ar,
        "home_rating": hr,
        "home_margin": home_margin,
        "model_home_spread": -home_margin,
        "model_total": total,
        "away_score": away_score,
        "home_score": home_score,
        "home_win_prob": home_wp,
        "away_win_prob": 1 - home_wp,
        "neutral": neutral,
        "hfa": applied_hfa,
        "week": week,
        "margin_sd": margin_sd,
        "total_sd": total_sd,
        "confidence": confidence,
        "data_completeness": completeness,
        "components": components,
    }

def cover_probability(home_margin_mean, market_home_spread, side="home", sigma=None):
    sigma = float(sigma or BASE_MARGIN_SD)
    threshold = -float(market_home_spread)
    p_home = 1.0 - NormalDist(mu=home_margin_mean, sigma=sigma).cdf(threshold)
    return p_home if side == "home" else 1 - p_home

def total_probability(model_total, market_total, side="over", sigma=None):
    sigma = float(sigma or BASE_TOTAL_SD)
    p_over = 1.0 - NormalDist(mu=model_total, sigma=sigma).cdf(float(market_total))
    return p_over if side == "over" else 1 - p_over

def implied_prob(odds):
    odds = float(odds)
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)

def fair_ml(prob):
    p = max(.001, min(.999, float(prob)))
    return -round(100*p/(1-p)) if p >= .5 else round(100*(1-p)/p)

def expected_value(prob, odds):
    odds = float(odds)
    profit = odds/100 if odds > 0 else 100/abs(odds)
    return prob*profit - (1-prob)

def juice_thresholds(odds):
    odds = int(odds)
    if odds >= -149: return .025, .05
    if odds >= -179: return .03, .06
    if odds >= -199: return .04, .07
    if odds >= -249: return .05, .08
    return .07, .12

def grade(prob, odds, confidence=75):
    imp = implied_prob(odds)
    edge = prob - imp
    ev = expected_value(prob, odds)
    me, mv = juice_thresholds(odds)

    # Early/low-confidence model states can still show positive-EV leans,
    # but require >=70 confidence for a formal BET.
    if confidence >= 80 and edge >= me + .02 and ev >= mv + .03:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= me and ev >= mv:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"
    return verdict, edge, ev, imp

def fetch_lines(api_key, year=None, week=None, game_id=None, provider=None):
    params = {"seasonType": "regular"}
    if game_id is not None:
        params["gameId"] = int(game_id)
    elif year is not None:
        params["year"] = int(year)
        if week is not None:
            params["week"] = int(week)
    if provider:
        params["provider"] = provider
    return cfbd_get("/lines", api_key, params)

def _line_obj_values(obj):
    lines = obj.get("lines")
    if isinstance(lines, list):
        return lines
    if isinstance(lines, dict):
        return [lines]
    return []

def normalize_game_lines(rows, game_id=None):
    providers = []
    for row in rows or []:
        if game_id is not None:
            try:
                if int(row.get("id")) != int(game_id):
                    continue
            except Exception:
                pass

        away = row.get("awayTeam")
        home = row.get("homeTeam")
        for ln in _line_obj_values(row):
            provider = ln.get("provider") or "Unknown"
            spread = _num(ln.get("spread"))
            total = _num(ln.get("overUnder"))
            try:
                away_ml = int(ln.get("awayMoneyline")) if ln.get("awayMoneyline") is not None else None
            except Exception:
                away_ml = None
            try:
                home_ml = int(ln.get("homeMoneyline")) if ln.get("homeMoneyline") is not None else None
            except Exception:
                home_ml = None

            providers.append({
                "provider": str(provider),
                "away": away,
                "home": home,
                "home_spread": spread,
                "away_spread": -spread if spread is not None else None,
                "total": total,
                "away_ml": away_ml,
                "home_ml": home_ml,
            })
    return providers
# ===== End embedded model engine =====


st.set_page_config(page_title="CFB Model", page_icon="🏈", layout="centered")
st.title("🏈 CFB Model")
st.caption("Version 0.2.1-MATCHUP-FIX • SP+ anchor + matchup/efficiency/roster adjustments")

try:
    API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error("Missing CFBD_API_KEY in Streamlit Secrets.")
    st.stop()

@st.cache_data(ttl=1800)
def get_games(year):
    return fetch_games(API_KEY, year)

@st.cache_data(ttl=3600)
def get_model_data(year):
    return load_model_data(API_KEY, year)


@st.cache_data(ttl=300)
def get_market_lines(game_id, year):
    return fetch_lines(API_KEY, year=year, game_id=game_id)

def game_date_et(g):
    s=g.get("startDate")
    if not s: return None
    try: return pd.to_datetime(s, utc=True).tz_convert("America/New_York").date()
    except: return None


def ios_save_button(label, csv_text, filename):
    """
    Browser-native download button.
    Uses a Blob + download attribute instead of Streamlit's file response,
    which avoids iOS rendering CSV as a full-screen document preview.
    """
    payload = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    safe_label = html.escape(label)
    safe_filename = html.escape(filename, quote=True)

    components.html(
        f"""
        <div style="width:100%;padding:0;margin:0;">
          <button id="saveBtn"
            style="
              width:100%;
              min-height:44px;
              border:1px solid rgba(49,51,63,.2);
              border-radius:8px;
              background:white;
              color:rgb(49,51,63);
              font-size:16px;
              font-weight:600;
              cursor:pointer;
              padding:10px 14px;">
            {safe_label}
          </button>
        </div>
        <script>
        document.getElementById("saveBtn").addEventListener("click", function() {{
            const b64 = "{payload}";
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {{
                bytes[i] = binary.charCodeAt(i);
            }}

            // Deliberately use binary MIME so iOS doesn't Quick Look the CSV.
            const blob = new Blob([bytes], {{type: "application/octet-stream"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{safe_filename}";
            a.rel = "noopener";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 5000);
        }});
        </script>
        """,
        height=58,
    )

selected_date = st.date_input("Game date", value=date.today())
year = selected_date.year

try:
    games=get_games(year)
except Exception as e:
    st.error(f"CFBD games request failed: {e}")
    st.stop()

daily_all=[g for g in games if game_date_et(g)==selected_date]

MAJOR_CONFERENCES = {
    "ACC", "SEC", "Big Ten", "Big 12", "Pac-12"
}
MAJOR_INDEPENDENTS = {"Notre Dame"}

def _classification(g, side):
    return str(g.get(f"{side}Classification") or "").lower()

def _conference(g, side):
    return str(g.get(f"{side}Conference") or "")

def _team(g, side):
    return str(g.get(f"{side}Team") or "")

def is_fbs_team(g, side):
    c = _classification(g, side)
    if c:
        return c == "fbs"
    # Fallback for payloads where classification is missing.
    return bool(_conference(g, side))

def is_major_team(g, side):
    return (
        _conference(g, side) in MAJOR_CONFERENCES
        or _team(g, side) in MAJOR_INDEPENDENTS
    )

slate_filter = st.selectbox(
    "Game level",
    ["Major FBS", "All FBS", "All college games"],
    index=0,
    help=(
        "Major FBS = Power-conference teams plus Notre Dame. "
        "All FBS removes FCS-vs-FCS games. All college games shows everything returned by CFBD."
    ),
)

if slate_filter == "Major FBS":
    # Keep games involving at least one major-program team, but require the opponent
    # to be FBS unless the major team is hosting an FCS tune-up.
    daily = [
        g for g in daily_all
        if is_major_team(g, "home") or is_major_team(g, "away")
    ]
elif slate_filter == "All FBS":
    daily = [
        g for g in daily_all
        if is_fbs_team(g, "home") or is_fbs_team(g, "away")
    ]
else:
    daily = daily_all

if not daily:
    st.warning("No games found for that date with the selected game-level filter.")
    st.stop()

st.caption(f"Showing {len(daily)} of {len(daily_all)} games on {selected_date:%b %d}.")


def kickoff_et(g):
    s = g.get("startDate")
    if not s:
        return None
    try:
        return pd.to_datetime(s, utc=True).tz_convert("America/New_York")
    except Exception:
        return None

def slate_bucket(g):
    k = kickoff_et(g)
    if k is None:
        return "Unknown"
    mins = k.hour * 60 + k.minute
    if mins < 15 * 60 + 30:
        return "Early"
    if mins < 19 * 60:
        return "Midday"
    return "Night"

def consensus_line(rows):
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    def med(col):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.median())

    away_ml = med("away_ml")
    home_ml = med("home_ml")
    home_spread = med("home_spread")
    total = med("total")

    return {
        "provider": "Consensus median",
        "away_ml": int(round(away_ml)) if away_ml is not None else None,
        "home_ml": int(round(home_ml)) if home_ml is not None else None,
        "home_spread": home_spread,
        "away_spread": -home_spread if home_spread is not None else None,
        "total": total,
    }

run_mode = st.radio(
    "Run mode",
    ["Single Game", "Slate"],
    horizontal=True,
    index=0,
)

if run_mode == "Slate":
    slate_choice = st.selectbox(
        "Slate",
        ["Early", "Midday", "Night", "All Day"],
        index=0,
        help=(
            "Early = before 3:30 PM ET • Midday = 3:30 PM–6:59 PM ET • "
            "Night = 7:00 PM ET or later."
        ),
    )

    if slate_choice == "All Day":
        slate_games = list(daily)
    else:
        slate_games = [g for g in daily if slate_bucket(g) == slate_choice]

    slate_games = sorted(
        slate_games,
        key=lambda g: kickoff_et(g) if kickoff_et(g) is not None else pd.Timestamp.max.tz_localize("UTC"),
    )

    st.caption(f"{len(slate_games)} games in the {slate_choice} slate.")

    if not slate_games:
        st.warning("No games are in this slate with the current game-level filter.")
        st.stop()

    include_lines = st.checkbox(
        "Include generic market lines",
        value=True,
        help="Uses CFBD market lines and builds a consensus median across available providers."
    )

    if st.button("Run Slate", type="primary", use_container_width=True):
        try:
            model_data_s = get_model_data(year)
        except Exception as e:
            st.error(f"CFBD model-data request failed: {e}")
            st.stop()

        # Efficiently pull all line data by unique week rather than one request per game.
        line_cache = {}
        if include_lines:
            weeks = sorted({g.get("week") for g in slate_games if g.get("week") is not None})
            for wk in weeks:
                try:
                    raw = fetch_lines(API_KEY, year=year, week=int(wk))
                    line_cache[int(wk)] = raw
                except Exception:
                    line_cache[int(wk)] = []

        slate_rows = []

        for g in slate_games:
            gp = project_game(g, model_data_s, hfa=2.5)
            k = kickoff_et(g)

            market = {}
            provider_rows = []
            if include_lines and g.get("week") is not None:
                provider_rows = normalize_game_lines(
                    line_cache.get(int(g.get("week")), []),
                    game_id=g.get("id")
                )
                market = consensus_line(provider_rows)

            best_verdict = "NO LINE"
            best_market = ""
            best_odds = None
            best_edge = None
            best_ev = None

            candidates = []

            if market.get("away_ml") is not None:
                v,e,ev,_ = grade(gp["away_win_prob"], market["away_ml"], gp["confidence"])
                candidates.append((v, f"{gp['away']} ML", market["away_ml"], e, ev))
            if market.get("home_ml") is not None:
                v,e,ev,_ = grade(gp["home_win_prob"], market["home_ml"], gp["confidence"])
                candidates.append((v, f"{gp['home']} ML", market["home_ml"], e, ev))

            if market.get("home_spread") is not None:
                hp = cover_probability(gp["home_margin"], market["home_spread"], "home", gp["margin_sd"])
                ap = 1 - hp
                v,e,ev,_ = grade(hp, -110, gp["confidence"])
                candidates.append((v, f"{gp['home']} {market['home_spread']:+.1f}", -110, e, ev))
                v,e,ev,_ = grade(ap, -110, gp["confidence"])
                candidates.append((v, f"{gp['away']} {-market['home_spread']:+.1f}", -110, e, ev))

            if market.get("total") is not None:
                op = total_probability(gp["model_total"], market["total"], "over", gp["total_sd"])
                up = 1 - op
                v,e,ev,_ = grade(op, -110, gp["confidence"])
                candidates.append((v, f"Over {market['total']:g}", -110, e, ev))
                v,e,ev,_ = grade(up, -110, gp["confidence"])
                candidates.append((v, f"Under {market['total']:g}", -110, e, ev))

            if candidates:
                rank = {"STRONG BET":3, "BET":2, "LEAN":1, "PASS":0}
                candidates.sort(key=lambda x:(rank.get(x[0], -1), x[4]), reverse=True)
                b = candidates[0]
                best_verdict, best_market, best_odds, best_edge, best_ev = b

            slate_rows.append({
                "model_version": "0.2.1-MATCHUP-FIX",
                "game_date": str(selected_date),
                "slate": slate_choice,
                "kickoff_et": k.strftime("%I:%M %p") if k is not None else "",
                "game_id": g.get("id"),
                "away_team": gp["away"],
                "home_team": gp["home"],
                "neutral_site": gp["neutral"],
                "away_rating_source": gp["away_rating"]["source"],
                "home_rating_source": gp["home_rating"]["source"],
                "away_sp_rating": gp["away_rating"]["rating"],
                "home_sp_rating": gp["home_rating"]["rating"],
                "projected_away_score": round(gp["away_score"], 2),
                "projected_home_score": round(gp["home_score"], 2),
                "model_home_spread": round(gp["model_home_spread"], 2),
                "model_total": round(gp["model_total"], 2),
                "away_win_prob": round(gp["away_win_prob"], 6),
                "home_win_prob": round(gp["home_win_prob"], 6),
                "model_confidence": gp["confidence"],
                "margin_sd": round(gp["margin_sd"], 3),
                "total_sd": round(gp["total_sd"], 3),
                "data_completeness": round(gp["data_completeness"], 4),
                "base_power_margin": round(gp["components"]["base_power_margin"], 4),
                "matchup_margin_adjustment": round(gp["components"]["matchup_margin_adjustment"], 4),
                "sp_total_base": round(gp["components"]["sp_total_base"], 4),
                "efficiency_total_adjustment": round(gp["components"]["efficiency_total_adjustment"], 4),
                "pace_total_adjustment": round(gp["components"]["pace_total_adjustment"], 4),
                "market_source": market.get("provider"),
                "market_away_ml": market.get("away_ml"),
                "market_home_ml": market.get("home_ml"),
                "market_home_spread": market.get("home_spread"),
                "market_total": market.get("total"),
                "best_verdict": best_verdict,
                "best_market": best_market,
                "best_odds": best_odds,
                "best_edge": round(best_edge, 6) if best_edge is not None else None,
                "best_ev": round(best_ev, 6) if best_ev is not None else None,
            })

        slate_df = pd.DataFrame(slate_rows)

        st.subheader(f"{slate_choice} Slate Results")
        display_cols = [
            "kickoff_et","away_team","home_team","model_home_spread","model_total",
            "market_home_spread","market_total","best_verdict","best_market","best_edge","best_ev"
        ]
        st.dataframe(slate_df[display_cols], use_container_width=True, hide_index=True)

        actionable = slate_df[slate_df["best_verdict"].isin(["BET","STRONG BET"])]
        if len(actionable):
            st.success(f"{len(actionable)} game(s) have a BET or STRONG BET call.")
        else:
            st.info("No games in this slate currently clear the BET threshold.")

        ios_save_button(
            f"Save {slate_choice} Slate CSV",
            slate_df.to_csv(index=False),
            f"cfb_v020_{selected_date}_{slate_choice.lower().replace(' ','_')}_slate.csv",
        )

        st.caption(
            "Slate lines use a median across available CFBD providers. "
            "Spread and total pricing are assumed at -110 in slate mode unless actual ML prices are available."
        )

    st.stop()

labels={}
for g in daily:
    label=f"{g.get('awayTeam','Away')} @ {g.get('homeTeam','Home')}"
    if g.get("neutralSite"): label += " (Neutral)"
    labels[label]=g

game=labels[st.selectbox("Game", list(labels.keys()))]

try:
    model_data = get_model_data(year)
except Exception as e:
    st.error(f"CFBD model-data request failed: {e}")
    st.stop()

hfa=st.number_input("Home-field advantage", min_value=0.0, max_value=6.0, value=2.5, step=.25, disabled=bool(game.get("neutralSite")))
p=project_game(game,model_data,hfa=hfa)

st.subheader("Model projection")
a,b,c=st.columns(3)
a.metric(p["away"],f"{p['away_score']:.1f}")
b.metric(p["home"],f"{p['home_score']:.1f}")
c.metric("Total",f"{p['model_total']:.1f}")
st.write(f"**Model spread:** {p['home']} {p['model_home_spread']:+.1f}")
st.write(f"**Win probability:** {p['home']} {p['home_win_prob']*100:.1f}% / {p['away']} {p['away_win_prob']*100:.1f}%")
st.caption(f"{p['away']} source: {p['away_rating']['source']} • {p['home']} source: {p['home_rating']['source']}")

d1,d2,d3=st.columns(3)
d1.metric("Model confidence", f"{p['confidence']}/100")
d2.metric("Margin σ", f"{p['margin_sd']:.1f}")
d3.metric("Total σ", f"{p['total_sd']:.1f}")

with st.expander("Projection components"):
    c = p["components"]
    st.write(f"Base power margin: {c['base_power_margin']:+.2f}")
    st.write(f"Matchup adjustment: {c['matchup_margin_adjustment']:+.2f}")
    st.write(f"HFA adjustment: {c['hfa_adjustment']:+.2f}")
    st.write(f"SP+ matchup total base: {c['sp_total_base']:.2f}")
    st.write(f"Efficiency total adjustment: {c['efficiency_total_adjustment']:+.2f}")
    st.write(f"Pace total adjustment: {c['pace_total_adjustment']:+.2f}")


def build_export_row(p, game, selected_date, market=None):
    market = market or {}
    row = {
        "model_version": "0.2.1-MATCHUP-FIX",
        "game_date": str(selected_date),
        "game_id": game.get("id"),
        "away_team": p["away"],
        "home_team": p["home"],
        "neutral_site": p["neutral"],
        "hfa_points": p["hfa"],

        "away_rating_source": p["away_rating"]["source"],
        "home_rating_source": p["home_rating"]["source"],
        "away_sp_rating": p["away_rating"]["rating"],
        "home_sp_rating": p["home_rating"]["rating"],
        "away_sp_offense": p["away_rating"]["offense"],
        "home_sp_offense": p["home_rating"]["offense"],
        "away_sp_defense": p["away_rating"]["defense"],
        "home_sp_defense": p["home_rating"]["defense"],
        "away_sp_special_teams": p["away_rating"]["special"],
        "home_sp_special_teams": p["home_rating"]["special"],
        "away_sp_pace": p["away_rating"]["pace"],
        "home_sp_pace": p["home_rating"]["pace"],

        "projected_away_score": round(p["away_score"], 3),
        "projected_home_score": round(p["home_score"], 3),
        "projected_total": round(p["model_total"], 3),
        "projected_home_margin": round(p["home_margin"], 3),
        "model_home_spread": round(p["model_home_spread"], 3),
        "away_win_prob": round(p["away_win_prob"], 6),
        "home_win_prob": round(p["home_win_prob"], 6),
        "away_fair_ml": fair_ml(p["away_win_prob"]),
        "home_fair_ml": fair_ml(p["home_win_prob"]),

        "week": p["week"],
        "model_confidence": p["confidence"],
        "margin_sd": round(p["margin_sd"], 3),
        "total_sd": round(p["total_sd"], 3),
        "data_completeness": round(p["data_completeness"], 4),

        "base_power_margin": round(p["components"]["base_power_margin"], 4),
        "matchup_margin_adjustment": round(p["components"]["matchup_margin_adjustment"], 4),
        "sp_total_base": round(p["components"]["sp_total_base"], 4),
        "efficiency_total_adjustment": round(p["components"]["efficiency_total_adjustment"], 4),
        "pace_total_adjustment": round(p["components"]["pace_total_adjustment"], 4),

        "away_srs": p["away_rating"].get("srs"),
        "home_srs": p["home_rating"].get("srs"),
        "away_talent": p["away_rating"].get("talent"),
        "home_talent": p["home_rating"].get("talent"),
        "away_returning_ppa_pct": p["away_rating"].get("returning"),
        "home_returning_ppa_pct": p["home_rating"].get("returning"),
        "away_current_data_weight": p["away_rating"].get("current_data_weight"),
        "home_current_data_weight": p["home_rating"].get("current_data_weight"),
        "away_ppa_off": p["away_rating"].get("ppa",{}).get("off"),
        "home_ppa_off": p["home_rating"].get("ppa",{}).get("off"),
        "away_ppa_def": p["away_rating"].get("ppa",{}).get("def"),
        "home_ppa_def": p["home_rating"].get("ppa",{}).get("def"),
        "away_success_rate": p["away_rating"].get("adv",{}).get("off_success"),
        "home_success_rate": p["home_rating"].get("adv",{}).get("off_success"),
        "away_def_success_rate": p["away_rating"].get("adv",{}).get("def_success"),
        "home_def_success_rate": p["home_rating"].get("adv",{}).get("def_success"),
    }

    row.update(market)
    return row

st.divider()
st.subheader("Sportsbook lines")

st.caption(
    "Pull a generic market line automatically from CFBD, then edit anything that differs from your book."
)

line_rows = []
providers = []
game_id = game.get("id")

if st.button("Pull Market Lines", use_container_width=True):
    try:
        raw_lines = get_market_lines(game_id, year)
        line_rows = normalize_game_lines(raw_lines, game_id=game_id)
        st.session_state["cfb_line_rows"] = line_rows
    except Exception as e:
        st.error(f"Line pull failed: {e}")

line_rows = st.session_state.get("cfb_line_rows", [])

selected_line = {}
if line_rows:
    provider_names = sorted({x["provider"] for x in line_rows})
    provider = st.selectbox("Line source", provider_names)

    matches = [x for x in line_rows if x["provider"] == provider]
    selected_line = matches[0] if matches else line_rows[0]

    pulled_bits = []
    if selected_line.get("away_ml") is not None:
        pulled_bits.append(f"{p['away']} ML {selected_line['away_ml']:+d}")
    if selected_line.get("home_ml") is not None:
        pulled_bits.append(f"{p['home']} ML {selected_line['home_ml']:+d}")
    if selected_line.get("home_spread") is not None:
        pulled_bits.append(f"{p['home']} {selected_line['home_spread']:+.1f}")
    if selected_line.get("total") is not None:
        pulled_bits.append(f"Total {selected_line['total']:g}")

    if pulled_bits:
        st.success("Pulled: " + " • ".join(pulled_bits))
    else:
        st.warning("A line source was returned, but the main values were blank.")

    with st.expander("Available providers"):
        st.dataframe(pd.DataFrame(line_rows), use_container_width=True, hide_index=True)
else:
    st.info("Tap **Pull Market Lines** to populate the fields automatically.")

st.caption("All pulled values remain editable. Spread/total prices default to -110 because CFBD's generic line feed may not include side-specific juice.")

default_home_spread = float(
    selected_line.get("home_spread")
    if selected_line.get("home_spread") is not None
    else round(p["model_home_spread"]*2)/2
)
default_away_spread = float(
    selected_line.get("away_spread")
    if selected_line.get("away_spread") is not None
    else -default_home_spread
)
default_total = float(
    selected_line.get("total")
    if selected_line.get("total") is not None
    else round(p["model_total"]*2)/2
)

m1,m2=st.columns(2)
away_ml=m1.number_input(
    f"{p['away']} ML",
    value=int(selected_line.get("away_ml") if selected_line.get("away_ml") is not None else 100),
    step=5
)
home_ml=m2.number_input(
    f"{p['home']} ML",
    value=int(selected_line.get("home_ml") if selected_line.get("home_ml") is not None else -110),
    step=5
)

s1,s2=st.columns(2)
home_spread=s1.number_input(f"{p['home']} spread", value=default_home_spread, step=.5)
home_spread_odds=s2.number_input("Home spread odds", value=-110, step=5)

s3,s4=st.columns(2)
away_spread=s3.number_input(f"{p['away']} spread", value=default_away_spread, step=.5)
away_spread_odds=s4.number_input("Away spread odds", value=-110, step=5)

t1,t2,t3=st.columns(3)
market_total=t1.number_input("Total", value=default_total, step=.5)
over_odds=t2.number_input("Over odds", value=-110, step=5)
under_odds=t3.number_input("Under odds", value=-110, step=5)


projection_only_df = pd.DataFrame([build_export_row(p, game, selected_date)])
ios_save_button(
    "Save Projection CSV",
    projection_only_df.to_csv(index=False),
    f"cfb_projection_v020_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
)
st.caption("Use this to save the projection file for audit/upload.")

if st.button("Should I Bet?",type="primary",use_container_width=True):
    markets=[]
    for name,prob,odds in [
        (f"{p['away']} ML",p["away_win_prob"],away_ml),
        (f"{p['home']} ML",p["home_win_prob"],home_ml)
    ]:
        v,e,ev,imp=grade(prob,odds,p["confidence"]); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    hc=cover_probability(p["home_margin"],home_spread,"home",p["margin_sd"])
    ac=cover_probability(p["home_margin"],home_spread,"away",p["margin_sd"])
    for name,prob,odds in [
        (f"{p['home']} {home_spread:+.1f}",hc,home_spread_odds),
        (f"{p['away']} {away_spread:+.1f}",ac,away_spread_odds)
    ]:
        v,e,ev,imp=grade(prob,odds,p["confidence"]); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    op=total_probability(p["model_total"],market_total,"over",p["total_sd"])
    up=1-op
    for name,prob,odds in [(f"Over {market_total:g}",op,over_odds),(f"Under {market_total:g}",up,under_odds)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"]); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    rank={"STRONG BET":3,"BET":2,"LEAN":1,"PASS":0}
    markets.sort(key=lambda x:(rank[x[0]],x[5]),reverse=True)
    best=markets[0]
    if best[0] in {"BET","STRONG BET"}:
        st.success(f"🟢 **{best[0]}: {best[1]} {int(best[2]):+d}**\n\nModel {best[3]*100:.1f}% • Edge {best[4]*100:+.1f}% • EV {best[5]*100:+.1f}%")
    elif best[0]=="LEAN":
        st.warning(f"🟡 **LEAN: {best[1]} {int(best[2]):+d}**")
    else:
        st.info("⚪ **PASS — no market clears the threshold.**")

    st.subheader("Every market")
    for v,name,odds,prob,e,ev,fair in markets:
        icon="🟢" if v in {"BET","STRONG BET"} else ("🟡" if v=="LEAN" else "⚪")
        st.markdown(f"**{icon} {v} — {name} {int(odds):+d}**  \nModel {prob*100:.1f}% • Edge {e*100:+.1f}% • EV {ev*100:+.1f}% • Fair {int(fair):+d}")

    market_export = {
        "line_provider": selected_line.get("provider") if selected_line else None,
        "sportsbook_away_ml": away_ml,
        "sportsbook_home_ml": home_ml,
        "sportsbook_away_spread": away_spread,
        "sportsbook_away_spread_odds": away_spread_odds,
        "sportsbook_home_spread": home_spread,
        "sportsbook_home_spread_odds": home_spread_odds,
        "sportsbook_total": market_total,
        "sportsbook_over_odds": over_odds,
        "sportsbook_under_odds": under_odds,
        "model_away_cover_prob": round(ac, 6),
        "model_home_cover_prob": round(hc, 6),
        "model_over_prob": round(op, 6),
        "model_under_prob": round(up, 6),
        "best_verdict": best[0],
        "best_market": best[1],
        "best_odds": best[2],
        "best_model_prob": round(best[3], 6),
        "best_edge": round(best[4], 6),
        "best_ev": round(best[5], 6),
        "best_fair_line": best[6],
    }

    export_row = build_export_row(p, game, selected_date, market_export)
    export_df = pd.DataFrame([export_row])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.subheader("Export model check")
    st.caption("Download this CSV and upload it back into ChatGPT so the inputs, projection, market comparison, and betting call can be audited.")
    ios_save_button(
        "Save Game CSV",
        export_df.to_csv(index=False),
        f"cfb_model_v020_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
    )

st.divider()
st.caption("v0.2.1 keeps SP+ as the anchor, adds SRS/talent/returning-production and matchup efficiency, rebuilds totals from offense-vs-defense components, and widens uncertainty early in the season. It still needs backtesting/calibration before production betting.")
