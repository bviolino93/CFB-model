
import math
import requests
from statistics import NormalDist, mean, pstdev

BASE_URL = "https://api.collegefootballdata.com"
MODEL_VERSION = "1.6.2-TRACKER-INIT-HOTFIX"

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
    home_wp = NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)

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


# ===== v0.4 residual-market architecture =====
# The Streamlit app trains the residual coefficients point-in-time on historical
# seasons. This helper applies a supplied correction to the sportsbook baseline.

RESIDUAL_SPREAD_CAP = 4.0
RESIDUAL_TOTAL_CAP = 3.0

def apply_market_residual(market_home_spread=None, market_total=None,
                          spread_correction=0.0, total_correction=0.0):
    spread_correction = max(-RESIDUAL_SPREAD_CAP, min(RESIDUAL_SPREAD_CAP, float(spread_correction)))
    total_correction = max(-RESIDUAL_TOTAL_CAP, min(RESIDUAL_TOTAL_CAP, float(total_correction)))
    out = {
        "spread_correction": spread_correction,
        "total_correction": total_correction,
    }
    if market_home_spread is not None:
        market_margin = -float(market_home_spread)
        out["fair_home_spread"] = -(market_margin + spread_correction)
    if market_total is not None:
        out["fair_total"] = float(market_total) + total_correction
    return out


# v0.4.1 feature-audit note:
# Production residual corrections remain unchanged in model.py. The app's
# historical audit tests interpretable football feature groups strictly
# walk-forward before any future promotion into the live residual layer.
FEATURE_AUDIT_VERSION = "v2.6.0-feature-audit"


# v0.4.2 sparse-model research note:
# The live model remains on the prior market-baseline residual architecture.
# app.py compares locked sparse candidates out-of-sample before any promotion.
SPARSE_BAKEOFF_VERSION = "v2.7.0-sparse-bakeoff"


# v0.4.3 situational-discovery research note:
# Live logic remains unchanged. app.py tests predeclared spread situations and
# same-game matched samples before any future promotion.
SITUATIONAL_DISCOVERY_VERSION = "v2.8.0-situational-discovery"


# v2.8.1 UX note:
# Validation output can now be downloaded as a single ZIP bundle from app.py.
DOWNLOAD_BUNDLE_VERSION = "v2.8.1-download-bundle"


# v1.0.0 point-in-time rebuild note:
# app.py now contains the research-grade v3 engine using weekly advanced stats
# only through the prior week, pregame Elo, and richer preseason priors.
# Live production promotion remains gated on out-of-sample validation.
POINT_IN_TIME_REBUILD_VERSION = "v3.0.0-point-in-time"


# v1.1.0 nonlinear ML bake-off note:
# app.py now compares regularized linear models with tree/boosting models
# for market-residual regression and direct ATS / O-U classification.
# Live production logic remains unchanged pending out-of-sample confirmation.
NONLINEAR_ML_BAKEOFF_VERSION = "v3.1.0-nonlinear-ml-bakeoff"


# v1.2.0 signal-stability / ensemble note:
# v3.2 audits v3.1 probabilities for fixed holdout betting evidence,
# monotonic calibration, situational stability and cross-model agreement.
# No live production promotion occurs automatically.
SIGNAL_STABILITY_ENSEMBLE_VERSION = "v3.2.0-signal-stability-ensemble"


# v1.3.0 game-day selector note:
# v3.3 ranks each weekly slate cross-sectionally and evaluates selective
# top-N/top-percentile betting workflows with ROI, drawdown, streaks,
# season stability and a locked holdout gate.
GAMEDAY_SELECTOR_VERSION = "v3.3.0-gameday-selector"


# v1.4.0 slate-aware finalist note:
# v3.4 simulates Early / Midday / Late game-day slates, compares a small fixed
# set of ranking architectures using 2022-2024 only, locks the development
# winner, and then evaluates that locked winner on the holdout.
SLATE_AWARE_FINALIST_VERSION = "v3.4.0-slate-aware-finalist"


# v1.5.0 adaptive daily card note:
# v3.5 ranks every game on the calendar day, applies fixed quality thresholds,
# and uses day size only for presentation grouping. Small days can produce
# zero bets; large days can produce several, but the qualification bar never moves.
ADAPTIVE_DAILY_CARD_VERSION = "v3.5.0-adaptive-daily-card"


# v1.5.1 date-cache hotfix:
# If an older cached point-in-time history frame lacks calendar date/kickoff
# fields, v3.5.1 rehydrates those fields from the historical CFBD game schedule
# by game_id before constructing the daily card.
ADAPTIVE_DAILY_CARD_HOTFIX_VERSION = "v3.5.1-date-cache-hotfix"


# v1.5.2 cache-rebuild hotfix:
# Streamlit parent caches can survive helper changes. v3.5.2 explicitly clears
# the parent v3.1 ML/history cache before a daily-card validation run, forcing
# the historical feature frame to be rebuilt with calendar kickoff fields.
ADAPTIVE_DAILY_CARD_CACHE_REBUILD_VERSION = "v3.5.2-cache-rebuild"


# v1.5.3 bridge hotfix:
# Daily-card construction now starts from the proven v3.3 selector frame
# (_v33_rank_frame), attaches calendar kickoff by game_id, then ranks within day.
# This removes the separate v3.4 base merge that yielded zero daily rows.
ADAPTIVE_DAILY_CARD_BRIDGE_VERSION = "v3.5.3-v33-bridge"


# v1.5.4 robust daily-card fallback:
# v3.5.4 uses the exact v3.3 selector frame as the sole upstream card source,
# removes forced upstream cache clearing, and emits stage-level diagnostics:
# history -> predictions -> spread classification -> v3.3 -> calendar join.
ADAPTIVE_DAILY_CARD_ROBUST_VERSION = "v3.5.4-robust-fallback"


# v1.5.5 slate-window compatibility hotfix:
# Shared drawdown metrics expected a slate_window/slate_rank schema from v3.4.
# v3.5 uses display_group/day_rank.  v3.5.5 adds aliases and makes the shared
# helper schema-tolerant.
ADAPTIVE_DAILY_CARD_SLATE_WINDOW_HOTFIX = "v3.5.5"


# v1.6.0 production daily card:
# Historical research is locked into a practical live selector.
# All games on a calendar day are ranked together using the v3.3/v3.5
# Balanced Ensemble. Spread scores >=0.84 are official 1u plays.
# The highest-scoring official play is labeled Best Bet for presentation only;
# it does not receive a larger stake.
PRODUCTION_DAILY_CARD_VERSION = "v3.6.0-production-daily-card"
PRODUCTION_DAILY_SCORE_FLOOR = 0.84


# v1.6.1 automatic official performance tracker
# The live app freezes every official >=0.84 spread recommendation before kickoff,
# auto-grades ATS results after final scores post, and tracks units/ROI permanently.
PERFORMANCE_TRACKER_VERSION = "v3.6.1-auto-performance-tracker"


# v1.6.2 tracker initialization hotfix
# No model logic, scoring, threshold, or staking changes.
# Fixes Streamlit startup NameError caused by tracker-path definition order.
TRACKER_INIT_HOTFIX_VERSION = "v3.6.2"
