
import streamlit as st
import pandas as pd
import numpy as np
import base64
import html
import json
import io
import zipfile
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
import re
import streamlit.components.v1 as components
from datetime import date
from pathlib import Path

# ===== Embedded CFB v0.2.0 model engine =====

import math
import requests
from statistics import NormalDist, mean, pstdev
from functools import lru_cache
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api.collegefootballdata.com"
MODEL_VERSION = "4.4.0-REAL-TEAM-LOGOS"

# Fully enclosed/domed stadiums. Outdoor weather adjustments are suppressed here.
ENCLOSED_VENUES = {
    "allegiant stadium",
    "mercedes-benz stadium",
    "ford field",
    "lucas oil stadium",
    "u.s. bank stadium",
    "us bank stadium",
    "caesars superdome",
    "the dome at america's center",
    "alamodome",
    "jma wireless dome",
    "carrier dome",
}

def is_enclosed_venue(venue_name):
    if not venue_name:
        return False
    v = str(venue_name).strip().lower()
    return any(name in v for name in ENCLOSED_VENUES)


DEFAULT_HFA = 2.5

# Distribution widths are intentionally wider early in the season.
BASE_MARGIN_SD = 15.8
BASE_TOTAL_SD = 12.8

# v0.3.2 calibration guardrails
# These are deliberately conservative and should be re-estimated after a larger
# historical sample. They change the betting layer, not the underlying football ratings.
EARLY_SIDE_SHRINK = {1: 0.55, 2: 0.65, 3: 0.75, 4: 0.85}
EARLY_TOTAL_SHRINK = {1: 0.40, 2: 0.50, 3: 0.60, 4: 0.75}

def _week_num(week):
    try:
        return max(1, int(week))
    except Exception:
        return 1

def _shrink_weight(week, market_type):
    w = _week_num(week)
    table = EARLY_TOTAL_SHRINK if market_type == "total" else EARLY_SIDE_SHRINK
    return table.get(w, 1.0)

def calibrated_market_projection(raw_value, market_value, week, market_type):
    """
    Pull an early-season model projection toward the observed market.
    Returns (adjusted_value, model_weight, shrink_points).
    """
    if market_value is None:
        return float(raw_value), 1.0, 0.0
    weight = _shrink_weight(week, market_type)
    adjusted = float(market_value) + weight * (float(raw_value) - float(market_value))
    return adjusted, weight, float(raw_value) - adjusted


# ===== v0.4 residual-market layer =====
# The sportsbook line is now the baseline forecast. The football model only
# supplies a regularized estimate of the market's residual error.
RESIDUAL_TRAIN_START = 2018
RESIDUAL_RIDGE_ALPHA = 12.0
RESIDUAL_SPREAD_CAP = 4.0
RESIDUAL_TOTAL_CAP = 3.0
RESIDUAL_MIN_ROWS = 300

RESIDUAL_SPREAD_FEATURES = [
    "raw_gap",
    "raw_margin",
    "market_margin",
    "abs_market_margin",
    "base_power_margin",
    "matchup_margin_adjustment",
    "hfa_adjustment",
    "week",
    "confidence",
]

RESIDUAL_TOTAL_FEATURES = [
    "raw_gap",
    "raw_total",
    "market_total",
    "sp_total_base",
    "efficiency_total_adjustment",
    "pace_total_adjustment",
    "week",
    "confidence",
]


def _ridge_fit_numpy(X, y, alpha=RESIDUAL_RIDGE_ALPHA):
    """Small dependency-free standardized ridge regression."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-8), 1.0, sd)
    Xs = (X - mu) / sd
    y_mu = float(np.nanmean(y))
    yc = y - y_mu

    xtx = Xs.T @ Xs
    penalty = float(alpha) * np.eye(Xs.shape[1])
    beta = np.linalg.solve(xtx + penalty, Xs.T @ yc)

    fitted = y_mu + Xs @ beta
    resid = y - fitted
    sigma = float(np.nanstd(resid, ddof=max(1, min(Xs.shape[1], len(y)-1))))
    return {
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "intercept": y_mu,
        "sigma": sigma,
        "n": int(len(y)),
        "alpha": float(alpha),
    }


def _ridge_predict_numpy(fit, values):
    if not fit:
        return 0.0
    x = np.asarray(values, dtype=float)
    xs = (x - fit["mu"]) / fit["sd"]
    return float(fit["intercept"] + xs @ fit["beta"])


def _residual_feature_dict(p, market):
    comps = p.get("components") or {}
    raw_home_spread = float(p["model_home_spread"])
    raw_margin = -raw_home_spread

    hs = market.get("home_spread")
    mt = market.get("total")

    spread = None
    if hs is not None:
        market_margin = -float(hs)
        spread = {
            "raw_gap": raw_margin - market_margin,
            "raw_margin": raw_margin,
            "market_margin": market_margin,
            "abs_market_margin": abs(market_margin),
            "base_power_margin": float(comps.get("base_power_margin") or 0.0),
            "matchup_margin_adjustment": float(comps.get("matchup_margin_adjustment") or 0.0),
            "hfa_adjustment": float(comps.get("hfa_adjustment") or 0.0),
            "week": float(_week_num(p.get("week"))),
            "confidence": float(p.get("confidence") or 70.0),
        }

    total = None
    if mt is not None:
        total = {
            "raw_gap": float(p["model_total"]) - float(mt),
            "raw_total": float(p["model_total"]),
            "market_total": float(mt),
            "sp_total_base": float(comps.get("sp_total_base") or 0.0),
            "efficiency_total_adjustment": float(comps.get("efficiency_total_adjustment") or 0.0),
            "pace_total_adjustment": float(comps.get("pace_total_adjustment") or 0.0),
            "week": float(_week_num(p.get("week"))),
            "confidence": float(p.get("confidence") or 70.0),
        }
    return spread, total


def residual_market_projection(p, market, residual_models=None):
    """
    Return a market-baseline fair spread/total plus residual probabilities.

    Spread target:
        actual home margin - market implied home margin

    Total target:
        actual total - market total

    If no trained residual model is available, the correction is zero and the
    market remains the forecast. This is deliberately conservative.
    """
    spread_features, total_features = _residual_feature_dict(p, market)
    sm = (residual_models or {}).get("spread")
    tm = (residual_models or {}).get("total")

    spread_correction = 0.0
    spread_sigma = 15.0
    if spread_features is not None and sm and sm.get("n", 0) >= RESIDUAL_MIN_ROWS:
        values = [spread_features[k] for k in RESIDUAL_SPREAD_FEATURES]
        spread_correction = _ridge_predict_numpy(sm, values)
        spread_correction = max(-RESIDUAL_SPREAD_CAP, min(RESIDUAL_SPREAD_CAP, spread_correction))
        spread_sigma = max(12.5, min(18.5, float(sm.get("sigma") or 15.0)))

    total_correction = 0.0
    total_sigma = 13.0
    if total_features is not None and tm and tm.get("n", 0) >= RESIDUAL_MIN_ROWS:
        values = [total_features[k] for k in RESIDUAL_TOTAL_FEATURES]
        total_correction = _ridge_predict_numpy(tm, values)
        total_correction = max(-RESIDUAL_TOTAL_CAP, min(RESIDUAL_TOTAL_CAP, total_correction))
        total_sigma = max(10.5, min(16.5, float(tm.get("sigma") or 13.0)))

    out = {
        "spread_correction": spread_correction,
        "spread_sigma": spread_sigma,
        "spread_model_n": int(sm.get("n", 0)) if sm else 0,
        "total_correction": total_correction,
        "total_sigma": total_sigma,
        "total_model_n": int(tm.get("n", 0)) if tm else 0,
    }

    if market.get("home_spread") is not None:
        market_margin = -float(market["home_spread"])
        fair_margin = market_margin + spread_correction
        out["adjusted_home_spread"] = -fair_margin
        out["home_cover_prob"] = 1.0 - NormalDist(
            mu=spread_correction, sigma=spread_sigma
        ).cdf(0.0)
        out["away_cover_prob"] = 1.0 - out["home_cover_prob"]
    else:
        out["adjusted_home_spread"] = float(p["model_home_spread"])
        out["home_cover_prob"] = None
        out["away_cover_prob"] = None

    if market.get("total") is not None:
        out["adjusted_total"] = float(market["total"]) + total_correction
        out["over_prob"] = 1.0 - NormalDist(
            mu=total_correction, sigma=total_sigma
        ).cdf(0.0)
        out["under_prob"] = 1.0 - out["over_prob"]
    else:
        out["adjusted_total"] = float(p["model_total"])
        out["over_prob"] = None
        out["under_prob"] = None

    return out


def cap_total_research_verdict(verdict):
    """Totals remain research-only until the residual model passes validation."""
    return "LEAN" if verdict in {"BET", "STRONG BET"} else verdict

def calibrated_sigmas(margin_sd, total_sd, week):
    """Extra uncertainty in Weeks 0-2/1-2 while priors dominate."""
    w = _week_num(week)
    if w <= 1:
        return max(float(margin_sd), 18.0), max(float(total_sd), 15.0)
    if w == 2:
        return max(float(margin_sd), 17.2), max(float(total_sd), 14.3)
    if w == 3:
        return max(float(margin_sd), 16.5), max(float(total_sd), 13.7)
    return float(margin_sd), float(total_sd)

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

def fetch_fbs_teams(api_key, year):
    return cfbd_get("/teams/fbs", api_key, {"year": year})

def fetch_venues(api_key):
    return cfbd_get("/venues", api_key, {})

# ===== v3.0 richer-data endpoints =====
def fetch_core(api_key, year):
    return cfbd_get("/ratings/core", api_key, {"year": int(year)})

def fetch_fpi(api_key, year):
    return cfbd_get("/ratings/fpi", api_key, {"year": int(year)})

def fetch_recruiting_teams(api_key, year):
    return cfbd_get("/recruiting/teams", api_key, {"year": int(year)})

def fetch_portal(api_key, year):
    return cfbd_get("/player/portal", api_key, {"year": int(year)})

def fetch_advanced_through_week(api_key, year, end_week):
    if end_week is None or int(end_week) < 1:
        return []
    return cfbd_get(
        "/stats/season/advanced",
        api_key,
        {
            "year": int(year),
            "endWeek": int(end_week),
            "excludeGarbageTime": "true",
            "classification": "fbs",
        },
    )


def _team_logo_url(data, team):
    try:
        row = (data.get("teams") or {}).get(team) or {}
        logos = row.get("logos") or []
        if isinstance(logos, list) and logos:
            return str(logos[0])
    except Exception:
        pass
    return ""


def _logo_html(url, team, size=30):
    initials = "".join([w[0] for w in str(team).split()[:2] if w])[:2].upper() or "CF"
    if url:
        safe_url = html.escape(str(url), quote=True)
        safe_team = html.escape(str(team), quote=True)
        return f'<img class="team-logo" src="{safe_url}" alt="{safe_team}" style="width:{size}px;height:{size}px;">'
    return f'<div class="team-logo-fallback" style="width:{size}px;height:{size}px;">{html.escape(initials)}</div>'


def _pick_logo_html(row, size=32):
    mtype = str(row.get("market_type", "")).upper()
    away = str(row.get("away_team", ""))
    home = str(row.get("home_team", ""))
    away_logo = str(row.get("away_logo", "") or "")
    home_logo = str(row.get("home_logo", "") or "")
    market = str(row.get("market", ""))

    if mtype == "TOTAL":
        return '<div class="logo-pair">' + _logo_html(away_logo, away, size) + _logo_html(home_logo, home, size) + '</div>'
    if market.startswith(home):
        return _logo_html(home_logo, home, size)
    return _logo_html(away_logo, away, size)

def _school_map(rows):
    return {str(r.get("school")): r for r in rows or [] if r.get("school")}

def _venue_map(rows):
    out = {}
    for r in rows or []:
        name = r.get("name")
        if name:
            out[str(name).strip().lower()] = r
        vid = r.get("id")
        if vid is not None:
            out[str(vid)] = r
    return out

def _haversine_miles(lat1, lon1, lat2, lon2):
    vals = [lat1, lon1, lat2, lon2]
    if any(v is None for v in vals):
        return None
    try:
        from math import radians, sin, cos, asin, sqrt
        lat1, lon1, lat2, lon2 = map(radians, map(float, vals))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 3958.8 * 2 * asin(sqrt(a))
    except Exception:
        return None

def _loc_fields(obj):
    if not isinstance(obj, dict):
        return {}
    loc = obj.get("location") if isinstance(obj.get("location"), dict) else obj
    return {
        "name": loc.get("name"),
        "city": loc.get("city"),
        "state": loc.get("state"),
        "country": loc.get("countryCode") or loc.get("country"),
        "latitude": _num(loc.get("latitude")),
        "longitude": _num(loc.get("longitude")),
        "elevation": _num(loc.get("elevation")),
        "timezone": loc.get("timezone"),
        "dome": loc.get("dome"),
    }


# Curated stadium/school coordinates for edge cases and common major-FBS teams.
# These are used only when CFBD does not resolve a usable game venue.
# lat/lon are approximate stadium coordinates and are sufficient for travel/weather.
FBS_STADIUM_OVERRIDES = {
    "Stanford": {
        "name": "Stanford Stadium",
        "city": "Stanford",
        "state": "CA",
        "countryCode": "US",
        "latitude": 37.4345,
        "longitude": -122.1611,
        "elevation": 95,
    },
    "Hawai'i": {
        "name": "Clarence T.C. Ching Athletics Complex",
        "city": "Honolulu",
        "state": "HI",
        "countryCode": "US",
        "latitude": 21.2947,
        "longitude": -157.8174,
        "elevation": 20,
    },
    "Hawaii": {
        "name": "Clarence T.C. Ching Athletics Complex",
        "city": "Honolulu",
        "state": "HI",
        "countryCode": "US",
        "latitude": 21.2947,
        "longitude": -157.8174,
        "elevation": 20,
    },
    "Notre Dame": {
        "name": "Notre Dame Stadium",
        "city": "Notre Dame",
        "state": "IN",
        "countryCode": "US",
        "latitude": 41.6984,
        "longitude": -86.2339,
        "elevation": 730,
    },
}

def _normalize_team_name(name):
    s = str(name or "").strip()
    # Normalize common apostrophe/encoding variants.
    return (s.replace("’", "'")
             .replace("`", "'")
             .replace("Hawai’i", "Hawai'i")
             .replace("Hawaiʻi", "Hawai'i"))

@lru_cache(maxsize=512)
def _nominatim_geocode(query):
    """
    Last-resort geocoder for a home stadium/campus.
    Uses OpenStreetMap Nominatim with a descriptive User-Agent.
    Cached to avoid repeat calls.
    """
    if not query:
        return {}
    try:
        headers = {"User-Agent": "CFBModel/0.2.4 stadium-weather resolver"}
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "us",
            "addressdetails": 1,
        }
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json() or []
        if not rows:
            return {}
        row = rows[0]
        address = row.get("address") or {}
        return {
            "name": row.get("name") or query,
            "city": address.get("city") or address.get("town") or address.get("village"),
            "state": address.get("state"),
            "countryCode": str(address.get("country_code") or "us").upper(),
            "latitude": _num(row.get("lat")),
            "longitude": _num(row.get("lon")),
            "display_name": row.get("display_name"),
        }
    except Exception:
        return {}

def _stadium_fallback(team):
    """
    Resolve a team's home stadium/location independently of CFBD.
    1) curated stadium override
    2) geocode '<team> football stadium'
    3) geocode '<team> university'
    """
    team = _normalize_team_name(team)

    if team in FBS_STADIUM_OVERRIDES:
        out = dict(FBS_STADIUM_OVERRIDES[team])
        out["_venue_source"] = "curated_stadium_override"
        return out

    for query in (
        f"{team} football stadium",
        f"{team} university football stadium",
        f"{team} university",
    ):
        out = _nominatim_geocode(query)
        if out.get("latitude") is not None and out.get("longitude") is not None:
            out["_venue_source"] = "openstreetmap_geocode"
            out["_geocode_query"] = query
            return out

    return {}

def _resolve_venue(game, data):
    """
    Venue resolution order:
      1. explicit game venue object
      2. CFBD venue lookup by name / id
      3. CFBD home-team location (non-neutral)
      4. curated FBS stadium override
      5. OpenStreetMap stadium/campus geocode
    Neutral games do not silently fall back to the home-designated team's campus.
    """
    raw = game.get("venue")

    if isinstance(raw, dict):
        loc = _loc_fields(raw)
        if loc.get("latitude") is not None and loc.get("longitude") is not None:
            out = dict(raw)
            out["_venue_source"] = "game_venue"
            return out

    venues = data.get("venues", {})
    if raw is not None:
        key = str(raw).strip().lower()
        candidate = venues.get(key) or venues.get(str(raw))
        if candidate:
            loc = _loc_fields(candidate)
            if loc.get("latitude") is not None and loc.get("longitude") is not None:
                out = dict(candidate)
                out["_venue_source"] = "cfbd_venue_lookup"
                return out

    vid = game.get("venueId") or game.get("venue_id")
    if vid is not None and str(vid) in venues:
        candidate = venues[str(vid)]
        loc = _loc_fields(candidate)
        if loc.get("latitude") is not None and loc.get("longitude") is not None:
            out = dict(candidate)
            out["_venue_source"] = "cfbd_venue_id"
            return out

    if not bool(game.get("neutralSite")):
        home = _normalize_team_name(game.get("homeTeam"))

        home_obj = data.get("teams", {}).get(home) or data.get("teams", {}).get(str(game.get("homeTeam"))) or {}
        home_loc = _loc_fields(home_obj)
        if home_loc.get("latitude") is not None and home_loc.get("longitude") is not None:
            out = dict(home_obj)
            if not out.get("name"):
                out["name"] = f"{home} home location"
            out["_venue_source"] = "cfbd_home_team_location"
            return out

        fallback = _stadium_fallback(home)
        if fallback:
            return fallback

    # Neutral game: if the raw venue name exists, try geocoding that exact venue.
    if bool(game.get("neutralSite")) and raw:
        fallback = _nominatim_geocode(str(raw))
        if fallback:
            fallback["_venue_source"] = "neutral_venue_geocode"
            return fallback

    return {"_venue_source": "unresolved"}

def _weather_from_game(game):
    weather = game.get("weather") if isinstance(game.get("weather"), dict) else {}
    description = str(
        weather.get("description")
        or game.get("weatherDescription")
        or game.get("weather_condition")
        or ""
    ).lower()
    wind = _num(weather.get("windSpeed"), _num(game.get("windSpeed")))
    temp = _num(weather.get("temperature"), _num(game.get("temperature")))
    return {"description": description, "wind_mph": wind, "temperature_f": temp}


def _parse_game_datetime(game):
    raw = game.get("startDate") or game.get("start_date") or game.get("startTime")
    if not raw:
        return None
    try:
        s = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

@lru_cache(maxsize=512)
def _open_meteo_hourly(lat_key, lon_key, date_key):
    """
    Free/no-key forecast fallback. Cache by rounded venue coordinates + local date.
    """
    try:
        params = {
            "latitude": float(lat_key),
            "longitude": float(lon_key),
            "hourly": "temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",
            "start_date": date_key,
            "end_date": date_key,
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _forecast_weather(game, venue):
    lat = venue.get("latitude")
    lon = venue.get("longitude")
    dt = _parse_game_datetime(game)
    if lat is None or lon is None or dt is None:
        return {}

    # Ask Open-Meteo for the venue's local calendar date. We use the game date
    # from the ISO string as a safe first pass and select the nearest forecast hour.
    date_key = dt.date().isoformat()
    data = _open_meteo_hourly(round(float(lat), 3), round(float(lon), 3), date_key)
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {}

    # Convert kickoff to venue-local time using Open-Meteo's UTC offset when present.
    try:
        offset_seconds = int(data.get("utc_offset_seconds") or 0)
        local_dt = dt.astimezone(timezone.utc) + timedelta(seconds=offset_seconds)
        target = local_dt.replace(tzinfo=None)
    except Exception:
        target = dt.replace(tzinfo=None)

    nearest_i = None
    nearest_seconds = None
    for i, ts in enumerate(times):
        try:
            t = datetime.fromisoformat(ts)
            diff = abs((t - target).total_seconds())
            if nearest_seconds is None or diff < nearest_seconds:
                nearest_seconds = diff
                nearest_i = i
        except Exception:
            continue
    if nearest_i is None:
        return {}

    def pick(key):
        vals = hourly.get(key) or []
        if nearest_i < len(vals):
            return _num(vals[nearest_i])
        return None

    precip_prob = pick("precipitation_probability")
    precip = pick("precipitation")
    wind = pick("wind_speed_10m")
    gust = pick("wind_gusts_10m")
    temp = pick("temperature_2m")

    desc_bits = []
    if precip_prob is not None:
        desc_bits.append(f"{precip_prob:.0f}% precip")
    if precip is not None and precip >= 0.02:
        desc_bits.append(f"{precip:.2f} in precip")
    if wind is not None:
        desc_bits.append(f"{wind:.0f} mph wind")
    if gust is not None:
        desc_bits.append(f"gusts {gust:.0f}")
    if temp is not None:
        desc_bits.append(f"{temp:.0f}F")

    return {
        "source": "Open-Meteo",
        "description": ", ".join(desc_bits),
        "wind_mph": wind,
        "wind_gust_mph": gust,
        "temperature_f": temp,
        "precip_probability": precip_prob,
        "precipitation_in": precip,
        "forecast_hour": times[nearest_i],
    }

def _environment_adjustment(game, data):
    """
    Conservative pregame environment layer.
    It uses CFBD team/venue coordinates and game weather when available.
    These are small, capped adjustments until we have a larger backtest.
    """
    away = game.get("awayTeam")
    home = game.get("homeTeam")
    neutral = bool(game.get("neutralSite"))

    venue_obj = _resolve_venue(game, data)
    venue = _loc_fields(venue_obj)

    resolved_venue_name = (
        venue_obj.get("name") if isinstance(venue_obj, dict) else None
    ) or game.get("venue")
    enclosed_venue = is_enclosed_venue(resolved_venue_name)

    away_team_obj = data.get("teams", {}).get(str(away), {})
    home_team_obj = data.get("teams", {}).get(str(home), {})
    away_loc = _loc_fields(away_team_obj)
    home_loc = _loc_fields(home_team_obj)

    # If CFBD does not expose campus/stadium coordinates, use the independent
    # stadium resolver as each team's travel origin.
    if away_loc.get("latitude") is None or away_loc.get("longitude") is None:
        away_loc = _loc_fields(_stadium_fallback(away))
    if home_loc.get("latitude") is None or home_loc.get("longitude") is None:
        home_loc = _loc_fields(_stadium_fallback(home))

    away_miles = _haversine_miles(
        away_loc.get("latitude"), away_loc.get("longitude"),
        venue.get("latitude"), venue.get("longitude")
    )
    home_miles = _haversine_miles(
        home_loc.get("latitude"), home_loc.get("longitude"),
        venue.get("latitude"), venue.get("longitude")
    )

    margin_adj = 0.0
    total_adj = 0.0
    confidence_penalty = 0
    reasons = []

    # Travel: differential matters more than raw distance.
    # Keep point impact modest until calibrated.
    if away_miles is not None and home_miles is not None:
        differential = away_miles - home_miles

        if differential >= 3000:
            margin_adj += 0.75
            confidence_penalty += 2
            reasons.append("extreme away travel disadvantage")
        elif differential >= 1800:
            margin_adj += 0.50
            confidence_penalty += 1
            reasons.append("long away travel disadvantage")
        elif differential >= 900:
            margin_adj += 0.25
            reasons.append("moderate away travel disadvantage")
        elif differential <= -3000:
            margin_adj -= 0.75
            confidence_penalty += 2
            reasons.append("extreme home-designated travel disadvantage")
        elif differential <= -1800:
            margin_adj -= 0.50
            confidence_penalty += 1
            reasons.append("long home-designated travel disadvantage")
        elif differential <= -900:
            margin_adj -= 0.25
            reasons.append("moderate home-designated travel disadvantage")

        # Very long trips can suppress offensive efficiency slightly.
        if max(away_miles, home_miles) >= 3000:
            total_adj -= 0.75
        elif max(away_miles, home_miles) >= 2000:
            total_adj -= 0.35

    # International / unusual neutral-site context.
    venue_blob = " ".join(str(x or "") for x in [
        venue_obj.get("name") if isinstance(venue_obj, dict) else "",
        venue.get("city"), venue.get("state"), venue.get("country"),
        game.get("venue")
    ]).lower()

    international_keywords = ["ireland", "dublin", "aviva", "australia", "sydney", "japan", "tokyo"]
    international = any(k in venue_blob for k in international_keywords)
    if international:
        # Informational only in v0.2.7.1. International/neutral games are rare,
        # so we do not apply a bespoke scoring penalty without calibration data.
        reasons.append("international/atypical venue (informational)")

    # Weather: wind is the strongest total suppressor here.
    weather = _weather_from_game(game)
    weather_source = "CFBD game payload" if any([
        weather.get("wind_mph") is not None,
        weather.get("temperature_f") is not None,
        bool(weather.get("description"))
    ]) else None

    # If CFBD does not carry usable weather, fetch forecast weather from Open-Meteo.
    if weather_source is None:
        forecast = _forecast_weather(game, venue)
        if forecast:
            weather = {
                "description": forecast.get("description") or "",
                "wind_mph": forecast.get("wind_mph"),
                "temperature_f": forecast.get("temperature_f"),
                "wind_gust_mph": forecast.get("wind_gust_mph"),
                "precip_probability": forecast.get("precip_probability"),
                "precipitation_in": forecast.get("precipitation_in"),
                "forecast_hour": forecast.get("forecast_hour"),
            }
            weather_source = forecast.get("source")

    wind = weather.get("wind_mph")
    temp = weather.get("temperature_f")
    desc = weather.get("description", "")

    if not enclosed_venue and wind is not None:
        if wind >= 25:
            total_adj -= 4.0
            confidence_penalty += 2
            reasons.append("very high wind")
        elif wind >= 20:
            total_adj -= 3.0
            confidence_penalty += 1
            reasons.append("high wind")
        elif wind >= 15:
            total_adj -= 1.5
            reasons.append("meaningful wind")

    gust = _num(weather.get("wind_gust_mph"))
    if not enclosed_venue and gust is not None and gust >= 35 and (wind is None or wind < 20):
        total_adj -= 0.75
        reasons.append("strong wind gusts")

    # Only count precipitation when there is actual evidence of it.
    # Do NOT key off the generic word "precip" because Open-Meteo descriptions
    # include strings like "0% precip", which caused a false adjustment in v0.2.4.
    weather_terms = ["rain", "shower", "storm", "snow", "sleet"]
    precip_prob = _num(weather.get("precip_probability"))
    precip_amt = _num(weather.get("precipitation_in"))

    text_precip = any(term in desc for term in weather_terms)
    numeric_precip = (
        (precip_amt is not None and precip_amt >= 0.02)
        or (
            precip_prob is not None
            and precip_prob >= 55
            and precip_amt is not None
            and precip_amt > 0
        )
    )

    meaningful_precip = text_precip or numeric_precip
    if not enclosed_venue and meaningful_precip:
        total_adj -= 0.75
        reasons.append("precipitation")

    if not enclosed_venue and temp is not None:
        if temp <= 25:
            total_adj -= 1.0
            reasons.append("extreme cold")
        elif temp >= 100:
            total_adj -= 0.75
            reasons.append("extreme heat")

    # Elevation: small home-side acclimation edge when a low-travel visitor goes high.
    elev = venue.get("elevation")
    if elev is not None and elev >= 4500 and away_miles is not None and away_miles >= 700 and not neutral:
        margin_adj += 0.25
        total_adj -= 0.25
        reasons.append("high-altitude venue")

    margin_adj = max(-1.25, min(1.25, margin_adj))
    total_adj = max(-5.0, min(1.0, total_adj))

    if enclosed_venue:
        reasons.append("weather suppressed: enclosed venue")

    return {
        "margin_adjustment": margin_adj,
        "total_adjustment": total_adj,
        "confidence_penalty": confidence_penalty,
        "away_travel_miles": away_miles,
        "home_travel_miles": home_miles,
        "venue_name": venue_obj.get("name") if isinstance(venue_obj, dict) else game.get("venue"),
        "venue_city": venue.get("city"),
        "venue_state": venue.get("state"),
        "venue_country": venue.get("country"),
        "venue_elevation": elev,
        "venue_source": venue_obj.get("_venue_source") if isinstance(venue_obj, dict) else None,
        "enclosed_venue": enclosed_venue,
        "venue_geocode_query": venue_obj.get("_geocode_query") if isinstance(venue_obj, dict) else None,
        "international": international,
        "weather_source": weather_source,
        "weather_description": weather.get("description"),
        "forecast_hour": weather.get("forecast_hour"),
        "wind_mph": wind,
        "wind_gust_mph": weather.get("wind_gust_mph"),
        "temperature_f": temp,
        "precip_probability": weather.get("precip_probability"),
        "precipitation_in": weather.get("precipitation_in"),
        "reasons": reasons,
    }

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
    teams = _school_map(_safe_fetch(fetch_fbs_teams, api_key, year))
    venues = _venue_map(_safe_fetch(fetch_venues, api_key))

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
        "teams": teams,
        "venues": venues,
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


# ===== v1.4.2 FCS protection =====
# CFBD's FBS team endpoint is used as the membership list. If a scheduled team
# is not in that list and also lacks usable SP+ data, it must NOT be treated
# as an average FBS team.
FCS_FALLBACK_POWER = -20.0
FCS_CONFIDENCE_CAP = 62
FCS_OFFICIAL_BET_CAP = "LEAN"

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

    is_fbs = team in (data.get("teams") or {})
    has_sp = sp["rating"] is not None
    fcs_fallback = (not is_fbs) and (not has_sp)

    if fcs_fallback:
        # Critical fix: previously this became 0.0, effectively "average FBS".
        sp["rating"] = FCS_FALLBACK_POWER
        source = "FCS fallback"
    else:
        source = "Current-year SP+" if team in data["sp_current"] else (
            "Prior-year SP+ fallback" if team in data["sp_previous"] else "Average fallback"
        )

    return {
        "team": team,
        "rating": sp["rating"] if sp["rating"] is not None else 0.0,
        "offense": sp["offense"] if sp["offense"] is not None else 0.0,
        "defense": sp["defense"] if sp["defense"] is not None else 0.0,
        "special": sp["special"] if sp["special"] is not None else 0.0,
        "pace": sp["pace"] if sp["pace"] is not None else 0.0,
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
        "is_fbs": is_fbs,
        "fcs_fallback": fcs_fallback,
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
            "teams": {}, "venues": {},
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
    environment = _environment_adjustment(game, data)
    env_margin_adj = environment["margin_adjustment"]
    home_margin = base_margin + matchup_margin_adj + applied_hfa + env_margin_adj

    raw_sp_total = _total_from_sp(ar, hr)
    efficiency_adj = _total_efficiency_adjustment(ar, hr)
    pace_adj = _pace_adjustment(ar, hr)

    # Keep the total independent of the market.
    env_total_adj = environment["total_adjustment"]
    total = raw_sp_total + efficiency_adj + pace_adj + env_total_adj
    total = max(34.0, min(82.0, total))

    # Reconcile the scoring split to the independently estimated margin.
    home_score = max(7.0, (total + home_margin) / 2.0)
    away_score = max(7.0, (total - home_margin) / 2.0)
    total = home_score + away_score
    home_margin = home_score - away_score

    margin_sd, total_sd, confidence, completeness = _uncertainty(week, ar, hr)
    confidence = max(60, confidence - int(environment.get("confidence_penalty", 0)))

    fcs_fallback_used = bool(ar.get("fcs_fallback") or hr.get("fcs_fallback"))
    if fcs_fallback_used:
        confidence = min(confidence, FCS_CONFIDENCE_CAP)
        margin_sd = min(18.0, margin_sd + 1.0)

    home_wp = 1.0 - NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)

    components = {
        "base_power_margin": base_margin,
        "matchup_margin_adjustment": matchup_margin_adj,
        "hfa_adjustment": applied_hfa,
        "environment_margin_adjustment": env_margin_adj,
        "sp_total_base": raw_sp_total,
        "efficiency_total_adjustment": efficiency_adj,
        "pace_total_adjustment": pace_adj,
        "environment_total_adjustment": env_total_adj,
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
        "fcs_fallback_used": fcs_fallback_used,
        "away_is_fbs": ar.get("is_fbs"),
        "home_is_fbs": hr.get("is_fbs"),
        "components": components,
        "environment": environment,
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

def _valid_american_odds(odds):
    try:
        x = float(odds)
        return math.isfinite(x) and x != 0
    except Exception:
        return False

def implied_prob(odds):
    if not _valid_american_odds(odds):
        return None
    odds = float(odds)
    return 100/(odds+100) if odds > 0 else abs(odds)/(abs(odds)+100)

def fair_ml(prob):
    p = max(.001, min(.999, float(prob)))
    return -round(100*p/(1-p)) if p >= .5 else round(100*(1-p)/p)

def expected_value(prob, odds):
    if not _valid_american_odds(odds):
        return None
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

# ===== v1.0 ROBUST LIVE guardrails =====
# Historical research through 2019-2025 did not establish a durable ATS edge.
# v1.0 therefore separates "model opinion" from "official bet" and refuses to
# manufacture action. The purpose is to maximize decision quality, not bet count.

ROBUST_LIVE_MIN_CONFIDENCE = 78
ROBUST_LIVE_MIN_EDGE = 0.045       # 4.5 percentage points above implied probability
ROBUST_LIVE_MIN_EV = 0.075         # +7.5% expected value
ROBUST_LIVE_MAX_SIDE_GAP = 8.0     # very large model/market gaps are review-only
ROBUST_LIVE_MAX_TOTAL_GAP = 9.0
ROBUST_LIVE_EARLY_WEEKS = {1, 2, 3}
ROBUST_LIVE_EARLY_EXTRA_EDGE = 0.015
ROBUST_LIVE_EARLY_EXTRA_EV = 0.020

def robust_live_grade(prob, odds, confidence, market_type="spread",
                      model_line=None, market_line=None, week=None):
    """
    Production decision layer.
    Returns one of: BET / LEAN / PASS / REVIEW.

    BET is intentionally rare because the historical research did not prove
    a stable automated edge. Large disagreements are not treated as stronger
    evidence; past testing showed the opposite.
    """
    try:
        p = float(prob)
        o = int(odds)
        conf = float(confidence)
    except Exception:
        return "PASS", np.nan, np.nan, "invalid input"

    imp = implied_prob(o)
    ev = expected_value(p, o)
    if imp is None or ev is None:
        return "PASS", np.nan, np.nan, "invalid odds"

    edge = p - imp
    min_edge = ROBUST_LIVE_MIN_EDGE
    min_ev = ROBUST_LIVE_MIN_EV

    if week is not None:
        try:
            if int(week) in ROBUST_LIVE_EARLY_WEEKS:
                min_edge += ROBUST_LIVE_EARLY_EXTRA_EDGE
                min_ev += ROBUST_LIVE_EARLY_EXTRA_EV
        except Exception:
            pass

    # Totals remain research-only because they were consistently weak.
    if str(market_type).lower() == "total":
        if edge > 0 and ev > 0:
            return "LEAN", edge, ev, "totals remain research-only"
        return "PASS", edge, ev, "totals remain research-only"

    # Historical ML quality was not trustworthy enough for automatic promotion.
    if str(market_type).lower() in {"ml", "moneyline", "money line"}:
        if edge >= min_edge and ev >= min_ev and conf >= ROBUST_LIVE_MIN_CONFIDENCE:
            return "REVIEW", edge, ev, "moneyline requires manual review"
        if edge > 0 and ev > 0:
            return "LEAN", edge, ev, "moneyline requires manual review"
        return "PASS", edge, ev, "moneyline requires manual review"

    # Extreme disagreement is review-only, not an automatic bet.
    if model_line is not None and market_line is not None:
        try:
            gap = abs(float(model_line) - float(market_line))
            cap = ROBUST_LIVE_MAX_TOTAL_GAP if str(market_type).lower() == "total" else ROBUST_LIVE_MAX_SIDE_GAP
            if gap >= cap:
                if edge > 0 and ev > 0:
                    return "REVIEW", edge, ev, f"extreme model/market gap ({gap:.1f})"
                return "PASS", edge, ev, f"extreme model/market gap ({gap:.1f})"
        except Exception:
            pass

    if conf >= ROBUST_LIVE_MIN_CONFIDENCE and edge >= min_edge and ev >= min_ev:
        return "BET", edge, ev, "passes robust production gate"
    if edge >= 0.02 and ev >= 0.03:
        return "LEAN", edge, ev, "positive but below production gate"
    return "PASS", edge, ev, "insufficient edge"

def robust_live_stake(verdict, edge, confidence, bankroll_units=100.0):
    """
    Conservative flat/ramped sizing. No Kelly escalation.
    One unit = 1% of the user's designated betting bankroll.
    """
    if verdict != "BET":
        return 0.0
    try:
        e = float(edge)
        c = float(confidence)
    except Exception:
        return 0.0

    # 0.5u base, 0.75u for stronger qualified edges, 1.0u max.
    stake = 0.50
    if e >= 0.060 and c >= 82:
        stake = 0.75
    if e >= 0.080 and c >= 86:
        stake = 1.00
    return stake

# ===== End v1.0 ROBUST LIVE guardrails =====



def apply_fcs_guard(verdict, fcs_fallback_used):
    """
    Missing FCS team-specific inputs are too weak for an official A/B play.
    Preserve the model opinion, but cap STRONG BET/BET at LEAN.
    """
    if fcs_fallback_used and verdict in {"STRONG BET", "BET"}:
        return "LEAN"
    return verdict



def apply_moneyline_guard(verdict, odds, fcs_fallback_used=False):
    """
    Prevent fragile longshot moneyline probabilities from surfacing as official plays.

    Rules:
    - Any FCS-fallback moneyline is PASS.
    - Any dog at +1000 or longer is PASS.
    - Dogs from +500 to +999 can be no better than LEAN.
    - Normal ML behavior below +500 is unchanged.
    """
    try:
        odds = float(odds)
    except Exception:
        return verdict

    if fcs_fallback_used:
        return "PASS"

    if odds >= 1000:
        return "PASS"

    if odds >= 500 and verdict in {"STRONG BET", "BET"}:
        return "LEAN"

    return verdict

def _slate_market_type(name):
    s = str(name or "")
    if s.endswith(" ML"):
        return "MONEYLINE"
    if s.startswith("Over ") or s.startswith("Under "):
        return "TOTAL"
    return "SPREAD"


def _slate_grade_meta(verdict):
    return {
        "STRONG BET": ("A", 4, "BEST BET"),
        "BET": ("B", 3, "BET"),
        "LEAN": ("C", 2, "LEAN"),
        "PASS": ("D", 1, "PASS"),
        "NO LINE": ("D", 0, "NO LINE"),
    }.get(str(verdict), ("D", 0, "PASS"))


def _grade_label_from_grade(grade):
    return {
        "A": "BEST BET",
        "B": "BET",
        "C": "LEAN",
        "D": "PASS",
        "—": "NO LINE",
    }.get(str(grade), str(grade))


def _slate_rank_score(verdict, prob, edge, ev, confidence):
    """
    Rank official plays by "best chance to make money," not by raw EV alone.

    Grade remains the first gate. Within the same grade, ranking favors:
    1) higher model win probability,
    2) stronger edge vs break-even,
    3) positive EV,
    4) higher model/data confidence.

    This makes the Top 5/10 behave like a practical betting card rather than
    a list of high-variance mathematical longshots.
    """
    _, grade_rank, _ = _slate_grade_meta(verdict)

    try:
        p = max(0.0, min(1.0, float(prob)))
    except Exception:
        p = 0.50
    try:
        e = max(-0.25, min(0.25, float(edge)))
    except Exception:
        e = 0.0
    try:
        v = max(-0.50, min(0.75, float(ev)))
    except Exception:
        v = 0.0
    try:
        c = max(0.0, min(100.0, float(confidence)))
    except Exception:
        c = 60.0

    # Small reliability haircut when confidence is below 75.
    conservative_p = max(0.0, p - max(0.0, 75.0 - c) * 0.0015)

    # Grade dominates; then practical win probability dominates raw EV.
    return (
        grade_rank * 100.0
        + conservative_p * 45.0
        + e * 30.0
        + v * 12.0
        + c * 0.10
    )



# ===== v3.8.1 opportunistic moneyline overlay =====
# ML remains secondary to the validated spread production model.
# These rules only decide whether an ML deserves to be surfaced in the UI.
# They do NOT make it an official tracked recommendation.
V381_ML_MIN_EDGE = 0.05
V381_ML_MIN_EV = 0.08
V381_ML_MIN_PROB = 0.45
V381_ML_MIN_CONFIDENCE = 0.70
V381_ML_MIN_ODDS = -200
V381_ML_MAX_ODDS = 250

def _v381_ml_value_candidates(market_board):
    if market_board is None or market_board.empty:
        return pd.DataFrame()

    x = market_board.copy()
    x["market_type_norm"] = x["market_type"].astype(str).str.upper()
    for c in ["edge", "ev", "prob", "confidence", "odds"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    keep = (
        x["market_type_norm"].eq("MONEYLINE")
        & x["edge"].ge(V381_ML_MIN_EDGE)
        & x["ev"].ge(V381_ML_MIN_EV)
        & x["prob"].ge(V381_ML_MIN_PROB)
        & x["confidence"].ge(V381_ML_MIN_CONFIDENCE)
        & x["odds"].ge(V381_ML_MIN_ODDS)
        & x["odds"].le(V381_ML_MAX_ODDS)
        & (~x["fcs_fallback_used"].fillna(False).astype(bool))
    )
    out = x.loc[keep].copy()
    if len(out):
        out["ml_value_score"] = (
            0.45 * out["edge"].clip(lower=0)
            + 0.35 * out["ev"].clip(lower=0)
            + 0.20 * (out["prob"] - 0.50).clip(lower=0)
        )
        out = out.sort_values(
            ["ml_value_score", "ev", "edge"],
            ascending=False,
            na_position="last",
        )
    return out

def _ranked_market_board(slate_df):
    rows = []
    if slate_df is None or len(slate_df) == 0:
        return pd.DataFrame()

    for _, game_row in slate_df.iterrows():
        try:
            markets = json.loads(game_row.get("market_grades_json", "[]") or "[]")
        except Exception:
            markets = []

        game = f"{game_row['away_team']} @ {game_row['home_team']}"
        conf = game_row.get("model_confidence", 0)
        fcs = bool(game_row.get("fcs_fallback_used", False))

        for m in markets:
            verdict = str(m.get("verdict", "PASS"))
            grade, grade_rank, grade_label = _slate_grade_meta(verdict)
            odds = m.get("odds")
            edge = m.get("edge")
            ev = m.get("ev")
            try:
                prob = implied_prob(float(odds)) + float(edge)
            except Exception:
                prob = None

            rows.append({
                "game_id": game_row.get("game_id"),
                "game": game,
                "kickoff_et": game_row.get("kickoff_et", ""),
                "away_team": game_row.get("away_team"),
                "home_team": game_row.get("home_team"),
                "away_logo": game_row.get("away_logo", ""),
                "home_logo": game_row.get("home_logo", ""),
                "market": m.get("market", ""),
                "market_type": _slate_market_type(m.get("market")),
                "odds": odds,
                "prob": prob,
                "edge": edge,
                "ev": ev,
                "verdict": verdict,
                "grade": grade,
                "grade_rank": grade_rank,
                "grade_label": grade_label,
                "confidence": conf,
                "fcs_fallback_used": fcs,
                "rank_score": _slate_rank_score(verdict, prob, edge, ev, conf),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["grade_rank", "rank_score", "ev", "edge"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return out


def _fmt_market_pct(v, signed=False):
    try:
        x = float(v) * 100
        return f"{x:+.1f}%" if signed else f"{x:.1f}%"
    except Exception:
        return "—"


def _render_top_slate_bet(row, rank):
    grade = str(row.get("grade", "D"))
    cls = grade.lower()
    game = str(row.get("game", ""))
    market = str(row.get("market", ""))
    mtype = str(row.get("market_type", ""))
    kickoff = str(row.get("kickoff_et", ""))
    try:
        odds_txt = f"{int(float(row.get('odds'))):+d}"
    except Exception:
        odds_txt = ""

    if bool(row.get("fcs_fallback_used", False)) and mtype == "MONEYLINE":
        note = "FCS fallback • moneyline blocked"
    else:
        try:
            _o = float(row.get("odds"))
        except Exception:
            _o = 0.0
        if mtype == "MONEYLINE" and _o >= 1000:
            note = "Extreme longshot ML • blocked"
        elif mtype == "MONEYLINE" and _o >= 500:
            note = "Longshot ML • lean ceiling"
        else:
            note = f"{mtype}"

    pick_logo = _pick_logo_html(row, 32)
    st.markdown(
        f"""
        <div class="topbet-card {cls}">
          <div class="topbet-rank">#{rank}</div>
          <div class="topbet-logo">{pick_logo}</div>
          <div class="topbet-copy">
            <div class="topbet-game">{html.escape(kickoff)} • {html.escape(game)}</div>
            <div class="topbet-pick">{html.escape(market)} {html.escape(odds_txt)}</div>
            <div class="topbet-note">{html.escape(note)}</div>
          </div>
          <div class="topbet-grade {cls}">{row.get('grade_label','PASS')}</div>
          <div class="topbet-metrics">
            <div><span>Win Chance</span><b>{_fmt_market_pct(row.get("prob"))}</b></div>
            <div><span>Edge</span><b>{_fmt_market_pct(row.get("edge"), True)}</b></div>
            <div><span>EV</span><b>{_fmt_market_pct(row.get("ev"), True)}</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_game_market_stack(game_markets):
    g = game_markets.sort_values(
        ["grade_rank", "rank_score", "ev", "edge"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if len(g) == 0:
        st.caption("No market lines available.")
        return

    top = g.iloc[0]
    st.markdown('<div class="game-best-label">TOP BET FOR THIS GAME</div>', unsafe_allow_html=True)
    _render_top_slate_bet(top, 1)

    if len(g) > 1:
        st.markdown('<div class="game-best-label">OTHER MARKETS · RANKED</div>', unsafe_allow_html=True)
        for i in range(1, len(g)):
            r = g.iloc[i]
            grade = str(r.get("grade", "D"))
            try:
                odds_txt = f"{int(float(r.get('odds'))):+d}"
            except Exception:
                odds_txt = ""
            st.markdown(
                f"""
                <div class="game-market-row">
                  <div class="game-market-rank">#{i+1}</div>
                  <div class="game-market-grade {grade.lower()}">{_grade_label_from_grade(grade)}</div>
                  <div class="game-market-body">
                    <div class="game-market-pick">{html.escape(str(r.get('market','')))} {html.escape(odds_txt)}</div>
                    <div class="game-market-meta">{html.escape(str(r.get('market_type','')))} • Edge {_fmt_market_pct(r.get("edge"), True)} • EV {_fmt_market_pct(r.get("ev"), True)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def grade(prob, odds, confidence=75, market_type="spread", projection_gap=None, week=1):
    """
    v1.2 playable all-market grader.

    Internal verdicts remain STRONG BET / BET / LEAN / PASS so the existing
    slate, export and backtest UI remains compatible.

    User-facing interpretation:
      STRONG BET = A / Best Bet
      BET        = B / Bet
      LEAN       = C / Lean
      PASS       = D / Pass

    Confidence is now a modifier, not a hard veto. This is especially important
    in Weeks 1-3, where the engine naturally carries lower confidence.
    """
    if not _valid_american_odds(odds):
        return "PASS", 0.0, 0.0, None

    market_type = str(market_type or "spread").lower()
    imp = implied_prob(odds)
    if imp is None:
        return "PASS", 0.0, 0.0, None

    try:
        p = float(prob)
        conf = float(confidence)
    except Exception:
        return "PASS", 0.0, 0.0, imp

    edge = p - float(imp)
    ev = expected_value(p, odds)
    if ev is None:
        return "PASS", edge, 0.0, imp

    w = _week_num(week)
    o = int(odds)

    # Market-specific baseline thresholds. These are deliberately playable,
    # while still requiring positive price-adjusted value.
    if market_type in {"spread", "side"}:
        bet_edge, bet_ev = 0.035, 0.055
        strong_edge, strong_ev = 0.060, 0.095
        lean_edge, lean_ev = 0.012, 0.015
        review_gap = 10.0

    elif market_type == "total":
        # Totals were weaker historically, so they still need more evidence.
        bet_edge, bet_ev = 0.045, 0.070
        strong_edge, strong_ev = 0.070, 0.110
        lean_edge, lean_ev = 0.015, 0.020
        review_gap = 11.0

    elif market_type in {"moneyline", "ml", "money line"}:
        bet_edge, bet_ev = 0.035, 0.060
        strong_edge, strong_ev = 0.060, 0.105
        lean_edge, lean_ev = 0.010, 0.020
        review_gap = None

        # Price-aware protection against the historical long-dog artifact.
        if o >= 200:
            bet_edge += 0.010
            bet_ev += 0.020
            strong_edge += 0.015
            strong_ev += 0.025
        elif o <= -180:
            bet_edge += 0.0075
            bet_ev += 0.015
            strong_edge += 0.010
            strong_ev += 0.020
    else:
        bet_edge, bet_ev = 0.040, 0.065
        strong_edge, strong_ev = 0.065, 0.105
        lean_edge, lean_ev = 0.015, 0.020
        review_gap = None

    # Early season remains less certain, but no longer creates an impossible
    # confidence gate. We ask for a little more edge instead.
    if w <= 3:
        bet_edge += 0.0075
        bet_ev += 0.010
        strong_edge += 0.010
        strong_ev += 0.015
        if market_type == "total":
            bet_edge += 0.005
            bet_ev += 0.005

    # Confidence modifies the required evidence rather than vetoing the play.
    # Around 72 (common early-season confidence), a genuinely large edge can
    # still qualify. Low confidence requires progressively more value.
    if conf < 75:
        penalty = min(0.020, max(0.0, (75.0 - conf) * 0.0015))
        bet_edge += penalty
        strong_edge += penalty
        bet_ev += penalty * 1.25
        strong_ev += penalty * 1.25
    elif conf >= 82:
        bonus = min(0.0075, (conf - 82.0) * 0.00075)
        bet_edge = max(0.025, bet_edge - bonus)
        strong_edge = max(0.045, strong_edge - bonus)
        bet_ev = max(0.040, bet_ev - bonus)
        strong_ev = max(0.075, strong_ev - bonus)

    # Huge projection gaps are not rewarded. They can still be a normal BET
    # if the adjusted probability/EV clears the bar, but cannot be A/Best Bet.
    cap_strong = False
    if projection_gap is not None and review_gap is not None:
        try:
            if abs(float(projection_gap)) >= review_gap:
                cap_strong = True
        except Exception:
            pass

    # Very long dogs can be estimated and even shown as a lean, but our old
    # historical ML feed was too unreliable to auto-promote +300 or longer.
    long_dog_cap = market_type in {"moneyline", "ml", "money line"} and o >= 300

    if (
        not cap_strong
        and not long_dog_cap
        and edge >= strong_edge
        and ev >= strong_ev
        and conf >= 68
    ):
        verdict = "STRONG BET"
    elif (
        not long_dog_cap
        and edge >= bet_edge
        and ev >= bet_ev
        and conf >= 65
    ):
        verdict = "BET"
    elif edge >= lean_edge and ev >= lean_ev:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return verdict, edge, ev, imp


def display_grade(verdict):
    return {
        "STRONG BET": "BEST BET",
        "BET": "BET",
        "LEAN": "LEAN",
        "PASS": "PASS",
        "NO LINE": "NO LINE",
    }.get(str(verdict), str(verdict))


def playable_stake(verdict, edge, confidence):
    """
    Conservative entertainment-oriented unit guidance.
    No Kelly sizing and no forced bets.
    """
    try:
        e = float(edge)
        c = float(confidence)
    except Exception:
        return 0.0

    if verdict == "STRONG BET":
        return 1.00
    if verdict == "BET":
        # Better B plays can reach 0.75u; ordinary B plays stay 0.50u.
        return 0.75 if (e >= 0.055 and c >= 72) else 0.50
    if verdict == "LEAN":
        return 0.25
    return 0.0




def verdict_meta(verdict):
    mapping = {
        "STRONG BET": {
            "grade": "A",
            "label": "BEST BET",
            "emoji": "🟢",
            "class": "grade-a",
        },
        "BET": {
            "grade": "B",
            "label": "BET",
            "emoji": "🔵",
            "class": "grade-b",
        },
        "LEAN": {
            "grade": "C",
            "label": "LEAN",
            "emoji": "🟡",
            "class": "grade-c",
        },
        "PASS": {
            "grade": "D",
            "label": "PASS",
            "emoji": "⚪",
            "class": "grade-d",
        },
        "NO LINE": {
            "grade": "—",
            "label": "NO LINE",
            "emoji": "⚪",
            "class": "grade-d",
        },
    }
    return mapping.get(str(verdict), mapping["PASS"])


def render_recommendation_card(verdict, bet_label, odds, prob=None, edge=None, ev=None, fair=None, stake=None):
    meta = verdict_meta(verdict)
    cls = meta["grade"].lower() if meta["grade"] in {"A","B","C","D"} else "d"

    try:
        odds_txt = f"{int(odds):+d}"
    except Exception:
        odds_txt = str(odds)

    title = f"{bet_label} {odds_txt}".strip()

    if verdict == "PASS":
        st.markdown(
            f"""
            <div class="no-play">
              <div class="result-top">
                <div class="result-badge">PASS</div>
                <div>
                  <div class="result-label">NO PLAY RECOMMENDED</div>
                  <div class="no-play-title">{title}</div>
                </div>
              </div>
              <div class="no-play-sub">
                The best available market still falls below the model's playable threshold.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    label = {
        "STRONG BET":"BEST BET",
        "BET":"BET",
        "LEAN":"LEAN",
    }.get(verdict, verdict)

    metric_html = []
    if prob is not None and pd.notna(prob):
        metric_html.append(("Model", f"{float(prob)*100:.1f}%"))
    if edge is not None and pd.notna(edge):
        metric_html.append(("Edge", f"{float(edge)*100:+.1f}%"))
    if ev is not None and pd.notna(ev):
        metric_html.append(("EV", f"{float(ev)*100:+.1f}%"))
    if fair is not None:
        try:
            metric_html.append(("Fair", f"{int(round(float(fair))):+d}"))
        except Exception:
            pass

    metrics = "".join(
        f'<div class="metric-chip"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k,v in metric_html[:4]
    )

    stake_txt = ""
    if stake is not None and float(stake) > 0:
        stake_txt = f'<div class="result-stake">{float(stake):.2f}u</div>'

    st.markdown(
        f"""
        <div class="result-hero {cls}">
          <div class="result-top">
            <div class="result-badge">{meta['label']}</div>
            <div>
              <div class="result-label">{label}</div>
              <div class="result-pick">{title}</div>
            </div>
            {stake_txt}
          </div>
          <div class="result-metrics">{metrics}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_row(verdict, bet_label, odds, prob=None, edge=None, ev=None, fair=None):
    meta = verdict_meta(verdict)
    cls = meta["grade"].lower() if meta["grade"] in {"A","B","C","D"} else "d"

    try:
        odds_txt = f"{int(odds):+d}"
    except Exception:
        odds_txt = str(odds)

    pieces = []
    if prob is not None and pd.notna(prob):
        pieces.append(f"{float(prob)*100:.1f}% model")
    if edge is not None and pd.notna(edge):
        pieces.append(f"{float(edge)*100:+.1f}% edge")
    if ev is not None and pd.notna(ev):
        pieces.append(f"{float(ev)*100:+.1f}% EV")
    if fair is not None:
        try:
            pieces.append(f"fair {int(round(float(fair))):+d}")
        except Exception:
            pass

    sub = " • ".join(pieces)

    st.markdown(
        f"""
        <div class="market-card">
          <div class="market-grade {cls}">{meta['label']}</div>
          <div class="market-main">
            <div class="market-pick">{bet_label} {odds_txt}</div>
            <div class="market-sub">{sub}</div>
          </div>
          <div class="market-tag">{meta['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

# ===== v0.4.0 backtest engine =====

def grade_v031(prob, odds, confidence=75):
    """Exact v0.3.1 betting-layer grading retained for A/B comparison."""
    if not _valid_american_odds(odds):
        return "PASS", 0.0, 0.0, None
    imp = implied_prob(odds)
    edge = prob - imp
    ev = expected_value(prob, odds)
    me, mv = juice_thresholds(odds)
    if confidence >= 80 and edge >= me + .02 and ev >= mv + .03:
        verdict = "STRONG BET"
    elif confidence >= 70 and edge >= me and ev >= mv:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"
    return verdict, edge, ev, imp


def _bt_prior_only_data(data):
    """
    Leakage-safe Stage 1 input set.
    Removes current-season SP/SRS/PPA/advanced results and retains prior-season
    performance plus current-season talent/returning-production inputs.
    This is intentionally a preseason-prior backtest, not a claim that it exactly
    recreates the live in-season engine.
    """
    d = dict(data)
    d["sp_current"] = {}
    d["srs_current"] = {}
    d["ppa_current"] = {}
    d["adv_current"] = {}
    return d


def _bt_project_game(game, data, hfa=DEFAULT_HFA):
    """
    Historical projection path used by Backtest mode.
    It reproduces the football-rating math while setting travel/weather to zero,
    because the live environmental layer is not archived point-in-time in this app.
    """
    away, home = game["awayTeam"], game["homeTeam"]
    week = game.get("week", 1)
    neutral = bool(game.get("neutralSite"))
    applied_hfa = 0.0 if neutral else float(hfa)

    away_power, ar = _team_base_power(away, data, week)
    home_power, hr = _team_base_power(home, data, week)
    away_match = _matchup_adjustment(ar, hr)
    home_match = _matchup_adjustment(hr, ar)
    matchup_margin_adj = home_match - away_match
    base_margin = home_power - away_power
    home_margin = base_margin + matchup_margin_adj + applied_hfa

    raw_sp_total = _total_from_sp(ar, hr)
    efficiency_adj = _total_efficiency_adjustment(ar, hr)
    pace_adj = _pace_adjustment(ar, hr)
    total = raw_sp_total + efficiency_adj + pace_adj
    total = max(34.0, min(82.0, total))

    home_score = max(7.0, (total + home_margin) / 2.0)
    away_score = max(7.0, (total - home_margin) / 2.0)
    total = home_score + away_score
    home_margin = home_score - away_score

    margin_sd, total_sd, confidence, completeness = _uncertainty(week, ar, hr)
    home_wp = 1.0 - NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)

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
        "away_win_prob": 1.0 - home_wp,
        "neutral": neutral,
        "hfa": applied_hfa,
        "week": week,
        "margin_sd": margin_sd,
        "total_sd": total_sd,
        "confidence": confidence,
        "data_completeness": completeness,
        "components": {
            "base_power_margin": base_margin,
            "matchup_margin_adjustment": matchup_margin_adj,
            "hfa_adjustment": applied_hfa,
            "sp_total_base": raw_sp_total,
            "efficiency_total_adjustment": efficiency_adj,
            "pace_total_adjustment": pace_adj,
            "environment_margin_adjustment": 0.0,
            "environment_total_adjustment": 0.0,
        },
    }



def _residual_training_rows_for_season(season, scope="Major FBS"):
    """Build point-in-time-safe residual training rows for one completed season."""
    games = get_backtest_games(int(season))
    line_payload = get_backtest_lines(int(season))
    data = _bt_prior_only_data(get_backtest_model_data(int(season)))

    line_index = {}
    for lr in line_payload or []:
        gid = lr.get("id")
        if gid is None:
            continue
        try:
            key = int(gid)
        except Exception:
            key = gid
        line_index[key] = normalize_game_lines([lr], game_id=gid)

    rows = []
    for g in games or []:
        if g.get("completed") is not True:
            continue
        if g.get("homePoints") is None or g.get("awayPoints") is None:
            continue
        if not _bt_game_scope(g, scope):
            continue

        gid = g.get("id")
        try:
            key = int(gid)
        except Exception:
            key = gid
        market = _bt_consensus_line(line_index.get(key, []))
        if market.get("home_spread") is None and market.get("total") is None:
            continue

        try:
            p = _bt_project_game(g, data, hfa=DEFAULT_HFA)
        except Exception:
            continue

        sf, tf = _residual_feature_dict(p, market)
        actual_margin = float(g["homePoints"]) - float(g["awayPoints"])
        actual_total = float(g["homePoints"]) + float(g["awayPoints"])

        if sf is not None:
            rows.append({
                "market_type": "spread",
                "season": int(season),
                **sf,
                "target": actual_margin - (-float(market["home_spread"])),
            })
        if tf is not None:
            rows.append({
                "market_type": "total",
                "season": int(season),
                **tf,
                "target": actual_total - float(market["total"]),
            })
    return rows


@st.cache_data(ttl=86400, show_spinner=False)
def _fit_residual_models_cached(train_seasons_tuple, scope="Major FBS"):
    all_rows = []
    for season in train_seasons_tuple:
        all_rows.extend(_residual_training_rows_for_season(int(season), scope))
    df = pd.DataFrame(all_rows)

    out = {"spread": None, "total": None, "train_seasons": list(train_seasons_tuple)}
    if df.empty:
        return out

    sd = df[df["market_type"] == "spread"].dropna(
        subset=RESIDUAL_SPREAD_FEATURES + ["target"]
    )
    td = df[df["market_type"] == "total"].dropna(
        subset=RESIDUAL_TOTAL_FEATURES + ["target"]
    )

    if len(sd) >= RESIDUAL_MIN_ROWS:
        out["spread"] = _ridge_fit_numpy(
            sd[RESIDUAL_SPREAD_FEATURES].values,
            sd["target"].values,
        )
    if len(td) >= RESIDUAL_MIN_ROWS:
        out["total"] = _ridge_fit_numpy(
            td[RESIDUAL_TOTAL_FEATURES].values,
            td["target"].values,
        )
    return out


def fit_residual_models_before_season(test_season, scope="Major FBS"):
    test_season = int(test_season)
    train_seasons = tuple(range(RESIDUAL_TRAIN_START, test_season))
    return _fit_residual_models_cached(train_seasons, scope)


def fit_live_residual_models(current_season, scope="Major FBS"):
    """2026 live model trains only on completed prior seasons."""
    return fit_residual_models_before_season(int(current_season), scope)

def _bt_consensus_line(rows):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    def med(col):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return None if s.empty else float(s.median())
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


def _bt_profit(result, odds):
    if result == "PUSH":
        return 0.0
    if result != "WIN":
        return -1.0
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _bt_settle(market_type, side, home_points, away_points, line=None):
    hp, ap = float(home_points), float(away_points)
    if market_type == "moneyline":
        if hp == ap:
            return "PUSH"
        won = (hp > ap) if side == "home" else (ap > hp)
        return "WIN" if won else "LOSS"
    if market_type == "spread":
        home_spread = float(line)
        margin = hp - ap + home_spread
        if abs(margin) < 1e-9:
            home_result = "PUSH"
        else:
            home_result = "WIN" if margin > 0 else "LOSS"
        if side == "home":
            return home_result
        return "PUSH" if home_result == "PUSH" else ("LOSS" if home_result == "WIN" else "WIN")
    if market_type == "total":
        diff = hp + ap - float(line)
        if abs(diff) < 1e-9:
            return "PUSH"
        if side == "over":
            return "WIN" if diff > 0 else "LOSS"
        return "WIN" if diff < 0 else "LOSS"
    return None


def _bt_game_scope(game, scope):
    major_confs = {"ACC", "SEC", "Big Ten", "Big 12", "Pac-12"}
    major_ind = {"Notre Dame"}
    def cls(side):
        return str(game.get(f"{side}Classification") or "").lower()
    def conf(side):
        return str(game.get(f"{side}Conference") or "")
    def team(side):
        return str(game.get(f"{side}Team") or "")
    def is_fbs(side):
        c = cls(side)
        return c == "fbs" if c else bool(conf(side))
    def is_major(side):
        return conf(side) in major_confs or team(side) in major_ind
    if scope == "Major FBS":
        return is_major("home") or is_major("away")
    if scope == "All FBS":
        return is_fbs("home") or is_fbs("away")
    return True


def _bt_candidate_rows(game, p, market, season, version):
    """Create graded individual markets for one historical game/version."""
    rows = []
    week = int(game.get("week") or 1)
    raw_home_spread = float(p["model_home_spread"])
    raw_total = float(p["model_total"])

    if version == "v0.4.0":
        residual_models = fit_residual_models_before_season(int(season), "Major FBS")
        rp = residual_market_projection(p, market, residual_models)
        margin_sd, total_sd = rp["spread_sigma"], rp["total_sigma"]
        adj_home_spread = rp["adjusted_home_spread"]
        adj_total = rp["adjusted_total"]
        side_weight, side_shrink = 0.0, rp["spread_correction"]
        total_weight, total_shrink = 0.0, rp["total_correction"]
        home_margin = -adj_home_spread

        # ML remains conservative legacy-market calibration. v0.4 is a spread residual model.
        ml_spread, _, _ = calibrated_market_projection(
            raw_home_spread, market.get("home_spread"), week, "side"
        )
        ml_margin = -ml_spread
        home_wp = 1.0 - NormalDist(mu=ml_margin, sigma=max(margin_sd, 15.0)).cdf(0)
        away_wp = 1.0 - home_wp
        grader = grade
    elif version == "v0.3.2":
        margin_sd, total_sd = calibrated_sigmas(p["margin_sd"], p["total_sd"], week)
        adj_home_spread, side_weight, side_shrink = calibrated_market_projection(
            raw_home_spread, market.get("home_spread"), week, "side"
        )
        adj_total, total_weight, total_shrink = calibrated_market_projection(
            raw_total, market.get("total"), week, "total"
        )
        home_margin = -adj_home_spread
        home_wp = 1.0 - NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)
        away_wp = 1.0 - home_wp
        grader = grade
    else:
        margin_sd, total_sd = float(p["margin_sd"]), float(p["total_sd"])
        adj_home_spread, side_weight, side_shrink = raw_home_spread, 1.0, 0.0
        adj_total, total_weight, total_shrink = raw_total, 1.0, 0.0
        home_margin = -raw_home_spread
        home_wp = 1.0 - NormalDist(mu=home_margin, sigma=margin_sd).cdf(0)
        away_wp = 1.0 - home_wp
        grader = grade_v031

    base = {
        "season": season,
        "week": week,
        "game_id": game.get("id"),
        "away_team": game.get("awayTeam"),
        "home_team": game.get("homeTeam"),
        "home_points": game.get("homePoints"),
        "away_points": game.get("awayPoints"),
        "version": version,
        "confidence": p["confidence"],
        "raw_model_home_spread": raw_home_spread,
        "adjusted_model_home_spread": adj_home_spread,
        "market_home_spread": market.get("home_spread"),
        "raw_model_total": raw_total,
        "adjusted_model_total": adj_total,
        "market_total": market.get("total"),
        "side_market_weight": side_weight,
        "side_shrink_points": side_shrink,
        "total_market_weight": total_weight,
        "total_shrink_points": total_shrink,
        "margin_sd": margin_sd,
        "total_sd": total_sd,
    }

    # Moneyline
    for side, prob, odds, name in [
        ("away", away_wp, market.get("away_ml"), f"{p['away']} ML"),
        ("home", home_wp, market.get("home_ml"), f"{p['home']} ML"),
    ]:
        if not _valid_american_odds(odds):
            continue
        if version == "v0.3.2":
            verdict, edge, ev, imp = grader(prob, odds, p["confidence"], market_type="side", week=week)
        else:
            verdict, edge, ev, imp = grader(prob, odds, p["confidence"])
        result = _bt_settle("moneyline", side, game["homePoints"], game["awayPoints"])
        rows.append({**base, "market_type":"moneyline", "side":side, "market":name,
                     "line":None, "odds":int(odds), "prob":prob, "implied_prob":imp,
                     "edge":edge, "ev":ev, "verdict":verdict, "result":result,
                     "profit_units":_bt_profit(result, odds)})

    # Spread; CFBD generic feed does not reliably carry side-specific juice, so -110.
    if market.get("home_spread") is not None:
        line = float(market["home_spread"])
        if version == "v0.4.0":
            hp = rp["home_cover_prob"]
            ap = rp["away_cover_prob"]
            spread_gap = rp["spread_correction"]
        else:
            hp = cover_probability(home_margin, line, "home", margin_sd)
            ap = 1.0 - hp
            spread_gap = raw_home_spread - line
        for side, prob, name in [
            ("home", hp, f"{p['home']} {line:+.1f}"),
            ("away", ap, f"{p['away']} {-line:+.1f}"),
        ]:
            if version == "v0.3.2":
                verdict, edge, ev, imp = grader(prob, -110, p["confidence"], market_type="spread",
                                                 projection_gap=spread_gap, week=week)
            else:
                verdict, edge, ev, imp = grader(prob, -110, p["confidence"])
            result = _bt_settle("spread", side, game["homePoints"], game["awayPoints"], line)
            rows.append({**base, "market_type":"spread", "side":side, "market":name,
                         "line":line if side=="home" else -line, "odds":-110, "prob":prob,
                         "implied_prob":imp, "edge":edge, "ev":ev, "verdict":verdict,
                         "result":result, "profit_units":_bt_profit(result, -110)})

    # Total; -110 generic price assumption.
    if market.get("total") is not None:
        line = float(market["total"])
        if version == "v0.4.0":
            op = rp["over_prob"]
            up = rp["under_prob"]
            total_gap = rp["total_correction"]
        else:
            op = total_probability(adj_total, line, "over", total_sd)
            up = 1.0 - op
            total_gap = raw_total - line
        for side, prob, name in [
            ("over", op, f"Over {line:g}"),
            ("under", up, f"Under {line:g}"),
        ]:
            if version == "v0.3.2":
                verdict, edge, ev, imp = grader(prob, -110, p["confidence"], market_type="total",
                                                 projection_gap=total_gap, week=week)
            else:
                verdict, edge, ev, imp = grader(prob, -110, p["confidence"])
            if version == "v0.4.0":
                verdict = cap_total_research_verdict(verdict)
            result = _bt_settle("total", side, game["homePoints"], game["awayPoints"], line)
            rows.append({**base, "market_type":"total", "side":side, "market":name,
                         "line":line, "odds":-110, "prob":prob, "implied_prob":imp,
                         "edge":edge, "ev":ev, "verdict":verdict, "result":result,
                         "profit_units":_bt_profit(result, -110)})
    return rows


def _bt_best_per_game(df):
    if df.empty:
        return df
    d = df.copy()
    rank = {"STRONG BET": 4, "BET": 3, "LEAN": 2, "PASS": 1}
    d["_vrank"] = d["verdict"].map(rank).fillna(0)
    d = d.sort_values(["version","season","game_id","_vrank","ev","edge"],
                      ascending=[True,True,True,False,False,False])
    return d.groupby(["version","season","game_id"], as_index=False).head(1).drop(columns=["_vrank"])


def _bt_summary(df, label="All"):
    if df.empty:
        return {"Sample":label,"Bets":0,"W-L-P":"0-0-0","Win %":None,"ROI":None,"Profit (u)":0.0}
    bets = df[df["verdict"].isin(["BET","STRONG BET"])].copy()
    w = int((bets["result"]=="WIN").sum())
    l = int((bets["result"]=="LOSS").sum())
    p = int((bets["result"]=="PUSH").sum())
    denom = w+l
    roi = bets["profit_units"].sum()/len(bets) if len(bets) else None
    return {
        "Sample": label,
        "Bets": int(len(bets)),
        "W-L-P": f"{w}-{l}-{p}",
        "Win %": (w/denom) if denom else None,
        "ROI": roi,
        "Profit (u)": float(bets["profit_units"].sum()) if len(bets) else 0.0,
    }


def _bt_error_table(game_df):
    if game_df.empty:
        return pd.DataFrame()
    rows=[]
    for version, d in game_df.groupby("version"):
        actual_margin = d["home_points"] - d["away_points"]
        market_margin = -d["market_home_spread"]
        raw_margin = -d["raw_model_home_spread"]
        adj_margin = -d["adjusted_model_home_spread"]
        actual_total = d["home_points"] + d["away_points"]
        rows.append({
            "Version": version,
            "Games": len(d),
            "Raw spread MAE": (actual_margin-raw_margin).abs().mean(),
            "Adjusted spread MAE": (actual_margin-adj_margin).abs().mean(),
            "Market spread MAE": (actual_margin-market_margin).abs().mean(),
            "Raw total MAE": (actual_total-d["raw_model_total"]).abs().mean(),
            "Adjusted total MAE": (actual_total-d["adjusted_model_total"]).abs().mean(),
            "Market total MAE": (actual_total-d["market_total"]).abs().mean(),
        })
    return pd.DataFrame(rows)


# ===== v0.5.0 residual-market model =====

RESIDUAL_VERSION = "v0.5.0-residual"

def _ridge_fit(df, feature_cols, target_col, alpha=10.0):
    """Standardized ridge regression with an unpenalized intercept."""
    d = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < max(40, len(feature_cols) * 8):
        return None

    X = d[feature_cols].astype(float).to_numpy()
    y = d[target_col].astype(float).to_numpy()

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd

    X1 = np.column_stack([np.ones(len(Z)), Z])
    penalty = np.eye(X1.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X1.T @ X1 + penalty) @ X1.T @ y

    return {
        "features": list(feature_cols),
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "alpha": float(alpha),
        "n": int(len(d)),
    }

def _ridge_predict(model, df):
    if model is None or df.empty:
        return np.full(len(df), np.nan)
    X = df[model["features"]].replace([np.inf, -np.inf], np.nan).astype(float)
    good = X.notna().all(axis=1).to_numpy()
    out = np.full(len(df), np.nan)
    if good.any():
        Z = (X.to_numpy()[good] - model["mu"]) / model["sd"]
        X1 = np.column_stack([np.ones(len(Z)), Z])
        out[good] = X1 @ model["beta"]
    return out

def _residual_feature_frame(game_df):
    """
    One row per game with targets expressed as the amount by which the market
    missed the final margin/total. The model is trained to predict that residual,
    rather than trying to predict the entire score from scratch.
    """
    if game_df.empty:
        return pd.DataFrame()

    d = game_df.sort_values(["season", "game_id"]).drop_duplicates(["season", "game_id"]).copy()

    d["actual_margin"] = pd.to_numeric(d["home_points"], errors="coerce") - pd.to_numeric(d["away_points"], errors="coerce")
    d["actual_total"] = pd.to_numeric(d["home_points"], errors="coerce") + pd.to_numeric(d["away_points"], errors="coerce")
    d["market_margin"] = -pd.to_numeric(d["market_home_spread"], errors="coerce")
    d["raw_model_margin"] = -pd.to_numeric(d["raw_model_home_spread"], errors="coerce")

    d["spread_target_residual"] = d["actual_margin"] - d["market_margin"]
    d["spread_model_delta"] = d["raw_model_margin"] - d["market_margin"]
    d["total_target_residual"] = d["actual_total"] - pd.to_numeric(d["market_total"], errors="coerce")
    d["total_model_delta"] = pd.to_numeric(d["raw_model_total"], errors="coerce") - pd.to_numeric(d["market_total"], errors="coerce")

    d["abs_market_margin"] = d["market_margin"].abs()
    d["week_num"] = pd.to_numeric(d["week"], errors="coerce").clip(lower=1, upper=16)
    d["confidence_num"] = pd.to_numeric(d["confidence"], errors="coerce")

    d["sp_base_minus_market"] = pd.to_numeric(d["base_power_margin"], errors="coerce") - d["market_margin"]
    d["sp_matchup_adj"] = pd.to_numeric(d["matchup_margin_adjustment"], errors="coerce")
    d["sp_hfa"] = pd.to_numeric(d["hfa_adjustment"], errors="coerce")

    d["total_base_minus_market"] = pd.to_numeric(d["sp_total_base"], errors="coerce") - pd.to_numeric(d["market_total"], errors="coerce")
    d["total_eff_adj"] = pd.to_numeric(d["efficiency_total_adjustment"], errors="coerce")
    d["total_pace_adj"] = pd.to_numeric(d["pace_total_adjustment"], errors="coerce")

    # Interpretable football signals for v0.6.0 research.
    for col in [
        "away_sp_rating","home_sp_rating",
        "away_srs_adjustment","home_srs_adjustment",
        "away_talent_adjustment","home_talent_adjustment",
        "away_returning_adjustment","home_returning_adjustment",
    ]:
        if col not in d.columns:
            d[col] = np.nan

    d["sp_rating_diff"] = pd.to_numeric(d["home_sp_rating"], errors="coerce") - pd.to_numeric(d["away_sp_rating"], errors="coerce")
    d["srs_adjustment_diff"] = pd.to_numeric(d["home_srs_adjustment"], errors="coerce") - pd.to_numeric(d["away_srs_adjustment"], errors="coerce")
    d["talent_adjustment_diff"] = pd.to_numeric(d["home_talent_adjustment"], errors="coerce") - pd.to_numeric(d["away_talent_adjustment"], errors="coerce")
    d["returning_adjustment_diff"] = pd.to_numeric(d["home_returning_adjustment"], errors="coerce") - pd.to_numeric(d["away_returning_adjustment"], errors="coerce")

    # ===== v0.7 matchup-specific features =====
    def n(col):
        if col not in d.columns:
            d[col] = np.nan
        return pd.to_numeric(d[col], errors="coerce")

    # PPA: offense vs the opponent defense, then home matchup edge minus away matchup edge.
    d["home_pass_matchup"] = n("home_ppa_off_pass") - n("away_ppa_def_pass")
    d["away_pass_matchup"] = n("away_ppa_off_pass") - n("home_ppa_def_pass")
    d["net_pass_matchup"] = d["home_pass_matchup"] - d["away_pass_matchup"]

    d["home_rush_matchup"] = n("home_ppa_off_rush") - n("away_ppa_def_rush")
    d["away_rush_matchup"] = n("away_ppa_off_rush") - n("home_ppa_def_rush")
    d["net_rush_matchup"] = d["home_rush_matchup"] - d["away_rush_matchup"]

    # Advanced success-rate matchup. Lower defensive success allowed is better.
    d["home_success_matchup"] = n("home_adv_off_success") - n("away_adv_def_success")
    d["away_success_matchup"] = n("away_adv_off_success") - n("home_adv_def_success")
    d["net_success_matchup"] = d["home_success_matchup"] - d["away_success_matchup"]

    # Explosiveness matchup.
    d["home_expl_matchup"] = n("home_adv_off_expl") - n("away_adv_def_expl")
    d["away_expl_matchup"] = n("away_adv_off_expl") - n("home_adv_def_expl")
    d["net_expl_matchup"] = d["home_expl_matchup"] - d["away_expl_matchup"]

    # Advanced play-level PPA splits.
    d["home_adv_pass_matchup"] = n("home_adv_off_pass_ppa") - n("away_adv_def_pass_ppa")
    d["away_adv_pass_matchup"] = n("away_adv_off_pass_ppa") - n("home_adv_def_pass_ppa")
    d["net_adv_pass_matchup"] = d["home_adv_pass_matchup"] - d["away_adv_pass_matchup"]

    d["home_adv_rush_matchup"] = n("home_adv_off_rush_ppa") - n("away_adv_def_rush_ppa")
    d["away_adv_rush_matchup"] = n("away_adv_off_rush_ppa") - n("home_adv_def_rush_ppa")
    d["net_adv_rush_matchup"] = d["home_adv_rush_matchup"] - d["away_adv_rush_matchup"]

    # Finishing drives / points per opportunity.
    d["home_finishing_matchup"] = n("home_adv_off_ppo") - n("away_adv_def_ppo")
    d["away_finishing_matchup"] = n("away_adv_off_ppo") - n("home_adv_def_ppo")
    d["net_finishing_matchup"] = d["home_finishing_matchup"] - d["away_finishing_matchup"]

    # Defensive disruption advantage.
    d["havoc_diff"] = n("home_adv_def_havoc") - n("away_adv_def_havoc")

    # Pace/context.
    away_ppd = n("away_adv_off_plays") / n("away_adv_off_drives").replace(0, np.nan)
    home_ppd = n("home_adv_off_plays") / n("home_adv_off_drives").replace(0, np.nan)
    d["avg_plays_per_drive"] = (away_ppd + home_ppd) / 2.0
    d["plays_per_drive_diff"] = home_ppd - away_ppd

    # Preseason personnel continuity context.
    d["returning_pass_diff"] = n("home_returning_pass") - n("away_returning_pass")
    d["returning_usage_diff"] = n("home_returning_usage") - n("away_returning_usage")

    return d

SPREAD_RESIDUAL_FEATURES = [
    "spread_model_delta",
    "sp_base_minus_market",
    "sp_matchup_adj",
    "sp_hfa",
    "market_margin",
    "abs_market_margin",
    "week_num",
    "confidence_num",
]

TOTAL_RESIDUAL_FEATURES = [
    "total_model_delta",
    "total_base_minus_market",
    "total_eff_adj",
    "total_pace_adj",
    "market_total",
    "week_num",
    "confidence_num",
]

def _choose_ridge_alpha(dev, features, target, seasons):
    """
    Tune regularization on the latest development season only.
    Earlier development seasons train the candidate models.
    The untouched holdout is never used here.
    """
    seasons = sorted([int(s) for s in seasons])
    if len(seasons) < 2:
        return 10.0, pd.DataFrame()

    val_season = seasons[-1]
    train_seasons = seasons[:-1]
    train = dev[dev["season"].isin(train_seasons)].copy()
    val = dev[dev["season"] == val_season].copy()

    rows = []
    for alpha in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]:
        model = _ridge_fit(train, features, target, alpha=alpha)
        pred = _ridge_predict(model, val)
        y = pd.to_numeric(val[target], errors="coerce").to_numpy()
        mask = np.isfinite(pred) & np.isfinite(y)
        mae = float(np.mean(np.abs(y[mask] - pred[mask]))) if mask.any() else np.nan
        rows.append({"alpha": alpha, "validation_season": val_season, "validation_mae": mae})

    tbl = pd.DataFrame(rows)
    usable = tbl.dropna(subset=["validation_mae"])
    if usable.empty:
        return 10.0, tbl
    best = float(usable.sort_values(["validation_mae", "alpha"]).iloc[0]["alpha"])
    return best, tbl

def _fit_residual_models(feature_df, holdout):
    """
    1) Tune alpha only inside development seasons.
    2) Refit on every development season.
    3) Holdout season is predicted exactly once.
    """
    dev = feature_df[feature_df["season"] != holdout].copy()
    hold = feature_df[feature_df["season"] == holdout].copy()
    dev_seasons = sorted(dev["season"].dropna().astype(int).unique().tolist())

    sp_alpha, sp_cv = _choose_ridge_alpha(
        dev, SPREAD_RESIDUAL_FEATURES, "spread_target_residual", dev_seasons
    )
    tot_alpha, tot_cv = _choose_ridge_alpha(
        dev, TOTAL_RESIDUAL_FEATURES, "total_target_residual", dev_seasons
    )

    sp_model = _ridge_fit(dev, SPREAD_RESIDUAL_FEATURES, "spread_target_residual", sp_alpha)
    tot_model = _ridge_fit(dev, TOTAL_RESIDUAL_FEATURES, "total_target_residual", tot_alpha)

    hold = hold.copy()
    hold["pred_spread_residual"] = _ridge_predict(sp_model, hold)
    hold["pred_total_residual"] = _ridge_predict(tot_model, hold)

    # Empirical forecast-error widths from development data.
    sp_dev_pred = _ridge_predict(sp_model, dev)
    tot_dev_pred = _ridge_predict(tot_model, dev)
    sp_y = pd.to_numeric(dev["spread_target_residual"], errors="coerce").to_numpy()
    tot_y = pd.to_numeric(dev["total_target_residual"], errors="coerce").to_numpy()

    sp_mask = np.isfinite(sp_dev_pred) & np.isfinite(sp_y)
    tot_mask = np.isfinite(tot_dev_pred) & np.isfinite(tot_y)

    sp_err = sp_y[sp_mask] - sp_dev_pred[sp_mask]
    tot_err = tot_y[tot_mask] - tot_dev_pred[tot_mask]

    spread_sd = float(np.std(sp_err, ddof=1)) if len(sp_err) > 2 else BASE_MARGIN_SD
    total_sd = float(np.std(tot_err, ddof=1)) if len(tot_err) > 2 else BASE_TOTAL_SD
    spread_sd = max(spread_sd, 11.0)
    total_sd = max(total_sd, 11.0)

    diagnostics = {
        "spread_alpha": sp_alpha,
        "total_alpha": tot_alpha,
        "spread_sd": spread_sd,
        "total_sd": total_sd,
        "spread_n": 0 if sp_model is None else sp_model["n"],
        "total_n": 0 if tot_model is None else tot_model["n"],
        "spread_cv": sp_cv,
        "total_cv": tot_cv,
    }
    return hold, diagnostics

def _grade_residual(prob, odds, market_type, residual_points):
    """
    Residual version: official bets require both probability/EV support and a
    meaningful predicted market error in points. No moneyline bets are issued
    in v0.5.0 until historical ML quality is separately audited.
    """
    if not _valid_american_odds(odds):
        return "PASS", 0.0, 0.0, None

    imp = implied_prob(odds)
    edge = prob - imp
    ev = expected_value(prob, odds)
    r = abs(float(residual_points))

    if market_type == "spread":
        min_pts, bet_edge, bet_ev = 2.0, 0.030, 0.050
        strong_pts, strong_edge, strong_ev = 3.5, 0.055, 0.090
    else:
        min_pts, bet_edge, bet_ev = 2.5, 0.035, 0.060
        strong_pts, strong_edge, strong_ev = 4.0, 0.060, 0.100

    if r >= strong_pts and edge >= strong_edge and ev >= strong_ev:
        verdict = "STRONG BET"
    elif r >= min_pts and edge >= bet_edge and ev >= bet_ev:
        verdict = "BET"
    elif edge > 0 and ev > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"
    return verdict, edge, ev, imp

def _bt_residual_candidate_rows(row, spread_sd, total_sd):
    """
    Grade the untouched holdout from a market-first forecast:
        fair margin = market margin + predicted market error
        fair total  = market total  + predicted market error
    """
    rows = []
    season = int(row["season"])
    week = int(row["week"])
    hp = float(row["home_points"])
    ap = float(row["away_points"])
    market_spread = row.get("market_home_spread")
    market_total = row.get("market_total")

    base = {
        "season": season,
        "week": week,
        "game_id": row.get("game_id"),
        "away_team": row.get("away_team"),
        "home_team": row.get("home_team"),
        "home_points": hp,
        "away_points": ap,
        "version": RESIDUAL_VERSION,
        "confidence": row.get("confidence"),
        "raw_model_home_spread": row.get("raw_model_home_spread"),
        "adjusted_model_home_spread": np.nan,
        "market_home_spread": market_spread,
        "raw_model_total": row.get("raw_model_total"),
        "adjusted_model_total": np.nan,
        "market_total": market_total,
        "side_market_weight": np.nan,
        "side_shrink_points": np.nan,
        "total_market_weight": np.nan,
        "total_shrink_points": np.nan,
        "margin_sd": spread_sd,
        "total_sd": total_sd,
        "predicted_spread_residual": row.get("pred_spread_residual"),
        "predicted_total_residual": row.get("pred_total_residual"),
    }

    # v0.5.0 intentionally disables ML betting until the historical ML feed is audited.
    if pd.notna(market_spread) and pd.notna(row.get("pred_spread_residual")):
        line = float(market_spread)
        market_margin = -line
        pred_resid = float(row["pred_spread_residual"])
        fair_margin = market_margin + pred_resid
        fair_spread = -fair_margin
        base["adjusted_model_home_spread"] = fair_spread

        home_cover = 1.0 - NormalDist(mu=fair_margin, sigma=spread_sd).cdf(-line)
        away_cover = 1.0 - home_cover

        for side, prob, name, line_out in [
            ("home", home_cover, f"{row['home_team']} {line:+.1f}", line),
            ("away", away_cover, f"{row['away_team']} {-line:+.1f}", -line),
        ]:
            verdict, edge, ev, imp = _grade_residual(
                prob, -110, "spread", pred_resid
            )
            result = _bt_settle("spread", side, hp, ap, line)
            rows.append({
                **base,
                "market_type": "spread",
                "side": side,
                "market": name,
                "line": line_out,
                "odds": -110,
                "prob": prob,
                "implied_prob": imp,
                "edge": edge,
                "ev": ev,
                "verdict": verdict,
                "result": result,
                "profit_units": _bt_profit(result, -110),
            })

    if pd.notna(market_total) and pd.notna(row.get("pred_total_residual")):
        line = float(market_total)
        pred_resid = float(row["pred_total_residual"])
        fair_total = line + pred_resid
        base["adjusted_model_total"] = fair_total

        over_prob = 1.0 - NormalDist(mu=fair_total, sigma=total_sd).cdf(line)
        under_prob = 1.0 - over_prob

        for side, prob, name in [
            ("over", over_prob, f"Over {line:g}"),
            ("under", under_prob, f"Under {line:g}"),
        ]:
            verdict, edge, ev, imp = _grade_residual(
                prob, -110, "total", pred_resid
            )
            result = _bt_settle("total", side, hp, ap, line)
            rows.append({
                **base,
                "market_type": "total",
                "side": side,
                "market": name,
                "line": line,
                "odds": -110,
                "prob": prob,
                "implied_prob": imp,
                "edge": edge,
                "ev": ev,
                "verdict": verdict,
                "result": result,
                "profit_units": _bt_profit(result, -110),
            })
    return rows

def _residual_holdout_error_table(hold_df):
    if hold_df.empty:
        return pd.DataFrame()
    rows = []

    s = hold_df.dropna(subset=["market_margin", "spread_target_residual", "pred_spread_residual"])
    if len(s):
        market_mae = float(np.mean(np.abs(s["spread_target_residual"])))
        residual_mae = float(np.mean(np.abs(s["spread_target_residual"] - s["pred_spread_residual"])))
        rows.append({
            "Market": "Spread",
            "Games": len(s),
            "Market-only MAE": market_mae,
            "Residual-model MAE": residual_mae,
            "Improvement": market_mae - residual_mae,
        })

    t = hold_df.dropna(subset=["market_total", "total_target_residual", "pred_total_residual"])
    if len(t):
        market_mae = float(np.mean(np.abs(t["total_target_residual"])))
        residual_mae = float(np.mean(np.abs(t["total_target_residual"] - t["pred_total_residual"])))
        rows.append({
            "Market": "Total",
            "Games": len(t),
            "Market-only MAE": market_mae,
            "Residual-model MAE": residual_mae,
            "Improvement": market_mae - residual_mae,
        })
    return pd.DataFrame(rows)


# ===== v0.5.1 rolling walk-forward validation =====

WALKFORWARD_VERSION = "v0.5.1-walkforward"

def _fit_residual_models_for_cutoff(feature_df, test_season):
    """
    Train on all seasons strictly before test_season.
    Tune ridge alpha using the latest available development season only.
    Predict test_season once.
    """
    train_all = feature_df[feature_df["season"] < test_season].copy()
    test = feature_df[feature_df["season"] == test_season].copy()
    train_seasons = sorted(train_all["season"].dropna().astype(int).unique().tolist())

    if len(train_seasons) < 1 or test.empty:
        return pd.DataFrame(), {}

    # If only one prior season exists, use conservative fixed regularization.
    if len(train_seasons) == 1:
        sp_alpha = 10.0
        tot_alpha = 10.0
        sp_cv = pd.DataFrame()
        tot_cv = pd.DataFrame()
    else:
        sp_alpha, sp_cv = _choose_ridge_alpha(
            train_all, SPREAD_RESIDUAL_FEATURES, "spread_target_residual", train_seasons
        )
        tot_alpha, tot_cv = _choose_ridge_alpha(
            train_all, TOTAL_RESIDUAL_FEATURES, "total_target_residual", train_seasons
        )

    sp_model = _ridge_fit(train_all, SPREAD_RESIDUAL_FEATURES, "spread_target_residual", sp_alpha)
    tot_model = _ridge_fit(train_all, TOTAL_RESIDUAL_FEATURES, "total_target_residual", tot_alpha)

    out = test.copy()
    out["pred_spread_residual"] = _ridge_predict(sp_model, out)
    out["pred_total_residual"] = _ridge_predict(tot_model, out)

    sp_train_pred = _ridge_predict(sp_model, train_all)
    tot_train_pred = _ridge_predict(tot_model, train_all)

    sp_y = pd.to_numeric(train_all["spread_target_residual"], errors="coerce").to_numpy()
    tot_y = pd.to_numeric(train_all["total_target_residual"], errors="coerce").to_numpy()

    sp_mask = np.isfinite(sp_train_pred) & np.isfinite(sp_y)
    tot_mask = np.isfinite(tot_train_pred) & np.isfinite(tot_y)

    sp_err = sp_y[sp_mask] - sp_train_pred[sp_mask]
    tot_err = tot_y[tot_mask] - tot_train_pred[tot_mask]

    spread_sd = float(np.std(sp_err, ddof=1)) if len(sp_err) > 2 else BASE_MARGIN_SD
    total_sd = float(np.std(tot_err, ddof=1)) if len(tot_err) > 2 else BASE_TOTAL_SD
    spread_sd = max(spread_sd, 11.0)
    total_sd = max(total_sd, 11.0)

    diag = {
        "test_season": int(test_season),
        "train_seasons": train_seasons,
        "spread_alpha": sp_alpha,
        "total_alpha": tot_alpha,
        "spread_sd": spread_sd,
        "total_sd": total_sd,
        "spread_cv": sp_cv,
        "total_cv": tot_cv,
    }
    return out, diag

def _run_walkforward_residual(feature_df):
    """
    Rolling out-of-sample validation:
      train 2022 -> test 2023
      train 2022-23 -> test 2024
      train 2022-24 -> test 2025
    Generalizes automatically to the seasons present in the selected backtest.
    """
    seasons = sorted(feature_df["season"].dropna().astype(int).unique().tolist())
    if len(seasons) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    all_bets = []
    all_holdout_rows = []
    diagnostics = {}

    for test_season in seasons[1:]:
        test_df, diag = _fit_residual_models_for_cutoff(feature_df, test_season)
        if test_df.empty:
            continue

        diagnostics[test_season] = diag
        all_holdout_rows.append(test_df)

        for _, rr in test_df.iterrows():
            rows = _bt_residual_candidate_rows(rr, diag["spread_sd"], diag["total_sd"])
            for x in rows:
                x["version"] = WALKFORWARD_VERSION
                x["walkforward_test_season"] = int(test_season)

                # v0.5.1 is spread-first:
                # totals remain research-only and cannot become official BETs.
                if x.get("market_type") == "total" and x.get("verdict") in {"BET", "STRONG BET"}:
                    x["verdict"] = "LEAN"
                    x["research_only"] = True
                else:
                    x["research_only"] = False

                all_bets.append(x)

    bets_df = pd.DataFrame(all_bets)
    holdout_df = pd.concat(all_holdout_rows, ignore_index=True) if all_holdout_rows else pd.DataFrame()
    return bets_df, holdout_df, diagnostics

def _walkforward_season_summary(signal_df):
    if signal_df.empty:
        return pd.DataFrame()

    rows = []
    official = signal_df[
        (signal_df["version"] == WALKFORWARD_VERSION)
        & (signal_df["market_type"] == "spread")
        & (signal_df["verdict"].isin(["BET", "STRONG BET"]))
    ].copy()

    for season in sorted(official["season"].dropna().astype(int).unique().tolist()):
        s = official[official["season"] == season]
        wins = int((s["result"] == "WIN").sum())
        losses = int((s["result"] == "LOSS").sum())
        pushes = int((s["result"] == "PUSH").sum())
        decided = wins + losses
        units = float(s["profit_units"].sum()) if len(s) else 0.0
        roi = units / len(s) if len(s) else np.nan
        rows.append({
            "Test season": season,
            "Spread bets": len(s),
            "W": wins,
            "L": losses,
            "P": pushes,
            "Win %": wins / decided if decided else np.nan,
            "Units": units,
            "ROI": roi,
        })
    return pd.DataFrame(rows)

def _walkforward_error_summary(holdout_df):
    if holdout_df.empty:
        return pd.DataFrame()

    rows = []
    for season in sorted(holdout_df["season"].dropna().astype(int).unique().tolist()):
        s = holdout_df[holdout_df["season"] == season].copy()

        sp = s.dropna(subset=["spread_target_residual", "pred_spread_residual"])
        if len(sp):
            mkt = float(np.mean(np.abs(sp["spread_target_residual"])))
            res = float(np.mean(np.abs(sp["spread_target_residual"] - sp["pred_spread_residual"])))
            rows.append({
                "Test season": season,
                "Market": "Spread",
                "Games": len(sp),
                "Market-only MAE": mkt,
                "Residual MAE": res,
                "Improvement": mkt - res,
            })

        tot = s.dropna(subset=["total_target_residual", "pred_total_residual"])
        if len(tot):
            mkt = float(np.mean(np.abs(tot["total_target_residual"])))
            res = float(np.mean(np.abs(tot["total_target_residual"] - tot["pred_total_residual"])))
            rows.append({
                "Test season": season,
                "Market": "Total",
                "Games": len(tot),
                "Market-only MAE": mkt,
                "Residual MAE": res,
                "Improvement": mkt - res,
            })

    return pd.DataFrame(rows)


# ===== v0.6.0 signal research =====

SIGNAL_RESEARCH_VERSION = "v0.6.0-signal-research"

SIGNAL_CANDIDATES = {
    "Raw model disagreement": "spread_model_delta",
    "Base power vs market": "sp_base_minus_market",
    "SP+ rating differential": "sp_rating_diff",
    "SRS adjustment differential": "srs_adjustment_diff",
    "Talent adjustment differential": "talent_adjustment_diff",
    "Returning production differential": "returning_adjustment_diff",
    "Matchup adjustment": "sp_matchup_adj",
    "Home-field adjustment": "sp_hfa",
    "Market favorite size": "market_margin",
    "Absolute favorite size": "abs_market_margin",
    "Week": "week_num",
    "Model confidence": "confidence_num",
}

def _ats_profit(result, odds=-110):
    return _bt_profit(result, odds)

def _signal_directional_set(df, feature, direction, cutoff):
    """
    A research-only ATS set. Positive direction means back the home side when
    feature >= cutoff; negative direction means back away when feature <= cutoff.
    """
    d = df.copy()
    x = pd.to_numeric(d[feature], errors="coerce")
    if direction > 0:
        s = d[x >= cutoff].copy()
        side = "home"
    else:
        s = d[x <= cutoff].copy()
        side = "away"

    if s.empty:
        return pd.DataFrame()

    s["research_side"] = side
    s["research_result"] = s.apply(
        lambda r: _bt_settle(
            "spread",
            side,
            float(r["home_points"]),
            float(r["away_points"]),
            float(r["market_home_spread"]),
        ),
        axis=1,
    )
    s["research_profit_units"] = s["research_result"].map(lambda r: _ats_profit(r, -110))
    return s

def _signal_stats(s):
    if s.empty:
        return {"bets":0, "wins":0, "losses":0, "pushes":0, "win_pct":np.nan, "units":0.0, "roi":np.nan}
    w = int((s["research_result"]=="WIN").sum())
    l = int((s["research_result"]=="LOSS").sum())
    p = int((s["research_result"]=="PUSH").sum())
    n = len(s)
    decided = w + l
    units = float(s["research_profit_units"].sum())
    return {
        "bets": n, "wins": w, "losses": l, "pushes": p,
        "win_pct": w/decided if decided else np.nan,
        "units": units,
        "roi": units/n if n else np.nan,
    }

def _research_one_signal(dev, hold, label, feature):
    """
    Development sample chooses direction and extreme-quartile cutoff.
    Holdout is then evaluated with that exact frozen rule.
    This is descriptive signal research, not a promoted betting strategy.
    """
    cols = [feature, "spread_target_residual", "market_home_spread", "home_points", "away_points"]
    d = dev[cols].replace([np.inf,-np.inf], np.nan).dropna().copy()
    h = hold[cols].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(d) < 100 or len(h) < 50 or d[feature].nunique() < 5:
        return None

    corr = d[feature].corr(d["spread_target_residual"])
    if pd.isna(corr):
        return None
    direction = 1 if corr >= 0 else -1
    cutoff = float(d[feature].quantile(0.75 if direction > 0 else 0.25))

    dev_set = _signal_directional_set(d, feature, direction, cutoff)
    hold_set = _signal_directional_set(h, feature, direction, cutoff)
    ds = _signal_stats(dev_set)
    hs = _signal_stats(hold_set)

    return {
        "Signal": label,
        "Feature": feature,
        "Dev corr": float(corr),
        "Direction": "Home on high values" if direction > 0 else "Away on low values",
        "Frozen cutoff": cutoff,
        "Dev bets": ds["bets"],
        "Dev win %": ds["win_pct"],
        "Dev ROI": ds["roi"],
        "Holdout bets": hs["bets"],
        "Holdout W-L-P": f'{hs["wins"]}-{hs["losses"]}-{hs["pushes"]}',
        "Holdout win %": hs["win_pct"],
        "Holdout units": hs["units"],
        "Holdout ROI": hs["roi"],
    }

def _run_signal_research(feature_df, holdout):
    """
    Search only development seasons for simple, interpretable spread signals.
    The selected holdout is used only to report whether each frozen signal survives.
    """
    dev = feature_df[feature_df["season"] != holdout].copy()
    hold = feature_df[feature_df["season"] == holdout].copy()

    rows = []
    for label, feature in SIGNAL_CANDIDATES.items():
        if feature not in feature_df.columns:
            continue
        r = _research_one_signal(dev, hold, label, feature)
        if r is not None:
            rows.append(r)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Rank on development evidence only. Holdout performance is NEVER part of ranking.
    out["Dev score"] = (
        out["Dev ROI"].fillna(-9) * np.sqrt(out["Dev bets"].clip(lower=1))
        + out["Dev corr"].abs().fillna(0)
    )
    return out.sort_values(["Dev score","Dev bets"], ascending=[False,False]).reset_index(drop=True)

def _walkforward_signal_research(feature_df):
    """
    Repeats the frozen-signal exercise through time. Each test year only uses
    prior seasons to choose direction/cutoff.
    """
    seasons = sorted(feature_df["season"].dropna().astype(int).unique().tolist())
    rows = []
    for test_season in seasons[1:]:
        dev = feature_df[feature_df["season"] < test_season].copy()
        hold = feature_df[feature_df["season"] == test_season].copy()
        for label, feature in SIGNAL_CANDIDATES.items():
            if feature not in feature_df.columns:
                continue
            r = _research_one_signal(dev, hold, label, feature)
            if r is None:
                continue
            rows.append({
                "Test season": test_season,
                "Signal": label,
                "Direction": r["Direction"],
                "Frozen cutoff": r["Frozen cutoff"],
                "Bets": r["Holdout bets"],
                "W-L-P": r["Holdout W-L-P"],
                "Win %": r["Holdout win %"],
                "Units": r["Holdout units"],
                "ROI": r["Holdout ROI"],
            })
    return pd.DataFrame(rows)


# ===== v0.7.0 matchup residual model =====

MATCHUP_VERSION = "v0.7.0-matchup"

MATCHUP_SPREAD_FEATURES = [
    # Market/context anchor
    "market_margin",
    "abs_market_margin",
    "week_num",
    "sp_hfa",

    # Rating/personnel context
    "sp_rating_diff",
    "talent_adjustment_diff",
    "returning_adjustment_diff",
    "returning_pass_diff",
    "returning_usage_diff",

    # Actual matchup interactions
    "net_pass_matchup",
    "net_rush_matchup",
    "net_success_matchup",
    "net_expl_matchup",
    "net_adv_pass_matchup",
    "net_adv_rush_matchup",
    "net_finishing_matchup",
    "havoc_diff",

    # Pace/style
    "avg_plays_per_drive",
    "plays_per_drive_diff",
]

def _ridge_feature_importance(model):
    if not model:
        return pd.DataFrame()
    rows = []
    for feature, beta in zip(model["features"], model["beta"][1:]):
        rows.append({
            "Feature": feature,
            "Standardized coefficient": float(beta),
            "Absolute importance": abs(float(beta)),
        })
    return pd.DataFrame(rows).sort_values("Absolute importance", ascending=False).reset_index(drop=True)

def _choose_matchup_alpha(train_df, seasons):
    """
    Choose regularization entirely inside the development sample.
    The latest available development season is the validation fold.
    """
    return _choose_ridge_alpha(
        train_df,
        MATCHUP_SPREAD_FEATURES,
        "spread_target_residual",
        seasons,
    )

def _fit_matchup_model_for_cutoff(feature_df, test_season):
    train = feature_df[feature_df["season"] < test_season].copy()
    test = feature_df[feature_df["season"] == test_season].copy()
    seasons = sorted(train["season"].dropna().astype(int).unique().tolist())

    if train.empty or test.empty:
        return pd.DataFrame(), None, {}

    if len(seasons) <= 1:
        alpha, cv = 10.0, pd.DataFrame()
    else:
        alpha, cv = _choose_matchup_alpha(train, seasons)

    model = _ridge_fit(
        train,
        MATCHUP_SPREAD_FEATURES,
        "spread_target_residual",
        alpha=alpha,
    )
    test = test.copy()
    test["matchup_pred_residual"] = _ridge_predict(model, test)

    train_pred = _ridge_predict(model, train)
    y = pd.to_numeric(train["spread_target_residual"], errors="coerce").to_numpy()
    mask = np.isfinite(train_pred) & np.isfinite(y)
    err = y[mask] - train_pred[mask]
    sd = float(np.std(err, ddof=1)) if len(err) > 2 else BASE_MARGIN_SD
    sd = max(11.0, min(20.0, sd))

    diag = {
        "test_season": int(test_season),
        "train_seasons": seasons,
        "alpha": float(alpha),
        "forecast_sd": sd,
        "cv": cv,
        "importance": _ridge_feature_importance(model),
        "n_train": 0 if model is None else model["n"],
    }
    return test, model, diag

def _matchup_candidate_rows(test_df, diag):
    rows = []
    if test_df.empty:
        return pd.DataFrame()

    sd = float(diag.get("forecast_sd", BASE_MARGIN_SD))
    for _, r in test_df.iterrows():
        if pd.isna(r.get("market_home_spread")) or pd.isna(r.get("matchup_pred_residual")):
            continue

        line = float(r["market_home_spread"])
        market_margin = -line
        pred_resid = float(r["matchup_pred_residual"])
        fair_margin = market_margin + pred_resid
        fair_spread = -fair_margin

        home_cover = 1.0 - NormalDist(mu=fair_margin, sigma=sd).cdf(-line)
        away_cover = 1.0 - home_cover

        # Fixed research hurdle. No historical threshold optimization.
        side = "home" if pred_resid > 0 else "away"
        prob = home_cover if side == "home" else away_cover
        line_out = line if side == "home" else -line
        market_name = (
            f"{r['home_team']} {line:+.1f}" if side == "home"
            else f"{r['away_team']} {-line:+.1f}"
        )

        imp = implied_prob(-110)
        edge = prob - imp
        ev = expected_value(prob, -110)

        # v0.7 is still a lab: classifications are research labels, not live bets.
        abs_resid = abs(pred_resid)
        if abs_resid >= 3.0 and edge >= 0.035 and ev >= 0.05:
            verdict = "RESEARCH BET"
        elif abs_resid >= 2.0 and edge > 0 and ev > 0:
            verdict = "RESEARCH LEAN"
        else:
            verdict = "PASS"

        result = _bt_settle(
            "spread",
            side,
            float(r["home_points"]),
            float(r["away_points"]),
            line,
        )
        rows.append({
            "version": MATCHUP_VERSION,
            "season": int(r["season"]),
            "week": int(r["week"]),
            "game_id": r.get("game_id"),
            "away_team": r.get("away_team"),
            "home_team": r.get("home_team"),
            "market_type": "spread",
            "side": side,
            "market": market_name,
            "line": line_out,
            "odds": -110,
            "market_home_spread": line,
            "market_margin": market_margin,
            "matchup_pred_residual": pred_resid,
            "matchup_fair_home_spread": fair_spread,
            "prob": prob,
            "implied_prob": imp,
            "edge": edge,
            "ev": ev,
            "verdict": verdict,
            "result": result,
            "profit_units": _bt_profit(result, -110),
            "research_only": True,
        })
    return pd.DataFrame(rows)

def _run_matchup_walkforward(feature_df):
    seasons = sorted(feature_df["season"].dropna().astype(int).unique().tolist())
    all_tests = []
    all_bets = []
    diagnostics = {}

    for test_season in seasons[1:]:
        test, model, diag = _fit_matchup_model_for_cutoff(feature_df, test_season)
        if test.empty:
            continue
        diagnostics[test_season] = diag
        all_tests.append(test)
        bets = _matchup_candidate_rows(test, diag)
        if not bets.empty:
            all_bets.append(bets)

    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    return tests_df, bets_df, diagnostics

def _matchup_mae_table(tests_df):
    if tests_df.empty:
        return pd.DataFrame()
    rows = []
    for season in sorted(tests_df["season"].dropna().astype(int).unique()):
        s = tests_df[tests_df["season"] == season].dropna(
            subset=["spread_target_residual", "matchup_pred_residual"]
        )
        if s.empty:
            continue
        market_mae = float(np.mean(np.abs(s["spread_target_residual"])))
        matchup_mae = float(np.mean(np.abs(
            s["spread_target_residual"] - s["matchup_pred_residual"]
        )))
        rows.append({
            "Test season": int(season),
            "Games": len(s),
            "Market-only MAE": market_mae,
            "v0.7 matchup MAE": matchup_mae,
            "Improvement": market_mae - matchup_mae,
        })
    return pd.DataFrame(rows)

def _matchup_bet_table(bets_df):
    if bets_df.empty:
        return pd.DataFrame()

    rows = []
    q = bets_df[bets_df["verdict"] == "RESEARCH BET"].copy()
    for season in sorted(q["season"].dropna().astype(int).unique()):
        s = q[q["season"] == season]
        w = int((s["result"] == "WIN").sum())
        l = int((s["result"] == "LOSS").sum())
        p = int((s["result"] == "PUSH").sum())
        decided = w + l
        units = float(s["profit_units"].sum())
        rows.append({
            "Test season": int(season),
            "Bets": len(s),
            "W-L-P": f"{w}-{l}-{p}",
            "Win %": w / decided if decided else np.nan,
            "Units": units,
            "ROI": units / len(s) if len(s) else np.nan,
        })

    if len(q):
        w = int((q["result"] == "WIN").sum())
        l = int((q["result"] == "LOSS").sum())
        p = int((q["result"] == "PUSH").sum())
        decided = w + l
        units = float(q["profit_units"].sum())
        rows.append({
            "Test season": "Combined",
            "Bets": len(q),
            "W-L-P": f"{w}-{l}-{p}",
            "Win %": w / decided if decided else np.nan,
            "Units": units,
            "ROI": units / len(q) if len(q) else np.nan,
        })
    return pd.DataFrame(rows)

def _fit_matchup_final_holdout(feature_df, holdout):
    """
    Main final-exam view: train on every season before the user-selected holdout,
    then evaluate that holdout once.
    """
    test, model, diag = _fit_matchup_model_for_cutoff(feature_df, int(holdout))
    bets = _matchup_candidate_rows(test, diag) if not test.empty else pd.DataFrame()
    return test, bets, diag


# ===== v0.8.0 cover classification model =====

CLASSIFIER_VERSION = "v0.8.0-cover-classifier"

# Reuse the matchup-first feature set, but solve the actual wagering question:
# P(home covers the sportsbook spread) rather than "predict the final margin better."
CLASSIFIER_FEATURES = [
    "market_margin",
    "abs_market_margin",
    "week_num",
    "sp_hfa",

    "sp_rating_diff",
    "talent_adjustment_diff",
    "returning_adjustment_diff",
    "returning_pass_diff",
    "returning_usage_diff",

    "net_pass_matchup",
    "net_rush_matchup",
    "net_success_matchup",
    "net_expl_matchup",
    "net_adv_pass_matchup",
    "net_adv_rush_matchup",
    "net_finishing_matchup",
    "havoc_diff",

    "avg_plays_per_drive",
    "plays_per_drive_diff",
]

def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))

def _logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return np.nan
    y = y[m]
    p = p[m]
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def _brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return np.nan
    return float(np.mean((y[m] - p[m])**2))

def _classifier_feature_frame(feature_df):
    d = feature_df.copy()

    # Settlement target: 1 if home covers, 0 if away covers. Pushes are excluded
    # from fitting and probability-score evaluation.
    home_margin = pd.to_numeric(d["home_points"], errors="coerce") - pd.to_numeric(d["away_points"], errors="coerce")
    home_spread = pd.to_numeric(d["market_home_spread"], errors="coerce")
    ats_margin = home_margin + home_spread

    d["home_cover_target"] = np.where(
        ats_margin > 0, 1.0,
        np.where(ats_margin < 0, 0.0, np.nan)
    )

    # Convenience market labels.
    d["favorite_side"] = np.where(home_spread < 0, "home",
                           np.where(home_spread > 0, "away", "pickem"))
    return d

def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu, sd

def _logistic_fit(df, features, target_col, l2=1.0, max_iter=60, tol=1e-7):
    cols = [c for c in features if c in df.columns]
    if not cols:
        return None

    work = df[cols + [target_col]].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna()

    if len(work) < max(40, len(cols) * 3):
        return None

    X = work[cols].to_numpy(dtype=float)
    y = work[target_col].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return None

    mu, sd = _standardize_fit(X)
    Z = (X - mu) / sd
    Z = np.column_stack([np.ones(len(Z)), Z])

    beta = np.zeros(Z.shape[1], dtype=float)

    # Newton-Raphson / IRLS with L2 penalty on slopes, not intercept.
    penalty = np.eye(Z.shape[1]) * float(l2)
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        eta = Z @ beta
        p = _sigmoid(eta)
        w = np.clip(p * (1 - p), 1e-6, None)

        grad = Z.T @ (y - p) - penalty @ beta
        hess = -(Z.T @ (Z * w[:, None])) - penalty

        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad

        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return {
        "features": cols,
        "mu": mu,
        "sd": sd,
        "beta": beta,
        "l2": float(l2),
        "n": len(work),
    }

def _logistic_predict(model, df):
    if model is None or df.empty:
        return np.full(len(df), np.nan)

    X = np.column_stack([
        pd.to_numeric(df.get(c, np.nan), errors="coerce").to_numpy(dtype=float)
        for c in model["features"]
    ])
    finite = np.all(np.isfinite(X), axis=1)
    out = np.full(len(df), np.nan)
    if not finite.any():
        return out

    Z = (X[finite] - model["mu"]) / model["sd"]
    Z = np.column_stack([np.ones(len(Z)), Z])
    out[finite] = _sigmoid(Z @ model["beta"])
    return out

def _choose_logistic_l2(train_df, seasons):
    """
    Hyperparameter selection stays inside the development sample.
    The latest available development season becomes the validation fold.
    """
    grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    seasons = sorted([int(s) for s in seasons])

    if len(seasons) <= 1:
        return 10.0, pd.DataFrame()

    val_season = seasons[-1]
    tr = train_df[train_df["season"] < val_season].copy()
    va = train_df[train_df["season"] == val_season].copy()

    rows = []
    for l2 in grid:
        m = _logistic_fit(tr, CLASSIFIER_FEATURES, "home_cover_target", l2=l2)
        p = _logistic_predict(m, va)
        y = pd.to_numeric(va["home_cover_target"], errors="coerce").to_numpy()
        rows.append({
            "l2": l2,
            "log_loss": _logloss(y, p),
            "brier": _brier(y, p),
        })

    cv = pd.DataFrame(rows)
    valid = cv.dropna(subset=["log_loss"])
    if valid.empty:
        return 10.0, cv

    best = valid.sort_values(["log_loss", "brier", "l2"], ascending=[True, True, True]).iloc[0]
    return float(best["l2"]), cv

def _classifier_importance(model):
    if not model:
        return pd.DataFrame()
    rows = []
    for feature, beta in zip(model["features"], model["beta"][1:]):
        rows.append({
            "Feature": feature,
            "Standardized coefficient": float(beta),
            "Absolute importance": abs(float(beta)),
        })
    return pd.DataFrame(rows).sort_values("Absolute importance", ascending=False).reset_index(drop=True)

def _fit_classifier_for_cutoff(feature_df, test_season):
    d = _classifier_feature_frame(feature_df)
    train = d[d["season"] < test_season].copy()
    test = d[d["season"] == test_season].copy()
    seasons = sorted(train["season"].dropna().astype(int).unique().tolist())

    if train.empty or test.empty:
        return pd.DataFrame(), None, {}

    if len(seasons) <= 1:
        l2, cv = 10.0, pd.DataFrame()
    else:
        l2, cv = _choose_logistic_l2(train, seasons)

    model = _logistic_fit(train, CLASSIFIER_FEATURES, "home_cover_target", l2=l2)
    test = test.copy()
    test["p_home_cover"] = _logistic_predict(model, test)
    test["p_away_cover"] = 1.0 - test["p_home_cover"]

    diag = {
        "test_season": int(test_season),
        "train_seasons": seasons,
        "l2": float(l2),
        "cv": cv,
        "importance": _classifier_importance(model),
        "n_train": 0 if model is None else model["n"],
    }
    return test, model, diag

def _classifier_research_rows(test_df):
    rows = []
    if test_df.empty:
        return pd.DataFrame()

    breakeven = implied_prob(-110)

    for _, r in test_df.iterrows():
        ph = r.get("p_home_cover")
        if ph is None or pd.isna(ph):
            continue

        ph = float(ph)
        pa = 1.0 - ph
        side = "home" if ph >= pa else "away"
        prob = ph if side == "home" else pa
        confidence_edge = prob - 0.5

        line = float(r["market_home_spread"])
        side_line = line if side == "home" else -line
        market_name = (
            f"{r['home_team']} {line:+.1f}" if side == "home"
            else f"{r['away_team']} {-line:+.1f}"
        )

        edge = prob - breakeven
        ev = expected_value(prob, -110)

        # Fixed research thresholds. These are intentionally not optimized
        # against 2025 or any other holdout.
        if prob >= 0.58 and edge > 0 and ev > 0:
            verdict = "RESEARCH BET 58%+"
        elif prob >= 0.56 and edge > 0 and ev > 0:
            verdict = "RESEARCH BET 56-58%"
        elif prob >= 0.54 and edge > 0 and ev > 0:
            verdict = "RESEARCH LEAN 54-56%"
        elif prob >= breakeven and edge > 0 and ev > 0:
            verdict = "RESEARCH LEAN 52.4-54%"
        else:
            verdict = "PASS"

        result = _bt_settle(
            "spread",
            side,
            float(r["home_points"]),
            float(r["away_points"]),
            line,
        )

        rows.append({
            "version": CLASSIFIER_VERSION,
            "season": int(r["season"]),
            "week": int(r["week"]),
            "game_id": r.get("game_id"),
            "away_team": r.get("away_team"),
            "home_team": r.get("home_team"),
            "side": side,
            "market": market_name,
            "line": side_line,
            "odds": -110,
            "p_home_cover": ph,
            "p_away_cover": pa,
            "model_pick_prob": prob,
            "confidence_edge_vs_50": confidence_edge,
            "breakeven_prob": breakeven,
            "edge": edge,
            "ev": ev,
            "verdict": verdict,
            "result": result,
            "profit_units": _bt_profit(result, -110),
            "research_only": True,
        })

    return pd.DataFrame(rows)

def _run_classifier_walkforward(feature_df):
    d = _classifier_feature_frame(feature_df)
    seasons = sorted(d["season"].dropna().astype(int).unique().tolist())
    all_tests = []
    all_rows = []
    diagnostics = {}

    for test_season in seasons[1:]:
        test, model, diag = _fit_classifier_for_cutoff(feature_df, test_season)
        if test.empty:
            continue

        diagnostics[test_season] = diag
        all_tests.append(test)

        rr = _classifier_research_rows(test)
        if not rr.empty:
            all_rows.append(rr)

    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return tests_df, rows_df, diagnostics

def _classifier_score_table(tests_df):
    if tests_df.empty:
        return pd.DataFrame()

    rows = []
    for season in sorted(tests_df["season"].dropna().astype(int).unique()):
        s = tests_df[tests_df["season"] == season].copy()
        y = pd.to_numeric(s["home_cover_target"], errors="coerce").to_numpy()
        p = pd.to_numeric(s["p_home_cover"], errors="coerce").to_numpy()

        # Naive benchmark: every game 50/50 ATS.
        base = np.full(len(s), 0.5)
        rows.append({
            "Test season": int(season),
            "Games": int(np.isfinite(y).sum()),
            "Classifier log loss": _logloss(y, p),
            "50/50 log loss": _logloss(y, base),
            "Log-loss improvement": _logloss(y, base) - _logloss(y, p),
            "Classifier Brier": _brier(y, p),
            "50/50 Brier": _brier(y, base),
            "Brier improvement": _brier(y, base) - _brier(y, p),
        })
    return pd.DataFrame(rows)

def _classifier_bucket_table(rows_df):
    if rows_df.empty:
        return pd.DataFrame()

    d = rows_df.copy()
    d["bucket"] = pd.cut(
        d["model_pick_prob"],
        bins=[0.5, implied_prob(-110), 0.54, 0.56, 0.58, 1.0],
        labels=["50-52.4%", "52.4-54%", "54-56%", "56-58%", "58%+"],
        include_lowest=True,
        right=False,
    )

    out = []
    for season in list(sorted(d["season"].dropna().astype(int).unique())) + ["Combined"]:
        s = d if season == "Combined" else d[d["season"] == season]
        for bucket, g in s.groupby("bucket", observed=False):
            if g.empty:
                continue
            w = int((g["result"] == "WIN").sum())
            l = int((g["result"] == "LOSS").sum())
            p = int((g["result"] == "PUSH").sum())
            decided = w + l
            units = float(g["profit_units"].sum())
            out.append({
                "Season": season,
                "Probability bucket": str(bucket),
                "Bets": len(g),
                "W-L-P": f"{w}-{l}-{p}",
                "Win %": w/decided if decided else np.nan,
                "Units": units,
                "ROI": units/len(g) if len(g) else np.nan,
                "Avg predicted prob": float(g["model_pick_prob"].mean()),
            })

    return pd.DataFrame(out)

def _classifier_calibration_table(tests_df):
    if tests_df.empty:
        return pd.DataFrame()

    rows = []
    for season in list(sorted(tests_df["season"].dropna().astype(int).unique())) + ["Combined"]:
        s = tests_df if season == "Combined" else tests_df[tests_df["season"] == season]
        s = s.dropna(subset=["p_home_cover", "home_cover_target"]).copy()
        if s.empty:
            continue

        s["cal_bin"] = pd.cut(
            s["p_home_cover"],
            bins=[0.0,0.40,0.45,0.50,0.55,0.60,1.0],
            labels=["<40%","40-45%","45-50%","50-55%","55-60%","60%+"],
            include_lowest=True,
        )
        for b, g in s.groupby("cal_bin", observed=False):
            if g.empty:
                continue
            rows.append({
                "Season": season,
                "Home-cover probability bin": str(b),
                "Games": len(g),
                "Avg predicted P(home cover)": float(g["p_home_cover"].mean()),
                "Actual home-cover rate": float(g["home_cover_target"].mean()),
                "Calibration gap": float(g["home_cover_target"].mean() - g["p_home_cover"].mean()),
            })
    return pd.DataFrame(rows)

def _fit_classifier_final_holdout(feature_df, holdout):
    test, model, diag = _fit_classifier_for_cutoff(feature_df, int(holdout))
    rows = _classifier_research_rows(test) if not test.empty else pd.DataFrame()
    return test, rows, diag


# ===== v0.8.1 56-58% signal audit =====

AUDIT_VERSION = "v0.8.1-signal-audit"

def _audit_spread_bucket(line_abs):
    if pd.isna(line_abs):
        return "Unknown"
    x = float(line_abs)
    if x <= 3.0:
        return "0-3"
    if x <= 7.0:
        return "3.5-7"
    if x <= 14.0:
        return "7.5-14"
    return "14+"

def _audit_week_bucket(week):
    try:
        w = int(week)
    except Exception:
        return "Unknown"
    if w <= 3:
        return "W1-3"
    if w <= 6:
        return "W4-6"
    if w <= 9:
        return "W7-9"
    return "W10+"

def _audit_prob_bucket(prob):
    if pd.isna(prob):
        return "Unknown"
    p = float(prob)
    if 0.56 <= p < 0.57:
        return "56-57%"
    if 0.57 <= p < 0.58:
        return "57-58%"
    return "Outside"

def _audit_binary_profile(val, label_pos, label_neg):
    if pd.isna(val):
        return "Unknown"
    return label_pos if float(val) >= 0 else label_neg

def _classifier_audit_frame(classifier_rows, classifier_tests):
    """
    Join v0.8 pick rows to the full classifier feature rows so the 56-58%
    bucket can be audited by market context and matchup characteristics.
    """
    if classifier_rows.empty or classifier_tests.empty:
        return pd.DataFrame()

    picks = classifier_rows.copy()
    picks = picks[
        (pd.to_numeric(picks["model_pick_prob"], errors="coerce") >= 0.56) &
        (pd.to_numeric(picks["model_pick_prob"], errors="coerce") < 0.58)
    ].copy()

    if picks.empty:
        return picks

    feature_cols = [
        "season","week","game_id","market_home_spread",
        "net_pass_matchup","net_rush_matchup","net_success_matchup",
        "net_expl_matchup","net_adv_pass_matchup","net_adv_rush_matchup",
        "net_finishing_matchup","havoc_diff","avg_plays_per_drive",
        "plays_per_drive_diff","sp_rating_diff","talent_adjustment_diff",
        "returning_adjustment_diff","returning_pass_diff","returning_usage_diff",
    ]
    available = [c for c in feature_cols if c in classifier_tests.columns]
    feat = classifier_tests[available].copy()

    keys = [c for c in ["season","week","game_id"] if c in feat.columns and c in picks.columns]
    if not keys:
        return pd.DataFrame()

    d = picks.merge(feat, on=keys, how="left", suffixes=("","_feature"))

    # Core market context.
    d["home_away"] = np.where(d["side"] == "home", "Home", "Away")

    home_spread = pd.to_numeric(d["market_home_spread"], errors="coerce")
    picked_is_favorite = np.where(
        d["side"] == "home",
        home_spread < 0,
        home_spread > 0
    )
    d["fav_dog"] = np.where(picked_is_favorite, "Favorite", "Underdog")
    d["abs_spread"] = home_spread.abs()
    d["spread_bucket"] = d["abs_spread"].map(_audit_spread_bucket)
    d["week_bucket"] = d["week"].map(_audit_week_bucket)
    d["prob_bucket"] = d["model_pick_prob"].map(_audit_prob_bucket)

    # Team-direction matchup signals from the perspective of the selected side.
    sign = np.where(d["side"] == "home", 1.0, -1.0)

    raw_signal_cols = [
        "net_pass_matchup","net_rush_matchup","net_success_matchup",
        "net_expl_matchup","net_adv_pass_matchup","net_adv_rush_matchup",
        "net_finishing_matchup","havoc_diff","plays_per_drive_diff",
        "sp_rating_diff","talent_adjustment_diff","returning_adjustment_diff",
        "returning_pass_diff","returning_usage_diff",
    ]
    for c in raw_signal_cols:
        if c in d.columns:
            d[f"pick_{c}"] = pd.to_numeric(d[c], errors="coerce") * sign

    # Interpretable positive/negative profiles.
    if "pick_net_pass_matchup" in d.columns:
        d["pass_profile"] = d["pick_net_pass_matchup"].map(
            lambda x: _audit_binary_profile(x, "Pass edge", "Pass disadvantage")
        )
    if "pick_net_rush_matchup" in d.columns:
        d["rush_profile"] = d["pick_net_rush_matchup"].map(
            lambda x: _audit_binary_profile(x, "Rush edge", "Rush disadvantage")
        )
    if "pick_net_expl_matchup" in d.columns:
        d["expl_profile"] = d["pick_net_expl_matchup"].map(
            lambda x: _audit_binary_profile(x, "Explosiveness edge", "Explosiveness disadvantage")
        )
    if "pick_havoc_diff" in d.columns:
        d["havoc_profile"] = d["pick_havoc_diff"].map(
            lambda x: _audit_binary_profile(x, "Havoc edge", "Havoc disadvantage")
        )
    if "pick_net_finishing_matchup" in d.columns:
        d["finishing_profile"] = d["pick_net_finishing_matchup"].map(
            lambda x: _audit_binary_profile(x, "Finishing edge", "Finishing disadvantage")
        )

    return d

def _audit_group_stats(df, group_cols, min_bets=1):
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False, observed=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        if n < min_bets:
            continue

        w = int((g["result"] == "WIN").sum())
        l = int((g["result"] == "LOSS").sum())
        p = int((g["result"] == "PUSH").sum())
        decided = w + l
        units = float(pd.to_numeric(g["profit_units"], errors="coerce").fillna(0).sum())

        row = {c: k for c, k in zip(group_cols, keys)}
        row.update({
            "Bets": n,
            "W-L-P": f"{w}-{l}-{p}",
            "Win %": w/decided if decided else np.nan,
            "Units": units,
            "ROI": units/n if n else np.nan,
            "Avg model prob": float(pd.to_numeric(g["model_pick_prob"], errors="coerce").mean()),
        })
        rows.append(row)

    return pd.DataFrame(rows)

def _audit_multiseason_survival(df, subgroup_col, min_bets_per_season=8):
    """
    Rank subgroups on multi-season survivability, not one-year peak ROI.
    A subgroup gets credit for each unseen season with enough sample and positive ROI.
    """
    if df.empty or subgroup_col not in df.columns:
        return pd.DataFrame()

    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    rows = []

    for subgroup, g in df.groupby(subgroup_col, dropna=False, observed=False):
        season_records = []
        positive = 0
        qualifying = 0
        total_units = 0.0
        total_bets = len(g)

        for season in seasons:
            s = g[g["season"] == season]
            if len(s) < min_bets_per_season:
                season_records.append(f"{season}: n={len(s)}")
                continue

            qualifying += 1
            units = float(pd.to_numeric(s["profit_units"], errors="coerce").fillna(0).sum())
            roi = units/len(s) if len(s) else np.nan
            if roi > 0:
                positive += 1
            total_units += units
            season_records.append(f"{season}: {len(s)} bets, {roi*100:+.1f}%")

        if qualifying == 0:
            continue

        combined_units = float(pd.to_numeric(g["profit_units"], errors="coerce").fillna(0).sum())
        combined_roi = combined_units / total_bets if total_bets else np.nan

        rows.append({
            "Subgroup": subgroup,
            "Total bets": total_bets,
            "Qualifying seasons": qualifying,
            "Positive seasons": positive,
            "Positive-season rate": positive/qualifying if qualifying else np.nan,
            "Combined units": combined_units,
            "Combined ROI": combined_roi,
            "Season detail": " | ".join(season_records),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["Positive-season rate","Qualifying seasons","Combined ROI","Total bets"],
        ascending=[False,False,False,False]
    ).reset_index(drop=True)

def _build_signal_audit_tables(classifier_rows, classifier_tests):
    d = _classifier_audit_frame(classifier_rows, classifier_tests)
    if d.empty:
        return d, {}

    tables = {}
    tables["home_away"] = _audit_group_stats(d, ["home_away"])
    tables["fav_dog"] = _audit_group_stats(d, ["fav_dog"])
    tables["spread_bucket"] = _audit_group_stats(d, ["spread_bucket"])
    tables["week_bucket"] = _audit_group_stats(d, ["week_bucket"])
    tables["prob_bucket"] = _audit_group_stats(d, ["prob_bucket"])

    for c in ["pass_profile","rush_profile","expl_profile","havoc_profile","finishing_profile"]:
        if c in d.columns:
            tables[c] = _audit_group_stats(d, [c])

    # Two-way splits that can reveal stable niches without exploding degrees of freedom.
    tables["homeaway_favdog"] = _audit_group_stats(d, ["home_away","fav_dog"], min_bets=5)
    tables["favdog_spread"] = _audit_group_stats(d, ["fav_dog","spread_bucket"], min_bets=5)
    tables["homeaway_spread"] = _audit_group_stats(d, ["home_away","spread_bucket"], min_bets=5)
    tables["prob_favdog"] = _audit_group_stats(d, ["prob_bucket","fav_dog"], min_bets=5)

    # Multi-season survival summaries for the main categorical dimensions.
    survival = []
    for col in [
        "home_away","fav_dog","spread_bucket","week_bucket","prob_bucket",
        "pass_profile","rush_profile","expl_profile","havoc_profile","finishing_profile"
    ]:
        if col not in d.columns:
            continue
        s = _audit_multiseason_survival(d, col, min_bets_per_season=8)
        if not s.empty:
            s.insert(0, "Dimension", col)
            survival.append(s)

    tables["survival"] = pd.concat(survival, ignore_index=True) if survival else pd.DataFrame()
    return d, tables


# ===== v0.9.0 locked candidate validation =====

CANDIDATE_VERSION = "v0.9.0-candidate-validation"
CANDIDATE_MIN_PROB = 0.56
CANDIDATE_MAX_PROB = 0.57
COVID_SEASON = 2020

def _wilson_interval(wins, losses, z=1.96):
    n = wins + losses
    if n <= 0:
        return (np.nan, np.nan)
    phat = wins / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z*np.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n))) / denom
    return max(0.0, center-half), min(1.0, center+half)

def _candidate_rows_from_classifier(rows_df):
    if rows_df.empty:
        return pd.DataFrame()
    d = rows_df.copy()
    p = pd.to_numeric(d["model_pick_prob"], errors="coerce")
    d = d[(p >= CANDIDATE_MIN_PROB) & (p < CANDIDATE_MAX_PROB)].copy()
    d["candidate_rule"] = "56.0%-<57.0%"
    d["is_covid_2020"] = pd.to_numeric(d["season"], errors="coerce") == COVID_SEASON
    return d

def _candidate_stats(df, label="All"):
    if df.empty:
        return {
            "Sample": label, "Bets": 0, "W-L-P": "0-0-0",
            "Win %": np.nan, "Units": 0.0, "ROI": np.nan,
            "Wilson low": np.nan, "Wilson high": np.nan,
            "Edge over -110 BE": np.nan,
        }

    w = int((df["result"] == "WIN").sum())
    l = int((df["result"] == "LOSS").sum())
    p = int((df["result"] == "PUSH").sum())
    decided = w + l
    units = float(pd.to_numeric(df["profit_units"], errors="coerce").fillna(0).sum())
    lo, hi = _wilson_interval(w, l)
    winp = w/decided if decided else np.nan
    be = implied_prob(-110)

    return {
        "Sample": label,
        "Bets": len(df),
        "W-L-P": f"{w}-{l}-{p}",
        "Win %": winp,
        "Units": units,
        "ROI": units/len(df) if len(df) else np.nan,
        "Wilson low": lo,
        "Wilson high": hi,
        "Edge over -110 BE": winp - be if pd.notna(winp) and be is not None else np.nan,
    }

def _candidate_season_table(candidate_df):
    if candidate_df.empty:
        return pd.DataFrame()

    rows = []
    for season in sorted(candidate_df["season"].dropna().astype(int).unique()):
        s = candidate_df[candidate_df["season"] == season]
        row = _candidate_stats(s, str(season))
        row["COVID flag"] = "COVID / separate" if season == COVID_SEASON else ""
        rows.append(row)

    # Primary aggregate explicitly excludes 2020.
    primary = candidate_df[candidate_df["season"] != COVID_SEASON]
    rows.append(_candidate_stats(primary, "Primary combined (ex-2020)"))
    rows.append(_candidate_stats(candidate_df, "Combined incl. 2020"))
    return pd.DataFrame(rows)

def _candidate_era_table(candidate_df):
    if candidate_df.empty:
        return pd.DataFrame()

    d = candidate_df.copy()
    season = pd.to_numeric(d["season"], errors="coerce")
    d["Era"] = np.select(
        [
            season.isin([2019]),
            season.isin([2020]),
            season.isin([2021, 2022]),
            season.isin([2023, 2024, 2025]),
        ],
        [
            "2019 pre-COVID test",
            "2020 COVID",
            "2021-22 post-COVID",
            "2023-25 recent",
        ],
        default="Other"
    )

    rows = []
    for era, g in d.groupby("Era", observed=False):
        rows.append(_candidate_stats(g, era))
    return pd.DataFrame(rows)

def _candidate_side_table(candidate_df):
    if candidate_df.empty:
        return pd.DataFrame()

    rows = []
    for col, values in [
        ("side", ["home", "away"]),
    ]:
        for value in values:
            g = candidate_df[candidate_df[col] == value]
            if not g.empty:
                rows.append(_candidate_stats(g, value.title()))

    # Favorite / underdog from selected side relative to home spread.
    if "market_home_spread" in candidate_df.columns:
        d = candidate_df.copy()
        hs = pd.to_numeric(d["market_home_spread"], errors="coerce")
        is_fav = np.where(d["side"] == "home", hs < 0, hs > 0)
        d["candidate_favdog"] = np.where(is_fav, "Favorite", "Underdog")
        for value, g in d.groupby("candidate_favdog", observed=False):
            rows.append(_candidate_stats(g, value))

    return pd.DataFrame(rows)

def _candidate_leave_one_season_out(candidate_df):
    """
    Stress test the locked candidate's dependence on any one non-COVID season.
    This is not model refitting; it is robustness reporting only.
    """
    if candidate_df.empty:
        return pd.DataFrame()

    base = candidate_df[candidate_df["season"] != COVID_SEASON].copy()
    seasons = sorted(base["season"].dropna().astype(int).unique())
    rows = []

    for omitted in seasons:
        g = base[base["season"] != omitted]
        row = _candidate_stats(g, f"Excluding {omitted}")
        row["Omitted season"] = omitted
        rows.append(row)
    return pd.DataFrame(rows)

def _run_classifier_walkforward_excluding_2020(feature_df):
    """
    COVID-excluded training stress test:
      - 2020 is never used as a training season
      - 2020 is not tested
      - every other test year is predicted only from earlier non-2020 seasons

    This addresses the concern that the unusual 2020 season could distort
    parameters carried into 2021+.
    """
    d = _classifier_feature_frame(feature_df)
    seasons = sorted(d["season"].dropna().astype(int).unique().tolist())
    test_seasons = [s for s in seasons[1:] if s != COVID_SEASON]

    all_tests = []
    all_rows = []
    diagnostics = {}

    for test_season in test_seasons:
        train = d[(d["season"] < test_season) & (d["season"] != COVID_SEASON)].copy()
        test = d[d["season"] == test_season].copy()
        train_seasons = sorted(train["season"].dropna().astype(int).unique().tolist())

        if train.empty or test.empty:
            continue

        if len(train_seasons) <= 1:
            l2, cv = 10.0, pd.DataFrame()
        else:
            # Hyperparameter choice remains fully inside the non-COVID development set.
            l2, cv = _choose_logistic_l2(train, train_seasons)

        model = _logistic_fit(train, CLASSIFIER_FEATURES, "home_cover_target", l2=l2)
        test["p_home_cover"] = _logistic_predict(model, test)
        test["p_away_cover"] = 1.0 - test["p_home_cover"]

        diagnostics[test_season] = {
            "test_season": int(test_season),
            "train_seasons": train_seasons,
            "l2": float(l2),
            "cv": cv,
            "importance": _classifier_importance(model),
            "n_train": 0 if model is None else model["n"],
            "covid_excluded_training": True,
        }

        all_tests.append(test)
        rr = _classifier_research_rows(test)
        if not rr.empty:
            all_rows.append(rr)

    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return tests_df, rows_df, diagnostics

def _candidate_validation_bundle(classifier_rows, classifier_tests, feature_df):
    """
    Produce two independent stress-test views of the same predeclared rule:

    A) Standard rolling walk-forward, with 2020 reported separately.
    B) COVID-excluded-training walk-forward, where 2020 cannot influence 2021+.
    """
    standard_candidate = _candidate_rows_from_classifier(classifier_rows)

    covid_tests, covid_rows, covid_diag = _run_classifier_walkforward_excluding_2020(feature_df)
    covid_candidate = _candidate_rows_from_classifier(covid_rows)

    bundle = {
        "standard_candidate": standard_candidate,
        "standard_seasons": _candidate_season_table(standard_candidate),
        "standard_eras": _candidate_era_table(standard_candidate),
        "standard_sides": _candidate_side_table(standard_candidate),
        "standard_loso": _candidate_leave_one_season_out(standard_candidate),

        "covid_excluded_tests": covid_tests,
        "covid_excluded_candidate": covid_candidate,
        "covid_excluded_seasons": _candidate_season_table(covid_candidate),
        "covid_excluded_eras": _candidate_era_table(covid_candidate),
        "covid_excluded_sides": _candidate_side_table(covid_candidate),
        "covid_excluded_loso": _candidate_leave_one_season_out(covid_candidate),
        "covid_excluded_diag": covid_diag,
    }
    return bundle

def _candidate_pass_fail(candidate_df, min_bets=150, min_positive_seasons=4):
    """
    Conservative, predeclared promotion screen.
    This is a research gate, not a claim of statistical certainty.
    """
    if candidate_df.empty:
        return "FAIL", "No candidate bets."

    d = candidate_df[candidate_df["season"] != COVID_SEASON].copy()
    stats = _candidate_stats(d, "Primary")
    seasons = sorted(d["season"].dropna().astype(int).unique())
    positive = 0
    qualifying = 0

    for season in seasons:
        s = d[d["season"] == season]
        if len(s) < 12:
            continue
        qualifying += 1
        roi = float(pd.to_numeric(s["profit_units"], errors="coerce").fillna(0).sum()) / len(s)
        if roi > 0:
            positive += 1

    checks = [
        len(d) >= min_bets,
        pd.notna(stats["Win %"]) and stats["Win %"] > implied_prob(-110),
        stats["ROI"] is not None and pd.notna(stats["ROI"]) and stats["ROI"] > 0,
        qualifying >= min_positive_seasons,
        positive >= min_positive_seasons,
    ]

    if all(checks):
        return "PASS TO LIVE-CANDIDATE REVIEW", (
            f"{len(d)} ex-2020 bets; {positive}/{qualifying} qualifying seasons positive; "
            f"combined ROI {stats['ROI']*100:+.1f}%."
        )

    return "RESEARCH ONLY", (
        f"{len(d)} ex-2020 bets; {positive}/{qualifying} qualifying seasons positive; "
        f"combined ROI {stats['ROI']*100:+.1f}% if available."
    )

# ===== End v0.9.0 locked candidate validation =====

# ===== End v0.8.1 signal audit =====

# ===== End v0.8.0 cover classification model =====

# ===== End v0.7.0 matchup residual model =====

# ===== End v0.6.0 signal research =====

# ===== End v0.5.1 rolling walk-forward validation =====

# ===== End v0.5.0 residual-market model =====

# ===== End v0.4.0 backtest engine =====

# ===== End embedded model engine =====


st.set_page_config(
    page_title="Saturday Edge",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Sleek mobile-first app theme ----------



st.markdown("""
<style>
/* v4.1.1 unified ranked slate */
.v411-ranked-row{
  display:flex;
  gap:12px;
  align-items:flex-start;
  padding:15px 2px;
  border-bottom:1px solid rgba(148,163,184,.10);
}
.v411-ranked-row.best{
  border:1px solid rgba(74,174,255,.35);
  border-radius:16px;
  padding:16px;
  margin:8px 0 10px;
  background:rgba(16,38,63,.55);
}
.v411-rank{
  flex:0 0 24px;
  color:#657b92;
  font-size:.76rem;
  font-weight:900;
  padding-top:3px;
}
.v411-body{min-width:0;flex:1}
.v411-topline{
  display:flex;
  justify-content:space-between;
  gap:10px;
  align-items:center;
}
.v411-market{
  color:#6e859d;
  font-size:.59rem;
  font-weight:900;
  letter-spacing:.10em;
}
.v411-verdict{
  font-size:.59rem;
  font-weight:950;
  letter-spacing:.08em;
}
.v411-verdict.best{color:#6bc1ff}
.v411-verdict.bet{color:#80d7b0}
.v411-verdict.lean{color:#d6bb72}
.v411-pick{
  color:#f7fbff;
  font-size:1.02rem;
  font-weight:900;
  margin-top:4px;
}
.v411-game{
  color:#73889f;
  font-size:.64rem;
  margin-top:3px;
}
.v411-metrics{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:8px;
  color:#70869e;
  font-size:.62rem;
}
.v411-metrics b{color:#bccddd}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.v403-watch-row{
  padding:12px 2px;
  border-bottom:1px solid rgba(148,163,184,.09);
}
.v403-watch-pick{
  color:#f3f7fb;
  font-size:.98rem;
  font-weight:850;
}
.v403-watch-meta{
  display:flex;
  gap:6px;
  flex-wrap:wrap;
  color:#7389a1;
  font-size:.66rem;
  margin-top:4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{
  --bg:#06111f;
  --panel:#0b1728;
  --panel2:#0e1d31;
  --line:rgba(148,163,184,.12);
  --text:#f8fafc;
  --muted:#8fa3ba;
  --blue:#3b82f6;
  --cyan:#38bdf8;
  --green:#22c55e;
  --yellow:#facc15;
  --red:#ef4444;
}
.stApp{
  background:
    radial-gradient(circle at 18% -4%, rgba(59,130,246,.18), transparent 30%),
    radial-gradient(circle at 88% 4%, rgba(56,189,248,.08), transparent 22%),
    linear-gradient(180deg,#071321 0%,#06111f 44%,#050d18 100%);
  color:var(--text);
}
.block-container{
  max-width:940px;
  padding-top:1rem;
  padding-bottom:4rem;
}
header[data-testid="stHeader"]{
  background:rgba(6,17,31,.76);
  backdrop-filter:blur(14px);
  border-bottom:1px solid rgba(148,163,184,.08);
}
h1,h2,h3{
  letter-spacing:-.03em;
}
[data-testid="stMarkdownContainer"] p{
  line-height:1.45;
}

/* Hero */
.cfb-hero{
  padding:20px 2px 12px;
}
.cfb-kicker{
  font-size:.72rem;
  font-weight:900;
  letter-spacing:.18em;
  color:#7dd3fc;
  margin-bottom:6px;
}
.cfb-title{
  font-size:2.55rem;
  line-height:1;
  font-weight:900;
  letter-spacing:-.055em;
  color:#fff;
}
.cfb-subtitle{
  margin-top:10px;
  color:#8fa3ba;
  max-width:620px;
  font-size:.95rem;
}
.version-pill{
  display:inline-flex;
  margin-top:12px;
  padding:5px 9px;
  border-radius:999px;
  background:rgba(59,130,246,.11);
  border:1px solid rgba(96,165,250,.22);
  color:#bfdbfe;
  font-size:.68rem;
  font-weight:800;
  letter-spacing:.05em;
}
.status-strip{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:10px;
  padding:10px 12px;
  margin:0 0 14px;
  border-radius:13px;
  background:rgba(11,23,40,.68);
  border:1px solid rgba(148,163,184,.10);
  color:#8fa3ba;
  font-size:.76rem;
}
.status-live{
  display:flex;
  align-items:center;
  gap:7px;
  color:#86efac;
  font-weight:850;
}
.status-dot{
  width:7px;height:7px;border-radius:999px;background:#22c55e;
  box-shadow:0 0 0 4px rgba(34,197,94,.10);
}

/* Workflow */
.workflow-step{
  margin:1.05rem 0 .5rem;
  display:flex;
  align-items:center;
  gap:.7rem;
}
.workflow-num{
  width:30px;height:30px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  background:linear-gradient(180deg,rgba(59,130,246,.25),rgba(37,99,235,.12));
  border:1px solid rgba(96,165,250,.28);
  color:#dbeafe;
  font-size:.76rem;font-weight:900;
}
.workflow-title{
  font-size:1.03rem;font-weight:850;color:#fff;letter-spacing:-.02em;
}
.workflow-sub{
  color:#7f94ae;font-size:.75rem;margin-top:1px;
}

/* Inputs */
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] > label,
[data-testid="stCheckbox"] label{
  font-size:.76rem!important;
  font-weight:800!important;
  color:#93a8c2!important;
}
[data-baseweb="select"] > div,
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input{
  background:#0a1728!important;
  border-color:rgba(148,163,184,.12)!important;
}
[data-testid="stTabs"] button{
  font-weight:800!important;
}
[data-testid="stExpander"]{
  border:1px solid rgba(148,163,184,.10)!important;
  border-radius:14px!important;
  background:rgba(10,22,40,.36)!important;
}

/* Buttons */
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button{
  border-radius:14px!important;
  font-weight:850!important;
  min-height:3.05rem!important;
  border:1px solid rgba(255,255,255,.08)!important;
}
div[data-testid="stButton"] > button[kind="primary"]{
  background:linear-gradient(135deg,#2563eb,#3b82f6)!important;
  box-shadow:0 10px 30px rgba(37,99,235,.24)!important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover{
  transform:translateY(-1px);
  box-shadow:0 13px 36px rgba(37,99,235,.30)!important;
}

/* Projection metrics */
div[data-testid="stMetric"]{
  background:linear-gradient(180deg,rgba(14,29,49,.95),rgba(10,23,40,.95));
  border:1px solid rgba(148,163,184,.10);
  border-radius:14px;
  padding:11px 13px;
  box-shadow:0 8px 20px rgba(0,0,0,.14);
}

/* Compact legend */
.grade-legend-inline{
  display:flex;
  gap:7px;
  flex-wrap:wrap;
  margin:5px 0 10px;
}
.grade-pill{
  padding:6px 9px;
  border-radius:999px;
  background:#0b1728;
  border:1px solid rgba(148,163,184,.10);
  font-size:.70rem;
  color:#93a8c2;
  font-weight:760;
}
.grade-pill b{color:#fff;margin-right:4px}
.grade-pill.a{border-color:rgba(34,197,94,.24)}
.grade-pill.b{border-color:rgba(56,189,248,.24)}
.grade-pill.c{border-color:rgba(250,204,21,.20)}
.grade-pill.d{border-color:rgba(148,163,184,.12)}

/* Top result card */
.result-hero{
  position:relative;
  overflow:hidden;
  border-radius:20px;
  padding:18px;
  margin:8px 0 16px;
  background:
    linear-gradient(135deg,rgba(15,31,54,.98),rgba(9,22,39,.98));
  border:1px solid rgba(148,163,184,.12);
  box-shadow:0 18px 50px rgba(0,0,0,.26);
}
.result-hero:after{
  content:"";
  position:absolute;
  inset:auto -60px -90px auto;
  width:180px;height:180px;border-radius:999px;
  background:radial-gradient(circle,rgba(59,130,246,.16),transparent 66%);
}
.result-hero.a{border-color:rgba(34,197,94,.32);box-shadow:0 18px 50px rgba(20,83,45,.16)}
.result-hero.b{border-color:rgba(56,189,248,.30);box-shadow:0 18px 50px rgba(14,116,144,.13)}
.result-hero.c{border-color:rgba(250,204,21,.22)}
.result-hero.d{border-color:rgba(148,163,184,.12)}
.result-top{
  display:flex;align-items:center;gap:12px;
}
.result-badge{
  min-width:84px;min-height:44px;padding:0 10px;border-radius:13px;
  display:flex;align-items:center;justify-content:center;
  font-size:.66rem;font-weight:950;color:#fff;letter-spacing:.08em;text-transform:uppercase;
  border:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.06);
  text-align:center;line-height:1.05;
}
.result-label{
  font-size:.68rem;letter-spacing:.14em;font-weight:900;color:#8fa3ba;
}
.result-pick{
  font-size:1.28rem;font-weight:900;color:#fff;letter-spacing:-.025em;margin-top:1px;
}
.result-stake{
  margin-left:auto;
  padding:7px 10px;border-radius:999px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.08);
  font-size:.73rem;font-weight:850;color:#cbd5e1;
}
.result-metrics{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:8px;
  margin-top:14px;
}
.metric-chip{
  background:rgba(255,255,255,.035);
  border:1px solid rgba(255,255,255,.06);
  border-radius:12px;
  padding:9px 10px;
}
.metric-chip .k{
  font-size:.63rem;text-transform:uppercase;letter-spacing:.10em;color:#6f849d;font-weight:850;
}
.metric-chip .v{
  margin-top:2px;font-size:.90rem;color:#e2e8f0;font-weight:850;
}

/* No-play state */
.no-play{
  border-radius:20px;
  padding:18px;
  margin:8px 0 16px;
  background:linear-gradient(135deg,rgba(15,23,42,.92),rgba(9,18,32,.92));
  border:1px solid rgba(148,163,184,.11);
}
.no-play-title{
  font-size:1.16rem;font-weight:900;color:#f8fafc;
}
.no-play-sub{
  margin-top:5px;color:#8fa3ba;font-size:.83rem;
}

/* Market board */
.market-board{
  display:flex;
  flex-direction:column;
  gap:9px;
  margin-top:8px;
}
.market-card{
  display:flex;
  align-items:center;
  gap:12px;
  padding:12px 13px;
  border-radius:15px;
  background:linear-gradient(180deg,rgba(14,29,49,.92),rgba(10,23,40,.92));
  border:1px solid rgba(148,163,184,.10);
}
.market-card:hover{
  border-color:rgba(96,165,250,.20);
}
.market-grade{
  flex:0 0 auto;
  min-width:72px;min-height:38px;padding:0 9px;
  border-radius:11px;
  display:flex;align-items:center;justify-content:center;
  font-size:.56rem;font-weight:950;color:#fff;letter-spacing:.07em;text-transform:uppercase;
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.08);
  text-align:center;line-height:1.05;
}
.market-grade.a{border-color:rgba(34,197,94,.28)}
.market-grade.b{border-color:rgba(56,189,248,.28)}
.market-grade.c{border-color:rgba(250,204,21,.23)}
.market-grade.d{border-color:rgba(148,163,184,.13)}
.market-main{min-width:0;flex:1}
.market-pick{
  font-size:.98rem;font-weight:850;color:#f8fafc;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.market-sub{
  margin-top:3px;
  font-size:.74rem;color:#7f94ae;
}
.market-tag{
  margin-left:auto;
  font-size:.66rem;font-weight:900;letter-spacing:.08em;
  color:#94a3b8;
}

/* Section titles */
.section-kicker{
  font-size:.72rem;
  letter-spacing:.17em;
  font-weight:900;
  color:#7dd3fc;
  margin-top:23px;
  margin-bottom:6px;
}

/* Audit/export */
.help-card{
  padding:11px 13px;
  border-radius:12px;
  background:rgba(15,31,54,.55);
  border:1px solid rgba(148,163,184,.09);
  color:#8fa3ba;
  font-size:.76rem;
  line-height:1.45;
  margin:.35rem 0 .75rem;
}

@media (max-width:700px){
  .block-container{padding-left:1rem;padding-right:1rem}
  .cfb-title{font-size:2.15rem}
  .status-strip{align-items:flex-start;flex-direction:column;gap:5px}
  .result-hero{padding:15px}
  .result-pick{font-size:1.10rem}
  .result-stake{display:none}
  .result-metrics{grid-template-columns:repeat(2,1fr)}
  .market-card{padding:11px 12px}
  .market-pick{font-size:.94rem}
  .market-sub{font-size:.71rem}
}

/* Premium slate cards */
.slate-card{
  border-radius:18px;
  padding:14px;
  margin:9px 0;
  background:linear-gradient(180deg,rgba(14,29,49,.95),rgba(9,21,37,.95));
  border:1px solid rgba(148,163,184,.11);
  box-shadow:0 10px 28px rgba(0,0,0,.16);
}
.slate-card.a{border-color:rgba(34,197,94,.30)}
.slate-card.b{border-color:rgba(56,189,248,.28)}
.slate-card.c{border-color:rgba(250,204,21,.20)}
.slate-card.d{border-color:rgba(148,163,184,.10);opacity:.82}
.slate-card-top{
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:flex-start;
}
.slate-time{
  font-size:.66rem;
  color:#7890aa;
  font-weight:850;
  letter-spacing:.08em;
  text-transform:uppercase;
}
.slate-matchup{
  margin-top:3px;
  font-size:1.04rem;
  font-weight:900;
  color:#f8fafc;
  letter-spacing:-.02em;
}
.slate-matchup span{color:#607792;font-weight:700}
.slate-grade{
  width:40px;height:40px;
  border-radius:12px;
  display:flex;align-items:center;justify-content:center;
  font-weight:950;font-size:1.05rem;
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.09);
}
.slate-grade.a{border-color:rgba(34,197,94,.30)}
.slate-grade.b{border-color:rgba(56,189,248,.28)}
.slate-grade.c{border-color:rgba(250,204,21,.23)}
.slate-grade.d{border-color:rgba(148,163,184,.13)}
.slate-reco{
  margin-top:11px;
  padding:11px 12px;
  border-radius:13px;
  background:rgba(255,255,255,.035);
  border:1px solid rgba(255,255,255,.055);
}
.slate-reco-label{
  font-size:.62rem;
  color:#71869f;
  font-weight:900;
  letter-spacing:.12em;
  text-transform:uppercase;
}
.slate-reco-value{
  margin-top:2px;
  font-size:1rem;
  font-weight:900;
  color:#f8fafc;
}
.slate-reco-meta{
  margin-top:3px;
  font-size:.72rem;
  color:#8398b1;
}
.slate-grid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:7px;
  margin-top:9px;
}
.slate-box{
  padding:8px 9px;
  border-radius:10px;
  background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.045);
}
.slate-box-label{
  font-size:.57rem;
  color:#687e98;
  font-weight:850;
  text-transform:uppercase;
  letter-spacing:.07em;
}
.slate-box-value{
  margin-top:2px;
  font-size:.78rem;
  color:#dbe5f1;
  font-weight:820;
}
.slate-footer{
  margin-top:9px;
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  font-size:.68rem;
  color:#70869e;
}
.slate-footer span{color:#445a72}
.slate-market-row{
  display:flex;
  align-items:center;
  gap:10px;
  padding:10px 2px;
  border-bottom:1px solid rgba(148,163,184,.07);
}
.slate-market-row:last-child{border-bottom:none}
.slate-market-grade{
  width:30px;height:30px;
  border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  font-weight:900;
  font-size:.82rem;
  background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.07);
}
.slate-market-body{flex:1;min-width:0}
.slate-market-pick{
  font-size:.87rem;
  font-weight:820;
  color:#e9f0f8;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.slate-market-sub{
  margin-top:2px;
  font-size:.68rem;
  color:#72879f;
}
@media(max-width:700px){
  .slate-grid{grid-template-columns:repeat(2,1fr)}
  .slate-card{padding:12px}
  .slate-matchup{font-size:.96rem}
}


.slate-summary{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:8px;
  margin:8px 0 16px 0;
}
.slate-summary-box{
  padding:11px 12px;
  border-radius:12px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(148,163,184,.09);
}
.slate-summary-label{
  font-size:.58rem;
  color:#71869f;
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:.08em;
}
.slate-summary-value{
  margin-top:3px;
  font-size:1.06rem;
  color:#f8fafc;
  font-weight:950;
}
.slate-note{
  margin:7px 0 13px 0;
  padding:10px 12px;
  border-radius:11px;
  background:rgba(56,189,248,.05);
  border:1px solid rgba(56,189,248,.10);
  color:#8ca2ba;
  font-size:.73rem;
}
@media(max-width:700px){
  .slate-summary{grid-template-columns:repeat(2,1fr)}
}


.topbet-card{
  display:grid;
  grid-template-columns:38px 1fr auto;
  gap:9px;
  padding:12px;
  margin:8px 0;
  border-radius:15px;
  background:linear-gradient(180deg,rgba(14,29,49,.97),rgba(9,21,37,.97));
  border:1px solid rgba(148,163,184,.10);
}
.topbet-card.a{border-color:rgba(34,197,94,.34)}
.topbet-card.b{border-color:rgba(56,189,248,.30)}
.topbet-card.c{border-color:rgba(250,204,21,.22)}
.topbet-rank{font-weight:950;color:#7890aa;font-size:.82rem;padding-top:2px}
.topbet-copy{min-width:0}
.topbet-game{
  color:#70869f;font-size:.60rem;font-weight:850;letter-spacing:.05em;
  text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.topbet-pick{
  margin-top:3px;color:#f7f9fc;font-weight:950;font-size:1rem;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.topbet-note{margin-top:3px;color:#7890aa;font-size:.66rem}
.topbet-grade{
  min-width:78px;min-height:38px;padding:0 9px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:.56rem;font-weight:950;letter-spacing:.07em;text-transform:uppercase;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  text-align:center;line-height:1.05;
}
.topbet-grade.a{border-color:rgba(34,197,94,.34)}
.topbet-grade.b{border-color:rgba(56,189,248,.30)}
.topbet-grade.c{border-color:rgba(250,204,21,.23)}
.topbet-metrics{
  grid-column:2 / 4;display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:1px;
}
.topbet-metrics div{
  padding:6px 8px;border-radius:9px;background:rgba(255,255,255,.025);
  border:1px solid rgba(255,255,255,.045);
}
.topbet-metrics span{
  display:block;color:#60768e;font-size:.54rem;font-weight:850;text-transform:uppercase;letter-spacing:.06em;
}
.topbet-metrics b{display:block;margin-top:2px;color:#dfe8f2;font-size:.74rem}
.game-best-label{
  margin:4px 0 7px;color:#6f859d;font-size:.61rem;font-weight:900;letter-spacing:.10em;text-transform:uppercase;
}
.game-market-row{
  display:grid;grid-template-columns:28px auto 1fr;gap:8px;align-items:center;
  padding:9px 2px;border-bottom:1px solid rgba(148,163,184,.07);
}
.game-market-row:last-child{border-bottom:none}
.game-market-rank{color:#657b93;font-size:.67rem;font-weight:850}
.game-market-grade{
  min-width:58px;min-height:28px;padding:0 8px;border-radius:8px;display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
  color:#e8eef6;font-size:.50rem;font-weight:900;letter-spacing:.06em;text-transform:uppercase;
  text-align:center;line-height:1.05;
}
.game-market-grade.a{border-color:rgba(34,197,94,.30)}
.game-market-grade.b{border-color:rgba(56,189,248,.28)}
.game-market-grade.c{border-color:rgba(250,204,21,.22)}
.game-market-pick{color:#e7eef7;font-size:.84rem;font-weight:850}
.game-market-meta{margin-top:2px;color:#71869f;font-size:.64rem}
@media(max-width:700px){
  .topbet-card{grid-template-columns:34px 1fr auto;padding:11px}
  .topbet-pick{font-size:.93rem}
  .topbet-metrics{grid-column:1 / 4}
}


/* ===== v1.6 legit-app visual layer ===== */
.app-shell-head{
  display:flex;align-items:flex-end;justify-content:space-between;gap:12px;
  padding:14px 15px;margin:2px 0 14px;border-radius:18px;
  background:linear-gradient(145deg,rgba(13,31,52,.99),rgba(7,18,32,.99));
  border:1px solid rgba(73,188,255,.16);
  box-shadow:0 14px 36px rgba(0,0,0,.22);
}
.app-eyebrow{color:#67d1ff;font-size:.59rem;font-weight:950;letter-spacing:.16em}
.app-title{color:#f8fafc;font-size:1.35rem;font-weight:950;letter-spacing:-.035em;margin-top:2px}
.app-subtitle{color:#748da7;font-size:.67rem;margin-top:3px}
.app-live{display:flex;align-items:center;gap:6px;color:#9fb4ca;font-size:.59rem;font-weight:900;letter-spacing:.08em;white-space:nowrap}
.app-live span{width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 10px rgba(34,197,94,.65)}

.team-logo{
  object-fit:contain;border-radius:50%;background:rgba(255,255,255,.97);padding:2px;
  box-shadow:0 2px 10px rgba(0,0,0,.20);
}
.team-logo-fallback{
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.10);
  color:#b7c7d8;font-size:.60rem;font-weight:950;
}
.logo-pair{display:flex;align-items:center;padding-left:4px}
.logo-pair .team-logo,.logo-pair .team-logo-fallback{margin-left:-7px;border:2px solid #0a1929}

.topbet-card{
  grid-template-columns:30px 38px 1fr 42px !important;
  align-items:start;
}
.topbet-logo{padding-top:0}
.topbet-metrics{grid-column:3 / 5 !important}

.game-detail-head{
  display:grid;grid-template-columns:1fr 24px 1fr;gap:8px;align-items:center;
  padding:11px 12px;margin:2px 0 6px;border-radius:14px;
  background:rgba(255,255,255,.026);border:1px solid rgba(255,255,255,.055);
}
.game-team{display:flex;align-items:center;gap:9px;min-width:0}
.game-team.home{justify-content:flex-end;text-align:right}
.game-team span{display:block;color:#667e97;font-size:.51rem;font-weight:850;text-transform:uppercase;letter-spacing:.07em}
.game-team b{display:block;color:#ecf2f8;font-size:.78rem;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.game-at{text-align:center;color:#58718b;font-size:.72rem;font-weight:900}
.game-detail-sub{color:#71879f;font-size:.64rem;margin:0 2px 10px}

div[data-testid="stExpander"]{
  border-radius:15px !important;
  border:1px solid rgba(148,163,184,.09) !important;
  background:rgba(7,18,32,.50) !important;
  overflow:hidden;
}
div[data-testid="stExpander"] summary{min-height:49px}

@media(max-width:700px){
  .app-shell-head{padding:12px}
  .app-title{font-size:1.16rem}
  .topbet-card{grid-template-columns:25px 34px 1fr auto !important}
  .topbet-metrics{grid-column:1 / 5 !important}
  .game-team b{font-size:.70rem}
}


/* ===== CFB Edge v2.0 mobile app shell ===== */
.block-container{padding-bottom:112px !important;}

div[class*="st-key-cfb_main_navigation"]{
  position:fixed !important;left:0 !important;right:0 !important;bottom:0 !important;
  width:100vw !important;max-width:none !important;z-index:999999 !important;margin:0 !important;
  padding:7px 6px calc(7px + env(safe-area-inset-bottom)) !important;min-height:80px !important;
  background:linear-gradient(180deg,rgba(7,21,36,.965),rgba(4,14,25,.995)) !important;
  border-top:1px solid rgba(91,137,177,.30) !important;box-shadow:0 -12px 32px rgba(0,0,0,.34) !important;
  backdrop-filter:blur(18px) !important;
}
div[class*="st-key-cfb_main_navigation"] [role="radiogroup"]{
  display:grid !important;grid-template-columns:repeat(5,minmax(0,1fr)) !important;
  gap:0 !important;width:100% !important;max-width:none !important;margin:0 !important;
}
div[class*="st-key-cfb_main_navigation"] label{
  display:flex !important;flex-direction:column !important;align-items:center !important;justify-content:center !important;
  min-width:0 !important;min-height:62px !important;gap:4px !important;padding:4px 2px !important;margin:0 !important;
  border:0 !important;border-radius:0 !important;background:transparent !important;box-shadow:none !important;
}
div[class*="st-key-cfb_main_navigation"] label:has(input:checked){background:transparent !important;box-shadow:none !important;}
div[class*="st-key-cfb_main_navigation"] input,
div[class*="st-key-cfb_main_navigation"] label > div:first-child,
div[class*="st-key-cfb_main_navigation"] [data-baseweb="radio"]{display:none !important;}
div[class*="st-key-cfb_main_navigation"] label p{
  color:#74889d !important;font-size:.56rem !important;font-weight:760 !important;line-height:1 !important;
  margin:0 !important;padding:0 !important;text-transform:none !important;white-space:nowrap !important;
}
div[class*="st-key-cfb_main_navigation"] label:has(input:checked) p{color:#4ea8ff !important;font-weight:900 !important;}
div[class*="st-key-cfb_main_navigation"] label::before{
  content:"" !important;display:block !important;width:25px !important;height:25px !important;background-color:#6f8498 !important;
  -webkit-mask-size:contain !important;-webkit-mask-repeat:no-repeat !important;-webkit-mask-position:center !important;
  mask-size:contain !important;mask-repeat:no-repeat !important;mask-position:center !important;
}
div[class*="st-key-cfb_main_navigation"] label:has(input:checked)::before{
  background-color:#4ea8ff !important;filter:drop-shadow(0 0 8px rgba(78,168,255,.28)) !important;
}
div[class*="st-key-cfb_main_navigation"] label:nth-child(1)::before{
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E") !important;
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E") !important;
}
div[class*="st-key-cfb_main_navigation"] label:nth-child(2)::before{
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E") !important;
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E") !important;
}
div[class*="st-key-cfb_main_navigation"] label:nth-child(3)::before{
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E") !important;
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E") !important;
}
div[class*="st-key-cfb_main_navigation"] label:nth-child(4)::before{
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E") !important;
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E") !important;
}
div[class*="st-key-cfb_main_navigation"] label:nth-child(5)::before{
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E") !important;
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E") !important;
}

.mobile-page-head{margin:4px 0 18px;}
.mobile-page-kicker{color:#6ed0ff;font-size:.56rem;font-weight:950;letter-spacing:.14em;}
.mobile-page-title{margin-top:3px;color:#fff;font-size:1.72rem;line-height:1.06;font-weight:950;letter-spacing:-.04em;}
.mobile-page-sub{margin-top:6px;color:#8fa4b8;font-size:.75rem;}
.page-count{display:inline-flex;min-width:27px;height:27px;align-items:center;justify-content:center;padding:0 7px;border-radius:8px;background:#20364f;color:#fff;font-size:.76rem;vertical-align:middle;}

.cfb-live-card{margin:10px 0;padding:14px 15px;border-radius:18px;background:radial-gradient(circle at 94% 5%,rgba(57,189,248,.08),transparent 25%),linear-gradient(180deg,#11283f,#0b1c2e);border:1px solid #31536f;box-shadow:0 14px 30px rgba(0,0,0,.18);}
.cfb-live-top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;}
.cfb-live-teams{flex:1;min-width:0;}
.cfb-score-row{display:grid;grid-template-columns:1fr auto;align-items:center;gap:10px;color:#fff;font-size:.96rem;font-weight:900;line-height:1.55;}
.cfb-score-row b{font-size:1.05rem;}
.cfb-game-state{flex:0 0 auto;padding:6px 9px;border-radius:999px;color:#79efaa;background:#0b3022;border:1px solid #2d7250;font-size:.57rem;font-weight:950;}
.cfb-game-state.final{color:#c8d6e4;background:#1b2b3b;border-color:#42576b;}
.cfb-live-meta{margin-top:8px;color:#8fa4b8;font-size:.66rem;}

.saved-bets-shell{padding:14px;margin:8px 0 14px;border-radius:18px;background:linear-gradient(135deg,#102940,#0a1e31);border:1px solid #31516d;}
.saved-bets-title{font-size:1.05rem;font-weight:950;color:#fff;}
.saved-bets-sub{margin-top:4px;color:#91a6ba;font-size:.68rem;}
.saved-bet-row{display:flex;align-items:center;gap:10px;padding:11px 0;border-top:1px solid rgba(148,163,184,.10);}
.saved-bet-row:first-of-type{margin-top:10px;}
.saved-bet-rank{flex:0 0 28px;color:#69c8ff;font-size:.68rem;font-weight:950;}
.saved-bet-main{flex:1;min-width:0;}
.saved-bet-pick{color:#fff;font-size:.88rem;font-weight:900;}
.saved-bet-sub{margin-top:2px;color:#8197aa;font-size:.60rem;}
.saved-grade{padding:6px 10px;border-radius:999px;font-size:.52rem;font-weight:950;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;}
.saved-grade.a{color:#7cf0ad;background:#0d3325;border:1px solid #2a7550;}
.saved-grade.b{color:#7fd6ff;background:#0c2b43;border:1px solid #2f6f98;}

.cfb-info-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:10px 0 16px;}
.cfb-info-grid>div{padding:11px 12px;border-radius:12px;background:#0d2032;border:1px solid #29475f;}
.cfb-info-grid span{display:block;color:#8ea4b8;font-size:.51rem;font-weight:900;letter-spacing:.09em;}
.cfb-info-grid b{display:block;margin-top:4px;color:#fff;font-size:.69rem;word-break:break-word;}

@media(max-width:700px){
  div[class*="st-key-cfb_main_navigation"]{padding-left:4px !important;padding-right:4px !important;}
  div[class*="st-key-cfb_main_navigation"] label::before{width:23px !important;height:23px !important;}
  div[class*="st-key-cfb_main_navigation"] label p{font-size:.53rem !important;}
}


/* ===== CFB v2.1 live recommendation tracker ===== */
.cfb-slate-pulse{
  margin:0 0 15px;padding:14px;border-radius:17px;
  background:radial-gradient(circle at 90% 0%,rgba(71,190,255,.12),transparent 30%),linear-gradient(135deg,#102a42,#0a1e31);
  border:1px solid #315470;box-shadow:0 14px 30px rgba(0,0,0,.18);
}
.cfb-pulse-head{display:flex;align-items:center;justify-content:space-between;gap:10px;}
.cfb-pulse-head span{display:block;color:#69d8ca;font-size:.50rem;font-weight:950;letter-spacing:.12em;}
.cfb-pulse-head b{display:block;color:#fff;font-size:1.02rem;margin-top:3px;}
.cfb-pulse-chip{padding:6px 8px;border-radius:999px;color:#83cfff;background:#0b2b45;border:1px solid #2b638b;font-size:.52rem;font-weight:950;}
.cfb-pulse-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;margin-top:11px;}
.cfb-pulse-grid div{padding:8px 6px;border-radius:10px;background:rgba(5,17,29,.58);border:1px solid #25455f;}
.cfb-pulse-grid span{display:block;color:#899fb3;font-size:.45rem;font-weight:900;letter-spacing:.06em;}
.cfb-pulse-grid b{display:block;color:#fff;font-size:.78rem;margin-top:3px;}

.cfb-tracker-card{
  margin:10px 0;padding:14px;border-radius:18px;
  background:radial-gradient(circle at 94% 5%,rgba(57,189,248,.08),transparent 26%),linear-gradient(180deg,#11293f,#0b1d2f);
  border:1px solid #345771;box-shadow:0 15px 34px rgba(0,0,0,.19);
}
.cfb-track-score{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;}
.cfb-live-chip{padding:6px 8px;border-radius:999px;color:#77efaa;background:#0c3022;border:1px solid #2c7551;font-size:.53rem;font-weight:950;}
.cfb-track-divider{height:1px;background:#29465f;margin:12px 0;}
.cfb-track-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;}
.cfb-track-head span{display:block;color:#72cfff;font-size:.49rem;letter-spacing:.10em;font-weight:950;}
.cfb-track-head b{display:block;color:#fff;font-size:.96rem;margin-top:4px;}
.cfb-track-status{padding:5px 8px;border-radius:999px;font-size:.50rem;font-weight:950;white-space:nowrap;border:1px solid;}
.cfb-track-status.good{color:#79efaa;background:#0c3324;border-color:#2b7650;}
.cfb-track-status.risk{color:#ff858a;background:#371619;border-color:#81343a;}
.cfb-track-status.neutral{color:#f0da82;background:#30280e;border-color:#75621b;}

.cfb-track-stats{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:14px;}
.cfb-track-stats>div{padding:9px 10px;border-radius:11px;background:#0a1c2e;border:1px solid #294962;}
.cfb-track-stats span{display:block;color:#8fa5b8;font-size:.47rem;letter-spacing:.08em;font-weight:900;}
.cfb-track-stats b{display:block;color:#fff;font-size:1.28rem;line-height:1.05;margin-top:4px;}

.cfb-progress{position:relative;height:8px;margin:14px 2px 4px;border-radius:999px;background:#223d54;}
.cfb-progress-fill{height:8px;border-radius:999px;}
.cfb-progress-fill.good{background:linear-gradient(90deg,#36dd8e,#72f2b6);box-shadow:0 0 13px rgba(74,230,155,.28);}
.cfb-progress-fill.risk{background:linear-gradient(90deg,#ff626c,#ff8d92);}
.cfb-progress-fill.neutral{background:linear-gradient(90deg,#d7ba43,#f0d973);}
.cfb-progress i{position:absolute;top:-6px;width:3px;height:20px;background:#fff;transform:translateX(-1px);box-shadow:0 0 10px rgba(255,255,255,.25);}

.cfb-spread-meter{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;margin-top:14px;color:#8398ab;font-size:.47rem;font-weight:850;}
.cfb-spread-meter i{height:7px;border-radius:999px;background:#42566b;}
.cfb-spread-meter i.good{background:linear-gradient(90deg,#31485d,#46e096);}
.cfb-spread-meter i.risk{background:linear-gradient(90deg,#ff666f,#3b4f63);}
.cfb-spread-meter i.neutral{background:#d6b845;}

.cfb-upcoming-track,.cfb-final-track{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 2px;border-bottom:1px solid rgba(148,163,184,.10);}
.cfb-upcoming-track b,.cfb-final-track b{display:block;color:#fff;font-size:.79rem;}
.cfb-upcoming-track span,.cfb-final-track span{display:block;color:#879caf;font-size:.57rem;margin-top:2px;}

@media(max-width:700px){
  .cfb-pulse-grid{grid-template-columns:repeat(3,minmax(0,1fr));}
  .cfb-pulse-grid div:last-child{grid-column:span 2;}
  .cfb-track-stats b{font-size:1.18rem;}
}


/* v2.1.2 nav hotfix — remove Streamlit radio artifacts completely */
div[class*="st-key-cfb_main_navigation"] [data-testid="stWidgetLabel"],
div[class*="st-key-cfb_main_navigation"] legend,
div[class*="st-key-cfb_main_navigation"] > label {
    display:none !important;
}
div[class*="st-key-cfb_main_navigation"] label > div:first-child,
div[class*="st-key-cfb_main_navigation"] label [data-baseweb="radio"],
div[class*="st-key-cfb_main_navigation"] label [role="radio"] > div,
div[class*="st-key-cfb_main_navigation"] label input + div,
div[class*="st-key-cfb_main_navigation"] label input + div > div,
div[class*="st-key-cfb_main_navigation"] label svg {
    display:none !important;
}
div[class*="st-key-cfb_main_navigation"] label {
    cursor:pointer !important;
}


/* ===== CFB Edge v2.2 clean mobile shell ===== */

/* Hide obsolete radio-nav shell if a stale DOM instance exists */
div[class*="st-key-cfb_main_navigation"]{display:none !important;}

/* Fixed five-button native-style bottom nav */
div[class*="st-key-cfb_nav_"]{
    position:fixed !important;
    bottom:0 !important;
    z-index:999999 !important;
    width:20vw !important;
    margin:0 !important;
    padding:0 !important;
    background:#061523 !important;
    border-top:1px solid #29455e !important;
}
div[class*="st-key-cfb_nav_home_"]{left:0 !important;}
div[class*="st-key-cfb_nav_live_"]{left:20vw !important;}
div[class*="st-key-cfb_nav_tracker_"]{left:40vw !important;}
div[class*="st-key-cfb_nav_bets_"]{left:60vw !important;}
div[class*="st-key-cfb_nav_more_"]{left:80vw !important;}

div[class*="st-key-cfb_nav_"] button{
    width:100% !important;
    height:78px !important;
    min-height:78px !important;
    padding:8px 2px calc(7px + env(safe-area-inset-bottom)) !important;
    border:0 !important;
    border-radius:0 !important;
    background:#061523 !important;
    box-shadow:none !important;
    color:#74899d !important;
    display:flex !important;
    flex-direction:column !important;
    justify-content:center !important;
    align-items:center !important;
    gap:5px !important;
    font-size:.56rem !important;
    font-weight:800 !important;
}
div[class*="st-key-cfb_nav_"] button::before{
    content:"" !important;
    display:block !important;
    width:25px !important;
    height:25px !important;
    background-color:#70869a !important;
    -webkit-mask-size:contain !important;
    -webkit-mask-repeat:no-repeat !important;
    -webkit-mask-position:center !important;
    mask-size:contain !important;
    mask-repeat:no-repeat !important;
    mask-position:center !important;
}
div[class*="st-key-cfb_nav_"][class*="_active"] button{
    color:#48a9ff !important;
}
div[class*="st-key-cfb_nav_"][class*="_active"] button::before{
    background-color:#48a9ff !important;
    filter:drop-shadow(0 0 8px rgba(72,169,255,.30));
}

div[class*="st-key-cfb_nav_home_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 10.5 12 3l9 7.5'/%3E%3Cpath d='M5 9.5V21h5v-6h4v6h5V9.5'/%3E%3C/svg%3E");
}
div[class*="st-key-cfb_nav_live_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='2.2'/%3E%3Cpath d='M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.7 4.7a10.4 10.4 0 0 0 0 14.6M19.3 4.7a10.4 10.4 0 0 1 0 14.6'/%3E%3C/svg%3E");
}
div[class*="st-key-cfb_nav_tracker_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10h4v10M10 20V6h4v14M16 20V12h4v8'/%3E%3Cpath d='m4 7 5-3 4 3 7-5'/%3E%3C/svg%3E");
}
div[class*="st-key-cfb_nav_bets_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M8 7h8M8 11h8M8 15h5'/%3E%3C/svg%3E");
}
div[class*="st-key-cfb_nav_more_"] button::before{
    -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E");
    mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E");
}

/* Streamlit buttons add paragraph wrappers */
div[class*="st-key-cfb_nav_"] button p{
    margin:0 !important;
    font-size:.56rem !important;
    color:inherit !important;
    line-height:1 !important;
}

/* Expander surfaces must remain dark */
div[data-testid="stExpander"],
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"]{
    background:#081a2b !important;
    color:#e9f1f8 !important;
}
div[data-testid="stExpander"] summary{
    border-bottom:1px solid rgba(148,163,184,.08) !important;
}
div[data-testid="stExpander"] summary p{
    color:#e9f1f8 !important;
}

/* Selected slate game replaces stacked dropdown cards */
.game-detail-shell{
    margin:10px 0 7px;
    padding:0;
    border-radius:16px;
    background:linear-gradient(180deg,#0e253a,#091a2b);
    border:1px solid #31516c;
    overflow:hidden;
}
.game-detail-shell .game-detail-head{
    margin:0 !important;
    border:0 !important;
    border-radius:0 !important;
    background:transparent !important;
}
.game-detail-shell .game-detail-sub{
    padding:0 12px 11px !important;
    margin:0 !important;
}

/* Recommendation labels are the status; avoid oversized letter-grade feel */
.topbet-note{
    text-transform:uppercase;
    letter-spacing:.06em;
}
.block-container{
    padding-bottom:105px !important;
}

</style>
""", unsafe_allow_html=True)




st.markdown("""
<style>
/* ===== v3.8.1 Spread Core + ML Value ===== */
.ml-overlay{
  display:flex;align-items:center;gap:7px;flex-wrap:wrap;
  margin-top:8px;padding-top:8px;border-top:1px solid rgba(148,163,184,.08);
}
.ml-overlay span{
  color:#91efc2;font-size:.52rem;font-weight:950;letter-spacing:.10em;
  border:1px solid rgba(66,211,146,.24);border-radius:999px;
  padding:3px 6px;background:rgba(66,211,146,.05)
}
.ml-overlay b{color:#cce8db;font-size:.70rem;font-weight:850}
.ml-overlay em{color:#6f8e81;font-size:.61rem;font-style:normal}
.v381-row{align-items:flex-start}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ===== CFB Edge v3.9 MLB-style Edge Engine ===== */
.edge-section-label{
  margin:22px 2px 10px;color:#6ec5ff;font-size:.64rem;font-weight:900;
  letter-spacing:.16em;text-transform:uppercase;
}
.edge-card{
  border:1px solid rgba(78,168,255,.20);border-radius:20px;
  background:linear-gradient(150deg,rgba(12,31,53,.98),rgba(8,20,35,.98));
  padding:19px;margin-bottom:12px;
}
.edge-card.best{border-color:rgba(78,168,255,.38)}
.edge-card-top{display:flex;justify-content:space-between;align-items:center}
.edge-status{color:#73c2ff;font-size:.61rem;font-weight:950;letter-spacing:.14em}
.edge-units{color:#8ea3bb;font-size:.68rem;font-weight:800}
.edge-pick{color:#fff;font-size:1.55rem;font-weight:950;letter-spacing:-.04em;margin-top:12px}
.edge-matchup{color:#7f94ad;font-size:.75rem;margin-top:5px}
.edge-grid{display:grid;gap:8px;margin-top:17px}
.edge-grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}
.edge-grid>div{
  background:rgba(255,255,255,.018);border:1px solid rgba(148,163,184,.09);
  border-radius:11px;padding:9px 7px;text-align:center
}
.edge-grid span{display:block;color:#667c94;font-size:.54rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
.edge-grid b{display:block;color:#eef6ff;font-size:.87rem;margin-top:4px}
.edge-card-bottom{
  display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
  margin-top:13px;color:#758ba4;font-size:.63rem
}
.edge-card-bottom b{color:#a9bdd1}
.edge-status-strip{
  display:flex;justify-content:space-between;gap:8px;color:#6f849d;
  font-size:.62rem;font-weight:750;padding:6px 2px 2px
}
.edge-empty{
  border:1px solid rgba(148,163,184,.13);border-radius:20px;
  background:linear-gradient(180deg,rgba(12,28,48,.96),rgba(8,20,34,.96));
  padding:20px;margin-bottom:12px
}
.edge-status-pass{color:#8ca0b7;font-size:.61rem;font-weight:950;letter-spacing:.14em}
.edge-empty-title{color:#fff;font-size:1.42rem;font-weight:900;margin-top:8px;letter-spacing:-.03em}
.edge-empty-copy{color:#8499b1;font-size:.8rem;line-height:1.48;margin-top:7px}
.edge-closest{padding-top:15px;margin-top:16px;border-top:1px solid rgba(148,163,184,.10)}
.edge-closest-pick{color:#f3f8ff;font-size:1rem;font-weight:900}
.edge-reliability{color:#7489a0;font-size:.67rem;margin-top:9px}
.edge-reliability b{color:#adbed0}
.edge-mini-row{
  display:flex;justify-content:space-between;gap:10px;align-items:center;
  padding:12px 1px;border-bottom:1px solid rgba(148,163,184,.08)
}
.edge-mini-row>div:first-child{display:flex;flex-direction:column;gap:3px}
.edge-mini-row b{color:#eaf2fb;font-size:.87rem}
.edge-mini-row span{color:#6f849b;font-size:.65rem}
.edge-mini-rel{color:#8da3b9;font-size:.61rem;font-weight:900;letter-spacing:.07em}
.edge-ranking-label{margin-top:25px}
.edge-board-row{
  display:flex;justify-content:space-between;gap:12px;align-items:flex-start;
  padding:14px 1px;border-bottom:1px solid rgba(148,163,184,.09)
}
.edge-board-main{min-width:0}
.edge-board-game{color:#6e849c;font-size:.63rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.edge-board-pick{color:#f7fbff;font-size:.98rem;font-weight:900;margin-top:3px}
.edge-board-sub{display:flex;gap:6px;flex-wrap:wrap;color:#7890a8;font-size:.64rem;margin-top:5px}
.edge-board-side{text-align:right;flex:0 0 auto}
.edge-board-rel{color:#a7bacd;font-size:.60rem;font-weight:900;letter-spacing:.07em;margin-top:1px}
.edge-method{
  border:1px solid rgba(148,163,184,.12);border-radius:16px;padding:15px;
  background:rgba(10,23,39,.55);margin:12px 0 16px
}
.edge-method-title{color:#f5f9fd;font-size:.86rem;font-weight:900;margin-bottom:11px}
.edge-method-row{display:grid;grid-template-columns:22px 72px 1fr;gap:8px;align-items:center;padding:5px 0}
.edge-method-row b{color:#61b6ff;font-size:.66rem}
.edge-method-row span{color:#dce8f4;font-size:.69rem;font-weight:850}
.edge-method-row em{color:#6f849c;font-size:.65rem;font-style:normal}
@media(max-width:640px){
  .edge-grid.four{grid-template-columns:repeat(2,minmax(0,1fr))}
  .edge-pick{font-size:1.42rem}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== CFB Edge v3.8 Decision UI ===== */
.v38-head{padding-bottom:6px!important}
.decision-section-label{margin:22px 2px 10px;color:#6ec5ff;font-size:.64rem;font-weight:900;letter-spacing:.16em;text-transform:uppercase}
.decision-empty{border:1px solid rgba(110,197,255,.18);border-radius:20px;background:linear-gradient(180deg,rgba(12,29,49,.96),rgba(8,21,36,.96));padding:20px;margin:0 0 12px}
.decision-state{color:#89a1bb;font-size:.63rem;font-weight:900;letter-spacing:.14em;margin-bottom:9px}
.decision-title{color:#f8fafc;font-size:1.5rem;font-weight:900;letter-spacing:-.03em}
.decision-copy{color:#8ea2ba;font-size:.84rem;line-height:1.5;margin-top:7px}
.closest-row{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;padding-top:18px;margin-top:18px;border-top:1px solid rgba(148,163,184,.12)}
.closest-row div:first-child{display:flex;flex-direction:column;gap:5px}
.closest-row span{color:#71879f;font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800}
.closest-row b{color:#f8fafc;font-size:1.04rem}
.closest-score{color:#f8fafc;font-size:1.6rem;font-weight:900;letter-spacing:-.04em}
.closest-gap{color:#71879f;font-size:.72rem;text-align:right;margin-top:4px}
.hero-bet{border:1px solid rgba(78,168,255,.28);border-radius:22px;background:linear-gradient(145deg,rgba(12,34,59,.98),rgba(8,22,39,.98));padding:21px;margin-bottom:12px}
.hero-bet-top{display:flex;justify-content:space-between;align-items:center;color:#73c2ff;font-size:.64rem;font-weight:900;letter-spacing:.14em}
.hero-bet-score{font-size:1rem}
.hero-bet-pick{color:#fff;font-size:1.72rem;font-weight:950;letter-spacing:-.045em;margin-top:15px}
.hero-bet-game{color:#8da1b8;font-size:.8rem;margin-top:6px}
.hero-bet-meta{display:flex;gap:8px;flex-wrap:wrap;color:#8fa6bf;font-size:.68rem;margin-top:18px;font-weight:750}
.compact-pick-row{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:14px 2px;border-bottom:1px solid rgba(148,163,184,.10)}
.compact-pick-row>div:first-child{display:flex;flex-direction:column;gap:4px}
.compact-pick-row b{color:#f8fafc;font-size:.98rem}
.compact-pick-row span{color:#71879f;font-size:.68rem}
.compact-pick-score{color:#dce8f6;font-size:1.05rem;font-weight:850}
.card-status-strip{display:flex;justify-content:space-between;gap:8px;color:#6f849d;font-size:.63rem;font-weight:750;padding:9px 2px 1px}
.slate-rank-row{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:15px 2px;border-bottom:1px solid rgba(148,163,184,.10)}
.slate-rank-main{min-width:0}
.slate-rank-game{color:#7d92aa;font-size:.66rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.slate-rank-pick{color:#f8fafc;font-size:1rem;font-weight:900;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.slate-rank-time{color:#62778f;font-size:.65rem;margin-top:4px}
.slate-rank-side{text-align:right;flex:0 0 auto}
.slate-rank-score{color:#dce9f8;font-size:1rem;font-weight:900}
.slate-rank-state{display:inline-block;margin-top:4px;padding:4px 7px;border-radius:999px;font-size:.54rem;font-weight:950;letter-spacing:.08em;border:1px solid rgba(148,163,184,.14);color:#7890aa}
.slate-rank-state.official{color:#91efc2;border-color:rgba(66,211,146,.25);background:rgba(66,211,146,.06)}
.slate-rank-state.watch{color:#ffd783;border-color:rgba(255,210,94,.22);background:rgba(255,210,94,.055)}
.slate-rank-state.pass{color:#6d829a}
@media(max-width:640px){.hero-bet-pick{font-size:1.55rem}.decision-title{font-size:1.34rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{--term-bg:#06111f;--term-line:rgba(148,163,184,.12);--term-text:#f8fafc;--term-muted:#8fa3bd;--term-green:#42d392;}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 50% -10%,rgba(46,118,196,.08),transparent 30%),var(--term-bg)!important;}
.block-container{max-width:860px!important;padding-top:1rem!important;padding-bottom:104px!important;}
.terminal-brand{padding:4px 2px 16px;margin:0 0 10px;border-bottom:1px solid var(--term-line);}
.terminal-brand-row{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;}
.terminal-wordmark{color:var(--term-text);font-size:1.02rem;font-weight:950;letter-spacing:.12em;line-height:1;}
.terminal-tagline{color:var(--term-muted);font-size:.73rem;margin-top:7px;}
.terminal-status{display:flex;align-items:center;gap:7px;color:#9ff0c9;font-size:.62rem;font-weight:900;letter-spacing:.12em;padding:6px 9px;border:1px solid rgba(66,211,146,.22);border-radius:999px;background:rgba(66,211,146,.055);}
.terminal-status span{width:6px;height:6px;border-radius:50%;background:var(--term-green);box-shadow:0 0 12px rgba(66,211,146,.7);}
.terminal-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;color:#6f859e;font-size:.61rem;font-weight:750;margin-top:11px;}
.terminal-meta i{width:3px;height:3px;border-radius:50%;background:#3f536b;display:inline-block;}
.mobile-page-head.terminal-page-head{padding:18px 2px 12px!important;margin:0!important;}
.mobile-page-head .mobile-page-kicker{color:#62b9ff!important;letter-spacing:.13em!important;font-size:.64rem!important;}
.mobile-page-head .mobile-page-title{font-size:2rem!important;line-height:1.05!important;letter-spacing:-.035em!important;margin-top:6px!important;}
.mobile-page-head .mobile-page-sub{max-width:620px;color:var(--term-muted)!important;font-size:.88rem!important;line-height:1.55!important;margin-top:9px!important;}
div[data-testid="stExpander"]{border:1px solid var(--term-line)!important;border-radius:16px!important;background:rgba(10,23,39,.58)!important;}
div[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(13,28,47,.95),rgba(9,21,36,.95));border:1px solid var(--term-line);border-radius:16px;padding:14px 15px;}
[data-testid="stButton"] button[kind="primary"]{min-height:56px!important;border-radius:15px!important;background:linear-gradient(135deg,#246ff0,#438cff)!important;border:0!important;box-shadow:0 10px 28px rgba(36,111,240,.17)!important;font-weight:850!important;}
.section-kicker{color:#7790aa!important;font-size:.63rem!important;letter-spacing:.13em!important;margin-top:18px!important;}
div[class*="st-key-cfb_nav_"]{width:25vw!important;}
div[class*="st-key-cfb_nav_slate_"]{left:0!important;}
div[class*="st-key-cfb_nav_game_"]{left:25vw!important;}
div[class*="st-key-cfb_nav_tracker_"]{left:50vw!important;}
div[class*="st-key-cfb_nav_more_"]{left:75vw!important;}
div[class*="st-key-cfb_nav_home_"],div[class*="st-key-cfb_nav_live_"],div[class*="st-key-cfb_nav_bets_"]{display:none!important;}
.cfb-hero,.status-strip{display:none!important;}
@media(max-width:640px){.block-container{padding-left:1rem!important;padding-right:1rem!important}.mobile-page-head .mobile-page-title{font-size:1.82rem!important}}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* CFB EDGE v4.2 — premium standalone app skin */
[data-testid="stHeader"]{display:none!important;height:0!important;min-height:0!important;background:transparent!important;}
[data-testid="stAppViewBlockContainer"],[data-testid="stMainBlockContainer"],.stMainBlockContainer,section.main > div.block-container{padding-top:calc(6px + env(safe-area-inset-top))!important;}
[data-testid="stMainBlockContainer"] > div:first-child,[data-testid="stAppViewBlockContainer"] > div:first-child{margin-top:0!important;padding-top:0!important;}
[data-testid="stMarkdownContainer"]:has(> style){display:none!important;}
[data-testid="stElementContainer"]:has(style):not(:has(*:not(style):not([data-testid="stMarkdownContainer"]))){display:none!important;height:0!important;margin:0!important;padding:0!important;}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],#MainMenu,footer{display:none!important;}
.block-container{max-width:760px!important;padding:calc(4px + env(safe-area-inset-top)) 18px calc(108px + env(safe-area-inset-bottom))!important;}
html,body,[data-testid="stAppViewContainer"]{
  background:radial-gradient(circle at 85% -10%,rgba(57,126,255,.14),transparent 28%),
             radial-gradient(circle at 0% 20%,rgba(40,191,255,.05),transparent 26%),#07111e!important;
}

/* top app bar */
.v420-appbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:8px 0 17px;margin-bottom:16px;border-bottom:1px solid rgba(148,163,184,.10);}
.v420-brandmark{display:flex;align-items:center;gap:11px;min-width:0}
.v420-logo{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:.72rem;font-weight:950;background:linear-gradient(145deg,#3a86ff,#1d5fd1);border:1px solid rgba(255,255,255,.18);box-shadow:0 8px 24px rgba(42,118,255,.25),inset 0 1px 0 rgba(255,255,255,.18)}
.v420-wordmark{color:#fff;font-size:.91rem;font-weight:950;letter-spacing:.11em;line-height:1}
.v420-brand-sub{color:#607790;font-size:.55rem;font-weight:750;margin-top:5px}
.v420-live{display:flex;align-items:center;gap:7px;padding:7px 10px;border-radius:999px;border:1px solid rgba(54,211,153,.25);background:rgba(35,197,132,.06);color:#9aefd0;font-size:.56rem;font-weight:950;letter-spacing:.12em}
.v420-live i{width:7px;height:7px;border-radius:50%;background:#48dfaa;box-shadow:0 0 13px rgba(72,223,170,.72)}

.v420-section-label{color:#59718b;font-size:.54rem;font-weight:950;letter-spacing:.15em;margin:2px 0 7px;text-transform:uppercase}
div[class*="st-key-v420_game_date"] [data-baseweb="input"],
div[class*="st-key-v420_game_level"] [data-baseweb="select"]>div{
  min-height:48px!important;border-radius:14px!important;background:rgba(13,27,44,.86)!important;
  border:1px solid rgba(122,156,188,.16)!important;box-shadow:none!important
}
div[class*="st-key-v420_game_date"] input{color:#edf5ff!important;-webkit-text-fill-color:#edf5ff!important;font-weight:800!important}
div[class*="st-key-v420_game_level"] [data-baseweb="select"] *{color:#eaf3fc!important}

.v420-day-summary{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:11px 0 18px;padding:13px 14px;border-radius:16px;background:linear-gradient(135deg,rgba(17,37,60,.88),rgba(10,25,42,.88));border:1px solid rgba(122,156,188,.13)}
.v420-day-summary>div:first-child{display:flex;flex-direction:column;gap:2px}
.v420-day-kicker{color:#68819a;font-size:.54rem;font-weight:900;letter-spacing:.09em;text-transform:uppercase}
.v420-day-summary b{color:#f7fbff;font-size:.88rem}
.v420-day-meta{display:flex;align-items:center;gap:7px;color:#8299af;font-size:.59rem;font-weight:750;text-align:right}
.v420-day-meta i,.v420-slate-info i{width:3px;height:3px;border-radius:50%;background:#3d556e;display:block}

.v420-slate-hero{padding:18px 17px;margin:2px 0 17px;border-radius:20px;background:radial-gradient(circle at 92% 0%,rgba(71,145,255,.18),transparent 32%),linear-gradient(145deg,#102743,#0b1d32 62%,#091827);border:1px solid rgba(77,135,194,.24);box-shadow:0 20px 44px rgba(0,0,0,.20)}
.v420-eyebrow{color:#62b9ff;font-size:.55rem;font-weight:950;letter-spacing:.16em}
.v420-hero-row{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-top:7px}
.v420-hero-title{color:#fff;font-size:1.72rem;font-weight:950;letter-spacing:-.05em;line-height:1}
.v420-hero-copy{color:#7891a9;font-size:.70rem;margin-top:7px}
.v420-market-pill{padding:6px 8px;border-radius:9px;color:#9db9d5;background:rgba(4,14,26,.45);border:1px solid rgba(126,162,196,.14);font-size:.52rem;font-weight:900;letter-spacing:.08em}

div[class*="st-key-v420_slate_segment"] [role="radiogroup"]{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;gap:4px!important;padding:4px!important;border-radius:14px!important;background:#0b1b2d!important;border:1px solid rgba(120,154,188,.13)!important}
div[class*="st-key-v420_slate_segment"] label{min-height:40px!important;display:flex!important;align-items:center!important;justify-content:center!important;border-radius:10px!important;margin:0!important;padding:0 4px!important;background:transparent!important}
div[class*="st-key-v420_slate_segment"] label p{margin:0!important;color:#728aa2!important;font-size:.65rem!important;font-weight:850!important}
div[class*="st-key-v420_slate_segment"] label:has(input:checked){background:linear-gradient(145deg,#1d4f86,#173e69)!important;box-shadow:0 4px 14px rgba(8,24,41,.35),inset 0 1px 0 rgba(255,255,255,.06)!important}
div[class*="st-key-v420_slate_segment"] label:has(input:checked) p{color:#f8fbff!important}
div[class*="st-key-v420_slate_segment"] [data-baseweb="radio"]{display:none!important}
.v420-slate-info{display:flex;align-items:center;gap:8px;margin:10px 2px 13px;color:#6f879f;font-size:.60rem;font-weight:740}
.v420-slate-info b{color:#c6d6e5;font-size:.68rem}

div[data-testid="stExpander"]{border-radius:15px!important;border:1px solid rgba(122,156,188,.11)!important;background:rgba(10,24,40,.55)!important;box-shadow:none!important}
div[data-testid="stExpander"] summary{min-height:48px!important;color:#b7c8d8!important}
div[data-testid="stExpander"] summary p{font-size:.72rem!important;font-weight:780!important}

div[class*="st-key-v420_run_slate"] button{min-height:54px!important;border-radius:16px!important;background:linear-gradient(135deg,#2f7cff,#1e66df)!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:0 14px 30px rgba(28,103,231,.28),inset 0 1px 0 rgba(255,255,255,.14)!important;font-size:.81rem!important;font-weight:900!important}

.v420-results-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:24px 2px 7px}
.v420-results-head span{color:#eef6ff;font-size:.72rem;font-weight:950;letter-spacing:.12em}
.v420-results-head em{color:#4f6a83;font-size:.49rem;font-style:normal;font-weight:900;letter-spacing:.10em}

.v411-ranked-row{display:grid!important;grid-template-columns:34px minmax(0,1fr)!important;gap:10px!important;align-items:flex-start!important;padding:15px 14px!important;margin:8px 0!important;border:1px solid rgba(125,158,190,.11)!important;border-radius:17px!important;background:linear-gradient(150deg,rgba(13,29,48,.90),rgba(8,20,34,.90))!important;box-shadow:0 11px 28px rgba(0,0,0,.13)!important}
.v411-ranked-row.best{border-color:rgba(74,157,255,.36)!important;background:radial-gradient(circle at 90% 0%,rgba(61,142,255,.18),transparent 33%),linear-gradient(145deg,#102b4a,#0b2038)!important;box-shadow:0 18px 42px rgba(0,0,0,.22)!important}
.v411-ranked-row.bet{border-color:rgba(75,196,151,.18)!important}
.v411-ranked-row.lean{opacity:.92}
.v411-rank{width:28px;height:28px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:#0a1c30;color:#6d8aa5!important;font-size:.62rem!important;font-weight:950!important;border:1px solid rgba(120,157,191,.10)}
.v411-ranked-row.best .v411-rank{background:#1e65ad;color:#fff!important}
.v411-market{display:inline-flex;padding:4px 6px;border-radius:7px;background:rgba(115,151,184,.08);color:#7691aa!important;font-size:.48rem!important;font-weight:950!important;letter-spacing:.10em!important}
.v411-verdict{padding:5px 8px;border-radius:999px;font-size:.49rem!important;font-weight:950!important;letter-spacing:.09em!important;border:1px solid transparent}
.v411-verdict.best{color:#94cfff!important;background:rgba(56,136,232,.10);border-color:rgba(73,151,245,.18)}
.v411-verdict.bet{color:#8fe2bc!important;background:rgba(46,186,126,.075);border-color:rgba(68,199,145,.16)}
.v411-verdict.lean{color:#dbc47e!important;background:rgba(188,154,56,.06);border-color:rgba(190,158,69,.14)}
.v411-pick{color:#fff!important;font-size:1.12rem!important;font-weight:950!important;letter-spacing:-.025em!important;margin-top:5px!important}
.v411-game{color:#627b94!important;font-size:.61rem!important;margin-top:4px!important}
.v420-metric-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin-top:12px}
.v420-metric-grid>div{min-width:0;padding:8px 5px;border-radius:10px;text-align:center;background:rgba(4,14,25,.36);border:1px solid rgba(113,150,184,.075)}
.v420-metric-grid span{display:block;color:#526b84;font-size:.43rem;font-weight:900;letter-spacing:.06em;text-transform:uppercase}
.v420-metric-grid b{display:block;color:#d8e6f2;font-size:.66rem;font-weight:900;margin-top:3px;white-space:nowrap}

/* four-tab native bottom nav */
div[class*="st-key-cfb_nav_"]{width:25vw!important;background:rgba(5,16,28,.97)!important;border-top:1px solid rgba(94,132,168,.18)!important;backdrop-filter:blur(22px)!important}
div[class*="st-key-cfb_nav_slate_"]{left:0!important}
div[class*="st-key-cfb_nav_game_"]{left:25vw!important}
div[class*="st-key-cfb_nav_tracker_"]{left:50vw!important}
div[class*="st-key-cfb_nav_more_"]{left:75vw!important}
div[class*="st-key-cfb_nav_"] button{height:72px!important;min-height:72px!important;background:transparent!important;padding:7px 2px calc(8px + env(safe-area-inset-bottom))!important;color:#62788d!important;border:0!important}
div[class*="st-key-cfb_nav_"] button::before{width:23px!important;height:23px!important;background-color:#62788d!important}
div[class*="st-key-cfb_nav_"][class*="_active"] button{color:#58adff!important}
div[class*="st-key-cfb_nav_"][class*="_active"] button::before{background-color:#58adff!important}
div[class*="st-key-cfb_nav_slate_"] button::before{-webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Crect x='3' y='4' width='18' height='4' rx='2'/%3E%3Crect x='3' y='10' width='18' height='4' rx='2'/%3E%3Crect x='3' y='16' width='18' height='4' rx='2'/%3E%3C/svg%3E")!important;mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Crect x='3' y='4' width='18' height='4' rx='2'/%3E%3Crect x='3' y='10' width='18' height='4' rx='2'/%3E%3Crect x='3' y='16' width='18' height='4' rx='2'/%3E%3C/svg%3E")!important}
div[class*="st-key-cfb_nav_game_"] button::before{-webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cellipse cx='12' cy='12' rx='8' ry='5' transform='rotate(-35 12 12)'/%3E%3Cpath d='m9 9 6 6M10.5 7.8l5.7 5.7M7.8 10.5l5.7 5.7'/%3E%3C/svg%3E")!important;mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cellipse cx='12' cy='12' rx='8' ry='5' transform='rotate(-35 12 12)'/%3E%3Cpath d='m9 9 6 6M10.5 7.8l5.7 5.7M7.8 10.5l5.7 5.7'/%3E%3C/svg%3E")!important}
div[class*="st-key-cfb_nav_tracker_"] button::before{-webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M4 19V11M10 19V6M16 19V9M22 19V3'/%3E%3C/svg%3E")!important;mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M4 19V11M10 19V6M16 19V9M22 19V3'/%3E%3C/svg%3E")!important}
div[class*="st-key-cfb_nav_more_"] button::before{-webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E")!important;mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'%3E%3Ccircle cx='5' cy='12' r='2'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3Ccircle cx='19' cy='12' r='2'/%3E%3C/svg%3E")!important}

@media(max-width:520px){
  .block-container{padding-left:14px!important;padding-right:14px!important}
  .v420-hero-title{font-size:1.55rem}
  .v420-market-pill{display:none}
  .v420-metric-grid{grid-template-columns:repeat(5,minmax(0,1fr));gap:4px}
  .v420-metric-grid>div{padding:7px 3px}
  .v420-metric-grid b{font-size:.61rem}
}





/* ===== GRIDIRON EDGE v4.4 REAL TEAM LOGOS ===== */
.ge440-logo-shell{
  position:relative;flex:0 0 auto;
  display:flex;align-items:center;justify-content:center;
  border-radius:12px;
  background:linear-gradient(145deg,#102a45,#0b1e33);
  border:1px solid rgba(111,155,195,.16);
  overflow:hidden;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
}
.ge440-team-logo{
  width:82%;height:82%;object-fit:contain;display:block;
  filter:drop-shadow(0 3px 5px rgba(0,0,0,.28));
}
.ge440-logo-fallback{
  position:absolute;inset:0;
  display:none;align-items:center;justify-content:center;
  color:#8fc7ff;font-size:.55rem;font-weight:950;letter-spacing:.03em;
}
.ge440-logo-pair{
  position:relative;display:flex;align-items:center;
  width:58px;height:42px;flex:0 0 58px;
}
.ge440-logo-pair .ge440-logo-shell:first-child{
  position:absolute;left:0;top:3px;z-index:2;
}
.ge440-logo-pair .ge440-logo-shell:last-child{
  position:absolute;right:0;bottom:0;z-index:1;
}
.ge440-logo-pair .ge440-logo-shell{
  border:2px solid #0c2440;
}
.ge440-lean-logo{display:flex;align-items:center;justify-content:center}
.ge440-matchup-logos{display:flex;align-items:center}
.ge440-matchup-logos .ge440-logo-shell + .ge440-logo-shell{margin-left:-7px}

.ge-official-card.best .ge440-logo-shell{
  background:linear-gradient(145deg,#173e68,#0d2a4a);
  border-color:rgba(96,166,229,.27);
  box-shadow:0 8px 20px rgba(0,0,0,.14);
}

.ge-lean-row{
  grid-template-columns:25px 60px minmax(0,1fr) auto auto!important;
}
.ge-lean-row.compact{
  grid-template-columns:25px 60px minmax(0,1fr) auto auto!important;
}

.ge440-pending-row{
  display:grid;grid-template-columns:29px minmax(0,1fr) auto;gap:9px;align-items:center;
  padding:9px 2px;border-bottom:1px solid rgba(110,150,185,.08);
}
.ge440-pending-row b{display:block;color:#eaf3fb;font-size:.64rem}
.ge440-pending-row span{display:block;color:#627d96;font-size:.46rem;margin-top:2px}
.ge440-pending-row em{
  font-style:normal;color:#7893aa;font-size:.42rem;font-weight:900;letter-spacing:.08em
}

@media(max-width:520px){
  .ge440-logo-pair{width:52px;height:38px;flex-basis:52px}
  .ge-lean-row{
    grid-template-columns:24px 54px minmax(0,1fr) auto auto!important;
  }
}

/* ===== GRIDIRON EDGE v4.3.3 PIXEL PASS ===== */

/* tighter top header */
.ge433-header{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  margin:0;padding:2px 0 6px;
}
.ge433-head-right{display:flex;align-items:center;gap:10px}
.ge433-cfb{color:#8fa7bc;font-size:.51rem;font-weight:950;letter-spacing:.12em}
.ge433-profile{
  width:27px;height:27px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  background:#0e2238;border:1px solid rgba(104,148,187,.18);
  color:#6e8aa3;font-size:.38rem;
}

/* top nav exactly one slim row, no label, no radio circles */
div[class*="st-key-ge432_topnav"] > label,
div[class*="st-key-ge432_topnav"] [data-testid="stWidgetLabel"],
div[class*="st-key-ge432_topnav"] [data-testid="stMarkdownContainer"]:has(p:empty){
  display:none!important;
}
div[class*="st-key-ge432_topnav"]{
  margin:2px 0 8px!important;
}
div[class*="st-key-ge432_topnav"] [role="radiogroup"]{
  display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;
  gap:3px!important;padding:3px!important;border-radius:13px!important;
  background:#0b1d30!important;border:1px solid rgba(104,148,187,.11)!important;
}
div[class*="st-key-ge432_topnav"] label{
  min-height:35px!important;border-radius:9px!important;
}
div[class*="st-key-ge432_topnav"] label p{
  font-size:.54rem!important;
}
div[class*="st-key-ge432_topnav"] label:has(input:checked){
  background:linear-gradient(145deg,#174879,#11365d)!important;
  border:1px solid rgba(62,143,235,.23)!important;
}
div[class*="st-key-ge432_topnav"] input,
div[class*="st-key-ge432_topnav"] [data-baseweb="radio"],
div[class*="st-key-ge432_topnav"] [role="radio"]>div,
div[class*="st-key-ge432_topnav"] svg{
  display:none!important;visibility:hidden!important;width:0!important;height:0!important;
}

/* FORCE filter columns to remain side-by-side on mobile */
div[class*="st-key-ge433_filter_row"] [data-testid="stHorizontalBlock"]{
  display:flex!important;flex-direction:column!important;gap:8px!important;
}
div[class*="st-key-ge433_filter_row"] [data-testid="column"]{
  width:100%!important;flex:1 1 100%!important;min-width:0!important;
}

/* kill Streamlit white filter styling */
div[class*="st-key-ge433_filter_row"] [data-testid="stDateInput"] div[data-baseweb="input"],
div[class*="st-key-ge433_filter_row"] [data-testid="stDateInput"] > div > div,
div[class*="st-key-ge433_filter_row"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[class*="st-key-ge433_filter_row"] [data-testid="stSelectbox"] > div > div{
  background:#0c2035!important;
  border-color:rgba(104,148,187,.14)!important;
  border-radius:13px!important;
  box-shadow:none!important;
  min-height:43px!important;
}
div[class*="st-key-ge433_filter_row"] [data-testid="stDateInput"] input{
  background:transparent!important;color:#edf5fb!important;-webkit-text-fill-color:#edf5fb!important;
  font-size:.63rem!important;font-weight:850!important;
}
div[class*="st-key-ge433_filter_row"] [data-testid="stSelectbox"] *{
  color:#edf4fa!important;
}
div[class*="st-key-ge433_filter_row"] [data-testid="stSelectbox"] [role="combobox"]{
  background:transparent!important;color:#edf4fa!important;font-size:.63rem!important;font-weight:850!important;
}

/* one compact mockup-style row for date / universe / live */
.ge433-summary{
  display:grid;grid-template-columns:1.1fr 1fr auto;gap:7px;align-items:center;
  margin:7px 0 8px;
}
.ge433-chip{
  min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  padding:8px 9px;border-radius:12px;
  background:#0c2035;border:1px solid rgba(104,148,187,.13);
  color:#dce8f2;font-size:.52rem;font-weight:850;
}
.ge433-live{
  display:inline-flex;align-items:center;justify-content:center;gap:5px;
  padding:8px 9px;border-radius:999px;
  border:1px solid rgba(48,208,151,.25);background:rgba(29,181,124,.055);
  color:#8de8c2;font-size:.48rem;font-weight:950;letter-spacing:.09em;
}
.ge433-live i{width:6px;height:6px;border-radius:50%;background:#4ae0aa;box-shadow:0 0 10px rgba(74,224,170,.6)}

/* slate segmented selector: absolutely no native circle or widget label */
div[class*="st-key-v420_slate_segment"] > label,
div[class*="st-key-v420_slate_segment"] [data-testid="stWidgetLabel"]{
  display:none!important;
}
div[class*="st-key-v420_slate_segment"]{
  margin-top:4px!important;
}
div[class*="st-key-v420_slate_segment"] [role="radiogroup"]{
  display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;
  gap:3px!important;padding:3px!important;border-radius:13px!important;
  background:#0c2035!important;border:1px solid rgba(104,148,187,.13)!important;
}
div[class*="st-key-v420_slate_segment"] label{
  min-height:39px!important;padding:0!important;margin:0!important;border-radius:10px!important;
  display:flex!important;justify-content:center!important;align-items:center!important;
}
div[class*="st-key-v420_slate_segment"] label p{
  font-size:.55rem!important;margin:0!important;color:#89a0b6!important;font-weight:850!important;
}
div[class*="st-key-v420_slate_segment"] label:has(input:checked){
  background:linear-gradient(145deg,#2b86f4,#1969d0)!important;
}
div[class*="st-key-v420_slate_segment"] label:has(input:checked) p{color:#fff!important}
div[class*="st-key-v420_slate_segment"] input,
div[class*="st-key-v420_slate_segment"] [data-baseweb="radio"],
div[class*="st-key-v420_slate_segment"] [role="radio"]>div,
div[class*="st-key-v420_slate_segment"] svg{
  display:none!important;visibility:hidden!important;width:0!important;height:0!important;
}
.ge433-slate-meta{
  display:flex;align-items:center;justify-content:flex-end;gap:8px;
  color:#7f97ad;font-size:.50rem;margin:6px 1px 8px;
}
.ge433-slate-meta span:first-child::after{
  content:"";display:inline-block;width:3px;height:3px;border-radius:50%;background:#40586f;margin-left:8px;vertical-align:middle;
}
.ge433-slate-meta b{color:#dce7f0}

/* compressed vertical rhythm */
.ge-official-head{margin-top:13px!important}
div[data-testid="stExpander"]{margin-top:5px!important}
div[class*="st-key-v420_run_slate"]{margin-bottom:2px!important}

@media(max-width:520px){
  .block-container{padding-top:calc(5px + env(safe-area-inset-top))!important}
  .ge433-summary{grid-template-columns:1.15fr .95fr auto!important}
  div[class*="st-key-ge433_filter_row"] [data-testid="stHorizontalBlock"]{
    display:flex!important;flex-direction:row!important;
  }
}

/* ===== v4.3.2 FIXED HORIZONTAL PRIMARY NAV ===== */
div[class*="st-key-ge432_topnav"]{
  width:100%!important;
  margin:2px 0 10px!important;
}
div[class*="st-key-ge432_topnav"] [role="radiogroup"]{
  display:grid!important;
  grid-template-columns:repeat(4,minmax(0,1fr))!important;
  width:100%!important;
  gap:4px!important;
  padding:4px!important;
  margin:0!important;
  border-radius:14px!important;
  background:#0a1b2d!important;
  border:1px solid rgba(105,145,181,.13)!important;
}
div[class*="st-key-ge432_topnav"] label{
  width:100%!important;
  min-width:0!important;
  min-height:38px!important;
  margin:0!important;
  padding:0 4px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  border-radius:10px!important;
  background:transparent!important;
  cursor:pointer!important;
}
div[class*="st-key-ge432_topnav"] label p{
  margin:0!important;
  color:#758ea6!important;
  font-size:.58rem!important;
  line-height:1!important;
  font-weight:850!important;
  text-align:center!important;
  white-space:nowrap!important;
}
div[class*="st-key-ge432_topnav"] label:has(input:checked){
  background:linear-gradient(145deg,#174676,#10345a)!important;
  border:1px solid rgba(66,148,239,.26)!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035)!important;
}
div[class*="st-key-ge432_topnav"] label:has(input:checked) p{
  color:#f7fbff!important;
}
div[class*="st-key-ge432_topnav"] input,
div[class*="st-key-ge432_topnav"] [data-baseweb="radio"],
div[class*="st-key-ge432_topnav"] svg{
  display:none!important;
  visibility:hidden!important;
  width:0!important;
  height:0!important;
  min-width:0!important;
  margin:0!important;
  padding:0!important;
}
@media(max-width:520px){
  div[class*="st-key-ge432_topnav"] [role="radiogroup"]{
    grid-template-columns:repeat(4,minmax(0,1fr))!important;
  }
  div[class*="st-key-ge432_topnav"] label{
    min-height:37px!important;
  }
  div[class*="st-key-ge432_topnav"] label p{
    font-size:.56rem!important;
  }
}

/* ===== GRIDIRON EDGE v4.3.1 MOCKUP MATCH ===== */
.block-container{max-width:760px!important;padding:calc(4px + env(safe-area-inset-top)) 16px 32px!important}

/* compact branded header */
.ge431-header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 5px;padding:4px 0 8px}
.ge431-brand{display:flex;align-items:center;gap:9px}
.ge431-header-right{display:flex;align-items:center;gap:9px}
.ge431-cfb{color:#8fa7bd;font-size:.54rem;font-weight:950;letter-spacing:.12em}
.ge-football-logo{width:40px!important;height:35px!important}
.ge-ball{width:34px!important;height:21px!important;top:6px!important}
.ge-wordmark span,.ge-wordmark strong{font-size:.96rem!important}
.ge-brand-sub{font-size:.40rem!important;margin-top:4px!important;letter-spacing:.14em!important}
.v420-live{padding:6px 9px!important;font-size:.49rem!important}


/* dark side-by-side filters */
div[class*="st-key-v420_game_date"] [data-baseweb="input"],
div[class*="st-key-v420_game_level"] [data-baseweb="select"]>div{min-height:45px!important;border-radius:14px!important;background:#0d2035!important;border:1px solid rgba(108,148,184,.15)!important;box-shadow:none!important}
div[class*="st-key-v420_game_date"] input{color:#eef5fb!important;-webkit-text-fill-color:#eef5fb!important;font-size:.71rem!important;font-weight:850!important}
div[class*="st-key-v420_game_level"] [data-baseweb="select"] *{color:#edf4fa!important;font-size:.71rem!important;font-weight:850!important}

.ge431-filter-summary{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0 9px}
.ge431-date-chip{padding:8px 10px;border-radius:12px;background:#0c2035;border:1px solid rgba(108,148,184,.13);color:#d7e3ef;font-size:.56rem;font-weight:850}
.ge431-games{color:#8aa2b8;font-size:.56rem;font-weight:850}

/* remove bulky old header cards */
.v420-slate-hero,.v420-day-summary,.v420-section-label,.v420-hero-row,.v420-eyebrow,.v420-market-pill{display:none!important}

/* exact segmented feel */
div[class*="st-key-v420_slate_segment"] [role="radiogroup"]{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;gap:3px!important;padding:3px!important;border-radius:14px!important;background:#0d2035!important;border:1px solid rgba(108,148,184,.14)!important}
div[class*="st-key-v420_slate_segment"] label{min-height:40px!important;margin:0!important;padding:0 4px!important;display:flex!important;align-items:center!important;justify-content:center!important;border-radius:11px!important}
div[class*="st-key-v420_slate_segment"] label p{margin:0!important;color:#8ca3b8!important;font-size:.59rem!important;font-weight:850!important}
div[class*="st-key-v420_slate_segment"] label:has(input:checked){background:linear-gradient(145deg,#2c8bfb,#1767cb)!important;box-shadow:0 6px 16px rgba(28,105,214,.25)!important}
div[class*="st-key-v420_slate_segment"] label:has(input:checked) p{color:#fff!important}
div[class*="st-key-v420_slate_segment"] input,
div[class*="st-key-v420_slate_segment"] [data-baseweb="radio"],
div[class*="st-key-v420_slate_segment"] svg{display:none!important;visibility:hidden!important;width:0!important;height:0!important;margin:0!important;padding:0!important}

.ge431-slate-meta{display:flex;align-items:center;gap:8px;color:#7d95ac;font-size:.54rem;margin:7px 2px 9px}
.ge431-slate-meta span:not(:last-child)::after{content:"";display:inline-block;width:3px;height:3px;border-radius:50%;background:#405a72;margin-left:8px;vertical-align:middle}
.ge431-slate-meta b{color:#dbe7f0}

/* compact settings and build button */
div[data-testid="stExpander"]{border-radius:14px!important;background:#0c1f33!important;border:1px solid rgba(108,148,184,.12)!important}
div[data-testid="stExpander"] summary{min-height:42px!important}
div[class*="st-key-v420_run_slate"] button{min-height:48px!important;border-radius:14px!important;background:linear-gradient(135deg,#246cf0,#3a8bff)!important;font-size:.74rem!important;font-weight:850!important;box-shadow:0 11px 24px rgba(35,108,228,.22)!important}

/* connected official module */
.ge-official-head{margin:17px 0 0!important;padding:12px 13px!important;border-radius:17px 17px 0 0!important;background:#0d2137!important;border:1px solid rgba(108,148,184,.14)!important}
.ge-check{width:29px!important;height:29px!important;border-radius:9px!important}
.ge-section-title{font-size:.81rem!important}
.ge-section-sub{font-size:.51rem!important}
.ge-count{font-size:.43rem!important;padding:5px 7px!important}
.ge-official-card{margin:0!important;padding:14px!important;border-radius:0!important;border-top:0!important;background:linear-gradient(150deg,#0c2036,#09192b)!important}
.ge-official-card:last-of-type{border-radius:0 0 17px 17px!important}
.ge-official-card.best{background:radial-gradient(circle at 90% 0%,rgba(37,132,255,.20),transparent 33%),linear-gradient(145deg,#143960,#0c2949)!important}
.ge-rank{width:31px!important;height:31px!important}
.ge-verdict{font-size:.43rem!important;padding:5px 8px!important}
.ge431-pickline{display:flex;align-items:center;gap:10px;margin:11px 0 0 37px}
.ge431-team-mark{width:33px;height:33px;flex:0 0 auto;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#102b48;border:1px solid rgba(85,148,205,.18);color:#8cc5ff;font-size:.55rem;font-weight:950}
.ge431-pickcopy{min-width:0}
.ge-pick{margin:0!important;font-size:1rem!important}
.ge-game{margin:3px 0 0!important;font-size:.52rem!important}
.ge-metric-grid{margin-top:12px!important;gap:4px!important}
.ge-metric-grid>div{padding:7px 2px!important;border-radius:9px!important}
.ge-metric-grid span{font-size:.37rem!important}
.ge-metric-grid b{font-size:.58rem!important}

/* connected lean module */
.ge-leans-head{margin-top:15px!important;padding:11px 12px!important;border-radius:17px 17px 0 0!important;background:#0d2137!important}
.ge-lean-row{border-radius:0!important;background:#09192b!important;padding:10px 11px!important}
.ge-lean-row:last-of-type{border-radius:0 0 17px 17px!important}
.ge-lean-rank{width:25px!important;height:25px!important}
.ge-lean-main b{font-size:.60rem!important}
.ge-lean-main small{font-size:.44rem!important}
.ge-lean-stats{font-size:.45rem!important;gap:6px!important}
.ge-lean-pill{font-size:.39rem!important;padding:4px 6px!important}

/* tracker */
.ge-tracker-card{margin-top:16px!important;padding:12px!important;border-radius:17px!important}
.ge-tracker-grid{padding:8px 2px!important;margin-top:8px!important}
.ge-tracker-grid b{font-size:.66rem!important}
.ge-tracker-grid span{font-size:.39rem!important}

/* old nav gone */
div[class*="st-key-cfb_nav_"]{display:none!important}

@media(max-width:520px){
  .block-container{padding-left:13px!important;padding-right:13px!important}
  .ge-wordmark span,.ge-wordmark strong{font-size:.89rem!important}
  .ge-brand-sub{font-size:.37rem!important}
  .ge431-team-mark{width:30px;height:30px}
  .ge431-pickline{margin-left:35px}
  .ge-lean-stats{gap:5px!important}
}

/* ===== GRIDIRON EDGE v4.3 brand + generated-reference layout ===== */
.ge-appbar{padding:10px 0 18px!important;margin-bottom:15px!important}
.ge-brandmark{gap:10px!important}
.se-goalpost-logo{width:40px;height:40px;flex:0 0 auto;display:flex;align-items:center;justify-content:center}
.se-goalpost-logo svg{display:block}
.ge-football-logo{width:48px;height:42px;position:relative;flex:0 0 auto}
.ge-ball{position:absolute;width:39px;height:24px;left:2px;top:8px;border:3px solid #edf6ff;border-radius:50%;transform:rotate(-28deg);box-shadow:0 0 18px rgba(69,157,255,.13)}
.ge-ball:before{content:"";position:absolute;width:14px;height:3px;background:#edf6ff;left:10px;top:8px;border-radius:3px}
.ge-ball span,.ge-ball b,.ge-ball i{position:absolute;width:2px;height:7px;background:#edf6ff;top:6px;border-radius:2px}
.ge-ball span{left:13px}.ge-ball b{left:18px}.ge-ball i{left:23px}
.ge-wordmark{font-style:italic;line-height:1;white-space:nowrap}
.ge-wordmark span{color:#f7fbff;font-size:1.08rem;font-weight:950;letter-spacing:.035em}
.ge-wordmark strong{color:#388cff;font-size:1.08rem;font-weight:950;letter-spacing:.035em}
.ge-brand-sub{margin-top:6px;color:#7590aa;font-size:.48rem;font-weight:900;letter-spacing:.17em}
.ge-right{display:flex;align-items:center;gap:9px}
.ge-cfb{color:#8ba4bd;font-size:.59rem;font-weight:950;letter-spacing:.13em}

.ge-official-head,.ge-leans-head{display:flex;align-items:center;gap:10px;margin:23px 0 10px;padding:0 2px}
.ge-check{width:31px;height:31px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:linear-gradient(145deg,#2192ff,#1162d7);color:#fff;font-weight:950;box-shadow:0 8px 22px rgba(28,122,244,.25)}
.ge-check.muted{background:#11243a;color:#607991;box-shadow:none}
.ge-section-title{color:#f5f9ff;font-size:.91rem;font-weight:950;letter-spacing:-.01em}
.ge-section-title span{color:#8ba1b6;font-weight:850}
.ge-section-sub{color:#6e879f;font-size:.56rem;margin-top:3px}
.ge-count{margin-left:auto;color:#b9d8ff;background:#102b48;border:1px solid rgba(71,145,231,.16);padding:5px 8px;border-radius:999px;font-size:.48rem;font-weight:950;letter-spacing:.08em}
.ge-count.amber{color:#e7cd78;background:rgba(150,119,33,.08);border-color:rgba(196,158,55,.15)}

.ge-official-card{padding:16px 15px;margin:9px 0;border-radius:19px;background:linear-gradient(150deg,#0d2239,#09192b);border:1px solid rgba(91,135,175,.18);box-shadow:0 14px 35px rgba(0,0,0,.17)}
.ge-official-card.best{background:radial-gradient(circle at 88% 0%,rgba(40,131,255,.22),transparent 34%),linear-gradient(145deg,#12355b,#0b2340);border-color:rgba(67,151,249,.42);box-shadow:0 18px 44px rgba(0,0,0,.24)}
.ge-official-card.bet{border-color:rgba(61,203,151,.23)}
.ge-card-top{display:flex;align-items:center;gap:8px}
.ge-rank{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#0c2036;border:1px solid rgba(109,152,191,.13);color:#9bb3ca;font-size:.64rem;font-weight:950}
.ge-official-card.best .ge-rank{background:linear-gradient(145deg,#2788ed,#1762b7);color:#fff}
.ge-market{display:inline-flex;padding:5px 7px;border-radius:8px;background:rgba(100,145,186,.08);color:#7e9ab4;font-size:.46rem;font-weight:950;letter-spacing:.10em}
.ge-market.small{padding:3px 5px;margin-right:6px;font-size:.40rem}
.ge-verdict{margin-left:auto;padding:6px 9px;border-radius:999px;font-size:.47rem;font-weight:950;letter-spacing:.09em}
.ge-verdict.best{color:#8dccff;border:1px solid rgba(57,143,244,.34);background:rgba(31,119,225,.10)}
.ge-verdict.bet{color:#7ce5b4;border:1px solid rgba(58,202,145,.28);background:rgba(39,178,121,.08)}
.ge-pick{margin:12px 0 0 42px;color:#fff;font-size:1.17rem;font-weight:950;letter-spacing:-.035em}
.ge-game{margin:4px 0 0 42px;color:#6d869e;font-size:.59rem}
.ge-metric-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin-top:14px}
.ge-metric-grid>div{padding:8px 3px;text-align:center;border-radius:10px;background:rgba(3,13,24,.35);border:1px solid rgba(104,145,181,.08)}
.ge-metric-grid span{display:block;color:#58728b;font-size:.41rem;font-weight:950;letter-spacing:.07em;text-transform:uppercase}
.ge-metric-grid b{display:block;margin-top:3px;color:#e4eef8;font-size:.65rem;font-weight:950;white-space:nowrap}
.ge-official-card.best .ge-metric-grid>div:nth-child(3) b,.ge-official-card.bet .ge-metric-grid>div:nth-child(3) b{color:#5ce0a5}

.ge-leans-head{padding:13px 12px;margin-top:17px;border:1px solid rgba(116,151,183,.11);border-bottom:0;border-radius:17px 17px 0 0;background:#0b1d31}
.ge-lean-row{display:grid;grid-template-columns:28px minmax(0,1fr) auto auto;align-items:center;gap:8px;padding:11px 12px;border-left:1px solid rgba(116,151,183,.11);border-right:1px solid rgba(116,151,183,.11);border-top:1px solid rgba(116,151,183,.07);background:#09192b}
.ge-lean-row:last-of-type{border-radius:0 0 17px 17px;border-bottom:1px solid rgba(116,151,183,.11)}
.ge-lean-rank{width:27px;height:27px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:#0e2238;color:#6d87a0;font-size:.52rem;font-weight:950}
.ge-lean-main{min-width:0}.ge-lean-main b{color:#edf5fc;font-size:.68rem}.ge-lean-main small{display:block;color:#617b94;font-size:.49rem;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ge-lean-stats{display:flex;gap:8px;color:#93a9bd;font-size:.51rem;font-weight:800;white-space:nowrap}
.ge-lean-pill{padding:5px 7px;border-radius:999px;color:#e4c86d;border:1px solid rgba(195,158,54,.25);background:rgba(166,129,32,.07);font-size:.43rem;font-weight:950;letter-spacing:.08em}

.ge-tracker-card{margin:18px 0 8px;padding:14px;border-radius:18px;background:linear-gradient(145deg,#0d2238,#09192a);border:1px solid rgba(103,145,183,.13)}
.ge-tracker-head{display:flex;align-items:center;gap:8px;color:#eef6ff;font-size:.76rem}.ge-tracker-head span{color:#3f99ff}.ge-tracker-head b{font-weight:950}
.ge-tracker-grid{display:grid;grid-template-columns:repeat(4,1fr);margin-top:11px;padding:10px 2px;border-radius:12px;background:rgba(3,13,24,.25)}
.ge-tracker-grid>div{text-align:center;border-right:1px solid rgba(110,147,180,.09)}.ge-tracker-grid>div:last-child{border-right:0}
.ge-tracker-grid b{display:block;color:#edf6ff;font-size:.72rem;font-weight:950}.ge-tracker-grid>div:nth-child(2) b{color:#58dda5}
.ge-tracker-grid span{display:block;color:#637e97;font-size:.44rem;margin-top:3px}
.ge-empty{margin-top:8px!important}

/* stronger app-like spacing */
.v420-day-summary{margin-bottom:13px!important}
.v420-slate-hero{padding:15px!important;margin-bottom:13px!important}
.v420-hero-title{font-size:1.42rem!important}
.v420-hero-copy{font-size:.62rem!important}
.v420-section-label{margin-top:0!important}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="ge433-header">
      <div class="ge431-brand">
        <div class="se-goalpost-logo" aria-hidden="true">
          <svg viewBox="0 0 44 44" width="40" height="40" xmlns="http://www.w3.org/2000/svg">
            <rect x="1" y="1" width="42" height="42" rx="11" fill="#0b1b2d" stroke="#22405f" stroke-width="1.5"/>
            <line x1="22" y1="34" x2="22" y2="22" stroke="#edf6ff" stroke-width="3" stroke-linecap="round"/>
            <line x1="10" y1="22" x2="34" y2="22" stroke="#2f6bff" stroke-width="3" stroke-linecap="round"/>
            <line x1="10" y1="22" x2="10" y2="11" stroke="#2f6bff" stroke-width="3" stroke-linecap="round"/>
            <line x1="34" y1="22" x2="34" y2="11" stroke="#2f6bff" stroke-width="3" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="ge-wordmark"><span>SATURDAY</span> <strong>EDGE</strong></div>
          <div class="ge-brand-sub">DATA DRIVEN. GAME READY.</div>
        </div>
      </div>
      <div class="ge433-head-right">
        <span class="ge433-cfb">CFB⌄</span>
        <span class="ge433-profile">●</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# v4.3.2 native-style horizontal navigation
if "cfb_page" not in st.session_state:
    st.session_state["cfb_page"] = "Slate"

_ge432_alias = {"Home": "Game", "Bets": "Slate", "Live": "Slate"}
_ge432_view = _ge432_alias.get(
    st.session_state.get("cfb_page", "Slate"),
    st.session_state.get("cfb_page", "Slate"),
)
_ge432_options = ["Slate", "Games", "Tracker", "More"]

# Keep the segmented nav synchronized with navigation triggered elsewhere.
# Only resync when the page changed OUTSIDE this widget; otherwise a user's
# own tap gets overwritten before it can route.
_ge432_visible_view = "Games" if _ge432_view == "Game" else _ge432_view
if st.session_state.get("ge432_topnav") not in _ge432_options:
    st.session_state["ge432_topnav"] = _ge432_visible_view
elif st.session_state.get("_ge432_last_view") != _ge432_visible_view:
    st.session_state["ge432_topnav"] = _ge432_visible_view
st.session_state["_ge432_last_view"] = _ge432_visible_view

_ge432_choice = st.radio(
    "",
    _ge432_options,
    horizontal=True,
    label_visibility="collapsed",
    key="ge432_topnav",
)

_ge432_route = "Game" if _ge432_choice == "Games" else _ge432_choice
if _ge432_route != _ge432_view:
    st.session_state["cfb_page"] = _ge432_route
    st.rerun()



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

    Accepts either a Python string or bytes so research exports and legacy
    exports use the same safe download path.
    """
    if isinstance(csv_text, bytes):
        raw_bytes = csv_text
    else:
        raw_bytes = str(csv_text).encode("utf-8")
    payload = base64.b64encode(raw_bytes).decode("ascii")
    safe_label = html.escape(label)
    safe_filename = html.escape(filename, quote=True)

    components.html(
        f"""
        <div style="width:100%;padding:0;margin:0;">
          <button id="saveBtn"
            style="
              width:100%;
              min-height:44px;
              border:1px solid rgba(96,165,250,.28);
              border-radius:12px;
              background:linear-gradient(135deg,#0F2747,#12325B);
              color:#EAF2FF;
              font-size:15px;
              font-weight:750;
              box-shadow:0 10px 24px rgba(0,0,0,.18);
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


@st.cache_data(ttl=86400, show_spinner=False)
def get_backtest_games(year):
    return cfbd_get("/games", API_KEY, {"year": int(year), "seasonType": "regular"})

@st.cache_data(ttl=86400, show_spinner=False)
def get_backtest_lines(year):
    return fetch_lines(API_KEY, year=int(year))

@st.cache_data(ttl=86400, show_spinner=False)
def get_backtest_model_data(year):
    return load_model_data(API_KEY, int(year))


# ===== v3.0 point-in-time CFB rebuild =====
V3_VERSION = "v3.0.0-point-in-time"
V3_TRAIN_START = 2018
V3_RIDGE_ALPHA = 20.0
V3_MIN_TRAIN_ROWS = 500
V3_RESIDUAL_CAP_SPREAD = 5.0
V3_RESIDUAL_CAP_TOTAL = 4.0


def _v3_num(v, default=np.nan):
    try:
        if v is None or v == "":
            return default
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _v3_map(rows, key="team"):
    out = {}
    for r in rows or []:
        k = r.get(key)
        if k:
            out[str(k)] = r
    return out


def _v3_sp_fields(row):
    row = row or {}
    off = row.get("offense") or {}
    deff = row.get("defense") or {}
    havoc = deff.get("havoc") or {}
    stx = row.get("specialTeams") or {}
    return {
        "rating": _v3_num(row.get("rating")),
        "off_rating": _v3_num(off.get("rating")),
        "def_rating": _v3_num(deff.get("rating")),
        "st_rating": _v3_num(stx.get("rating")),
        "off_pass": _v3_num(off.get("passing")),
        "off_rush": _v3_num(off.get("rushing")),
        "off_success": _v3_num(off.get("success")),
        "off_expl": _v3_num(off.get("explosiveness")),
        "off_pace": _v3_num(off.get("pace")),
        "off_run_rate": _v3_num(off.get("runRate")),
        "off_standard": _v3_num(off.get("standardDowns")),
        "off_passing_downs": _v3_num(off.get("passingDowns")),
        "def_pass": _v3_num(deff.get("passing")),
        "def_rush": _v3_num(deff.get("rushing")),
        "def_success": _v3_num(deff.get("success")),
        "def_expl": _v3_num(deff.get("explosiveness")),
        "def_standard": _v3_num(deff.get("standardDowns")),
        "def_passing_downs": _v3_num(deff.get("passingDowns")),
        "def_havoc": _v3_num(havoc.get("total")),
    }


def _v3_core_fields(row):
    row = row or {}
    return {
        "overall": _v3_num(row.get("overall")),
        "offense": _v3_num(row.get("offense")),
        "defense": _v3_num(row.get("defense")),
    }


def _v3_fpi_fields(row):
    row = row or {}
    eff = row.get("efficiencies") or {}
    return {
        "fpi": _v3_num(row.get("fpi")),
        "overall": _v3_num(eff.get("overall")),
        "offense": _v3_num(eff.get("offense")),
        "defense": _v3_num(eff.get("defense")),
        "special": _v3_num(eff.get("specialTeams")),
    }


def _v3_adv_fields(row):
    row = row or {}
    off = row.get("offense") or {}
    deff = row.get("defense") or {}

    def split(side, key):
        return side.get(key) or {}

    op = split(off, "passingPlays")
    oru = split(off, "rushingPlays")
    osd = split(off, "standardDowns")
    opd = split(off, "passingDowns")
    oh = split(off, "havoc")
    ofp = split(off, "fieldPosition")

    dp = split(deff, "passingPlays")
    dru = split(deff, "rushingPlays")
    dsd = split(deff, "standardDowns")
    dpd = split(deff, "passingDowns")
    dh = split(deff, "havoc")
    dfp = split(deff, "fieldPosition")

    return {
        "off_ppa": _v3_num(off.get("ppa")),
        "off_success": _v3_num(off.get("successRate")),
        "off_expl": _v3_num(off.get("explosiveness")),
        "off_ppo": _v3_num(off.get("pointsPerOpportunity")),
        "off_plays": _v3_num(off.get("plays")),
        "off_drives": _v3_num(off.get("drives")),
        "off_pass_ppa": _v3_num(op.get("ppa")),
        "off_pass_success": _v3_num(op.get("successRate")),
        "off_pass_expl": _v3_num(op.get("explosiveness")),
        "off_pass_rate": _v3_num(op.get("rate")),
        "off_rush_ppa": _v3_num(oru.get("ppa")),
        "off_rush_success": _v3_num(oru.get("successRate")),
        "off_rush_expl": _v3_num(oru.get("explosiveness")),
        "off_rush_rate": _v3_num(oru.get("rate")),
        "off_std_ppa": _v3_num(osd.get("ppa")),
        "off_std_success": _v3_num(osd.get("successRate")),
        "off_passdown_ppa": _v3_num(opd.get("ppa")),
        "off_passdown_success": _v3_num(opd.get("successRate")),
        "off_line_yards": _v3_num(off.get("lineYards")),
        "off_stuff_rate": _v3_num(off.get("stuffRate")),
        "off_power_success": _v3_num(off.get("powerSuccess")),
        "off_open_field": _v3_num(off.get("openFieldYards")),
        "off_second_level": _v3_num(off.get("secondLevelYards")),
        "off_field_start": _v3_num(ofp.get("averageStart")),
        "off_field_ep": _v3_num(ofp.get("averagePredictedPoints")),
        "off_havoc": _v3_num(oh.get("total")),

        "def_ppa": _v3_num(deff.get("ppa")),
        "def_success": _v3_num(deff.get("successRate")),
        "def_expl": _v3_num(deff.get("explosiveness")),
        "def_ppo": _v3_num(deff.get("pointsPerOpportunity")),
        "def_plays": _v3_num(deff.get("plays")),
        "def_drives": _v3_num(deff.get("drives")),
        "def_pass_ppa": _v3_num(dp.get("ppa")),
        "def_pass_success": _v3_num(dp.get("successRate")),
        "def_pass_expl": _v3_num(dp.get("explosiveness")),
        "def_rush_ppa": _v3_num(dru.get("ppa")),
        "def_rush_success": _v3_num(dru.get("successRate")),
        "def_rush_expl": _v3_num(dru.get("explosiveness")),
        "def_std_ppa": _v3_num(dsd.get("ppa")),
        "def_std_success": _v3_num(dsd.get("successRate")),
        "def_passdown_ppa": _v3_num(dpd.get("ppa")),
        "def_passdown_success": _v3_num(dpd.get("successRate")),
        "def_line_yards": _v3_num(deff.get("lineYards")),
        "def_stuff_rate": _v3_num(deff.get("stuffRate")),
        "def_power_success": _v3_num(deff.get("powerSuccess")),
        "def_open_field": _v3_num(deff.get("openFieldYards")),
        "def_second_level": _v3_num(deff.get("secondLevelYards")),
        "def_field_start": _v3_num(dfp.get("averageStart")),
        "def_field_ep": _v3_num(dfp.get("averagePredictedPoints")),
        "def_havoc": _v3_num(dh.get("total")),
    }


def _v3_portal_team_map(rows):
    out = {}
    for r in rows or []:
        rating = _v3_num(r.get("rating"), 0.0)
        stars = _v3_num(r.get("stars"), 0.0)
        origin = r.get("origin")
        dest = r.get("destination")

        if origin:
            x = out.setdefault(str(origin), {
                "in_count": 0, "out_count": 0,
                "in_rating": 0.0, "out_rating": 0.0,
                "in_stars": 0.0, "out_stars": 0.0,
            })
            x["out_count"] += 1
            x["out_rating"] += rating
            x["out_stars"] += stars
        if dest:
            x = out.setdefault(str(dest), {
                "in_count": 0, "out_count": 0,
                "in_rating": 0.0, "out_rating": 0.0,
                "in_stars": 0.0, "out_stars": 0.0,
            })
            x["in_count"] += 1
            x["in_rating"] += rating
            x["in_stars"] += stars

    for team, x in out.items():
        x["net_count"] = x["in_count"] - x["out_count"]
        x["net_rating"] = x["in_rating"] - x["out_rating"]
        x["net_stars"] = x["in_stars"] - x["out_stars"]
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def _v3_preseason_data(year):
    """
    Only information knowable before or at the start of the target season:
    prior-season ratings/advanced data plus current recruiting, talent,
    returning production and portal movement.
    """
    y = int(year)
    prev_sp = _v3_map(_safe_fetch(fetch_sp, API_KEY, y - 1))
    prev_core = _v3_map(_safe_fetch(fetch_core, API_KEY, y - 1))
    prev_fpi = _v3_map(_safe_fetch(fetch_fpi, API_KEY, y - 1))
    prev_adv = _v3_map(_safe_fetch(fetch_advanced, API_KEY, y - 1))
    talent = _v3_map(_safe_fetch(fetch_talent, API_KEY, y))
    returning = _v3_map(_safe_fetch(fetch_returning, API_KEY, y))
    recruiting = _v3_map(_safe_fetch(fetch_recruiting_teams, API_KEY, y))
    portal = _v3_portal_team_map(_safe_fetch(fetch_portal, API_KEY, y))
    return {
        "prev_sp": prev_sp,
        "prev_core": prev_core,
        "prev_fpi": prev_fpi,
        "prev_adv": prev_adv,
        "talent": talent,
        "returning": returning,
        "recruiting": recruiting,
        "portal": portal,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def _v3_advanced_through(year, end_week):
    if int(end_week) < 1:
        return {}
    return _v3_map(_safe_fetch(fetch_advanced_through_week, API_KEY, int(year), int(end_week)))


def _v3_team_snapshot(team, preseason, current_adv):
    t = str(team)
    sp = _v3_sp_fields((preseason["prev_sp"] or {}).get(t))
    core = _v3_core_fields((preseason["prev_core"] or {}).get(t))
    fpi = _v3_fpi_fields((preseason["prev_fpi"] or {}).get(t))
    pa = _v3_adv_fields((preseason["prev_adv"] or {}).get(t))
    ca = _v3_adv_fields((current_adv or {}).get(t))
    talent = (preseason["talent"] or {}).get(t) or {}
    ret = (preseason["returning"] or {}).get(t) or {}
    rec = (preseason["recruiting"] or {}).get(t) or {}
    portal = (preseason["portal"] or {}).get(t) or {}
    return {
        "sp": sp,
        "core": core,
        "fpi": fpi,
        "prior_adv": pa,
        "cur_adv": ca,
        "talent": _v3_num(talent.get("talent")),
        "returning_ppa": _v3_num(ret.get("percentPPA")),
        "returning_pass": _v3_num(ret.get("percentPassingPPA")),
        "returning_rush": _v3_num(ret.get("percentRushingPPA")),
        "returning_usage": _v3_num(ret.get("usage")),
        "recruit_points": _v3_num(rec.get("points")),
        "recruit_rank": _v3_num(rec.get("rank")),
        "portal_net_count": _v3_num(portal.get("net_count"), 0.0),
        "portal_net_rating": _v3_num(portal.get("net_rating"), 0.0),
        "portal_net_stars": _v3_num(portal.get("net_stars"), 0.0),
    }


def _v3_diff(row, h, a, key):
    return _v3_num(h.get(key)) - _v3_num(a.get(key))


def _v3_matchup(home_adv, away_adv, prefix="cur"):
    """
    Margin-oriented matchup features: home offense vs away defense minus
    away offense vs home defense. Positive generally favors the home team.
    """
    def n(d, k):
        return _v3_num(d.get(k))

    return {
        f"{prefix}_ppa_match": (n(home_adv, "off_ppa") - n(away_adv, "def_ppa")) - (n(away_adv, "off_ppa") - n(home_adv, "def_ppa")),
        f"{prefix}_pass_ppa_match": (n(home_adv, "off_pass_ppa") - n(away_adv, "def_pass_ppa")) - (n(away_adv, "off_pass_ppa") - n(home_adv, "def_pass_ppa")),
        f"{prefix}_rush_ppa_match": (n(home_adv, "off_rush_ppa") - n(away_adv, "def_rush_ppa")) - (n(away_adv, "off_rush_ppa") - n(home_adv, "def_rush_ppa")),
        f"{prefix}_success_match": (n(home_adv, "off_success") - n(away_adv, "def_success")) - (n(away_adv, "off_success") - n(home_adv, "def_success")),
        f"{prefix}_pass_success_match": (n(home_adv, "off_pass_success") - n(away_adv, "def_pass_success")) - (n(away_adv, "off_pass_success") - n(home_adv, "def_pass_success")),
        f"{prefix}_rush_success_match": (n(home_adv, "off_rush_success") - n(away_adv, "def_rush_success")) - (n(away_adv, "off_rush_success") - n(home_adv, "def_rush_success")),
        f"{prefix}_expl_match": (n(home_adv, "off_expl") - n(away_adv, "def_expl")) - (n(away_adv, "off_expl") - n(home_adv, "def_expl")),
        f"{prefix}_std_match": (n(home_adv, "off_std_ppa") - n(away_adv, "def_std_ppa")) - (n(away_adv, "off_std_ppa") - n(home_adv, "def_std_ppa")),
        f"{prefix}_passdown_match": (n(home_adv, "off_passdown_ppa") - n(away_adv, "def_passdown_ppa")) - (n(away_adv, "off_passdown_ppa") - n(home_adv, "def_passdown_ppa")),
        f"{prefix}_line_match": (n(home_adv, "off_line_yards") - n(away_adv, "def_line_yards")) - (n(away_adv, "off_line_yards") - n(home_adv, "def_line_yards")),
        f"{prefix}_stuff_match": (n(away_adv, "def_stuff_rate") - n(home_adv, "off_stuff_rate")) - (n(home_adv, "def_stuff_rate") - n(away_adv, "off_stuff_rate")),
        f"{prefix}_ppo_match": (n(home_adv, "off_ppo") - n(away_adv, "def_ppo")) - (n(away_adv, "off_ppo") - n(home_adv, "def_ppo")),
        f"{prefix}_havoc_diff": n(home_adv, "def_havoc") - n(away_adv, "def_havoc"),
        f"{prefix}_field_pos_diff": n(home_adv, "off_field_ep") - n(away_adv, "off_field_ep"),
        f"{prefix}_pace_sum": (
            n(home_adv, "off_plays") / max(n(home_adv, "off_drives"), 1.0)
            + n(away_adv, "off_plays") / max(n(away_adv, "off_drives"), 1.0)
        ),
    }


def _v3_total_context(home_adv, away_adv, prefix="cur"):
    def n(d, k):
        return _v3_num(d.get(k))
    return {
        f"{prefix}_total_ppa": n(home_adv, "off_ppa") + n(away_adv, "off_ppa") + n(home_adv, "def_ppa") + n(away_adv, "def_ppa"),
        f"{prefix}_total_success": n(home_adv, "off_success") + n(away_adv, "off_success") + n(home_adv, "def_success") + n(away_adv, "def_success"),
        f"{prefix}_total_expl": n(home_adv, "off_expl") + n(away_adv, "off_expl") + n(home_adv, "def_expl") + n(away_adv, "def_expl"),
        f"{prefix}_total_ppo": n(home_adv, "off_ppo") + n(away_adv, "off_ppo") + n(home_adv, "def_ppo") + n(away_adv, "def_ppo"),
        f"{prefix}_total_pass": n(home_adv, "off_pass_ppa") + n(away_adv, "off_pass_ppa") + n(home_adv, "def_pass_ppa") + n(away_adv, "def_pass_ppa"),
        f"{prefix}_total_rush": n(home_adv, "off_rush_ppa") + n(away_adv, "off_rush_ppa") + n(home_adv, "def_rush_ppa") + n(away_adv, "def_rush_ppa"),
        f"{prefix}_total_pace": (
            n(home_adv, "off_plays") / max(n(home_adv, "off_drives"), 1.0)
            + n(away_adv, "off_plays") / max(n(away_adv, "off_drives"), 1.0)
        ),
        f"{prefix}_total_havoc": n(home_adv, "def_havoc") + n(away_adv, "def_havoc"),
    }


def _v3_game_feature_row(game, market, preseason, current_adv):
    home = game.get("homeTeam")
    away = game.get("awayTeam")
    hs = _v3_team_snapshot(home, preseason, current_adv)
    as_ = _v3_team_snapshot(away, preseason, current_adv)

    week = int(game.get("week") or 1)
    neutral = 1.0 if bool(game.get("neutralSite")) else 0.0
    hfa = 0.0 if neutral else DEFAULT_HFA

    # Preserve historical kickoff so downstream validation can reproduce
    # the exact early / midday / late slate the user would have seen.
    _kick_raw = game.get("startDate") or game.get("start_date") or game.get("startTime")
    _kick_et = pd.NaT
    try:
        if _kick_raw:
            _kick_et = pd.to_datetime(_kick_raw, utc=True).tz_convert("America/New_York")
    except Exception:
        _kick_et = pd.NaT

    _kick_hour = np.nan
    _kick_label = "Unknown"
    _kick_date = ""
    if pd.notna(_kick_et):
        _kick_hour = float(_kick_et.hour) + float(_kick_et.minute) / 60.0
        _kick_date = _kick_et.strftime("%Y-%m-%d")
        # Practical Saturday windows:
        # Early: before 2:30 PM ET
        # Midday: 2:30 PM through 6:29 PM ET
        # Late: 6:30 PM ET and later
        if _kick_hour < 14.5:
            _kick_label = "Early"
        elif _kick_hour < 18.5:
            _kick_label = "Midday"
        else:
            _kick_label = "Late"

    row = {
        "season": int(game.get("season") or 0),
        "week": week,
        "game_id": game.get("id"),
        "kickoff_et": "" if pd.isna(_kick_et) else _kick_et.strftime("%Y-%m-%d %I:%M %p"),
        "game_date_et": _kick_date,
        "kickoff_hour_et": _kick_hour,
        "slate_window": _kick_label,
        "home_team": home,
        "away_team": away,
        "home_points": _v3_num(game.get("homePoints")),
        "away_points": _v3_num(game.get("awayPoints")),
        "actual_margin": _v3_num(game.get("homePoints")) - _v3_num(game.get("awayPoints")),
        "actual_total": _v3_num(game.get("homePoints")) + _v3_num(game.get("awayPoints")),
        "market_margin": -_v3_num(market.get("home_spread")) if market and market.get("home_spread") is not None else np.nan,
        "market_total": _v3_num(market.get("total")) if market else np.nan,
        "week_num": float(week),
        "early_week": 1.0 if week <= 3 else 0.0,
        "hfa": float(hfa),
        "neutral": neutral,
        "conference_game": 1.0 if bool(game.get("conferenceGame")) else 0.0,
        "home_pregame_elo": _v3_num(game.get("homePregameElo")),
        "away_pregame_elo": _v3_num(game.get("awayPregameElo")),
        "elo_diff": _v3_num(game.get("homePregameElo")) - _v3_num(game.get("awayPregameElo")),
        "talent_diff": hs["talent"] - as_["talent"],
        "returning_ppa_diff": hs["returning_ppa"] - as_["returning_ppa"],
        "returning_pass_diff": hs["returning_pass"] - as_["returning_pass"],
        "returning_rush_diff": hs["returning_rush"] - as_["returning_rush"],
        "returning_usage_diff": hs["returning_usage"] - as_["returning_usage"],
        "recruit_points_diff": hs["recruit_points"] - as_["recruit_points"],
        "recruit_rank_diff": as_["recruit_rank"] - hs["recruit_rank"],  # positive = better home rank
        "portal_net_count_diff": hs["portal_net_count"] - as_["portal_net_count"],
        "portal_net_rating_diff": hs["portal_net_rating"] - as_["portal_net_rating"],
        "portal_net_stars_diff": hs["portal_net_stars"] - as_["portal_net_stars"],
    }

    # Prior rating differentials.
    for k in [
        "rating","off_rating","def_rating","st_rating","off_pass","off_rush",
        "off_success","off_expl","off_pace","off_standard","off_passing_downs",
        "def_pass","def_rush","def_success","def_expl","def_standard",
        "def_passing_downs","def_havoc",
    ]:
        row[f"sp_{k}_diff"] = _v3_num(hs["sp"].get(k)) - _v3_num(as_["sp"].get(k))

    for k in ["overall","offense","defense"]:
        row[f"core_{k}_diff"] = _v3_num(hs["core"].get(k)) - _v3_num(as_["core"].get(k))

    for k in ["fpi","overall","offense","defense","special"]:
        row[f"fpi_{k}_diff"] = _v3_num(hs["fpi"].get(k)) - _v3_num(as_["fpi"].get(k))

    row.update(_v3_matchup(hs["prior_adv"], as_["prior_adv"], "prior"))
    row.update(_v3_matchup(hs["cur_adv"], as_["cur_adv"], "cur"))
    row.update(_v3_total_context(hs["prior_adv"], as_["prior_adv"], "prior"))
    row.update(_v3_total_context(hs["cur_adv"], as_["cur_adv"], "cur"))

    return row


V3_MARGIN_FEATURES = [
    "week_num","early_week","hfa","neutral","conference_game","elo_diff",
    "sp_rating_diff","sp_off_rating_diff","sp_def_rating_diff","sp_st_rating_diff",
    "sp_off_pass_diff","sp_off_rush_diff","sp_def_pass_diff","sp_def_rush_diff",
    "sp_off_success_diff","sp_def_success_diff","sp_off_expl_diff","sp_def_expl_diff",
    "sp_def_havoc_diff",
    "core_overall_diff","core_offense_diff","core_defense_diff",
    "fpi_fpi_diff","fpi_offense_diff","fpi_defense_diff","fpi_special_diff",
    "talent_diff","returning_ppa_diff","returning_pass_diff","returning_rush_diff",
    "returning_usage_diff","recruit_points_diff","recruit_rank_diff",
    "portal_net_count_diff","portal_net_rating_diff","portal_net_stars_diff",
    "prior_ppa_match","prior_pass_ppa_match","prior_rush_ppa_match",
    "prior_success_match","prior_expl_match","prior_std_match","prior_passdown_match",
    "prior_line_match","prior_stuff_match","prior_ppo_match","prior_havoc_diff",
    "cur_ppa_match","cur_pass_ppa_match","cur_rush_ppa_match",
    "cur_success_match","cur_pass_success_match","cur_rush_success_match",
    "cur_expl_match","cur_std_match","cur_passdown_match","cur_line_match",
    "cur_stuff_match","cur_ppo_match","cur_havoc_diff","cur_field_pos_diff",
]

V3_TOTAL_FEATURES = [
    "week_num","early_week","neutral",
    "sp_off_pace_diff",
    "prior_total_ppa","prior_total_success","prior_total_expl","prior_total_ppo",
    "prior_total_pass","prior_total_rush","prior_total_pace","prior_total_havoc",
    "cur_total_ppa","cur_total_success","cur_total_expl","cur_total_ppo",
    "cur_total_pass","cur_total_rush","cur_total_pace","cur_total_havoc",
    "talent_diff","returning_ppa_diff","recruit_points_diff",
]


def _v3_impute_train_test(train_df, test_df, features, min_coverage=0.45):
    usable = []
    medians = {}
    for f in features:
        if f not in train_df.columns or f not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[f], errors="coerce")
        coverage = float(tr.notna().mean()) if len(tr) else 0.0
        if coverage < min_coverage:
            continue
        med = float(tr.median()) if tr.notna().any() else 0.0
        usable.append(f)
        medians[f] = med

    if not usable:
        return None, None, []

    Xtr = np.column_stack([
        pd.to_numeric(train_df[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    Xte = np.column_stack([
        pd.to_numeric(test_df[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    return Xtr, Xte, usable


def _v3_ridge_predict(train_df, test_df, features, target, alpha=V3_RIDGE_ALPHA):
    tr = train_df.dropna(subset=[target]).copy()
    te = test_df.dropna(subset=[target]).copy()
    if len(tr) < V3_MIN_TRAIN_ROWS or te.empty:
        return None

    Xtr, Xte, usable = _v3_impute_train_test(tr, te, features)
    if Xtr is None or len(usable) < 5:
        return None

    ytr = pd.to_numeric(tr[target], errors="coerce").to_numpy(dtype=float)
    fit = _ridge_fit_numpy(Xtr, ytr, alpha)
    preds = np.array([_ridge_predict_numpy(fit, x) for x in Xte], dtype=float)
    return {
        "preds": preds,
        "test_index": te.index.to_numpy(),
        "usable_features": usable,
        "n_train": len(tr),
        "n_test": len(te),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def _v3_history_frame(start_year, end_year, scope):
    """
    Build a true point-in-time game frame.
    For a Week N game, current-season advanced stats are aggregated only
    through Week N-1. Pregame Elo comes directly from the historical game row.
    """
    rows = []
    for year in range(int(start_year), int(end_year) + 1):
        games = get_backtest_games(year)
        line_payload = get_backtest_lines(year)
        preseason = _v3_preseason_data(year)

        line_index = {}
        for lr in line_payload or []:
            gid = lr.get("id")
            if gid is None:
                continue
            try:
                key = int(gid)
            except Exception:
                key = gid
            line_index[key] = normalize_game_lines([lr], game_id=gid)

        weeks = sorted(set(
            int(g.get("week") or 1)
            for g in games or []
            if g.get("completed") is True
        ))

        adv_by_week = {}
        for w in weeks:
            adv_by_week[w] = _v3_advanced_through(year, w - 1)

        for g in games or []:
            if g.get("completed") is not True:
                continue
            if g.get("homePoints") is None or g.get("awayPoints") is None:
                continue
            if not _bt_game_scope(g, scope):
                continue

            gid = g.get("id")
            try:
                key = int(gid)
            except Exception:
                key = gid
            market = _bt_consensus_line(line_index.get(key, []))
            if not market:
                continue

            w = int(g.get("week") or 1)
            try:
                rows.append(_v3_game_feature_row(
                    g, market, preseason, adv_by_week.get(w, {})
                ))
            except Exception:
                continue

    return pd.DataFrame(rows)


def _v3_model_suite(history, test_seasons):
    if history is None or history.empty:
        return pd.DataFrame(), pd.DataFrame()

    result_rows = []
    pred_rows = []

    for season in sorted(set(int(s) for s in test_seasons)):
        train = history[history["season"].astype(int) < season].copy()
        test = history[history["season"].astype(int) == season].copy()
        if test.empty:
            continue

        # Direct football margin.
        direct_sp = _v3_ridge_predict(
            train, test, V3_MARGIN_FEATURES, "actual_margin"
        )

        # Market residual margin.
        train_sp_resid = train.copy()
        test_sp_resid = test.copy()
        train_sp_resid["spread_residual_target"] = (
            pd.to_numeric(train_sp_resid["actual_margin"], errors="coerce")
            - pd.to_numeric(train_sp_resid["market_margin"], errors="coerce")
        )
        test_sp_resid["spread_residual_target"] = (
            pd.to_numeric(test_sp_resid["actual_margin"], errors="coerce")
            - pd.to_numeric(test_sp_resid["market_margin"], errors="coerce")
        )
        residual_sp_features = V3_MARGIN_FEATURES + [
            "market_margin",
        ]
        resid_sp = _v3_ridge_predict(
            train_sp_resid, test_sp_resid, residual_sp_features,
            "spread_residual_target"
        )

        # Direct football total.
        direct_tot = _v3_ridge_predict(
            train, test, V3_TOTAL_FEATURES, "actual_total"
        )

        # Market residual total.
        train_tot_resid = train.copy()
        test_tot_resid = test.copy()
        train_tot_resid["total_residual_target"] = (
            pd.to_numeric(train_tot_resid["actual_total"], errors="coerce")
            - pd.to_numeric(train_tot_resid["market_total"], errors="coerce")
        )
        test_tot_resid["total_residual_target"] = (
            pd.to_numeric(test_tot_resid["actual_total"], errors="coerce")
            - pd.to_numeric(test_tot_resid["market_total"], errors="coerce")
        )
        residual_tot_features = V3_TOTAL_FEATURES + ["market_total"]
        resid_tot = _v3_ridge_predict(
            train_tot_resid, test_tot_resid, residual_tot_features,
            "total_residual_target"
        )

        # Common scoring helper.
        def add_market_rows(market_type, target_col, market_col):
            ss = test.dropna(subset=[target_col, market_col]).copy()
            if ss.empty:
                return
            y = pd.to_numeric(ss[target_col], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ss[market_col], errors="coerce").to_numpy(dtype=float)
            mae = float(np.mean(np.abs(y - m)))
            result_rows.append({
                "Market": market_type,
                "Model": "A · Market Control",
                "Season": season,
                "Games": len(ss),
                "MAE": mae,
                "Improvement vs Market": 0.0,
            })

        add_market_rows("SPREAD", "actual_margin", "market_margin")
        add_market_rows("TOTAL", "actual_total", "market_total")

        if direct_sp is not None:
            ss = test.loc[direct_sp["test_index"]].copy()
            y = pd.to_numeric(ss["actual_margin"], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ss["market_margin"], errors="coerce").to_numpy(dtype=float)
            p = direct_sp["preds"]
            model_mae = float(np.mean(np.abs(y - p)))
            market_mae = float(np.mean(np.abs(y - m)))
            result_rows.append({
                "Market": "SPREAD",
                "Model": "B · Football Direct",
                "Season": season,
                "Games": len(ss),
                "MAE": model_mae,
                "Improvement vs Market": market_mae - model_mae,
            })
            for idx, yy, mm, pp in zip(ss.index, y, m, p):
                pred_rows.append({
                    "market_type": "spread", "model": "B · Football Direct",
                    "season": season, "row_index": int(idx),
                    "actual": yy, "market": mm, "prediction": pp,
                    "abs_error": abs(yy - pp),
                })

        if resid_sp is not None:
            ss = test_sp_resid.loc[resid_sp["test_index"]].copy()
            y = pd.to_numeric(ss["actual_margin"], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ss["market_margin"], errors="coerce").to_numpy(dtype=float)
            corr = np.clip(resid_sp["preds"], -V3_RESIDUAL_CAP_SPREAD, V3_RESIDUAL_CAP_SPREAD)
            p = m + corr
            model_mae = float(np.mean(np.abs(y - p)))
            market_mae = float(np.mean(np.abs(y - m)))
            result_rows.append({
                "Market": "SPREAD",
                "Model": "C · Point-in-Time Hybrid",
                "Season": season,
                "Games": len(ss),
                "MAE": model_mae,
                "Improvement vs Market": market_mae - model_mae,
            })
            for idx, yy, mm, pp, cc in zip(ss.index, y, m, p, corr):
                pred_rows.append({
                    "market_type": "spread", "model": "C · Point-in-Time Hybrid",
                    "season": season, "row_index": int(idx),
                    "actual": yy, "market": mm, "prediction": pp,
                    "correction": cc, "abs_error": abs(yy - pp),
                })

        if direct_tot is not None:
            ss = test.loc[direct_tot["test_index"]].copy()
            y = pd.to_numeric(ss["actual_total"], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ss["market_total"], errors="coerce").to_numpy(dtype=float)
            p = direct_tot["preds"]
            model_mae = float(np.mean(np.abs(y - p)))
            market_mae = float(np.mean(np.abs(y - m)))
            result_rows.append({
                "Market": "TOTAL",
                "Model": "B · Football Direct",
                "Season": season,
                "Games": len(ss),
                "MAE": model_mae,
                "Improvement vs Market": market_mae - model_mae,
            })

        if resid_tot is not None:
            ss = test_tot_resid.loc[resid_tot["test_index"]].copy()
            y = pd.to_numeric(ss["actual_total"], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ss["market_total"], errors="coerce").to_numpy(dtype=float)
            corr = np.clip(resid_tot["preds"], -V3_RESIDUAL_CAP_TOTAL, V3_RESIDUAL_CAP_TOTAL)
            p = m + corr
            model_mae = float(np.mean(np.abs(y - p)))
            market_mae = float(np.mean(np.abs(y - m)))
            result_rows.append({
                "Market": "TOTAL",
                "Model": "C · Point-in-Time Hybrid",
                "Season": season,
                "Games": len(ss),
                "MAE": model_mae,
                "Improvement vs Market": market_mae - model_mae,
            })

    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows)


def _v3_summary(results, holdout):
    if results is None or results.empty:
        return pd.DataFrame()
    rows = []
    for (market, model_name), g in results.groupby(["Market", "Model"]):
        overall = float(np.average(
            pd.to_numeric(g["MAE"], errors="coerce"),
            weights=pd.to_numeric(g["Games"], errors="coerce"),
        ))
        avg_impr = float(np.average(
            pd.to_numeric(g["Improvement vs Market"], errors="coerce"),
            weights=pd.to_numeric(g["Games"], errors="coerce"),
        ))
        h = g[g["Season"].astype(int) == int(holdout)]
        hold_mae = float(h.iloc[0]["MAE"]) if not h.empty else np.nan
        hold_impr = float(h.iloc[0]["Improvement vs Market"]) if not h.empty else np.nan
        wins = int((pd.to_numeric(g["Improvement vs Market"], errors="coerce") > 0).sum())
        n = int(len(g))

        if model_name == "A · Market Control":
            status = "CONTROL"
        elif (
            market == "SPREAD"
            and n >= 3
            and wins >= max(2, n - 1)
            and avg_impr > 0
            and pd.notna(hold_impr) and hold_impr > 0
        ):
            status = "PROMOTION REVIEW"
        elif avg_impr > 0:
            status = "RESEARCH POSITIVE"
        else:
            status = "RESEARCH"

        rows.append({
            "Market": market,
            "Model": model_name,
            "Overall MAE": overall,
            "Weighted Improvement": avg_impr,
            f"{holdout} Holdout MAE": hold_mae,
            f"{holdout} Holdout Improvement": hold_impr,
            "Seasons Beat Market": f"{wins}/{n}",
            "Status": status,
        })
    return pd.DataFrame(rows)


def _v3_data_readiness(history):
    if history is None or history.empty:
        return pd.DataFrame()
    checks = {
        "Pregame Elo": ["elo_diff"],
        "Prior SP+": ["sp_rating_diff"],
        "Prior CORE": ["core_overall_diff"],
        "Prior FPI": ["fpi_fpi_diff"],
        "Talent": ["talent_diff"],
        "Returning production": ["returning_ppa_diff"],
        "Recruiting": ["recruit_points_diff"],
        "Transfer portal": ["portal_net_rating_diff"],
        "Prior advanced": ["prior_ppa_match", "prior_success_match"],
        "Point-in-time advanced": ["cur_ppa_match", "cur_success_match", "cur_line_match"],
        "Historical market": ["market_margin", "market_total"],
    }
    rows = []
    for label, cols in checks.items():
        vals = []
        for c in cols:
            if c in history.columns:
                vals.append(float(pd.to_numeric(history[c], errors="coerce").notna().mean()))
        coverage = float(np.mean(vals)) if vals else 0.0
        rows.append({
            "Data Source": label,
            "Coverage": coverage,
            "Status": "READY" if coverage >= 0.80 else ("PARTIAL" if coverage >= 0.35 else "MISSING"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v3_point_in_time_lab(test_seasons_tuple, scope, holdout, train_start):
    test_seasons = sorted(set(int(s) for s in test_seasons_tuple))
    end_year = max(test_seasons)
    start_year = min(int(train_start), min(test_seasons) - 1)
    history = _v3_history_frame(start_year, end_year, scope)
    results, preds = _v3_model_suite(history, test_seasons)
    summary = _v3_summary(results, holdout)
    readiness = _v3_data_readiness(history)
    return history, results, summary, preds, readiness


def _render_v3_lab(history, results, summary, preds, readiness, holdout):
    st.markdown("#### v3.0 Point-in-Time Rebuild")
    st.caption(
        "This is the first validation layer that uses current-season advanced stats only through the week BEFORE each historical game, "
        "plus pregame Elo and preseason-safe roster/recruiting/portal priors."
    )

    if summary is None or summary.empty:
        st.info("Run the v3.0 lab to build the richer point-in-time historical feature set.")
        return

    st.markdown("##### Data readiness")
    rshow = readiness.copy()
    rshow["Coverage"] = rshow["Coverage"].map(lambda v: f"{100*v:.1f}%")
    st.dataframe(rshow, use_container_width=True, hide_index=True)

    st.markdown("##### Model comparison")
    sshow = summary.copy()
    for c in [
        "Overall MAE", "Weighted Improvement",
        f"{holdout} Holdout MAE", f"{holdout} Holdout Improvement",
    ]:
        if c in sshow.columns:
            sshow[c] = sshow[c].map(
                lambda v: f"{v:+.4f}" if "Improvement" in c else (f"{v:.4f}" if pd.notna(v) else "—")
            )
    st.dataframe(sshow, use_container_width=True, hide_index=True)

    promo = summary[
        (summary["Market"] == "SPREAD") &
        (summary["Status"] == "PROMOTION REVIEW")
    ]
    if not promo.empty:
        st.success(
            "At least one v3 spread architecture passed the locked projection screen. "
            "Do not change live betting thresholds yet; the next step is ATS calibration and confirmation."
        )
    else:
        st.warning(
            "No v3 spread architecture passed the locked projection screen yet. "
            "The richer data may still improve specific calibrated betting signals, but there is no automatic promotion."
        )

    with st.expander("Season-by-season v3 results", expanded=False):
        x = results.copy()
        x["MAE"] = x["MAE"].map(lambda v: f"{v:.4f}")
        x["Improvement vs Market"] = x["Improvement vs Market"].map(lambda v: f"{v:+.4f}")
        st.dataframe(x, use_container_width=True, hide_index=True)

    with st.expander("v3.0 Downloads", expanded=True):
        bundle = _csv_download_bundle({
            "cfb_v300_summary.csv": summary,
            "cfb_v300_seasons.csv": results,
            "cfb_v300_predictions.csv": preds,
            "cfb_v300_data_readiness.csv": readiness,
            "cfb_v300_history_features.csv": history,
        })
        st.download_button(
            "Download All v3.0 Files",
            data=bundle,
            file_name="cfb_v300_point_in_time_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v300_all",
        )
        st.caption("One ZIP contains every file needed for review.")

# ===== v3.1 nonlinear ML bake-off + ATS classification =====
V31_VERSION = "v3.1.0-nonlinear-ml-bakeoff"
V31_MIN_TRAIN_ROWS = 500
V31_RANDOM_STATE = 41
V31_SPREAD_CORRECTION_CAP = 6.0
V31_TOTAL_CORRECTION_CAP = 5.0
V31_BET_PROB_THRESHOLDS = (0.54, 0.56, 0.58)
V31_STANDARD_JUICE = -110


def _v31_feature_groups():
    spread = {
        "Market context": ["market_margin", "week_num", "early_week", "hfa", "neutral", "conference_game"],
        "Pregame strength": [
            "elo_diff", "sp_rating_diff", "core_overall_diff", "fpi_fpi_diff",
        ],
        "Offense / defense priors": [
            "sp_off_rating_diff", "sp_def_rating_diff", "sp_st_rating_diff",
            "core_offense_diff", "core_defense_diff",
            "fpi_offense_diff", "fpi_defense_diff", "fpi_special_diff",
        ],
        "Personnel": [
            "talent_diff", "returning_ppa_diff", "returning_pass_diff",
            "returning_rush_diff", "returning_usage_diff",
            "recruit_points_diff", "recruit_rank_diff",
            "portal_net_count_diff", "portal_net_rating_diff", "portal_net_stars_diff",
        ],
        "Prior matchup": [
            "prior_ppa_match", "prior_pass_ppa_match", "prior_rush_ppa_match",
            "prior_success_match", "prior_expl_match", "prior_std_match",
            "prior_passdown_match", "prior_line_match", "prior_stuff_match",
            "prior_ppo_match", "prior_havoc_diff",
        ],
        "Current matchup": [
            "cur_ppa_match", "cur_pass_ppa_match", "cur_rush_ppa_match",
            "cur_success_match", "cur_pass_success_match", "cur_rush_success_match",
            "cur_expl_match", "cur_std_match", "cur_passdown_match",
            "cur_line_match", "cur_stuff_match", "cur_ppo_match",
            "cur_havoc_diff", "cur_field_pos_diff",
        ],
    }
    total = {
        "Market context": ["market_total", "week_num", "early_week", "neutral"],
        "Personnel": [
            "talent_diff", "returning_ppa_diff", "recruit_points_diff",
            "portal_net_rating_diff",
        ],
        "Prior efficiency / pace": [
            "prior_total_ppa", "prior_total_success", "prior_total_expl",
            "prior_total_ppo", "prior_total_pass", "prior_total_rush",
            "prior_total_pace", "prior_total_havoc",
        ],
        "Current efficiency / pace": [
            "cur_total_ppa", "cur_total_success", "cur_total_expl",
            "cur_total_ppo", "cur_total_pass", "cur_total_rush",
            "cur_total_pace", "cur_total_havoc",
        ],
    }
    return spread, total


def _v31_flatten(groups):
    out = []
    for vals in groups.values():
        for v in vals:
            if v not in out:
                out.append(v)
    return out


def _v31_prepare_xy(train_df, test_df, features, target, min_coverage=0.35):
    tr = train_df.copy()
    te = test_df.copy()
    usable, medians = [], {}
    for f in features:
        if f not in tr.columns or f not in te.columns:
            continue
        s = pd.to_numeric(tr[f], errors="coerce")
        if float(s.notna().mean()) < min_coverage:
            continue
        med = float(s.median()) if s.notna().any() else 0.0
        usable.append(f)
        medians[f] = med

    tr_y = pd.to_numeric(tr[target], errors="coerce")
    te_y = pd.to_numeric(te[target], errors="coerce")
    tr_mask = tr_y.notna()
    te_mask = te_y.notna()
    tr = tr.loc[tr_mask].copy()
    te = te.loc[te_mask].copy()
    tr_y = tr_y.loc[tr_mask].to_numpy(dtype=float)
    te_y = te_y.loc[te_mask].to_numpy(dtype=float)

    if len(tr) < V31_MIN_TRAIN_ROWS or len(te) == 0 or len(usable) < 5:
        return None

    Xtr = np.column_stack([
        pd.to_numeric(tr[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    Xte = np.column_stack([
        pd.to_numeric(te[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    return {
        "Xtr": Xtr,
        "Xte": Xte,
        "ytr": tr_y,
        "yte": te_y,
        "train_index": tr.index.to_numpy(),
        "test_index": te.index.to_numpy(),
        "features": usable,
    }


def _v31_regressors():
    models = {
        "Ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=20.0)),
        ]),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.025, max_depth=2,
            min_samples_leaf=20, loss="huber", random_state=V31_RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=25, l2_regularization=2.0,
            random_state=V31_RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=12,
            max_features=0.70, n_jobs=-1, random_state=V31_RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=9, min_samples_leaf=12,
            max_features=0.70, n_jobs=-1, random_state=V31_RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=350, max_depth=3, learning_rate=0.025,
            subsample=0.80, colsample_bytree=0.75,
            reg_alpha=1.0, reg_lambda=6.0,
            objective="reg:squarederror", n_jobs=2,
            random_state=V31_RANDOM_STATE,
        )
    return models


def _v31_classifiers():
    models = {
        "Logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.35, max_iter=2000, solver="lbfgs",
                random_state=V31_RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.025, max_depth=2,
            min_samples_leaf=25, random_state=V31_RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=170, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=3.0,
            random_state=V31_RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            max_features=0.70, class_weight="balanced",
            n_jobs=-1, random_state=V31_RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=9, min_samples_leaf=15,
            max_features=0.70, class_weight="balanced",
            n_jobs=-1, random_state=V31_RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.025,
            subsample=0.80, colsample_bytree=0.75,
            reg_alpha=1.0, reg_lambda=6.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=2, random_state=V31_RANDOM_STATE,
        )
    return models


def _v31_platt_fit(raw_prob, y):
    p = np.clip(np.asarray(raw_prob, dtype=float), 1e-5, 1 - 1e-5)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    yy = np.asarray(y, dtype=int)
    if len(np.unique(yy)) < 2 or len(yy) < 100:
        return None
    cal = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000)
    cal.fit(x, yy)
    return cal


def _v31_platt_apply(cal, raw_prob):
    p = np.clip(np.asarray(raw_prob, dtype=float), 1e-5, 1 - 1e-5)
    if cal is None:
        return p
    x = np.log(p / (1 - p)).reshape(-1, 1)
    return cal.predict_proba(x)[:, 1]


def _v31_inner_calibration(train_df, features, target, model_factory_name):
    """
    Point-in-time Platt calibration:
    use the latest season inside the training set as calibration only,
    train the base classifier on earlier training seasons, then fit a
    one-variable logistic calibrator on that season's probabilities.
    """
    seasons = sorted(set(pd.to_numeric(train_df["season"], errors="coerce").dropna().astype(int)))
    if len(seasons) < 3:
        return None
    cal_season = seasons[-1]
    base = train_df[train_df["season"].astype(int) < cal_season].copy()
    cal_df = train_df[train_df["season"].astype(int) == cal_season].copy()
    prep = _v31_prepare_xy(base, cal_df, features, target)
    if prep is None:
        return None
    model = _v31_classifiers().get(model_factory_name)
    if model is None:
        return None
    try:
        model.fit(prep["Xtr"], prep["ytr"].astype(int))
        raw = model.predict_proba(prep["Xte"])[:, 1]
        return _v31_platt_fit(raw, prep["yte"].astype(int))
    except Exception:
        return None


def _v31_roi_from_wins(wins, losses, pushes=0, odds=-110):
    if wins + losses <= 0:
        return np.nan
    if odds < 0:
        win_profit = 100.0 / abs(float(odds))
    else:
        win_profit = float(odds) / 100.0
    units = wins * win_profit - losses
    return units / float(wins + losses)


def _v31_classification_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "AUC": np.nan,
    }
    if len(np.unique(y)) == 2:
        try:
            out["AUC"] = float(roc_auc_score(y, p))
        except Exception:
            pass
    return out


def _v31_bet_rows(test_rows, market_type, model_name, season, probs):
    """
    Convert calibrated home-cover / over probabilities into symmetric betting
    opportunities. Thresholds are predeclared; no threshold is selected on
    the holdout.
    """
    rows = []
    probs = np.asarray(probs, dtype=float)
    if market_type == "spread":
        y_home = np.asarray(test_rows["home_cover_target"], dtype=float)
    else:
        y_home = np.asarray(test_rows["over_target"], dtype=float)

    for threshold in V31_BET_PROB_THRESHOLDS:
        for i, p in enumerate(probs):
            if not np.isfinite(y_home[i]):
                continue
            pick = None
            won = None
            pick_prob = None
            if p >= threshold:
                pick = "HOME" if market_type == "spread" else "OVER"
                won = int(y_home[i] == 1)
                pick_prob = p
            elif p <= 1.0 - threshold:
                pick = "AWAY" if market_type == "spread" else "UNDER"
                won = int(y_home[i] == 0)
                pick_prob = 1.0 - p
            if pick is None:
                continue
            rows.append({
                "market_type": market_type,
                "model": model_name,
                "season": int(season),
                "threshold": float(threshold),
                "pick": pick,
                "pick_probability": float(pick_prob),
                "won": int(won),
                "game_id": test_rows.iloc[i].get("game_id"),
                "home_team": test_rows.iloc[i].get("home_team"),
                "away_team": test_rows.iloc[i].get("away_team"),
            })
    return rows


def _v31_run_one_market(history, test_seasons, market_type):
    spread_groups, total_groups = _v31_feature_groups()
    if market_type == "spread":
        groups = spread_groups
        features = _v31_flatten(groups)
        market_col = "market_margin"
        actual_col = "actual_margin"
        residual_target = "spread_residual_target"
        class_target = "home_cover_target"
        cap = V31_SPREAD_CORRECTION_CAP
    else:
        groups = total_groups
        features = _v31_flatten(groups)
        market_col = "market_total"
        actual_col = "actual_total"
        residual_target = "total_residual_target"
        class_target = "over_target"
        cap = V31_TOTAL_CORRECTION_CAP

    h = history.copy()
    h[residual_target] = (
        pd.to_numeric(h[actual_col], errors="coerce")
        - pd.to_numeric(h[market_col], errors="coerce")
    )
    if market_type == "spread":
        diff = h[residual_target]
    else:
        diff = h[residual_target]
    h[class_target] = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, np.nan))

    reg_results, class_results, pred_rows, bet_rows = [], [], [], []

    for season in sorted(set(int(s) for s in test_seasons)):
        train = h[h["season"].astype(int) < season].copy()
        test = h[h["season"].astype(int) == season].copy()
        if train.empty or test.empty:
            continue

        # Market control MAE on rows with a market and outcome.
        ctrl = test.dropna(subset=[actual_col, market_col]).copy()
        if not ctrl.empty:
            y = pd.to_numeric(ctrl[actual_col], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(ctrl[market_col], errors="coerce").to_numpy(dtype=float)
            reg_results.append({
                "Market": market_type.upper(),
                "Task": "Regression",
                "Model": "Market Control",
                "Season": season,
                "Games": len(ctrl),
                "MAE": float(mean_absolute_error(y, m)),
                "RMSE": float(np.sqrt(mean_squared_error(y, m))),
                "Improvement vs Market": 0.0,
            })

        # Residual regression.
        reg_features = features + [market_col]
        prep = _v31_prepare_xy(train, test, reg_features, residual_target)
        if prep is not None:
            test_rows = test.loc[prep["test_index"]].copy()
            actual = pd.to_numeric(test_rows[actual_col], errors="coerce").to_numpy(dtype=float)
            market = pd.to_numeric(test_rows[market_col], errors="coerce").to_numpy(dtype=float)
            market_mae = float(mean_absolute_error(actual, market))
            for name, estimator in _v31_regressors().items():
                try:
                    estimator.fit(prep["Xtr"], prep["ytr"])
                    corr = np.asarray(estimator.predict(prep["Xte"]), dtype=float)
                    corr = np.clip(corr, -cap, cap)
                    pred = market + corr
                    mae = float(mean_absolute_error(actual, pred))
                    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
                except Exception:
                    continue
                reg_results.append({
                    "Market": market_type.upper(),
                    "Task": "Regression",
                    "Model": name,
                    "Season": season,
                    "Games": len(test_rows),
                    "MAE": mae,
                    "RMSE": rmse,
                    "Improvement vs Market": market_mae - mae,
                })
                for idx, yy, mm, cc, pp in zip(
                    test_rows.index, actual, market, corr, pred
                ):
                    pred_rows.append({
                        "market_type": market_type,
                        "task": "regression",
                        "model": name,
                        "season": season,
                        "row_index": int(idx),
                        "actual": float(yy),
                        "market": float(mm),
                        "correction": float(cc),
                        "prediction": float(pp),
                    })

        # ATS / O-U classification. Pushes excluded.
        class_train = train.dropna(subset=[class_target]).copy()
        class_test = test.dropna(subset=[class_target]).copy()
        prep_c = _v31_prepare_xy(class_train, class_test, features + [market_col], class_target)
        if prep_c is not None:
            test_rows = class_test.loc[prep_c["test_index"]].copy()
            y = prep_c["yte"].astype(int)

            # Naive 50% control.
            ctrl_metrics = _v31_classification_metrics(y, np.full(len(y), 0.5))
            class_results.append({
                "Market": market_type.upper(),
                "Task": "Classification",
                "Model": "50% Control",
                "Season": season,
                "Games": len(y),
                **ctrl_metrics,
            })

            for name, estimator in _v31_classifiers().items():
                try:
                    calibrator = _v31_inner_calibration(
                        class_train, features + [market_col], class_target, name
                    )
                    estimator.fit(prep_c["Xtr"], prep_c["ytr"].astype(int))
                    raw = estimator.predict_proba(prep_c["Xte"])[:, 1]
                    p = _v31_platt_apply(calibrator, raw)
                    metrics = _v31_classification_metrics(y, p)
                except Exception:
                    continue

                class_results.append({
                    "Market": market_type.upper(),
                    "Task": "Classification",
                    "Model": name,
                    "Season": season,
                    "Games": len(y),
                    **metrics,
                })
                for idx, yy, pp in zip(test_rows.index, y, p):
                    pred_rows.append({
                        "market_type": market_type,
                        "task": "classification",
                        "model": name,
                        "season": season,
                        "row_index": int(idx),
                        "actual_class": int(yy),
                        "probability": float(pp),
                    })
                bet_rows.extend(_v31_bet_rows(
                    test_rows, market_type, name, season, p
                ))

    return (
        pd.DataFrame(reg_results),
        pd.DataFrame(class_results),
        pd.DataFrame(pred_rows),
        pd.DataFrame(bet_rows),
    )


def _v31_regression_summary(reg):
    if reg is None or reg.empty:
        return pd.DataFrame()
    rows = []
    for (market, model), g in reg.groupby(["Market", "Model"]):
        games = pd.to_numeric(g["Games"], errors="coerce").to_numpy(dtype=float)
        mae = pd.to_numeric(g["MAE"], errors="coerce").to_numpy(dtype=float)
        impr = pd.to_numeric(g["Improvement vs Market"], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "Market": market,
            "Model": model,
            "Games": int(np.sum(games)),
            "Weighted MAE": float(np.average(mae, weights=games)),
            "Weighted Improvement vs Market": float(np.average(impr, weights=games)),
            "Seasons Beat Market": f"{int(np.sum(impr > 0))}/{len(g)}",
        })
    return pd.DataFrame(rows)


def _v31_classification_summary(cls):
    if cls is None or cls.empty:
        return pd.DataFrame()
    rows = []
    for (market, model), g in cls.groupby(["Market", "Model"]):
        w = pd.to_numeric(g["Games"], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "Market": market,
            "Model": model,
            "Games": int(np.sum(w)),
            "Weighted Brier": float(np.average(pd.to_numeric(g["Brier"], errors="coerce"), weights=w)),
            "Weighted LogLoss": float(np.average(pd.to_numeric(g["LogLoss"], errors="coerce"), weights=w)),
            "Weighted AUC": float(np.average(
                pd.to_numeric(g["AUC"], errors="coerce").fillna(0.5), weights=w
            )),
        })
    return pd.DataFrame(rows)


def _v31_betting_summary(bets):
    if bets is None or bets.empty:
        return pd.DataFrame()
    rows = []
    for (market, model, threshold), g in bets.groupby(
        ["market_type", "model", "threshold"]
    ):
        wins = int(pd.to_numeric(g["won"], errors="coerce").fillna(0).sum())
        n = int(len(g))
        losses = n - wins
        roi = _v31_roi_from_wins(wins, losses, 0, V31_STANDARD_JUICE)
        season_rois = []
        positive = 0
        for season, sg in g.groupby("season"):
            sw = int(pd.to_numeric(sg["won"], errors="coerce").fillna(0).sum())
            sl = int(len(sg) - sw)
            sr = _v31_roi_from_wins(sw, sl, 0, V31_STANDARD_JUICE)
            if np.isfinite(sr):
                season_rois.append(sr)
                if sr > 0:
                    positive += 1
        rows.append({
            "Market": market.upper(),
            "Model": model,
            "Threshold": float(threshold),
            "Bets": n,
            "Wins": wins,
            "Losses": losses,
            "Win Rate": wins / n if n else np.nan,
            "ROI": roi,
            "Avg Pick Probability": float(pd.to_numeric(g["pick_probability"], errors="coerce").mean()),
            "Positive ROI Seasons": f"{positive}/{len(season_rois)}",
        })
    return pd.DataFrame(rows)


def _v31_holdout_gate(reg, cls, bets_raw, holdout):
    """
    v3.2 hotfix: every "holdout" betting metric is now calculated from the
    actual holdout-season bet rows, never from the all-season aggregate table.
    """
    rows = []
    for market in ["SPREAD", "TOTAL"]:
        rr = reg[
            (reg["Market"] == market) &
            (reg["Season"].astype(int) == int(holdout))
        ].copy()

        cc = cls[
            (cls["Market"] == market) &
            (cls["Season"].astype(int) == int(holdout))
        ].copy()

        raw_market = market.lower()
        hb = (
            bets_raw[
                (bets_raw["market_type"] == raw_market) &
                (bets_raw["season"].astype(int) == int(holdout))
            ].copy()
            if bets_raw is not None and not bets_raw.empty
            else pd.DataFrame()
        )

        best_reg = None
        if not rr.empty:
            r2 = rr[rr["Model"] != "Market Control"].sort_values("MAE")
            if not r2.empty:
                best_reg = r2.iloc[0]

        best_cls = None
        if not cc.empty:
            c2 = cc[cc["Model"] != "50% Control"].sort_values("Brier")
            if not c2.empty:
                best_cls = c2.iloc[0]

        best_bet = None
        if not hb.empty:
            hold_rows = []
            for (model_name, threshold), g in hb.groupby(["model", "threshold"]):
                wins = int(pd.to_numeric(g["won"], errors="coerce").fillna(0).sum())
                n = int(len(g))
                losses = n - wins
                hold_rows.append({
                    "Model": model_name,
                    "Threshold": float(threshold),
                    "Bets": n,
                    "Wins": wins,
                    "Losses": losses,
                    "Win Rate": wins / n if n else np.nan,
                    "ROI": _v31_roi_from_wins(wins, losses, 0, V31_STANDARD_JUICE),
                })
            hold_df = pd.DataFrame(hold_rows)
            eligible = hold_df[hold_df["Bets"] >= 25].copy()
            if not eligible.empty:
                best_bet = eligible.sort_values(
                    ["ROI", "Bets"], ascending=[False, False]
                ).iloc[0]

        reg_pass = bool(
            best_reg is not None and
            float(best_reg["Improvement vs Market"]) > 0
        )
        cls_pass = bool(
            best_cls is not None and
            float(best_cls["Brier"]) < 0.25
        )
        bet_pass = bool(
            market == "SPREAD" and
            best_bet is not None and
            int(best_bet["Bets"]) >= 25 and
            float(best_bet["ROI"]) > 0
        )

        if market == "SPREAD" and reg_pass and cls_pass and bet_pass:
            verdict = "CONFIRMATION CANDIDATE"
        elif reg_pass or cls_pass or bet_pass:
            verdict = "RESEARCH POSITIVE"
        else:
            verdict = "KEEP IN RESEARCH"

        rows.append({
            "Market": market,
            "Best Holdout Regression": None if best_reg is None else best_reg["Model"],
            "Holdout Regression Improvement": np.nan if best_reg is None else float(best_reg["Improvement vs Market"]),
            "Best Holdout Classifier": None if best_cls is None else best_cls["Model"],
            "Holdout Brier": np.nan if best_cls is None else float(best_cls["Brier"]),
            "Best Holdout Betting Model": None if best_bet is None else best_bet["Model"],
            "Best Holdout Betting Threshold": np.nan if best_bet is None else float(best_bet["Threshold"]),
            "Holdout Betting ROI": np.nan if best_bet is None else float(best_bet["ROI"]),
            "Holdout Betting N": 0 if best_bet is None else int(best_bet["Bets"]),
            "Verdict": verdict,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v31_ml_bakeoff(test_seasons_tuple, scope, holdout, train_start):
    test_seasons = sorted(set(int(s) for s in test_seasons_tuple))
    history = _v3_history_frame(
        min(int(train_start), min(test_seasons) - 1),
        max(test_seasons),
        scope,
    )

    sr, sc, sp, sb = _v31_run_one_market(history, test_seasons, "spread")
    tr, tc, tp, tb = _v31_run_one_market(history, test_seasons, "total")

    reg = pd.concat([sr, tr], ignore_index=True) if not sr.empty or not tr.empty else pd.DataFrame()
    cls = pd.concat([sc, tc], ignore_index=True) if not sc.empty or not tc.empty else pd.DataFrame()
    preds = pd.concat([sp, tp], ignore_index=True) if not sp.empty or not tp.empty else pd.DataFrame()
    bets_raw = pd.concat([sb, tb], ignore_index=True) if not sb.empty or not tb.empty else pd.DataFrame()

    reg_summary = _v31_regression_summary(reg)
    cls_summary = _v31_classification_summary(cls)
    bet_summary = _v31_betting_summary(bets_raw)
    gate = _v31_holdout_gate(reg, cls, bets_raw, holdout)
    return history, reg, reg_summary, cls, cls_summary, bets_raw, bet_summary, preds, gate


def _render_v31_ml_bakeoff(
    history, reg, reg_summary, cls, cls_summary,
    bets_raw, bet_summary, preds, gate, holdout
):
    st.markdown("#### v3.1 Nonlinear ML Bake-Off + ATS Classification")
    st.caption(
        "Regression predicts market residual error. Classification predicts HOME cover / OVER directly. "
        "All test seasons are walk-forward; classifier probabilities are Platt-calibrated using only an earlier training season."
    )

    if reg_summary is None or reg_summary.empty:
        st.info("Run the v3.1 ML bake-off to compare linear and nonlinear models.")
        return

    st.markdown("##### Residual regression")
    r = reg_summary.copy()
    r["Weighted MAE"] = r["Weighted MAE"].map(lambda v: f"{v:.4f}")
    r["Weighted Improvement vs Market"] = r["Weighted Improvement vs Market"].map(lambda v: f"{v:+.4f}")
    st.dataframe(r, use_container_width=True, hide_index=True)

    st.markdown("##### ATS / O-U classification")
    c = cls_summary.copy()
    c["Weighted Brier"] = c["Weighted Brier"].map(lambda v: f"{v:.4f}")
    c["Weighted LogLoss"] = c["Weighted LogLoss"].map(lambda v: f"{v:.4f}")
    c["Weighted AUC"] = c["Weighted AUC"].map(lambda v: f"{v:.4f}")
    st.dataframe(c, use_container_width=True, hide_index=True)

    st.markdown("##### Predeclared betting thresholds")
    if bet_summary is None or bet_summary.empty:
        st.info("No classified opportunities met the fixed 54%, 56%, or 58% probability thresholds.")
    else:
        b = bet_summary.copy()
        b["Threshold"] = b["Threshold"].map(lambda v: f"{100*v:.0f}%")
        b["Win Rate"] = b["Win Rate"].map(lambda v: f"{100*v:.1f}%")
        b["ROI"] = b["ROI"].map(lambda v: f"{100*v:+.1f}%")
        b["Avg Pick Probability"] = b["Avg Pick Probability"].map(lambda v: f"{100*v:.1f}%")
        st.dataframe(b, use_container_width=True, hide_index=True)

    st.markdown("##### Locked holdout gate")
    g = gate.copy()
    if not g.empty:
        for col in ["Holdout Regression Improvement", "Holdout Brier", "Holdout Betting ROI"]:
            if col in g.columns:
                if col == "Holdout Betting ROI":
                    g[col] = g[col].map(lambda v: "—" if pd.isna(v) else f"{100*v:+.1f}%")
                else:
                    g[col] = g[col].map(lambda v: "—" if pd.isna(v) else f"{v:+.4f}")
        st.dataframe(g, use_container_width=True, hide_index=True)

    spread_gate = gate[gate["Market"] == "SPREAD"] if gate is not None and not gate.empty else pd.DataFrame()
    if not spread_gate.empty and spread_gate.iloc[0]["Verdict"] == "CONFIRMATION CANDIDATE":
        st.success(
            "A spread architecture cleared the v3.1 research gate. This still does not change live betting automatically; "
            "the next step is a locked confirmation/forward test."
        )
    else:
        st.warning(
            "No automatic live promotion. We keep the market as the benchmark until a model clears regression, calibration, and betting evidence together."
        )

    with st.expander("v3.1 season detail", expanded=False):
        st.markdown("**Regression**")
        st.dataframe(reg, use_container_width=True, hide_index=True)
        st.markdown("**Classification**")
        st.dataframe(cls, use_container_width=True, hide_index=True)

    with st.expander("v3.1 Downloads", expanded=True):
        bundle = _csv_download_bundle({
            "cfb_v310_regression_summary.csv": reg_summary,
            "cfb_v310_regression_seasons.csv": reg,
            "cfb_v310_classification_summary.csv": cls_summary,
            "cfb_v310_classification_seasons.csv": cls,
            "cfb_v310_betting_summary.csv": bet_summary,
            "cfb_v310_bets.csv": bets_raw,
            "cfb_v310_predictions.csv": preds,
            "cfb_v310_gate.csv": gate,
        })
        st.download_button(
            "Download All v3.1 Files",
            data=bundle,
            file_name="cfb_v310_ml_bakeoff_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v310_all",
        )
        st.caption("Upload this single ZIP back to ChatGPT for review.")

# ===== v3.2 signal stability + ensemble discovery =====
V32_VERSION = "v3.2.0-signal-stability-ensemble"
V32_CORE_MODELS = ("Gradient Boosting", "Extra Trees", "Logistic")
V32_MIN_SEGMENT_BETS = 20
V32_DISCOVERY_SEASONS = (2022, 2023, 2024)
V32_PROB_BUCKETS = [
    (0.50, 0.52, "50–51.9%"),
    (0.52, 0.54, "52–53.9%"),
    (0.54, 0.56, "54–55.9%"),
    (0.56, 0.58, "56–57.9%"),
    (0.58, 0.60, "58–59.9%"),
    (0.60, 1.01, "60%+"),
]


def _v32_join_spread_predictions(history, preds):
    if history is None or history.empty or preds is None or preds.empty:
        return pd.DataFrame()

    p = preds[
        (preds["market_type"] == "spread") &
        (preds["task"] == "classification")
    ].copy()
    if p.empty:
        return pd.DataFrame()

    h = history.copy()
    h = h.reset_index().rename(columns={"index": "row_index"})
    keep = [
        "row_index", "season", "week", "game_id", "home_team", "away_team",
        "market_margin", "actual_margin", "conference_game", "elo_diff",
        "talent_diff", "returning_ppa_diff", "cur_ppa_match",
        "cur_success_match", "cur_line_match",
    ]
    keep = [c for c in keep if c in h.columns]
    h = h[keep].copy()

    d = p.merge(h, on=["row_index", "season"], how="left")
    residual = (
        pd.to_numeric(d["actual_margin"], errors="coerce")
        - pd.to_numeric(d["market_margin"], errors="coerce")
    )
    d["home_cover"] = np.where(residual > 0, 1.0, np.where(residual < 0, 0.0, np.nan))
    d["pick_side"] = np.where(
        pd.to_numeric(d["probability"], errors="coerce") >= 0.5, "HOME", "AWAY"
    )
    d["pick_probability"] = np.maximum(
        pd.to_numeric(d["probability"], errors="coerce"),
        1.0 - pd.to_numeric(d["probability"], errors="coerce"),
    )
    d["won"] = np.where(
        d["home_cover"].isna(),
        np.nan,
        np.where(
            d["pick_side"] == "HOME",
            d["home_cover"],
            1.0 - d["home_cover"],
        ),
    )
    return d


def _v32_add_segments(d):
    if d is None or d.empty:
        return pd.DataFrame()
    x = d.copy()
    abs_spread = pd.to_numeric(x["market_margin"], errors="coerce").abs()

    x["Week Segment"] = np.where(
        pd.to_numeric(x["week"], errors="coerce") <= 3,
        "Weeks 0–3", "Weeks 4+"
    )
    x["Favorite / Dog"] = np.where(
        pd.to_numeric(x["market_margin"], errors="coerce") > 0,
        "Home Favorite",
        np.where(pd.to_numeric(x["market_margin"], errors="coerce") < 0, "Home Dog", "Pick'em"),
    )
    x["Spread Range"] = pd.cut(
        abs_spread,
        bins=[-0.01, 3, 7, 14, np.inf],
        labels=["PK–3", "3–7", "7–14", "14+"],
        include_lowest=True,
        right=True,
    ).astype(str)
    x["Conference"] = np.where(
        pd.to_numeric(x.get("conference_game"), errors="coerce").fillna(0) > 0,
        "Conference", "Non-Conference"
    )

    elo = pd.to_numeric(x.get("elo_diff"), errors="coerce").abs()
    x["Elo Gap"] = pd.cut(
        elo,
        bins=[-0.01, 50, 100, 200, np.inf],
        labels=["<50", "50–99", "100–199", "200+"],
        include_lowest=True,
    ).astype(str)

    talent = pd.to_numeric(x.get("talent_diff"), errors="coerce").abs()
    q1 = float(talent.quantile(0.50)) if talent.notna().any() else 0.0
    q2 = float(talent.quantile(0.80)) if talent.notna().any() else 0.0
    x["Talent Gap"] = pd.cut(
        talent,
        bins=[-0.01, q1, q2, np.inf],
        labels=["Lower", "Medium", "High"],
        include_lowest=True,
        duplicates="drop",
    ).astype(str)

    ret = pd.to_numeric(x.get("returning_ppa_diff"), errors="coerce").abs()
    rq = float(ret.quantile(0.67)) if ret.notna().any() else 0.0
    x["Returning Uncertainty"] = np.where(
        ret >= rq, "High Differential", "Normal"
    )

    cur = pd.to_numeric(x.get("cur_ppa_match"), errors="coerce")
    x["Data Maturity"] = np.where(
        pd.to_numeric(x["week"], errors="coerce") <= 3,
        "Early / Thin",
        np.where(cur.notna(), "In-Season Mature", "Missing Current Advanced"),
    )
    return x


def _v32_prob_bucket(p):
    try:
        v = float(p)
    except Exception:
        return "Unknown"
    for lo, hi, label in V32_PROB_BUCKETS:
        if lo <= v < hi:
            return label
    return "Unknown"


def _v32_roi(wins, losses):
    return _v31_roi_from_wins(int(wins), int(losses), 0, V31_STANDARD_JUICE)


def _v32_segment_summary(joined, holdout):
    if joined is None or joined.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = _v32_add_segments(joined)
    x = x[x["model"].isin(V32_CORE_MODELS)].copy()

    segment_cols = [
        "Week Segment", "Favorite / Dog", "Spread Range", "Conference",
        "Elo Gap", "Talent Gap", "Returning Uncertainty", "Data Maturity",
    ]

    detail_rows = []
    summary_rows = []

    for model_name in V32_CORE_MODELS:
        md = x[x["model"] == model_name].copy()
        for seg_col in segment_cols:
            if seg_col not in md.columns:
                continue
            for seg_value, sg in md.groupby(seg_col, dropna=False):
                for threshold in V31_BET_PROB_THRESHOLDS:
                    bets = sg[
                        pd.to_numeric(sg["pick_probability"], errors="coerce") >= float(threshold)
                    ].dropna(subset=["won"]).copy()
                    if bets.empty:
                        continue

                    season_stats = []
                    for season, ss in bets.groupby("season"):
                        w = int(pd.to_numeric(ss["won"], errors="coerce").sum())
                        n = int(len(ss))
                        l = n - w
                        roi = _v32_roi(w, l)
                        row = {
                            "Model": model_name,
                            "Segment Type": seg_col,
                            "Segment": str(seg_value),
                            "Threshold": float(threshold),
                            "Season": int(season),
                            "Bets": n,
                            "Wins": w,
                            "Losses": l,
                            "Win Rate": w / n if n else np.nan,
                            "ROI": roi,
                        }
                        detail_rows.append(row)
                        season_stats.append(row)

                    all_w = int(pd.to_numeric(bets["won"], errors="coerce").sum())
                    all_n = int(len(bets))
                    all_l = all_n - all_w

                    dev = [r for r in season_stats if int(r["Season"]) in V32_DISCOVERY_SEASONS]
                    hld = [r for r in season_stats if int(r["Season"]) == int(holdout)]
                    dev_n = sum(r["Bets"] for r in dev)
                    dev_w = sum(r["Wins"] for r in dev)
                    dev_l = sum(r["Losses"] for r in dev)
                    hold_n = sum(r["Bets"] for r in hld)
                    hold_w = sum(r["Wins"] for r in hld)
                    hold_l = sum(r["Losses"] for r in hld)
                    pos_dev = sum(1 for r in dev if np.isfinite(r["ROI"]) and r["ROI"] > 0)

                    dev_roi = _v32_roi(dev_w, dev_l) if dev_n else np.nan
                    hold_roi = _v32_roi(hold_w, hold_l) if hold_n else np.nan

                    if (
                        dev_n >= 100 and
                        len(dev) >= 2 and
                        pos_dev >= 2 and
                        np.isfinite(dev_roi) and dev_roi > 0 and
                        hold_n >= 25 and
                        np.isfinite(hold_roi) and hold_roi > 0
                    ):
                        label = "HOLDOUT CONFIRMED"
                    elif (
                        dev_n >= 100 and
                        len(dev) >= 2 and
                        pos_dev >= 2 and
                        np.isfinite(dev_roi) and dev_roi > 0
                    ):
                        label = "DISCOVERY SIGNAL"
                    else:
                        label = "NO STABLE SIGNAL"

                    summary_rows.append({
                        "Model": model_name,
                        "Segment Type": seg_col,
                        "Segment": str(seg_value),
                        "Threshold": float(threshold),
                        "All Bets": all_n,
                        "All Win Rate": all_w / all_n if all_n else np.nan,
                        "All ROI": _v32_roi(all_w, all_l),
                        "Development Bets": dev_n,
                        "Development ROI": dev_roi,
                        "Positive Development Seasons": f"{pos_dev}/{len(dev)}",
                        "Holdout Bets": hold_n,
                        "Holdout ROI": hold_roi,
                        "Status": label,
                    })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def _v32_probability_monotonicity(joined):
    if joined is None or joined.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = joined[joined["model"].isin(V32_CORE_MODELS)].copy()
    x = x.dropna(subset=["won", "pick_probability"]).copy()
    x["Probability Bucket"] = x["pick_probability"].map(_v32_prob_bucket)

    rows = []
    diag = []
    order = [b[2] for b in V32_PROB_BUCKETS]

    for model_name, md in x.groupby("model"):
        bucket_rates = []
        for bucket in order:
            g = md[md["Probability Bucket"] == bucket]
            if g.empty:
                continue
            w = int(pd.to_numeric(g["won"], errors="coerce").sum())
            n = int(len(g))
            l = n - w
            realized = w / n if n else np.nan
            avg_pred = float(pd.to_numeric(g["pick_probability"], errors="coerce").mean())
            rows.append({
                "Model": model_name,
                "Probability Bucket": bucket,
                "Games": n,
                "Avg Predicted": avg_pred,
                "Realized Win Rate": realized,
                "Calibration Error": realized - avg_pred,
                "ROI": _v32_roi(w, l),
            })
            if n >= 50:
                bucket_rates.append((bucket, realized))

        realized_only = [v for _, v in bucket_rates]
        monotonic = all(
            realized_only[i] <= realized_only[i+1] + 0.005
            for i in range(len(realized_only)-1)
        ) if len(realized_only) >= 3 else False
        diag.append({
            "Model": model_name,
            "Qualified Buckets": len(realized_only),
            "Monotonic": bool(monotonic),
            "Status": "PASS" if monotonic else "FAIL / INSUFFICIENT",
        })

    return pd.DataFrame(rows), pd.DataFrame(diag)


def _v32_ensemble_rows(joined, holdout):
    if joined is None or joined.empty:
        return pd.DataFrame(), pd.DataFrame()

    x = joined[joined["model"].isin(V32_CORE_MODELS)].copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = x.pivot_table(
        index=["season", "row_index", "game_id", "home_team", "away_team"],
        columns="model",
        values="probability",
        aggfunc="first",
    ).reset_index()

    needed = [m for m in V32_CORE_MODELS if m in pivot.columns]
    if len(needed) < 2:
        return pd.DataFrame(), pd.DataFrame()

    meta = x.sort_values("model").drop_duplicates(
        ["season", "row_index"]
    )[
        ["season", "row_index", "home_cover", "market_margin", "week"]
    ]
    pivot = pivot.merge(meta, on=["season", "row_index"], how="left")

    rows = []
    for _, r in pivot.iterrows():
        probs = {m: _v3_num(r.get(m)) for m in needed}
        valid = {m: p for m, p in probs.items() if np.isfinite(p)}
        if len(valid) < 2:
            continue

        votes = {m: ("HOME" if p >= 0.5 else "AWAY") for m, p in valid.items()}
        home_votes = sum(1 for s in votes.values() if s == "HOME")
        away_votes = len(votes) - home_votes
        if home_votes == away_votes:
            continue

        side = "HOME" if home_votes > away_votes else "AWAY"
        agreement = max(home_votes, away_votes)
        agreeing_probs = [
            (p if side == "HOME" else 1.0 - p)
            for m, p in valid.items()
            if votes[m] == side
        ]
        ens_prob = float(np.mean(agreeing_probs))
        y = _v3_num(r.get("home_cover"))
        if not np.isfinite(y):
            continue
        won = int(y == 1) if side == "HOME" else int(y == 0)

        rows.append({
            "season": int(r["season"]),
            "row_index": int(r["row_index"]),
            "game_id": r.get("game_id"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "market_margin": _v3_num(r.get("market_margin")),
            "week": _v3_num(r.get("week")),
            "models_available": len(valid),
            "agreement_count": agreement,
            "side": side,
            "ensemble_probability": ens_prob,
            "won": won,
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summary = []
    for rule_name, min_agree in [("2+ Model Agreement", 2), ("Unanimous", len(needed))]:
        dd = detail[detail["agreement_count"] >= min_agree].copy()
        for threshold in V31_BET_PROB_THRESHOLDS:
            bb = dd[dd["ensemble_probability"] >= threshold].copy()
            if bb.empty:
                continue

            for season, sg in bb.groupby("season"):
                w = int(pd.to_numeric(sg["won"], errors="coerce").sum())
                n = int(len(sg))
                l = n - w
                summary.append({
                    "Rule": rule_name,
                    "Threshold": float(threshold),
                    "Season": int(season),
                    "Bets": n,
                    "Wins": w,
                    "Losses": l,
                    "Win Rate": w / n if n else np.nan,
                    "ROI": _v32_roi(w, l),
                })

            # Aggregate and locked holdout rows.
            w = int(pd.to_numeric(bb["won"], errors="coerce").sum())
            n = int(len(bb))
            l = n - w
            hb = bb[bb["season"].astype(int) == int(holdout)]
            hw = int(pd.to_numeric(hb["won"], errors="coerce").sum()) if not hb.empty else 0
            hn = int(len(hb))
            hl = hn - hw
            summary.append({
                "Rule": rule_name,
                "Threshold": float(threshold),
                "Season": "ALL",
                "Bets": n,
                "Wins": w,
                "Losses": l,
                "Win Rate": w / n if n else np.nan,
                "ROI": _v32_roi(w, l),
                "Holdout Bets": hn,
                "Holdout Win Rate": hw / hn if hn else np.nan,
                "Holdout ROI": _v32_roi(hw, hl) if hn else np.nan,
            })

    return detail, pd.DataFrame(summary)


def _v32_feature_signal_report(history, joined):
    """
    Lightweight football-signal diagnostic. This does not choose a betting rule.
    It measures whether model confidence behaves differently as key predeclared
    football inputs become more extreme.
    """
    if joined is None or joined.empty:
        return pd.DataFrame()

    x = _v32_add_segments(joined)
    rows = []
    dims = ["Elo Gap", "Talent Gap", "Returning Uncertainty", "Data Maturity", "Spread Range"]
    for model_name in V32_CORE_MODELS:
        md = x[x["model"] == model_name].dropna(subset=["won", "pick_probability"])
        for dim in dims:
            if dim not in md.columns:
                continue
            for val, g in md.groupby(dim, dropna=False):
                if len(g) < 30:
                    continue
                w = int(pd.to_numeric(g["won"], errors="coerce").sum())
                n = int(len(g))
                rows.append({
                    "Model": model_name,
                    "Dimension": dim,
                    "Bucket": str(val),
                    "Games": n,
                    "Avg Confidence": float(pd.to_numeric(g["pick_probability"], errors="coerce").mean()),
                    "Realized Win Rate": w / n,
                    "ROI": _v32_roi(w, n - w),
                })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v32_signal_stability(test_seasons_tuple, scope, holdout, train_start):
    # Reuse exact v3.1 walk-forward predictions; no model retuning here.
    (
        history, reg, reg_summary, cls, cls_summary,
        bets_raw, bet_summary, preds, fixed_gate,
    ) = _run_v31_ml_bakeoff(
        tuple(sorted(set(int(s) for s in test_seasons_tuple))),
        scope,
        int(holdout),
        int(train_start),
    )

    joined = _v32_join_spread_predictions(history, preds)
    segment_summary, segment_seasons = _v32_segment_summary(joined, holdout)
    prob_buckets, monotonicity = _v32_probability_monotonicity(joined)
    ensemble_detail, ensemble_summary = _v32_ensemble_rows(joined, holdout)
    football_signals = _v32_feature_signal_report(history, joined)

    return (
        history, fixed_gate, joined,
        segment_summary, segment_seasons,
        prob_buckets, monotonicity,
        ensemble_detail, ensemble_summary,
        football_signals,
    )


def _render_v32_signal_stability(
    fixed_gate, joined, segment_summary, segment_seasons,
    prob_buckets, monotonicity, ensemble_detail, ensemble_summary,
    football_signals, holdout
):
    st.markdown("#### v3.2 Signal Stability + Ensemble Discovery")
    st.caption(
        "No new model is fit here. v3.2 audits the exact v3.1 walk-forward probabilities for "
        "season stability, probability monotonicity, predeclared situations, and multi-model agreement."
    )

    if joined is None or joined.empty:
        st.info("Run v3.2 after selecting the same 2022–2025 / 2025 holdout setup.")
        return

    st.markdown("##### Corrected holdout gate")
    g = fixed_gate.copy()
    if not g.empty:
        if "Holdout Betting ROI" in g.columns:
            g["Holdout Betting ROI"] = g["Holdout Betting ROI"].map(
                lambda v: "—" if pd.isna(v) else f"{100*v:+.1f}%"
            )
        st.dataframe(g, use_container_width=True, hide_index=True)

    st.markdown("##### Probability monotonicity")
    st.caption(
        "A legitimate probability model should generally produce better realized outcomes as predicted confidence rises."
    )
    pb = prob_buckets.copy()
    if not pb.empty:
        for c in ["Avg Predicted", "Realized Win Rate", "Calibration Error", "ROI"]:
            if c in pb.columns:
                pb[c] = pb[c].map(lambda v: f"{100*v:+.1f}%" if c in ["Calibration Error", "ROI"] else f"{100*v:.1f}%")
        st.dataframe(pb, use_container_width=True, hide_index=True)
    st.dataframe(monotonicity, use_container_width=True, hide_index=True)

    st.markdown("##### Situational stability")
    st.caption(
        "Discovery is restricted to 2022–2024. The selected holdout season is shown separately and is never used to create the status label."
    )
    ss = segment_summary.copy()
    if not ss.empty:
        for c in ["Threshold", "All Win Rate", "All ROI", "Development ROI", "Holdout ROI"]:
            if c in ss.columns:
                ss[c] = ss[c].map(
                    lambda v: "—" if pd.isna(v) else f"{100*v:+.1f}%"
                    if "ROI" in c else f"{100*v:.1f}%"
                )
        confirmed = ss[ss["Status"] == "HOLDOUT CONFIRMED"]
        discovery = ss[ss["Status"] == "DISCOVERY SIGNAL"]
        if not confirmed.empty:
            st.success(f"{len(confirmed)} predeclared model/segment combinations survived the holdout screen.")
        elif not discovery.empty:
            st.warning(
                f"{len(discovery)} discovery signals appeared in 2022–2024, but none met the locked holdout confirmation rule."
            )
        else:
            st.warning("No predeclared segment showed stable development-season evidence.")
        st.dataframe(
            ss.sort_values(
                ["Status", "Development Bets"],
                ascending=[True, False]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("##### Ensemble agreement")
    es = ensemble_summary.copy()
    if not es.empty:
        for c in ["Threshold", "Win Rate", "ROI", "Holdout Win Rate", "Holdout ROI"]:
            if c in es.columns:
                es[c] = es[c].map(
                    lambda v: "—" if pd.isna(v) else f"{100*v:+.1f}%"
                    if "ROI" in c else f"{100*v:.1f}%"
                )
        st.dataframe(es, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough common predictions to form the locked ensemble.")

    with st.expander("Football signal diagnostics", expanded=False):
        st.dataframe(football_signals, use_container_width=True, hide_index=True)

    with st.expander("v3.2 Downloads", expanded=True):
        bundle = _csv_download_bundle({
            "cfb_v320_fixed_holdout_gate.csv": fixed_gate,
            "cfb_v320_joined_predictions.csv": joined,
            "cfb_v320_segment_summary.csv": segment_summary,
            "cfb_v320_segment_seasons.csv": segment_seasons,
            "cfb_v320_probability_buckets.csv": prob_buckets,
            "cfb_v320_monotonicity.csv": monotonicity,
            "cfb_v320_ensemble_summary.csv": ensemble_summary,
            "cfb_v320_ensemble_predictions.csv": ensemble_detail,
            "cfb_v320_football_signal_diagnostics.csv": football_signals,
        })
        st.download_button(
            "Download All v3.2 Files",
            data=bundle,
            file_name="cfb_v320_signal_stability_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v320_all",
        )
        st.caption("Upload this single ZIP back to ChatGPT for review.")

# ===== v3.3 game-day selector / weekly ranking =====
V33_VERSION = "v3.3.0-gameday-selector"
V33_MIN_MODELS = 2
V33_TOP_N_CHOICES = (1, 3, 5)
V33_TOP_PCT_CHOICES = (0.10, 0.20)
V33_STANDARD_JUICE = -110


def _v33_model_probability_frame(history, preds):
    """
    Combine the exact walk-forward v3.1 classifier probabilities with the
    historical point-in-time game frame.
    """
    if history is None or history.empty or preds is None or preds.empty:
        return pd.DataFrame()

    p = preds[
        (preds["market_type"] == "spread") &
        (preds["task"] == "classification")
    ].copy()
    if p.empty:
        return pd.DataFrame()

    h = history.reset_index().rename(columns={"index": "row_index"}).copy()
    keep = [
        "row_index","season","week","game_id","home_team","away_team",
        "market_margin","actual_margin","conference_game","elo_diff",
        "talent_diff","returning_ppa_diff","cur_ppa_match","cur_success_match",
        "cur_line_match",
    ]
    keep = [c for c in keep if c in h.columns]
    h = h[keep].copy()

    x = p.merge(h, on=["row_index","season"], how="left")
    actual_resid = (
        pd.to_numeric(x["actual_margin"], errors="coerce")
        - pd.to_numeric(x["market_margin"], errors="coerce")
    )
    x["home_cover"] = np.where(
        actual_resid > 0, 1.0,
        np.where(actual_resid < 0, 0.0, np.nan)
    )
    x["model_side"] = np.where(
        pd.to_numeric(x["probability"], errors="coerce") >= 0.5,
        "HOME","AWAY"
    )
    x["model_conf"] = np.maximum(
        pd.to_numeric(x["probability"], errors="coerce"),
        1.0 - pd.to_numeric(x["probability"], errors="coerce"),
    )
    return x


def _v33_regression_consensus(preds):
    """
    Build a directional market-residual consensus from nonlinear regression
    models only. This is used as a ranking input, never as a standalone bet.
    """
    if preds is None or preds.empty:
        return pd.DataFrame()

    rp = preds[
        (preds["market_type"] == "spread") &
        (preds["task"] == "regression") &
        (preds["model"].isin(["Gradient Boosting","Extra Trees","Random Forest"]))
    ].copy()
    if rp.empty:
        return pd.DataFrame()

    rp["correction"] = pd.to_numeric(rp["correction"], errors="coerce")
    rp["reg_side"] = np.where(rp["correction"] >= 0, "HOME", "AWAY")
    rp["reg_strength"] = rp["correction"].abs()

    rows = []
    for (season, row_index), g in rp.groupby(["season","row_index"]):
        valid = g.dropna(subset=["correction"])
        if valid.empty:
            continue
        home_n = int((valid["reg_side"] == "HOME").sum())
        away_n = int((valid["reg_side"] == "AWAY").sum())
        side = "HOME" if home_n >= away_n else "AWAY"
        agree = max(home_n, away_n)
        strengths = valid.loc[valid["reg_side"] == side, "reg_strength"]
        rows.append({
            "season": int(season),
            "row_index": int(row_index),
            "reg_side": side,
            "reg_agreement": int(agree),
            "reg_strength": float(strengths.mean()) if len(strengths) else 0.0,
        })
    return pd.DataFrame(rows)


def _v33_rank_frame(history, preds):
    """
    Create one row per game with a predeclared ranking score.

    The selector intentionally avoids trusting absolute probabilities.
    It ranks games RELATIVE TO THAT WEEK'S SLATE using:
      1) classifier consensus confidence percentile,
      2) classifier agreement,
      3) nonlinear residual-regression strength percentile,
      4) classifier/regression directional agreement,
      5) data maturity.

    No weight is optimized on the holdout.
    """
    x = _v33_model_probability_frame(history, preds)
    if x.empty:
        return pd.DataFrame()

    core = x[x["model"].isin(["Gradient Boosting","Extra Trees","Logistic"])].copy()
    if core.empty:
        return pd.DataFrame()

    rows = []
    for (season, row_index), g in core.groupby(["season","row_index"]):
        valid = g.dropna(subset=["probability"])
        if len(valid) < V33_MIN_MODELS:
            continue

        home_votes = int((valid["model_side"] == "HOME").sum())
        away_votes = int((valid["model_side"] == "AWAY").sum())
        if home_votes == away_votes:
            continue

        side = "HOME" if home_votes > away_votes else "AWAY"
        agreeing = valid[valid["model_side"] == side].copy()
        if agreeing.empty:
            continue

        first = valid.iloc[0]
        y = _v3_num(first.get("home_cover"))
        if not np.isfinite(y):
            continue
        won = int(y == 1) if side == "HOME" else int(y == 0)

        rows.append({
            "season": int(season),
            "week": int(_v3_num(first.get("week"), 1)),
            "row_index": int(row_index),
            "game_id": first.get("game_id"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "market_margin": _v3_num(first.get("market_margin")),
            "pick_side": side,
            "won": won,
            "classifier_models": int(len(valid)),
            "classifier_agreement": int(len(agreeing)),
            "classifier_confidence": float(
                pd.to_numeric(agreeing["model_conf"], errors="coerce").mean()
            ),
            "conference_game": _v3_num(first.get("conference_game"), 0.0),
            "elo_diff": _v3_num(first.get("elo_diff")),
            "talent_diff": _v3_num(first.get("talent_diff")),
            "returning_ppa_diff": _v3_num(first.get("returning_ppa_diff")),
            "cur_ppa_match": _v3_num(first.get("cur_ppa_match")),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    reg = _v33_regression_consensus(preds)
    if not reg.empty:
        out = out.merge(reg, on=["season","row_index"], how="left")
    else:
        out["reg_side"] = None
        out["reg_agreement"] = 0
        out["reg_strength"] = 0.0

    out["reg_agreement"] = pd.to_numeric(
        out["reg_agreement"], errors="coerce"
    ).fillna(0.0)
    out["reg_strength"] = pd.to_numeric(
        out["reg_strength"], errors="coerce"
    ).fillna(0.0)

    out["direction_agreement"] = (
        out["pick_side"].astype(str) == out["reg_side"].astype(str)
    ).astype(float)

    # Week-by-week cross-sectional ranks.
    ranked = []
    for (season, week), g in out.groupby(["season","week"]):
        z = g.copy()
        z["conf_pct"] = z["classifier_confidence"].rank(
            method="average", pct=True
        )
        z["reg_pct"] = z["reg_strength"].rank(
            method="average", pct=True
        )

        # Data maturity is intentionally coarse and deterministic.
        maturity = 1.0 if int(week) >= 4 else 0.0
        z["data_maturity"] = maturity

        # Fixed score; no optimization on holdout.
        z["selector_score"] = (
            0.40 * z["conf_pct"]
            + 0.20 * (z["classifier_agreement"] / z["classifier_models"].clip(lower=1))
            + 0.20 * z["reg_pct"]
            + 0.10 * z["direction_agreement"]
            + 0.10 * z["data_maturity"]
        )
        z["slate_rank"] = z["selector_score"].rank(
            method="first", ascending=False
        ).astype(int)
        z["slate_size"] = int(len(z))
        z["slate_percentile"] = z["slate_rank"] / max(int(len(z)), 1)
        ranked.append(z)

    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def _v33_selection_masks(df):
    masks = {}
    if df is None or df.empty:
        return masks

    for n in V33_TOP_N_CHOICES:
        masks[f"Top {n}"] = df["slate_rank"] <= int(n)

    for pct in V33_TOP_PCT_CHOICES:
        label = f"Top {int(100*pct)}%"
        cutoff = np.maximum(
            1,
            np.ceil(pd.to_numeric(df["slate_size"], errors="coerce") * pct)
        )
        masks[label] = pd.to_numeric(df["slate_rank"], errors="coerce") <= cutoff
    return masks


def _v33_strategy_detail(rank_frame):
    if rank_frame is None or rank_frame.empty:
        return pd.DataFrame()

    masks = _v33_selection_masks(rank_frame)
    rows = []
    for strategy, mask in masks.items():
        x = rank_frame[mask].copy()
        for _, r in x.iterrows():
            rows.append({
                "Strategy": strategy,
                "season": int(r["season"]),
                "week": int(r["week"]),
                "game_id": r.get("game_id"),
                "away_team": r.get("away_team"),
                "home_team": r.get("home_team"),
                "market_margin": _v3_num(r.get("market_margin")),
                "pick_side": r.get("pick_side"),
                "selector_score": _v3_num(r.get("selector_score")),
                "slate_rank": int(r.get("slate_rank")),
                "slate_size": int(r.get("slate_size")),
                "won": int(r.get("won")),
            })
    return pd.DataFrame(rows)


def _v33_roi_from_rows(x):
    if x is None or x.empty:
        return np.nan
    wins = int(pd.to_numeric(x["won"], errors="coerce").sum())
    n = int(len(x))
    return _v31_roi_from_wins(wins, n - wins, 0, V33_STANDARD_JUICE)


def _v33_max_drawdown(detail):
    """
    Flat 1-unit stakes at -110. Each win earns 0.9091u, each loss -1u.
    """
    if detail is None or detail.empty:
        return np.nan
    d = detail.sort_values(
        ["season","week","selector_score"],
        ascending=[True,True,False]
    ).copy()
    d["unit_result"] = np.where(
        pd.to_numeric(d["won"], errors="coerce") == 1,
        100.0/110.0, -1.0
    )
    equity = d["unit_result"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else np.nan


def _v33_longest_losing_streak(detail):
    if detail is None or detail.empty:
        return 0
    d = detail.sort_values(
        ["season","week","selector_score"],
        ascending=[True,True,False]
    )
    longest = current = 0
    for w in pd.to_numeric(d["won"], errors="coerce").fillna(0).astype(int):
        if w == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _v33_strategy_summary(detail, holdout):
    if detail is None or detail.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    season_rows = []
    weekly_rows = []
    summary_rows = []

    for strategy, g in detail.groupby("Strategy"):
        for season, sg in g.groupby("season"):
            wins = int(pd.to_numeric(sg["won"], errors="coerce").sum())
            n = int(len(sg))
            season_rows.append({
                "Strategy": strategy,
                "Season": int(season),
                "Bets": n,
                "Wins": wins,
                "Losses": n - wins,
                "Win Rate": wins / n if n else np.nan,
                "ROI": _v33_roi_from_rows(sg),
            })

        for (season, week), wg in g.groupby(["season","week"]):
            wins = int(pd.to_numeric(wg["won"], errors="coerce").sum())
            n = int(len(wg))
            weekly_rows.append({
                "Strategy": strategy,
                "Season": int(season),
                "Week": int(week),
                "Bets": n,
                "Wins": wins,
                "Losses": n - wins,
                "Win Rate": wins / n if n else np.nan,
                "ROI": _v33_roi_from_rows(wg),
            })

        wins = int(pd.to_numeric(g["won"], errors="coerce").sum())
        n = int(len(g))
        season_df = pd.DataFrame([r for r in season_rows if r["Strategy"] == strategy])
        positive_seasons = int((pd.to_numeric(season_df["ROI"], errors="coerce") > 0).sum())

        hb = g[g["season"].astype(int) == int(holdout)]
        h_wins = int(pd.to_numeric(hb["won"], errors="coerce").sum()) if not hb.empty else 0
        h_n = int(len(hb))

        summary_rows.append({
            "Strategy": strategy,
            "Bets": n,
            "Wins": wins,
            "Losses": n - wins,
            "Win Rate": wins / n if n else np.nan,
            "ROI": _v33_roi_from_rows(g),
            "Positive Seasons": f"{positive_seasons}/{len(season_df)}",
            "Holdout Bets": h_n,
            "Holdout Win Rate": h_wins / h_n if h_n else np.nan,
            "Holdout ROI": _v33_roi_from_rows(hb) if h_n else np.nan,
            "Max Drawdown (u)": _v33_max_drawdown(g),
            "Longest Losing Streak": _v33_longest_losing_streak(g),
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(season_rows),
        pd.DataFrame(weekly_rows),
    )


def _v33_phase_summary(detail):
    if detail is None or detail.empty:
        return pd.DataFrame()
    rows = []
    x = detail.copy()
    x["Phase"] = np.where(
        pd.to_numeric(x["week"], errors="coerce") <= 3,
        "Weeks 0–3", "Weeks 4+"
    )
    for (strategy, phase), g in x.groupby(["Strategy","Phase"]):
        wins = int(pd.to_numeric(g["won"], errors="coerce").sum())
        n = int(len(g))
        rows.append({
            "Strategy": strategy,
            "Phase": phase,
            "Bets": n,
            "Wins": wins,
            "Losses": n - wins,
            "Win Rate": wins / n if n else np.nan,
            "ROI": _v33_roi_from_rows(g),
        })
    return pd.DataFrame(rows)


def _v33_locked_gate(summary, seasons, holdout):
    if summary is None or summary.empty:
        return pd.DataFrame()

    rows = []
    for _, r in summary.iterrows():
        pos_txt = str(r["Positive Seasons"])
        try:
            pos = int(pos_txt.split("/")[0])
            total_s = int(pos_txt.split("/")[1])
        except Exception:
            pos, total_s = 0, 0

        overall_pass = (
            int(r["Bets"]) >= 200
            and float(r["ROI"]) > 0
            and float(r["Win Rate"]) > (110/210)
        )
        season_pass = total_s >= 3 and pos >= max(2, total_s - 1)
        hold_pass = (
            int(r["Holdout Bets"]) >= 25
            and pd.notna(r["Holdout ROI"])
            and float(r["Holdout ROI"]) > 0
            and float(r["Holdout Win Rate"]) > (110/210)
        )

        if overall_pass and season_pass and hold_pass:
            verdict = "LOCKED CONFIRMATION CANDIDATE"
        elif overall_pass and season_pass:
            verdict = "DISCOVERY POSITIVE / HOLDOUT FAIL"
        else:
            verdict = "KEEP IN RESEARCH"

        rows.append({
            "Strategy": r["Strategy"],
            "Overall Pass": bool(overall_pass),
            "Season Stability Pass": bool(season_pass),
            f"{holdout} Holdout Pass": bool(hold_pass),
            "Verdict": verdict,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v33_gameday_selector(test_seasons_tuple, scope, holdout, train_start):
    (
        history, reg, reg_summary, cls, cls_summary,
        bets_raw, bet_summary, preds, fixed_gate,
    ) = _run_v31_ml_bakeoff(
        tuple(sorted(set(int(s) for s in test_seasons_tuple))),
        scope,
        int(holdout),
        int(train_start),
    )

    rank_frame = _v33_rank_frame(history, preds)
    detail = _v33_strategy_detail(rank_frame)
    summary, seasons, weeks = _v33_strategy_summary(detail, holdout)
    phases = _v33_phase_summary(detail)
    gate = _v33_locked_gate(summary, test_seasons_tuple, holdout)

    return (
        rank_frame, detail, summary, seasons, weeks, phases, gate
    )


def _render_v33_gameday_selector(
    rank_frame, detail, summary, seasons, weeks, phases, gate, holdout
):
    st.markdown("#### v3.3 Game-Day Selector")
    st.caption(
        "This is the workflow-oriented model: rank the slate, take only the strongest few games, "
        "and evaluate the exact historical game-day process rather than trying to forecast every game perfectly."
    )

    if summary is None or summary.empty:
        st.info("Run v3.3 to test top-N and top-percentile weekly selection.")
        return

    st.markdown("##### Historical game-day strategies")
    s = summary.copy()
    for c in ["Win Rate","ROI","Holdout Win Rate","Holdout ROI"]:
        if c in s.columns:
            s[c] = s[c].map(
                lambda v: "—" if pd.isna(v) else f"{100*v:.1f}%"
                if "Win Rate" in c else f"{100*v:+.1f}%"
            )
    if "Max Drawdown (u)" in s.columns:
        s["Max Drawdown (u)"] = s["Max Drawdown (u)"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.1f}u"
        )
    st.dataframe(s, use_container_width=True, hide_index=True)

    st.markdown("##### Locked confirmation gate")
    st.dataframe(gate, use_container_width=True, hide_index=True)

    passed = gate[
        gate["Verdict"] == "LOCKED CONFIRMATION CANDIDATE"
    ] if gate is not None and not gate.empty else pd.DataFrame()

    if not passed.empty:
        names = ", ".join(passed["Strategy"].astype(str).tolist())
        st.success(
            f"Selector strategy passed the locked historical screen: {names}. "
            "The next step is a forward 2026 tracker before increasing live confidence."
        )
    else:
        st.warning(
            "No selector strategy cleared the full locked screen. "
            "Do not force more bets; the selector stays research-only."
        )

    st.markdown("##### Early season vs mature season")
    p = phases.copy()
    if not p.empty:
        p["Win Rate"] = p["Win Rate"].map(lambda v: f"{100*v:.1f}%")
        p["ROI"] = p["ROI"].map(lambda v: f"{100*v:+.1f}%")
        st.dataframe(p, use_container_width=True, hide_index=True)

    with st.expander("Season-by-season selector results", expanded=False):
        x = seasons.copy()
        if not x.empty:
            x["Win Rate"] = x["Win Rate"].map(lambda v: f"{100*v:.1f}%")
            x["ROI"] = x["ROI"].map(lambda v: f"{100*v:+.1f}%")
        st.dataframe(x, use_container_width=True, hide_index=True)

    with st.expander("Week-by-week selector results", expanded=False):
        x = weeks.copy()
        if not x.empty:
            x["Win Rate"] = x["Win Rate"].map(lambda v: f"{100*v:.1f}%")
            x["ROI"] = x["ROI"].map(lambda v: f"{100*v:+.1f}%")
        st.dataframe(x, use_container_width=True, hide_index=True)

    with st.expander("v3.3 Downloads", expanded=True):
        bundle = _csv_download_bundle({
            "cfb_v330_selector_summary.csv": summary,
            "cfb_v330_selector_gate.csv": gate,
            "cfb_v330_selector_seasons.csv": seasons,
            "cfb_v330_selector_weeks.csv": weeks,
            "cfb_v330_selector_phases.csv": phases,
            "cfb_v330_selector_bets.csv": detail,
            "cfb_v330_ranked_games.csv": rank_frame,
        })
        st.download_button(
            "Download All v3.3 Files",
            data=bundle,
            file_name="cfb_v330_gameday_selector_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v330_all",
        )
        st.caption("Upload this single ZIP back to ChatGPT for review.")

# ===== v3.4 slate-aware finalist =====
V34_VERSION = "v3.4.0-slate-aware-finalist"
V34_DEV_SEASONS = (2022, 2023, 2024)
V34_DEFAULT_HOLDOUT = 2025
V34_BREAKEVEN = 110.0 / 210.0
V34_MIN_SLATE_GAMES = 3

# Fixed ranking architectures.  These are selected ONLY on development seasons.
# The holdout season is evaluated after the architecture + card rule are locked.
V34_ARCHITECTURES = {
    "Consensus First": {
        "confidence": 0.55,
        "agreement": 0.25,
        "regression": 0.10,
        "direction": 0.05,
        "maturity": 0.05,
    },
    "Balanced Ensemble": {
        "confidence": 0.40,
        "agreement": 0.20,
        "regression": 0.20,
        "direction": 0.10,
        "maturity": 0.10,
    },
    "Agreement Heavy": {
        "confidence": 0.35,
        "agreement": 0.35,
        "regression": 0.10,
        "direction": 0.10,
        "maturity": 0.10,
    },
    "Football Context": {
        "confidence": 0.35,
        "agreement": 0.20,
        "regression": 0.15,
        "direction": 0.10,
        "maturity": 0.20,
    },
}

V34_CARD_RULES = (
    "Top 1 / Slate",
    "Top 2 / Slate",
    "Top 3 / Slate",
    "Dynamic 1–3 / Slate",
)


def _v34_ensure_slate_columns(history):
    """
    v3.4 normally gets slate_window from the rebuilt v3 history frame.
    Fallbacks keep the app robust when an older cached history object is present.
    """
    h = history.copy()
    if "slate_window" not in h.columns:
        h["slate_window"] = "Unknown"
    if "game_date_et" not in h.columns:
        h["game_date_et"] = ""
    if "kickoff_hour_et" not in h.columns:
        h["kickoff_hour_et"] = np.nan
    return h


def _v34_base_game_frame(history, preds):
    history = _v34_ensure_slate_columns(history)
    if history is None or history.empty or preds is None or preds.empty:
        return pd.DataFrame()

    p = preds[
        (preds["market_type"] == "spread") &
        (preds["task"] == "classification") &
        (preds["model"].isin(["Gradient Boosting", "Extra Trees", "Logistic"]))
    ].copy()
    if p.empty:
        return pd.DataFrame()

    h = history.reset_index().rename(columns={"index": "row_index"}).copy()
    keep = [
        "row_index","season","week","game_id","home_team","away_team",
        "market_margin","actual_margin","kickoff_et","game_date_et",
        "kickoff_hour_et","slate_window","conference_game","elo_diff",
        "talent_diff","returning_ppa_diff","cur_ppa_match",
    ]
    keep = [c for c in keep if c in h.columns]
    h = h[keep]

    x = p.merge(h, on=["row_index","season"], how="left")
    residual = (
        pd.to_numeric(x["actual_margin"], errors="coerce")
        - pd.to_numeric(x["market_margin"], errors="coerce")
    )
    x["home_cover"] = np.where(
        residual > 0, 1.0,
        np.where(residual < 0, 0.0, np.nan)
    )
    x["side"] = np.where(
        pd.to_numeric(x["probability"], errors="coerce") >= 0.5,
        "HOME","AWAY"
    )
    x["side_confidence"] = np.maximum(
        pd.to_numeric(x["probability"], errors="coerce"),
        1.0 - pd.to_numeric(x["probability"], errors="coerce"),
    )

    reg = _v33_regression_consensus(preds)

    rows = []
    for (season, row_index), g in x.groupby(["season","row_index"]):
        valid = g.dropna(subset=["probability"])
        if len(valid) < 2:
            continue

        home_votes = int((valid["side"] == "HOME").sum())
        away_votes = int((valid["side"] == "AWAY").sum())
        if home_votes == away_votes:
            continue

        pick_side = "HOME" if home_votes > away_votes else "AWAY"
        agreeing = valid[valid["side"] == pick_side]
        if agreeing.empty:
            continue

        r0 = valid.iloc[0]
        y = _v3_num(r0.get("home_cover"))
        if not np.isfinite(y):
            continue

        won = int(y == 1) if pick_side == "HOME" else int(y == 0)
        rows.append({
            "season": int(season),
            "week": int(_v3_num(r0.get("week"), 1)),
            "row_index": int(row_index),
            "game_id": r0.get("game_id"),
            "home_team": r0.get("home_team"),
            "away_team": r0.get("away_team"),
            "market_margin": _v3_num(r0.get("market_margin")),
            "kickoff_et": r0.get("kickoff_et", ""),
            "game_date_et": r0.get("game_date_et", ""),
            "kickoff_hour_et": _v3_num(r0.get("kickoff_hour_et")),
            "slate_window": r0.get("slate_window", "Unknown"),
            "pick_side": pick_side,
            "won": won,
            "classifier_models": int(len(valid)),
            "classifier_agreement": int(len(agreeing)),
            "classifier_confidence": float(
                pd.to_numeric(agreeing["side_confidence"], errors="coerce").mean()
            ),
            "elo_abs": abs(_v3_num(r0.get("elo_diff"))),
            "talent_abs": abs(_v3_num(r0.get("talent_diff"))),
            "returning_abs": abs(_v3_num(r0.get("returning_ppa_diff"))),
            "current_data": 1.0 if np.isfinite(_v3_num(r0.get("cur_ppa_match"))) else 0.0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if reg is not None and not reg.empty:
        out = out.merge(reg, on=["season","row_index"], how="left")
    else:
        out["reg_side"] = None
        out["reg_agreement"] = 0.0
        out["reg_strength"] = 0.0

    out["reg_strength"] = pd.to_numeric(
        out["reg_strength"], errors="coerce"
    ).fillna(0.0)
    out["direction_agreement"] = (
        out["pick_side"].astype(str) == out["reg_side"].astype(str)
    ).astype(float)

    out["agreement_rate"] = (
        pd.to_numeric(out["classifier_agreement"], errors="coerce")
        / pd.to_numeric(out["classifier_models"], errors="coerce").clip(lower=1)
    )

    out["data_maturity"] = np.where(
        pd.to_numeric(out["week"], errors="coerce") >= 4,
        1.0,
        0.35 + 0.25 * pd.to_numeric(out["current_data"], errors="coerce").fillna(0.0),
    )

    # If historical kickoff is unavailable, preserve the game for architecture
    # work but do not allow it into slate-aware validation.
    out["slate_valid"] = out["slate_window"].isin(["Early","Midday","Late"])
    return out


def _v34_score_architecture(base, architecture_name):
    if base is None or base.empty:
        return pd.DataFrame()

    weights = V34_ARCHITECTURES[architecture_name]
    scored = []

    # Rankings are cross-sectional WITHIN each slate, not across the whole day.
    group_cols = ["season","week","slate_window"]
    for keys, g in base[base["slate_valid"]].groupby(group_cols):
        if len(g) < V34_MIN_SLATE_GAMES:
            continue

        z = g.copy()
        z["confidence_pct"] = z["classifier_confidence"].rank(
            method="average", pct=True
        )
        z["regression_pct"] = z["reg_strength"].rank(
            method="average", pct=True
        )

        z["selector_score"] = (
            weights["confidence"] * z["confidence_pct"]
            + weights["agreement"] * z["agreement_rate"]
            + weights["regression"] * z["regression_pct"]
            + weights["direction"] * z["direction_agreement"]
            + weights["maturity"] * z["data_maturity"]
        )

        z["architecture"] = architecture_name
        z["slate_rank"] = z["selector_score"].rank(
            method="first", ascending=False
        ).astype(int)
        z["slate_size"] = int(len(z))

        leader = float(z["selector_score"].max())
        z["leader_gap"] = leader - z["selector_score"]
        scored.append(z)

    return pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()


def _v34_card_mask(scored, card_rule):
    if scored is None or scored.empty:
        return pd.Series(False, index=scored.index if scored is not None else [])

    if card_rule == "Top 1 / Slate":
        return scored["slate_rank"] <= 1
    if card_rule == "Top 2 / Slate":
        return scored["slate_rank"] <= 2
    if card_rule == "Top 3 / Slate":
        return scored["slate_rank"] <= 3

    # Dynamic rule: always allow #1; #2 and #3 only when close enough to the
    # slate leader AND model agreement is at least 2/3.
    # These thresholds are fixed before holdout evaluation.
    return (
        (scored["slate_rank"] == 1)
        |
        (
            (scored["slate_rank"] == 2)
            & (pd.to_numeric(scored["leader_gap"], errors="coerce") <= 0.075)
            & (pd.to_numeric(scored["agreement_rate"], errors="coerce") >= (2/3))
        )
        |
        (
            (scored["slate_rank"] == 3)
            & (pd.to_numeric(scored["leader_gap"], errors="coerce") <= 0.040)
            & (pd.to_numeric(scored["agreement_rate"], errors="coerce") >= 1.0)
        )
    )


def _v34_flat_units(g):
    if g is None or g.empty:
        return np.nan
    wins = int(pd.to_numeric(g["won"], errors="coerce").sum())
    losses = int(len(g)) - wins
    return wins * (100.0 / 110.0) - losses


def _v34_roi(g):
    if g is None or g.empty:
        return np.nan
    return _v34_flat_units(g) / len(g)


def _v34_drawdown(g):
    if g is None or g.empty:
        return np.nan

    x = g.copy()
    if "slate_window" not in x.columns:
        if "display_group" in x.columns:
            x["slate_window"] = x["display_group"]
        else:
            x["slate_window"] = "Daily Slate"

    if "slate_rank" not in x.columns:
        if "day_rank" in x.columns:
            x["slate_rank"] = x["day_rank"]
        else:
            x["slate_rank"] = np.arange(1, len(x) + 1)

    x = x.sort_values(
        ["season","week","slate_window","slate_rank"],
        ascending=[True,True,True,True]
    ).copy()
    x["u"] = np.where(
        pd.to_numeric(x["won"], errors="coerce") == 1,
        100.0/110.0,
        -1.0
    )
    eq = x["u"].cumsum()
    peak = eq.cummax()
    return float((eq - peak).min())


def _v34_metrics(g):
    if g is None or g.empty:
        return {
            "Bets":0,"Wins":0,"Losses":0,"Win Rate":np.nan,
            "ROI":np.nan,"Units":0.0,"Max Drawdown":np.nan,
        }
    wins = int(pd.to_numeric(g["won"], errors="coerce").sum())
    n = int(len(g))
    return {
        "Bets": n,
        "Wins": wins,
        "Losses": n - wins,
        "Win Rate": wins / n if n else np.nan,
        "ROI": _v34_roi(g),
        "Units": _v34_flat_units(g),
        "Max Drawdown": _v34_drawdown(g),
    }


def _v34_candidate_table(base, holdout):
    """
    Select architecture + card rule on development seasons only.
    The holdout result is attached only after the development score is frozen.
    """
    rows = []
    bet_rows = []

    for arch in V34_ARCHITECTURES:
        scored = _v34_score_architecture(base, arch)
        if scored.empty:
            continue

        for rule in V34_CARD_RULES:
            chosen = scored[_v34_card_mask(scored, rule)].copy()
            if chosen.empty:
                continue

            chosen["card_rule"] = rule
            bet_rows.append(chosen)

            dev = chosen[chosen["season"].astype(int).isin(V34_DEV_SEASONS)].copy()
            hld = chosen[chosen["season"].astype(int) == int(holdout)].copy()

            dev_m = _v34_metrics(dev)
            hold_m = _v34_metrics(hld)

            season_rois = []
            for season in V34_DEV_SEASONS:
                sg = dev[dev["season"].astype(int) == int(season)]
                if len(sg):
                    season_rois.append(_v34_roi(sg))

            pos_dev = sum(1 for r in season_rois if np.isfinite(r) and r > 0)
            min_dev_roi = min(season_rois) if season_rois else np.nan

            # Development selection score. The objective rewards edge, stability,
            # useful volume and manageable drawdown without looking at holdout.
            volume_factor = min(dev_m["Bets"] / 250.0, 1.0)
            stability_factor = pos_dev / max(len(season_rois), 1)
            dd_penalty = min(abs(dev_m["Max Drawdown"]) / 25.0, 1.0) if np.isfinite(dev_m["Max Drawdown"]) else 1.0

            selection_score = (
                0.45 * (dev_m["ROI"] if np.isfinite(dev_m["ROI"]) else -1.0)
                + 0.20 * ((dev_m["Win Rate"] - V34_BREAKEVEN) if np.isfinite(dev_m["Win Rate"]) else -1.0)
                + 0.15 * stability_factor
                + 0.10 * volume_factor
                - 0.10 * dd_penalty
            )

            rows.append({
                "Architecture": arch,
                "Card Rule": rule,
                "Development Bets": dev_m["Bets"],
                "Development Win Rate": dev_m["Win Rate"],
                "Development ROI": dev_m["ROI"],
                "Development Units": dev_m["Units"],
                "Positive Dev Seasons": f"{pos_dev}/{len(season_rois)}",
                "Worst Dev Season ROI": min_dev_roi,
                "Development Max Drawdown": dev_m["Max Drawdown"],
                "Selection Score": selection_score,
                "Holdout Bets": hold_m["Bets"],
                "Holdout Win Rate": hold_m["Win Rate"],
                "Holdout ROI": hold_m["ROI"],
                "Holdout Units": hold_m["Units"],
                "Holdout Max Drawdown": hold_m["Max Drawdown"],
            })

    table = pd.DataFrame(rows)
    bets = pd.concat(bet_rows, ignore_index=True) if bet_rows else pd.DataFrame()

    if table.empty:
        return table, bets

    table = table.sort_values(
        ["Selection Score","Development ROI","Development Bets"],
        ascending=[False,False,False]
    ).reset_index(drop=True)
    table["Development Rank"] = np.arange(1, len(table)+1)
    table["Locked Winner"] = table["Development Rank"] == 1
    return table, bets


def _v34_locked_winner_detail(candidates, all_bets, holdout):
    if candidates is None or candidates.empty or all_bets is None or all_bets.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    winner = candidates.iloc[0]
    arch = winner["Architecture"]
    rule = winner["Card Rule"]

    x = all_bets[
        (all_bets["architecture"] == arch) &
        (all_bets["card_rule"] == rule)
    ].copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    season_rows = []
    for season, g in x.groupby("season"):
        m = _v34_metrics(g)
        season_rows.append({
            "Season": int(season),
            **m,
        })

    slate_rows = []
    for (season, slate), g in x.groupby(["season","slate_window"]):
        m = _v34_metrics(g)
        slate_rows.append({
            "Season": int(season),
            "Slate": slate,
            **m,
        })

    week_rows = []
    for (season, week), g in x.groupby(["season","week"]):
        m = _v34_metrics(g)
        week_rows.append({
            "Season": int(season),
            "Week": int(week),
            **m,
        })

    return (
        pd.DataFrame(season_rows),
        pd.DataFrame(slate_rows),
        pd.DataFrame(week_rows),
    )


def _v34_final_gate(candidates, holdout):
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    w = candidates.iloc[0]
    dev_bets = int(w["Development Bets"])
    dev_roi = _v3_num(w["Development ROI"])
    dev_wr = _v3_num(w["Development Win Rate"])
    hold_bets = int(w["Holdout Bets"])
    hold_roi = _v3_num(w["Holdout ROI"])
    hold_wr = _v3_num(w["Holdout Win Rate"])

    try:
        pos_dev = int(str(w["Positive Dev Seasons"]).split("/")[0])
        n_dev = int(str(w["Positive Dev Seasons"]).split("/")[1])
    except Exception:
        pos_dev, n_dev = 0, 0

    dev_pass = (
        dev_bets >= 150
        and np.isfinite(dev_roi) and dev_roi > 0
        and np.isfinite(dev_wr) and dev_wr > V34_BREAKEVEN
        and n_dev >= 3 and pos_dev >= 2
    )
    hold_pass = (
        hold_bets >= 35
        and np.isfinite(hold_roi) and hold_roi > 0
        and np.isfinite(hold_wr) and hold_wr > V34_BREAKEVEN
    )
    drawdown_ok = (
        np.isfinite(_v3_num(w["Development Max Drawdown"]))
        and abs(_v3_num(w["Development Max Drawdown"])) <= 20.0
    )

    if dev_pass and hold_pass and drawdown_ok:
        verdict = "FINALIST — FORWARD TRACK 2026"
    elif dev_pass:
        verdict = "DEVELOPMENT PASS / HOLDOUT FAIL"
    else:
        verdict = "KEEP IN RESEARCH"

    return pd.DataFrame([{
        "Locked Architecture": w["Architecture"],
        "Locked Card Rule": w["Card Rule"],
        "Development Pass": bool(dev_pass),
        f"{holdout} Holdout Pass": bool(hold_pass),
        "Drawdown Guardrail Pass": bool(drawdown_ok),
        "Verdict": verdict,
    }])


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v34_slate_finalist(test_seasons_tuple, scope, holdout, train_start):
    (
        history, reg, reg_summary, cls, cls_summary,
        bets_raw, bet_summary, preds, fixed_gate,
    ) = _run_v31_ml_bakeoff(
        tuple(sorted(set(int(s) for s in test_seasons_tuple))),
        scope,
        int(holdout),
        int(train_start),
    )

    base = _v34_base_game_frame(history, preds)
    candidates, all_bets = _v34_candidate_table(base, holdout)
    season_detail, slate_detail, week_detail = _v34_locked_winner_detail(
        candidates, all_bets, holdout
    )
    gate = _v34_final_gate(candidates, holdout)

    winner_bets = pd.DataFrame()
    winner_ranked = pd.DataFrame()
    if candidates is not None and not candidates.empty:
        winner = candidates.iloc[0]
        winner_ranked = _v34_score_architecture(base, winner["Architecture"])
        if winner_ranked is not None and not winner_ranked.empty:
            winner_bets = winner_ranked[
                _v34_card_mask(winner_ranked, winner["Card Rule"])
            ].copy()

    return (
        base, candidates, gate, winner_ranked, winner_bets,
        season_detail, slate_detail, week_detail,
    )


def _render_v34_slate_finalist(
    base, candidates, gate, winner_ranked, winner_bets,
    season_detail, slate_detail, week_detail, holdout
):
    st.markdown("#### v3.4 Slate-Aware Finalist")
    st.caption(
        "This is the closest historical simulation to the intended Saturday workflow: "
        "rank Early, Midday and Late separately, select the best ranking architecture on "
        "2022–2024 only, lock it, then reveal the selected architecture's 2025 result."
    )

    if candidates is None or candidates.empty:
        st.info("Run v3.4 to compare locked slate-aware ranking architectures.")
        return

    winner = candidates.iloc[0]
    st.markdown("##### Locked development winner")
    c1, c2, c3 = st.columns(3)
    c1.metric("Architecture", str(winner["Architecture"]))
    c2.metric("Card Rule", str(winner["Card Rule"]))
    c3.metric(
        "Development ROI",
        "—" if pd.isna(winner["Development ROI"])
        else f"{100*float(winner['Development ROI']):+.1f}%"
    )

    st.markdown("##### Final gate")
    st.dataframe(gate, use_container_width=True, hide_index=True)

    verdict = str(gate.iloc[0]["Verdict"]) if gate is not None and not gate.empty else ""
    if verdict.startswith("FINALIST"):
        st.success(
            "The locked slate-aware selector cleared the historical screen. "
            "Do not re-optimize it on 2025; the next step is to freeze this exact rule "
            "and track every 2026 recommendation forward."
        )
    elif "HOLDOUT FAIL" in verdict:
        st.warning(
            "The development winner did not survive the holdout. "
            "Do not promote it just because another candidate happened to look better in 2025."
        )
    else:
        st.warning(
            "No architecture currently meets the required development evidence. "
            "The model should remain selective/research-only."
        )

    st.markdown("##### Architecture + card-rule comparison")
    show = candidates.copy()
    pct_cols = [
        "Development Win Rate","Development ROI","Worst Dev Season ROI",
        "Holdout Win Rate","Holdout ROI",
    ]
    for c in pct_cols:
        if c in show.columns:
            show[c] = show[c].map(
                lambda v: "—" if pd.isna(v) else f"{100*float(v):+.1f}%"
                if "ROI" in c else f"{100*float(v):.1f}%"
            )
    for c in ["Development Max Drawdown","Holdout Max Drawdown"]:
        if c in show.columns:
            show[c] = show[c].map(
                lambda v: "—" if pd.isna(v) else f"{float(v):.1f}u"
            )
    st.dataframe(
        show.head(16),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### Locked winner by slate")
    sd = slate_detail.copy()
    if not sd.empty:
        sd["Win Rate"] = sd["Win Rate"].map(lambda v: f"{100*float(v):.1f}%")
        sd["ROI"] = sd["ROI"].map(lambda v: f"{100*float(v):+.1f}%")
        sd["Max Drawdown"] = sd["Max Drawdown"].map(lambda v: f"{float(v):.1f}u")
        st.dataframe(sd, use_container_width=True, hide_index=True)

    with st.expander("Locked winner by season", expanded=False):
        s = season_detail.copy()
        if not s.empty:
            s["Win Rate"] = s["Win Rate"].map(lambda v: f"{100*float(v):.1f}%")
            s["ROI"] = s["ROI"].map(lambda v: f"{100*float(v):+.1f}%")
        st.dataframe(s, use_container_width=True, hide_index=True)

    with st.expander("Locked winner week-by-week", expanded=False):
        w = week_detail.copy()
        if not w.empty:
            w["Win Rate"] = w["Win Rate"].map(lambda v: f"{100*float(v):.1f}%")
            w["ROI"] = w["ROI"].map(lambda v: f"{100*float(v):+.1f}%")
        st.dataframe(w, use_container_width=True, hide_index=True)

    with st.expander("v3.4 Downloads", expanded=True):
        bundle = _csv_download_bundle({
            "cfb_v340_candidate_architectures.csv": candidates,
            "cfb_v340_final_gate.csv": gate,
            "cfb_v340_locked_ranked_games.csv": winner_ranked,
            "cfb_v340_locked_bets.csv": winner_bets,
            "cfb_v340_locked_seasons.csv": season_detail,
            "cfb_v340_locked_slates.csv": slate_detail,
            "cfb_v340_locked_weeks.csv": week_detail,
            "cfb_v340_base_games.csv": base,
        })
        st.download_button(
            "Download All v3.4 Files",
            data=bundle,
            file_name="cfb_v340_slate_aware_finalist_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v340_all",
        )
        st.caption("Upload this ZIP back to ChatGPT for the final model review.")

# ===== v3.5 adaptive daily card / no forced bets =====
V35_VERSION = "v3.5.0-adaptive-daily-card"

# Fixed quality floors. These do NOT change based on how many games are available.
# The day size only affects presentation/grouping, never qualification.
V35_BEST_BET_SCORE = 0.84
V35_BET_SCORE = 0.78
V35_LEAN_SCORE = 0.72

# Additional quality guardrails.
V35_MIN_AGREEMENT_BEST = 2.0 / 3.0
V35_MIN_AGREEMENT_BET = 2.0 / 3.0
V35_MIN_AGREEMENT_LEAN = 2.0 / 3.0

# Daily grouping is presentation only.
V35_SMALL_DAY_MAX = 8
V35_MEDIUM_DAY_MAX = 20


def _v35_daily_group_label(game_count, kickoff_hour):
    """
    Day-size-aware display grouping.
    Qualification thresholds are never changed by this function.
    """
    try:
        n = int(game_count)
    except Exception:
        n = 0
    h = _v3_num(kickoff_hour)

    if n <= V35_SMALL_DAY_MAX:
        return "Daily Slate"

    if n <= V35_MEDIUM_DAY_MAX:
        if not np.isfinite(h):
            return "Daily Slate"
        return "Early" if h < 17.0 else "Late"

    if not np.isfinite(h):
        return "Daily Slate"
    if h < 14.5:
        return "Early"
    if h < 18.5:
        return "Midday"
    return "Late"



def _v354_attach_daily_calendar(v33_frame):
    """
    Convert a known-good v3.3 weekly selector frame into a calendar-day frame.
    This is intentionally independent of the v3.4/v3.5 history merge.
    """
    if v33_frame is None or v33_frame.empty:
        return pd.DataFrame()

    x = v33_frame.copy()
    schedule_rows = []

    seasons = sorted(
        pd.to_numeric(x["season"], errors="coerce").dropna().astype(int).unique()
    )

    for _season in seasons:
        source_rows = []

        # Prefer /games; then try historical lines as a second source.
        try:
            source_rows.extend(get_backtest_games(int(_season)) or [])
        except Exception:
            pass

        try:
            _line_rows = get_backtest_lines(int(_season)) or []
            for _lr in _line_rows:
                if isinstance(_lr, dict):
                    source_rows.append(_lr)
        except Exception:
            pass

        for _g in source_rows:
            if not isinstance(_g, dict):
                continue
            _gid = _g.get("id")
            _raw = (
                _g.get("startDate")
                or _g.get("start_date")
                or _g.get("startTime")
            )
            if _gid is None or not _raw:
                continue
            try:
                _dt = pd.to_datetime(_raw, utc=True).tz_convert("America/New_York")
            except Exception:
                continue
            schedule_rows.append({
                "season": int(_season),
                "game_id_key": str(_gid),
                "kickoff_et": _dt.strftime("%Y-%m-%d %I:%M %p"),
                "game_date_et": _dt.strftime("%Y-%m-%d"),
                "kickoff_hour_et": float(_dt.hour) + float(_dt.minute) / 60.0,
            })

    if not schedule_rows:
        return pd.DataFrame()

    sched = pd.DataFrame(schedule_rows).drop_duplicates(
        subset=["season", "game_id_key"], keep="first"
    )

    x["game_id_key"] = x["game_id"].astype(str)
    x = x.merge(sched, on=["season", "game_id_key"], how="inner")
    x = x.drop(columns=["game_id_key"], errors="ignore")

    if x.empty:
        return x

    ranked_days = []
    for (season, game_date), g in x.groupby(["season", "game_date_et"]):
        z = g.copy()

        # v3.3 carries classifier_confidence, classifier_agreement/models,
        # reg_strength, direction_agreement and data_maturity.
        z["confidence_pct"] = pd.to_numeric(
            z["classifier_confidence"], errors="coerce"
        ).rank(method="average", pct=True)

        z["regression_pct"] = pd.to_numeric(
            z["reg_strength"], errors="coerce"
        ).rank(method="average", pct=True)

        z["agreement_rate"] = (
            pd.to_numeric(z["classifier_agreement"], errors="coerce")
            / pd.to_numeric(z["classifier_models"], errors="coerce").clip(lower=1)
        )

        z["selector_score"] = (
            0.40 * z["confidence_pct"]
            + 0.20 * z["agreement_rate"]
            + 0.20 * z["regression_pct"]
            + 0.10 * pd.to_numeric(
                z["direction_agreement"], errors="coerce"
            ).fillna(0.0)
            + 0.10 * pd.to_numeric(
                z["data_maturity"], errors="coerce"
            ).fillna(0.0)
        )

        z["day_rank"] = z["selector_score"].rank(
            method="first", ascending=False
        ).astype(int)
        z["day_size"] = int(len(z))
        z["day_percentile"] = z["day_rank"] / max(int(len(z)), 1)
        z["display_group"] = [
            _v35_daily_group_label(len(z), h)
            for h in pd.to_numeric(z["kickoff_hour_et"], errors="coerce")
        ]

        # Compatibility alias for shared backtest metrics.  _v34_drawdown
        # historically sorts on slate_window; v3.5 uses display_group because
        # grouping is presentation-only.  They are equivalent for ordering.
        z["slate_window"] = z["display_group"]

        z["architecture"] = "Balanced Ensemble"
        ranked_days.append(z)

    return pd.concat(ranked_days, ignore_index=True) if ranked_days else pd.DataFrame()


def _v35_global_daily_rank_frame(history, preds, architecture_name="Balanced Ensemble"):
    """
    Build v3.3 selector rows first, then only attach calendar dates.
    """
    v33 = _v33_rank_frame(history, preds)
    return _v354_attach_daily_calendar(v33)


def _v35_quality_tier(row):
    score = _v3_num(row.get("selector_score"))
    agree = _v3_num(row.get("agreement_rate"))
    direction = _v3_num(row.get("direction_agreement"))

    # Hard no-bet if model direction is fragmented.
    if agree < V35_MIN_AGREEMENT_LEAN:
        return "PASS"

    if (
        score >= V35_BEST_BET_SCORE
        and agree >= V35_MIN_AGREEMENT_BEST
        and direction >= 1.0
    ):
        return "BEST BET"

    if score >= V35_BET_SCORE and agree >= V35_MIN_AGREEMENT_BET:
        return "BET"

    if score >= V35_LEAN_SCORE and agree >= V35_MIN_AGREEMENT_LEAN:
        return "LEAN"

    return "PASS"


def _v35_apply_quality_tiers(rank_frame):
    if rank_frame is None or rank_frame.empty:
        return pd.DataFrame()
    x = rank_frame.copy()
    x["verdict"] = x.apply(_v35_quality_tier, axis=1)

    # Conservative unit suggestions; no forced scaling with slate size.
    x["suggested_units"] = np.select(
        [
            x["verdict"] == "BEST BET",
            x["verdict"] == "BET",
            x["verdict"] == "LEAN",
        ],
        [1.0, 0.75, 0.0],
        default=0.0,
    )

    # User-facing matchup label.
    x["selection"] = np.where(
        x["pick_side"] == "HOME",
        x["home_team"].astype(str),
        x["away_team"].astype(str),
    )
    return x


def _v35_daily_card_backtest(tiered, holdout):
    if tiered is None or tiered.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    official = tiered[tiered["verdict"].isin(["BEST BET", "BET"])].copy()
    leans = tiered[tiered["verdict"] == "LEAN"].copy()

    daily_rows = []
    if not official.empty:
        for (season, day), g in official.groupby(["season", "game_date_et"]):
            m = _v34_metrics(g)
            daily_rows.append({
                "Season": int(season),
                "Date": day,
                "Available Games": int(g["day_size"].max()) if "day_size" in g else np.nan,
                "Official Bets": int(len(g)),
                "Best Bets": int((g["verdict"] == "BEST BET").sum()),
                "Bets": int((g["verdict"] == "BET").sum()),
                **m,
            })

    season_rows = []
    for season in sorted(tiered["season"].dropna().astype(int).unique()):
        sg = official[official["season"].astype(int) == int(season)]
        m = _v34_metrics(sg)
        total_days = tiered[tiered["season"].astype(int) == int(season)]["game_date_et"].nunique()
        bet_days = sg["game_date_et"].nunique() if not sg.empty else 0
        season_rows.append({
            "Season": int(season),
            "Official Bets": m["Bets"],
            "Wins": m["Wins"],
            "Losses": m["Losses"],
            "Win Rate": m["Win Rate"],
            "ROI": m["ROI"],
            "Units": m["Units"],
            "Max Drawdown": m["Max Drawdown"],
            "Days With Games": int(total_days),
            "Days With Official Bet": int(bet_days),
            "No-Bet Days": int(max(total_days - bet_days, 0)),
        })

    group_rows = []
    for group_name, g in official.groupby("display_group"):
        m = _v34_metrics(g)
        group_rows.append({
            "Display Group": group_name,
            **m,
        })

    holdout_rows = []
    h = official[official["season"].astype(int) == int(holdout)]
    for verdict, g in h.groupby("verdict"):
        m = _v34_metrics(g)
        holdout_rows.append({
            "Verdict": verdict,
            **m,
        })

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(season_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(holdout_rows),
    )


def _v35_threshold_audit(rank_frame, holdout):
    """
    Diagnostic only: show fixed-score buckets. This does not optimize thresholds.
    """
    if rank_frame is None or rank_frame.empty:
        return pd.DataFrame()

    bins = [
        (-np.inf, 0.72, "<0.72"),
        (0.72, 0.78, "0.72–0.779"),
        (0.78, 0.84, "0.78–0.839"),
        (0.84, np.inf, "0.84+"),
    ]
    rows = []
    for lo, hi, label in bins:
        g = rank_frame[
            (pd.to_numeric(rank_frame["selector_score"], errors="coerce") >= lo)
            & (pd.to_numeric(rank_frame["selector_score"], errors="coerce") < hi)
        ].copy()
        if g.empty:
            continue

        all_m = _v34_metrics(g)
        h = g[g["season"].astype(int) == int(holdout)]
        hm = _v34_metrics(h)
        rows.append({
            "Score Bucket": label,
            "Games": all_m["Bets"],
            "Win Rate": all_m["Win Rate"],
            "ROI": all_m["ROI"],
            "Holdout Games": hm["Bets"],
            "Holdout Win Rate": hm["Win Rate"],
            "Holdout ROI": hm["ROI"],
        })
    return pd.DataFrame(rows)


def _v35_final_gate(tiered, season_summary, holdout):
    if tiered is None or tiered.empty:
        return pd.DataFrame()

    official = tiered[tiered["verdict"].isin(["BEST BET", "BET"])].copy()
    dev = official[official["season"].astype(int).isin(V34_DEV_SEASONS)]
    hld = official[official["season"].astype(int) == int(holdout)]

    dm = _v34_metrics(dev)
    hm = _v34_metrics(hld)

    positive_dev = 0
    dev_seasons_present = 0
    if season_summary is not None and not season_summary.empty:
        d = season_summary[season_summary["Season"].astype(int).isin(V34_DEV_SEASONS)].copy()
        d = d[d["Official Bets"] > 0]
        dev_seasons_present = int(len(d))
        positive_dev = int((pd.to_numeric(d["ROI"], errors="coerce") > 0).sum())

    dev_pass = (
        dm["Bets"] >= 150
        and np.isfinite(dm["ROI"]) and dm["ROI"] > 0
        and np.isfinite(dm["Win Rate"]) and dm["Win Rate"] > V34_BREAKEVEN
        and dev_seasons_present >= 3
        and positive_dev >= 2
    )
    hold_pass = (
        hm["Bets"] >= 35
        and np.isfinite(hm["ROI"]) and hm["ROI"] > 0
        and np.isfinite(hm["Win Rate"]) and hm["Win Rate"] > V34_BREAKEVEN
    )
    drawdown_pass = (
        np.isfinite(dm["Max Drawdown"])
        and abs(dm["Max Drawdown"]) <= 20.0
    )

    if dev_pass and hold_pass and drawdown_pass:
        verdict = "PRODUCTION FINALIST — FORWARD TRACK"
    elif dev_pass:
        verdict = "DEVELOPMENT PASS / HOLDOUT FAIL"
    else:
        verdict = "KEEP IN RESEARCH"

    return pd.DataFrame([{
        "Architecture": "Balanced Ensemble",
        "Best Bet Score Floor": V35_BEST_BET_SCORE,
        "Bet Score Floor": V35_BET_SCORE,
        "Lean Score Floor": V35_LEAN_SCORE,
        "Development Official Bets": dm["Bets"],
        "Development Win Rate": dm["Win Rate"],
        "Development ROI": dm["ROI"],
        "Positive Development Seasons": f"{positive_dev}/{dev_seasons_present}",
        "Development Max Drawdown": dm["Max Drawdown"],
        "Holdout Official Bets": hm["Bets"],
        "Holdout Win Rate": hm["Win Rate"],
        "Holdout ROI": hm["ROI"],
        "Development Pass": bool(dev_pass),
        f"{holdout} Holdout Pass": bool(hold_pass),
        "Drawdown Pass": bool(drawdown_pass),
        "Verdict": verdict,
    }])


@st.cache_data(ttl=86400, show_spinner=False)
def _run_v35_adaptive_daily_card(test_seasons_tuple, scope, holdout, train_start):
    (
        history, reg, reg_summary, cls, cls_summary,
        bets_raw, bet_summary, preds, fixed_gate,
    ) = _run_v31_ml_bakeoff(
        tuple(sorted(set(int(s) for s in test_seasons_tuple))),
        scope,
        int(holdout),
        int(train_start),
    )

    v33 = _v33_rank_frame(history, preds)
    ranked = _v354_attach_daily_calendar(v33)
    tiered = _v35_apply_quality_tiers(ranked)

    daily, seasons, groups, holdout_tiers = _v35_daily_card_backtest(
        tiered, holdout
    )
    audit = _v35_threshold_audit(ranked, holdout)
    gate = _v35_final_gate(tiered, seasons, holdout)

    # Deep stage-by-stage diagnostics. These make the next failure actionable.
    cls_pred_rows = 0
    spread_cls_rows = 0
    if preds is not None and not preds.empty:
        cls_pred_rows = int((preds.get("task", pd.Series(index=preds.index, dtype=object)) == "classification").sum())
        try:
            spread_cls_rows = int((
                (preds["task"] == "classification")
                & (preds["market_type"] == "spread")
            ).sum())
        except Exception:
            spread_cls_rows = 0

    diag = pd.DataFrame([
        {"Check":"History rows", "Value": 0 if history is None else int(len(history))},
        {"Check":"Prediction rows", "Value": 0 if preds is None else int(len(preds))},
        {"Check":"Classification prediction rows", "Value": cls_pred_rows},
        {"Check":"Spread classification rows", "Value": spread_cls_rows},
        {"Check":"v3.3 selector rows", "Value": 0 if v33 is None else int(len(v33))},
        {"Check":"Ranked daily rows", "Value": 0 if ranked is None else int(len(ranked))},
        {"Check":"Verdict rows", "Value": 0 if tiered is None else int(len(tiered))},
    ])

    return (
        ranked, tiered, daily, seasons, groups,
        holdout_tiers, audit, gate, diag, v33
    )


def _render_v35_adaptive_daily_card(
    ranked, tiered, daily, seasons, groups,
    holdout_tiers, audit, gate, holdout,
    diagnostic=None, v33_frame=None
):
    st.markdown("#### v3.5 Adaptive Daily Card")
    st.caption(
        "The model ranks every game available that day and only recommends plays that clear fixed quality thresholds. "
        "A Friday night, bowl Tuesday, or full Saturday uses the same qualification bar. "
        "Day size changes organization only — never the number of required bets."
    )

    if tiered is None or tiered.empty:
        st.error(
            "v3.5.5 still did not produce daily-card rows. The stage-by-stage "
            "diagnostic below now shows exactly whether the loss occurred in "
            "history, predictions, the v3.3 selector, or the calendar join."
        )
        diag = diagnostic if diagnostic is not None else pd.DataFrame()
        if not diag.empty:
            st.dataframe(diag, use_container_width=True, hide_index=True)

        diag_bundle = _csv_download_bundle({
            "cfb_v355_stage_diagnostics.csv": diag,
            "cfb_v355_v33_selector_rows.csv": (
                v33_frame if v33_frame is not None else pd.DataFrame()
            ),
            "cfb_v355_ranked_rows.csv": (
                ranked if ranked is not None else pd.DataFrame()
            ),
            "cfb_v355_verdict_rows.csv": (
                tiered if tiered is not None else pd.DataFrame()
            ),
        })
        st.download_button(
            "Download v3.5.5 Diagnostic Bundle",
            data=diag_bundle,
            file_name="cfb_v355_daily_card_diagnostics.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_v354_diag",
        )
        return

    official = tiered[tiered["verdict"].isin(["BEST BET", "BET"])].copy()
    leans = tiered[tiered["verdict"] == "LEAN"].copy()
    passes = tiered[tiered["verdict"] == "PASS"].copy()

    st.markdown("##### Card behavior")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Official Bets", f"{len(official):,}")
    c2.metric("Best Bets", f"{int((tiered['verdict']=='BEST BET').sum()):,}")
    c3.metric("Leans", f"{len(leans):,}")
    c4.metric("Passes", f"{len(passes):,}")

    st.markdown("##### Locked thresholds")
    st.write(
        f"Best Bet ≥ {V35_BEST_BET_SCORE:.2f} score with directional agreement; "
        f"Bet ≥ {V35_BET_SCORE:.2f}; Lean ≥ {V35_LEAN_SCORE:.2f}. "
        "These floors do not drop on small days."
    )

    st.markdown("##### Final gate")
    show_gate = gate.copy()
    if not show_gate.empty:
        for c in ["Development Win Rate","Development ROI","Holdout Win Rate","Holdout ROI"]:
            if c in show_gate.columns:
                show_gate[c] = show_gate[c].map(
                    lambda v: "—" if pd.isna(v) else
                    (f"{100*float(v):+.1f}%" if "ROI" in c else f"{100*float(v):.1f}%")
                )
        if "Development Max Drawdown" in show_gate.columns:
            show_gate["Development Max Drawdown"] = show_gate["Development Max Drawdown"].map(
                lambda v: "—" if pd.isna(v) else f"{float(v):.1f}u"
            )
        st.dataframe(show_gate, use_container_width=True, hide_index=True)

        verdict = str(gate.iloc[0]["Verdict"])
        if verdict.startswith("PRODUCTION FINALIST"):
            st.success(
                "The fixed daily-card thresholds cleared the historical screen. "
                "Freeze them and begin 2026 forward tracking rather than re-optimizing."
            )
        elif "HOLDOUT FAIL" in verdict:
            st.warning(
                "The card looked acceptable in development but failed the holdout. "
                "Do not lower the thresholds just to create more action."
            )
        else:
            st.warning(
                "The card remains research-only. Weak days should continue to produce few or zero official bets."
            )

    st.markdown("##### Season-by-season")
    s = seasons.copy()
    if not s.empty:
        for c in ["Win Rate","ROI"]:
            s[c] = s[c].map(
                lambda v: "—" if pd.isna(v) else
                (f"{100*float(v):+.1f}%" if c == "ROI" else f"{100*float(v):.1f}%")
            )
        s["Max Drawdown"] = s["Max Drawdown"].map(
            lambda v: "—" if pd.isna(v) else f"{float(v):.1f}u"
        )
        st.dataframe(s, use_container_width=True, hide_index=True)

    st.markdown("##### Presentation groups")
    st.caption(
        "Small days stay as one Daily Slate. Medium/large days are grouped for readability only."
    )
    g = groups.copy()
    if not g.empty:
        for c in ["Win Rate","ROI"]:
            g[c] = g[c].map(
                lambda v: "—" if pd.isna(v) else
                (f"{100*float(v):+.1f}%" if c == "ROI" else f"{100*float(v):.1f}%")
            )
        st.dataframe(g, use_container_width=True, hide_index=True)

    with st.expander("Fixed score-bucket audit", expanded=False):
        a = audit.copy()
        if not a.empty:
            for c in ["Win Rate","ROI","Holdout Win Rate","Holdout ROI"]:
                a[c] = a[c].map(
                    lambda v: "—" if pd.isna(v) else
                    (f"{100*float(v):+.1f}%" if "ROI" in c else f"{100*float(v):.1f}%")
                )
        st.dataframe(a, use_container_width=True, hide_index=True)

    with st.expander("Daily card history", expanded=False):
        d = daily.copy()
        if not d.empty:
            for c in ["Win Rate","ROI"]:
                d[c] = d[c].map(
                    lambda v: "—" if pd.isna(v) else
                    (f"{100*float(v):+.1f}%" if c == "ROI" else f"{100*float(v):.1f}%")
                )
        st.dataframe(d, use_container_width=True, hide_index=True)

    st.markdown("##### v3.5 Result Bundle")
    bundle = _csv_download_bundle({
        "cfb_v355_final_gate.csv": gate,
        "cfb_v355_daily_card_summary.csv": daily,
        "cfb_v355_season_summary.csv": seasons,
        "cfb_v355_group_summary.csv": groups,
        "cfb_v355_holdout_tiers.csv": holdout_tiers,
        "cfb_v355_threshold_audit.csv": audit,
        "cfb_v355_all_ranked_games.csv": ranked,
        "cfb_v355_all_verdicts.csv": tiered,
        "cfb_v355_official_bets.csv": official,
    })
    st.download_button(
        "Download v3.5.5 Result Bundle",
        data=bundle,
        file_name="cfb_v355_adaptive_daily_card_bundle.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_v351_all",
    )
    st.caption(
        "This button appears only after v3.5.1 has produced ranked daily-card results. "
        "Upload the ZIP back to ChatGPT for final review."
    )

# ===== v2.4 current-production spread / total validation =====

VALIDATION_GAP_BUCKETS = [
    (0.0, 2.0, "0–1.9 pts"),
    (2.0, 3.0, "2–2.9 pts"),
    (3.0, 4.0, "3–3.9 pts"),
    (4.0, 6.0, "4–5.9 pts"),
    (6.0, 999.0, "6+ pts"),
]

VALIDATION_EDGE_BUCKETS = [
    (-999.0, 0.025, "<2.5%"),
    (0.025, 0.050, "2.5–4.9%"),
    (0.050, 0.075, "5.0–7.4%"),
    (0.075, 0.100, "7.5–9.9%"),
    (0.100, 999.0, "10%+"),
]


def _validation_bucket(value, buckets):
    try:
        x = float(value)
    except Exception:
        return "Unknown"
    for lo, hi, label in buckets:
        if lo <= x < hi:
            return label
    return buckets[-1][2]


def _validation_best_per_game_market(rows_df):
    """One strongest side per game per market, preserving spread + total separately."""
    if rows_df is None or rows_df.empty:
        return pd.DataFrame()
    d = rows_df.copy()
    d = d[d["market_type"].isin(["spread", "total"])].copy()
    if d.empty:
        return d

    rank = {"STRONG BET": 4, "BET": 3, "LEAN": 2, "PASS": 1}
    d["_rank"] = d["verdict"].map(rank).fillna(0)
    d = d.sort_values(
        ["season", "game_id", "market_type", "_rank", "edge", "ev"],
        ascending=[True, True, True, False, False, False],
    )
    return (
        d.groupby(["season", "game_id", "market_type"], as_index=False)
        .head(1)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def _validation_projection_rows(game_df):
    if game_df is None or game_df.empty:
        return pd.DataFrame()

    rows = []
    for season in sorted(game_df["season"].dropna().astype(int).unique()):
        d = game_df[game_df["season"] == season].copy()

        # Spread
        s = d.dropna(
            subset=["home_points", "away_points", "market_home_spread", "adjusted_model_home_spread"]
        ).copy()
        if not s.empty:
            actual = s["home_points"].astype(float) - s["away_points"].astype(float)
            market = -s["market_home_spread"].astype(float)
            model = -s["adjusted_model_home_spread"].astype(float)
            raw = -s["raw_model_home_spread"].astype(float)
            rows.append({
                "Market": "SPREAD",
                "Season": season,
                "Games": len(s),
                "Market MAE": float((actual-market).abs().mean()),
                "Adjusted Model MAE": float((actual-model).abs().mean()),
                "Raw Model MAE": float((actual-raw).abs().mean()),
                "Improvement vs Market": float((actual-market).abs().mean() - (actual-model).abs().mean()),
            })

        # Total
        t = d.dropna(
            subset=["home_points", "away_points", "market_total", "adjusted_model_total"]
        ).copy()
        if not t.empty:
            actual = t["home_points"].astype(float) + t["away_points"].astype(float)
            market = t["market_total"].astype(float)
            model = t["adjusted_model_total"].astype(float)
            raw = t["raw_model_total"].astype(float)
            rows.append({
                "Market": "TOTAL",
                "Season": season,
                "Games": len(t),
                "Market MAE": float((actual-market).abs().mean()),
                "Adjusted Model MAE": float((actual-model).abs().mean()),
                "Raw Model MAE": float((actual-raw).abs().mean()),
                "Improvement vs Market": float((actual-market).abs().mean() - (actual-model).abs().mean()),
            })

    return pd.DataFrame(rows)


def _validation_betting_summary(df, group_cols):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, g in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        w = int((g["result"] == "WIN").sum())
        l = int((g["result"] == "LOSS").sum())
        p = int((g["result"] == "PUSH").sum())
        decisions = w + l
        units = float(g["profit_units"].sum())
        bets = len(g)
        row = {c:k for c,k in zip(group_cols, keys)}
        row.update({
            "Bets": int(bets),
            "Record": f"{w}-{l}" + (f"-{p}P" if p else ""),
            "Win %": (w / decisions) if decisions else None,
            "Units": units,
            "ROI": (units / bets) if bets else None,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_gate(projection_df, picks_df, holdout):
    rows = []
    for market in ["SPREAD", "TOTAL"]:
        mkey = market.lower()
        proj = projection_df[projection_df["Market"] == market].copy()
        proj = proj.sort_values("Season")
        hold = proj[proj["Season"] == int(holdout)]

        season_wins = int((proj["Improvement vs Market"] > 0).sum()) if not proj.empty else 0
        season_count = len(proj)
        recent = proj.tail(min(3, season_count))
        recent_wins = int((recent["Improvement vs Market"] > 0).sum()) if not recent.empty else 0

        hold_mae_pass = bool(
            not hold.empty and float(hold.iloc[0]["Improvement vs Market"]) > 0
        )
        projection_pass = bool(
            hold_mae_pass
            and season_count >= 3
            and recent_wins >= 2
        )

        official = picks_df[
            (picks_df["market_type"] == mkey)
            & (picks_df["verdict"].isin(["BET", "STRONG BET"]))
        ].copy()

        all_s = _validation_betting_summary(official, ["market_type"])
        hold_s = _validation_betting_summary(
            official[official["season"] == int(holdout)],
            ["market_type"],
        )
        season_s = _validation_betting_summary(official, ["season"])

        all_bets = int(all_s.iloc[0]["Bets"]) if not all_s.empty else 0
        all_roi = float(all_s.iloc[0]["ROI"]) if not all_s.empty and pd.notna(all_s.iloc[0]["ROI"]) else 0.0
        hold_bets = int(hold_s.iloc[0]["Bets"]) if not hold_s.empty else 0
        hold_roi = float(hold_s.iloc[0]["ROI"]) if not hold_s.empty and pd.notna(hold_s.iloc[0]["ROI"]) else 0.0
        positive_seasons = int((season_s["ROI"] > 0).sum()) if not season_s.empty else 0
        tested_bet_seasons = len(season_s)

        # Fixed before seeing results; deliberately conservative.
        betting_pass = bool(
            all_bets >= 100
            and all_roi > 0
            and hold_bets >= 25
            and hold_roi >= 0
            and tested_bet_seasons >= 3
            and positive_seasons >= 2
        )

        if projection_pass and betting_pass:
            status = "PASS TO PROMOTION REVIEW"
        elif projection_pass:
            status = "PROJECTION PASS / BETTING FAIL"
        elif betting_pass:
            status = "BETTING PASS / PROJECTION FAIL"
        else:
            status = "KEEP IN RESEARCH"

        rows.append({
            "Market": market,
            "MAE Seasons Won": f"{season_wins}/{season_count}",
            "Recent MAE Wins": f"{recent_wins}/{len(recent)}",
            f"{holdout} MAE Beats Market": "YES" if hold_mae_pass else "NO",
            "Official Bets": all_bets,
            "Official ROI": all_roi,
            f"{holdout} Bets": hold_bets,
            f"{holdout} ROI": hold_roi,
            "Positive ROI Seasons": f"{positive_seasons}/{tested_bet_seasons}",
            "Projection Gate": "PASS" if projection_pass else "FAIL",
            "Betting Gate": "PASS" if betting_pass else "FAIL",
            "Status": status,
        })

    return pd.DataFrame(rows)


def _run_current_market_validation(seasons, scope, holdout, progress=None):
    """Evaluate the exact current v0.4 residual-market spread/total layer."""
    game_rows = []
    candidate_rows = []
    seasons = sorted(set(int(s) for s in seasons))

    for si, season in enumerate(seasons):
        if progress is not None:
            progress.progress(
                si / max(1, len(seasons)),
                text=f"Loading {season} games, consensus lines and preseason-safe inputs…",
            )

        games = get_backtest_games(season)
        line_payload = get_backtest_lines(season)
        data_full = get_backtest_model_data(season)
        data = _bt_prior_only_data(data_full)

        line_index = {}
        for lr in line_payload or []:
            gid = lr.get("id")
            if gid is None:
                continue
            try:
                key = int(gid)
            except Exception:
                key = gid
            line_index[key] = normalize_game_lines([lr], game_id=gid)

        season_games = [
            g for g in games or []
            if g.get("completed") is True
            and g.get("homePoints") is not None
            and g.get("awayPoints") is not None
            and _bt_game_scope(g, scope)
        ]

        for g in season_games:
            gid = g.get("id")
            try:
                lookup_gid = int(gid)
            except Exception:
                lookup_gid = gid

            market = _bt_consensus_line(line_index.get(lookup_gid, []))
            if not market:
                continue

            try:
                p = _bt_project_game(g, data, hfa=DEFAULT_HFA)
            except Exception:
                continue

            residual_models = fit_residual_models_before_season(int(season), scope)
            residual_p = residual_market_projection(p, market, residual_models)
            adj_spread = residual_p["adjusted_home_spread"]
            adj_total = residual_p["adjusted_total"]

            game_rows.append({
                "season": season,
                "week": p["week"],
                "game_id": gid,
                "away_team": p["away"],
                "home_team": p["home"],
                "away_conference": g.get("awayConference"),
                "home_conference": g.get("homeConference"),
                "away_points": float(g["awayPoints"]),
                "home_points": float(g["homePoints"]),
                "raw_model_home_spread": float(p["model_home_spread"]),
                "adjusted_model_home_spread": float(adj_spread),
                "market_home_spread": market.get("home_spread"),
                "raw_model_total": float(p["model_total"]),
                "adjusted_model_total": float(adj_total),
                "market_total": market.get("total"),
                "confidence": float(p["confidence"]),

                # Core residual-model decomposition
                "base_power_margin": float(p["components"].get("base_power_margin", 0.0)),
                "matchup_margin_adjustment": float(p["components"].get("matchup_margin_adjustment", 0.0)),
                "hfa_adjustment": float(p["components"].get("hfa_adjustment", 0.0)),
                "sp_total_base": float(p["components"].get("sp_total_base", 0.0)),
                "efficiency_total_adjustment": float(p["components"].get("efficiency_total_adjustment", 0.0)),
                "pace_total_adjustment": float(p["components"].get("pace_total_adjustment", 0.0)),

                # Team power / personnel
                "away_sp_rating": float(p["away_rating"].get("sp_rating") or 0.0),
                "home_sp_rating": float(p["home_rating"].get("sp_rating") or 0.0),
                "away_srs_adjustment": float(p["away_rating"].get("srs_adjustment") or 0.0),
                "home_srs_adjustment": float(p["home_rating"].get("srs_adjustment") or 0.0),
                "away_talent_adjustment": float(p["away_rating"].get("talent_adjustment") or 0.0),
                "home_talent_adjustment": float(p["home_rating"].get("talent_adjustment") or 0.0),
                "away_returning_adjustment": float(p["away_rating"].get("returning_adjustment") or 0.0),
                "home_returning_adjustment": float(p["home_rating"].get("returning_adjustment") or 0.0),
                "away_returning_pass": p["away_rating"].get("returning_pass"),
                "home_returning_pass": p["home_rating"].get("returning_pass"),
                "away_returning_usage": p["away_rating"].get("returning_usage"),
                "home_returning_usage": p["home_rating"].get("returning_usage"),

                # Prior-season PPA matchup inputs in leakage-safe validation mode
                "away_ppa_off_pass": p["away_rating"]["ppa"].get("off_pass"),
                "home_ppa_off_pass": p["home_rating"]["ppa"].get("off_pass"),
                "away_ppa_def_pass": p["away_rating"]["ppa"].get("def_pass"),
                "home_ppa_def_pass": p["home_rating"]["ppa"].get("def_pass"),
                "away_ppa_off_rush": p["away_rating"]["ppa"].get("off_rush"),
                "home_ppa_off_rush": p["home_rating"]["ppa"].get("off_rush"),
                "away_ppa_def_rush": p["away_rating"]["ppa"].get("def_rush"),
                "home_ppa_def_rush": p["home_rating"]["ppa"].get("def_rush"),

                # Advanced efficiency / explosiveness / finishing / pace
                "away_adv_off_success": p["away_rating"]["adv"].get("off_success"),
                "home_adv_off_success": p["home_rating"]["adv"].get("off_success"),
                "away_adv_def_success": p["away_rating"]["adv"].get("def_success"),
                "home_adv_def_success": p["home_rating"]["adv"].get("def_success"),
                "away_adv_off_expl": p["away_rating"]["adv"].get("off_expl"),
                "home_adv_off_expl": p["home_rating"]["adv"].get("off_expl"),
                "away_adv_def_expl": p["away_rating"]["adv"].get("def_expl"),
                "home_adv_def_expl": p["home_rating"]["adv"].get("def_expl"),
                "away_adv_off_pass_ppa": p["away_rating"]["adv"].get("off_pass_ppa"),
                "home_adv_off_pass_ppa": p["home_rating"]["adv"].get("off_pass_ppa"),
                "away_adv_def_pass_ppa": p["away_rating"]["adv"].get("def_pass_ppa"),
                "home_adv_def_pass_ppa": p["home_rating"]["adv"].get("def_pass_ppa"),
                "away_adv_off_rush_ppa": p["away_rating"]["adv"].get("off_rush_ppa"),
                "home_adv_off_rush_ppa": p["home_rating"]["adv"].get("off_rush_ppa"),
                "away_adv_def_rush_ppa": p["away_rating"]["adv"].get("def_rush_ppa"),
                "home_adv_def_rush_ppa": p["home_rating"]["adv"].get("def_rush_ppa"),
                "away_adv_off_ppo": p["away_rating"]["adv"].get("off_ppo"),
                "home_adv_off_ppo": p["home_rating"]["adv"].get("off_ppo"),
                "away_adv_def_ppo": p["away_rating"]["adv"].get("def_ppo"),
                "home_adv_def_ppo": p["home_rating"]["adv"].get("def_ppo"),
                "away_adv_def_havoc": p["away_rating"]["adv"].get("def_havoc"),
                "home_adv_def_havoc": p["home_rating"]["adv"].get("def_havoc"),
                "away_adv_off_plays": p["away_rating"]["adv"].get("off_plays"),
                "home_adv_off_plays": p["home_rating"]["adv"].get("off_plays"),
                "away_adv_off_drives": p["away_rating"]["adv"].get("off_drives"),
                "home_adv_off_drives": p["home_rating"]["adv"].get("off_drives"),
            })

            rows = _bt_candidate_rows(g, p, market, season, "v0.4.0")
            candidate_rows.extend(
                r for r in rows if r.get("market_type") in ("spread", "total")
            )

    if progress is not None:
        progress.progress(1.0, text="Validation complete.")

    games_df = pd.DataFrame(game_rows)
    raw_candidates = pd.DataFrame(candidate_rows)
    picks_df = _validation_best_per_game_market(raw_candidates)

    if not picks_df.empty:
        def gap_for_row(r):
            # v2.6: diagnose the residual correction actually applied to market.
            # For spread, fair spread - market spread has the same absolute size
            # as the predicted market-margin residual. For totals it is direct.
            if r["market_type"] == "spread":
                return abs(float(r["adjusted_model_home_spread"]) - float(r["market_home_spread"]))
            return abs(float(r["adjusted_model_total"]) - float(r["market_total"]))

        picks_df["projection_gap_pts"] = picks_df.apply(gap_for_row, axis=1)
        picks_df["gap_bucket"] = picks_df["projection_gap_pts"].apply(
            lambda x: _validation_bucket(x, VALIDATION_GAP_BUCKETS)
        )
        picks_df["edge_bucket"] = picks_df["edge"].apply(
            lambda x: _validation_bucket(x, VALIDATION_EDGE_BUCKETS)
        )

    projection_df = _validation_projection_rows(games_df)
    gate_df = _validation_gate(projection_df, picks_df, holdout)

    return games_df, picks_df, projection_df, gate_df



# ===== v2.6 residual feature audit =====
FEATURE_AUDIT_VERSION = "v2.6.0-feature-audit"
FEATURE_AUDIT_ALPHA = 12.0
FEATURE_AUDIT_MIN_TRAIN = 300

# Groups are intentionally interpretable. We remove one group at a time from
# the full model to see whether the group improves unseen-season MAE.
SPREAD_FEATURE_GROUPS = {
    "Raw model disagreement": ["spread_model_delta"],
    "SP+ power": ["sp_rating_diff"],
    "SRS": ["srs_adjustment_diff"],
    "Talent": ["talent_adjustment_diff"],
    "Returning production": [
        "returning_adjustment_diff",
        "returning_pass_diff",
        "returning_usage_diff",
    ],
    "Aggregate matchup": ["sp_matchup_adj"],
    "PPA pass/rush matchup": ["net_pass_matchup", "net_rush_matchup"],
    "Advanced efficiency": [
        "net_success_matchup",
        "net_expl_matchup",
        "net_adv_pass_matchup",
        "net_adv_rush_matchup",
        "net_finishing_matchup",
        "havoc_diff",
    ],
    "Home field": ["sp_hfa"],
    "Market favorite size": ["market_margin", "abs_market_margin"],
    "Week / early season": ["week_num", "early_week_flag"],
    "Model confidence": ["confidence_num"],
}

TOTAL_FEATURE_GROUPS = {
    "Raw total disagreement": ["total_model_delta"],
    "SP+ total base": ["total_base_minus_market"],
    "Efficiency adjustment": ["total_eff_adj"],
    "Pace adjustment": ["total_pace_adj"],
    "PPA pass/rush matchup": ["abs_net_pass_matchup", "abs_net_rush_matchup"],
    "Advanced efficiency": [
        "abs_net_success_matchup",
        "abs_net_expl_matchup",
        "abs_net_finishing_matchup",
    ],
    "Game pace": ["avg_plays_per_drive"],
    "Market total": ["market_total"],
    "Week / early season": ["week_num", "early_week_flag"],
    "Model confidence": ["confidence_num"],
}


def _feature_audit_frame(game_df):
    """Build leakage-safe, interpretable residual features from validation rows."""
    if game_df is None or game_df.empty:
        return pd.DataFrame()

    d = game_df.copy()
    d = d.sort_values(["season", "game_id"]).drop_duplicates(["season", "game_id"])

    def n(col):
        if col not in d.columns:
            d[col] = np.nan
        return pd.to_numeric(d[col], errors="coerce")

    d["actual_margin"] = n("home_points") - n("away_points")
    d["actual_total"] = n("home_points") + n("away_points")
    d["market_margin"] = -n("market_home_spread")
    d["raw_model_margin"] = -n("raw_model_home_spread")

    d["spread_target_residual"] = d["actual_margin"] - d["market_margin"]
    d["spread_model_delta"] = d["raw_model_margin"] - d["market_margin"]
    d["total_target_residual"] = d["actual_total"] - n("market_total")
    d["total_model_delta"] = n("raw_model_total") - n("market_total")

    d["abs_market_margin"] = d["market_margin"].abs()
    d["week_num"] = n("week").clip(lower=1, upper=16)
    d["early_week_flag"] = (d["week_num"] <= 3).astype(float)
    d["confidence_num"] = n("confidence")

    d["sp_rating_diff"] = n("home_sp_rating") - n("away_sp_rating")
    d["srs_adjustment_diff"] = n("home_srs_adjustment") - n("away_srs_adjustment")
    d["talent_adjustment_diff"] = n("home_talent_adjustment") - n("away_talent_adjustment")
    d["returning_adjustment_diff"] = n("home_returning_adjustment") - n("away_returning_adjustment")
    d["returning_pass_diff"] = n("home_returning_pass") - n("away_returning_pass")
    d["returning_usage_diff"] = n("home_returning_usage") - n("away_returning_usage")

    d["sp_matchup_adj"] = n("matchup_margin_adjustment")
    d["sp_hfa"] = n("hfa_adjustment")

    d["total_base_minus_market"] = n("sp_total_base") - n("market_total")
    d["total_eff_adj"] = n("efficiency_total_adjustment")
    d["total_pace_adj"] = n("pace_total_adjustment")

    # PPA matchup: home offense vs away defense minus away offense vs home defense.
    d["home_pass_matchup"] = n("home_ppa_off_pass") - n("away_ppa_def_pass")
    d["away_pass_matchup"] = n("away_ppa_off_pass") - n("home_ppa_def_pass")
    d["net_pass_matchup"] = d["home_pass_matchup"] - d["away_pass_matchup"]

    d["home_rush_matchup"] = n("home_ppa_off_rush") - n("away_ppa_def_rush")
    d["away_rush_matchup"] = n("away_ppa_off_rush") - n("home_ppa_def_rush")
    d["net_rush_matchup"] = d["home_rush_matchup"] - d["away_rush_matchup"]

    d["home_success_matchup"] = n("home_adv_off_success") - n("away_adv_def_success")
    d["away_success_matchup"] = n("away_adv_off_success") - n("home_adv_def_success")
    d["net_success_matchup"] = d["home_success_matchup"] - d["away_success_matchup"]

    d["home_expl_matchup"] = n("home_adv_off_expl") - n("away_adv_def_expl")
    d["away_expl_matchup"] = n("away_adv_off_expl") - n("home_adv_def_expl")
    d["net_expl_matchup"] = d["home_expl_matchup"] - d["away_expl_matchup"]

    d["home_adv_pass_matchup"] = n("home_adv_off_pass_ppa") - n("away_adv_def_pass_ppa")
    d["away_adv_pass_matchup"] = n("away_adv_off_pass_ppa") - n("home_adv_def_pass_ppa")
    d["net_adv_pass_matchup"] = d["home_adv_pass_matchup"] - d["away_adv_pass_matchup"]

    d["home_adv_rush_matchup"] = n("home_adv_off_rush_ppa") - n("away_adv_def_rush_ppa")
    d["away_adv_rush_matchup"] = n("away_adv_off_rush_ppa") - n("home_adv_def_rush_ppa")
    d["net_adv_rush_matchup"] = d["home_adv_rush_matchup"] - d["away_adv_rush_matchup"]

    d["home_finishing_matchup"] = n("home_adv_off_ppo") - n("away_adv_def_ppo")
    d["away_finishing_matchup"] = n("away_adv_off_ppo") - n("home_adv_def_ppo")
    d["net_finishing_matchup"] = d["home_finishing_matchup"] - d["away_finishing_matchup"]
    d["havoc_diff"] = n("home_adv_def_havoc") - n("away_adv_def_havoc")

    away_ppd = n("away_adv_off_plays") / n("away_adv_off_drives").replace(0, np.nan)
    home_ppd = n("home_adv_off_plays") / n("home_adv_off_drives").replace(0, np.nan)
    d["avg_plays_per_drive"] = (away_ppd + home_ppd) / 2.0

    # For total, direction is less important than mismatch magnitude.
    for c in [
        "net_pass_matchup",
        "net_rush_matchup",
        "net_success_matchup",
        "net_expl_matchup",
        "net_finishing_matchup",
    ]:
        d["abs_" + c] = d[c].abs()

    return d


def _flatten_feature_groups(groups):
    out = []
    for cols in groups.values():
        for c in cols:
            if c not in out:
                out.append(c)
    return out


def _audit_fit_predict(train, test, features, target):
    """Fit fixed-alpha ridge on train only and score unseen test rows."""
    usable = [f for f in features if f in train.columns and f in test.columns]
    if not usable:
        return None

    tr = train.dropna(subset=usable + [target]).copy()
    te = test.dropna(subset=usable + [target]).copy()
    if len(tr) < FEATURE_AUDIT_MIN_TRAIN or te.empty:
        return None

    fit = _ridge_fit_numpy(tr[usable].values, tr[target].values, FEATURE_AUDIT_ALPHA)
    preds = np.array([
        _ridge_predict_numpy(fit, row)
        for row in te[usable].to_numpy(dtype=float)
    ])
    actual = te[target].to_numpy(dtype=float)

    return {
        "n_train": len(tr),
        "n_test": len(te),
        "mae": float(np.mean(np.abs(actual - preds))),
        "market_mae": float(np.mean(np.abs(actual))),
        "preds": preds,
        "actual": actual,
        "index": te.index.to_numpy(),
    }


def _walkforward_feature_audit(feature_df, test_seasons, market_type, holdout):
    """
    Full-model and leave-one-group-out audit.

    Each test year is predicted exactly once using only rows from earlier seasons.
    No feature group is judged on in-sample fit.
    """
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if market_type == "spread":
        groups = SPREAD_FEATURE_GROUPS
        target = "spread_target_residual"
    else:
        groups = TOTAL_FEATURE_GROUPS
        target = "total_target_residual"

    full_features = _flatten_feature_groups(groups)
    season_rows = []
    ablation_rows = []
    prediction_rows = []

    for season in sorted(set(int(s) for s in test_seasons)):
        train = feature_df[feature_df["season"].astype(int) < season].copy()
        test = feature_df[feature_df["season"].astype(int) == season].copy()

        full = _audit_fit_predict(train, test, full_features, target)
        if full is None:
            continue

        season_rows.append({
            "Market": market_type.upper(),
            "Test Season": season,
            "Games": full["n_test"],
            "Train Games": full["n_train"],
            "Market MAE": full["market_mae"],
            "Full Model MAE": full["mae"],
            "Improvement vs Market": full["market_mae"] - full["mae"],
            "Holdout": "YES" if season == int(holdout) else "NO",
        })

        for idx, y, pred in zip(full["index"], full["actual"], full["preds"]):
            prediction_rows.append({
                "market_type": market_type,
                "season": season,
                "row_index": int(idx),
                "actual_market_residual": float(y),
                "predicted_residual": float(pred),
                "market_abs_error": abs(float(y)),
                "model_abs_error": abs(float(y) - float(pred)),
            })

        # Leave one interpretable group out at a time.
        for group_name, cols in groups.items():
            reduced = [f for f in full_features if f not in cols]
            ab = _audit_fit_predict(train, test, reduced, target)
            if ab is None:
                continue
            # Positive = full model is better, therefore the removed group helped.
            group_value = ab["mae"] - full["mae"]
            ablation_rows.append({
                "Market": market_type.upper(),
                "Test Season": season,
                "Feature Group": group_name,
                "Games": min(full["n_test"], ab["n_test"]),
                "Full Model MAE": full["mae"],
                "MAE Without Group": ab["mae"],
                "Feature Value": group_value,
                "Helps": "YES" if group_value > 0 else "NO",
                "Holdout": "YES" if season == int(holdout) else "NO",
            })

    return pd.DataFrame(season_rows), pd.DataFrame(ablation_rows), pd.DataFrame(prediction_rows)


def _feature_group_summary(ablation_df, holdout):
    if ablation_df is None or ablation_df.empty:
        return pd.DataFrame()

    rows = []
    for (market, group), g in ablation_df.groupby(["Market", "Feature Group"]):
        vals = pd.to_numeric(g["Feature Value"], errors="coerce").dropna()
        hold = g[g["Test Season"].astype(int) == int(holdout)]
        hold_val = (
            float(hold.iloc[0]["Feature Value"])
            if not hold.empty and pd.notna(hold.iloc[0]["Feature Value"])
            else np.nan
        )
        positive = int((vals > 0).sum())
        total = int(len(vals))
        avg = float(vals.mean()) if total else np.nan

        # Conservative audit label. No automatic production promotion.
        if total >= 3 and positive >= max(2, total - 1) and pd.notna(hold_val) and hold_val > 0 and avg > 0:
            verdict = "KEEP"
        elif pd.notna(hold_val) and hold_val < 0 and avg < 0:
            verdict = "REMOVE CANDIDATE"
        else:
            verdict = "MIXED"

        rows.append({
            "Market": market,
            "Feature Group": group,
            "Seasons Helped": f"{positive}/{total}",
            "Avg MAE Value": avg,
            f"{holdout} Holdout Value": hold_val,
            "Audit Verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values(
        ["Market", "Avg MAE Value"],
        ascending=[True, False],
    ).reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def _feature_audit_history(min_season, max_season, scope):
    """
    Build the complete point-in-time-safe historical frame needed for audit.
    Includes pre-test seasons so each selected test season can train only on past data.
    """
    rows = []
    for season in range(int(min_season), int(max_season) + 1):
        games = get_backtest_games(season)
        line_payload = get_backtest_lines(season)
        data = _bt_prior_only_data(get_backtest_model_data(season))

        line_index = {}
        for lr in line_payload or []:
            gid = lr.get("id")
            if gid is None:
                continue
            try:
                key = int(gid)
            except Exception:
                key = gid
            line_index[key] = normalize_game_lines([lr], game_id=gid)

        for g in games or []:
            if g.get("completed") is not True:
                continue
            if g.get("homePoints") is None or g.get("awayPoints") is None:
                continue
            if not _bt_game_scope(g, scope):
                continue

            gid = g.get("id")
            try:
                key = int(gid)
            except Exception:
                key = gid
            market = _bt_consensus_line(line_index.get(key, []))
            if not market:
                continue

            try:
                p = _bt_project_game(g, data, hfa=DEFAULT_HFA)
            except Exception:
                continue

            rows.append({
                "season": season,
                "week": p["week"],
                "game_id": gid,
                "away_team": p["away"],
                "home_team": p["home"],
                "away_points": float(g["awayPoints"]),
                "home_points": float(g["homePoints"]),
                "raw_model_home_spread": float(p["model_home_spread"]),
                "market_home_spread": market.get("home_spread"),
                "raw_model_total": float(p["model_total"]),
                "market_total": market.get("total"),
                "confidence": float(p["confidence"]),
                "base_power_margin": float(p["components"].get("base_power_margin", 0.0)),
                "matchup_margin_adjustment": float(p["components"].get("matchup_margin_adjustment", 0.0)),
                "hfa_adjustment": float(p["components"].get("hfa_adjustment", 0.0)),
                "sp_total_base": float(p["components"].get("sp_total_base", 0.0)),
                "efficiency_total_adjustment": float(p["components"].get("efficiency_total_adjustment", 0.0)),
                "pace_total_adjustment": float(p["components"].get("pace_total_adjustment", 0.0)),

                "away_sp_rating": float(p["away_rating"].get("sp_rating") or 0.0),
                "home_sp_rating": float(p["home_rating"].get("sp_rating") or 0.0),
                "away_srs_adjustment": float(p["away_rating"].get("srs_adjustment") or 0.0),
                "home_srs_adjustment": float(p["home_rating"].get("srs_adjustment") or 0.0),
                "away_talent_adjustment": float(p["away_rating"].get("talent_adjustment") or 0.0),
                "home_talent_adjustment": float(p["home_rating"].get("talent_adjustment") or 0.0),
                "away_returning_adjustment": float(p["away_rating"].get("returning_adjustment") or 0.0),
                "home_returning_adjustment": float(p["home_rating"].get("returning_adjustment") or 0.0),
                "away_returning_pass": p["away_rating"].get("returning_pass"),
                "home_returning_pass": p["home_rating"].get("returning_pass"),
                "away_returning_usage": p["away_rating"].get("returning_usage"),
                "home_returning_usage": p["home_rating"].get("returning_usage"),

                "away_ppa_off_pass": p["away_rating"]["ppa"].get("off_pass"),
                "home_ppa_off_pass": p["home_rating"]["ppa"].get("off_pass"),
                "away_ppa_def_pass": p["away_rating"]["ppa"].get("def_pass"),
                "home_ppa_def_pass": p["home_rating"]["ppa"].get("def_pass"),
                "away_ppa_off_rush": p["away_rating"]["ppa"].get("off_rush"),
                "home_ppa_off_rush": p["home_rating"]["ppa"].get("off_rush"),
                "away_ppa_def_rush": p["away_rating"]["ppa"].get("def_rush"),
                "home_ppa_def_rush": p["home_rating"]["ppa"].get("def_rush"),

                "away_adv_off_success": p["away_rating"]["adv"].get("off_success"),
                "home_adv_off_success": p["home_rating"]["adv"].get("off_success"),
                "away_adv_def_success": p["away_rating"]["adv"].get("def_success"),
                "home_adv_def_success": p["home_rating"]["adv"].get("def_success"),
                "away_adv_off_expl": p["away_rating"]["adv"].get("off_expl"),
                "home_adv_off_expl": p["home_rating"]["adv"].get("off_expl"),
                "away_adv_def_expl": p["away_rating"]["adv"].get("def_expl"),
                "home_adv_def_expl": p["home_rating"]["adv"].get("def_expl"),
                "away_adv_off_pass_ppa": p["away_rating"]["adv"].get("off_pass_ppa"),
                "home_adv_off_pass_ppa": p["home_rating"]["adv"].get("off_pass_ppa"),
                "away_adv_def_pass_ppa": p["away_rating"]["adv"].get("def_pass_ppa"),
                "home_adv_def_pass_ppa": p["home_rating"]["adv"].get("def_pass_ppa"),
                "away_adv_off_rush_ppa": p["away_rating"]["adv"].get("off_rush_ppa"),
                "home_adv_off_rush_ppa": p["home_rating"]["adv"].get("off_rush_ppa"),
                "away_adv_def_rush_ppa": p["away_rating"]["adv"].get("def_rush_ppa"),
                "home_adv_def_rush_ppa": p["home_rating"]["adv"].get("def_rush_ppa"),
                "away_adv_off_ppo": p["away_rating"]["adv"].get("off_ppo"),
                "home_adv_off_ppo": p["home_rating"]["adv"].get("off_ppo"),
                "away_adv_def_ppo": p["away_rating"]["adv"].get("def_ppo"),
                "home_adv_def_ppo": p["home_rating"]["adv"].get("def_ppo"),
                "away_adv_def_havoc": p["away_rating"]["adv"].get("def_havoc"),
                "home_adv_def_havoc": p["home_rating"]["adv"].get("def_havoc"),
                "away_adv_off_plays": p["away_rating"]["adv"].get("off_plays"),
                "home_adv_off_plays": p["home_rating"]["adv"].get("off_plays"),
                "away_adv_off_drives": p["away_rating"]["adv"].get("off_drives"),
                "home_adv_off_drives": p["home_rating"]["adv"].get("off_drives"),
            })

    return _feature_audit_frame(pd.DataFrame(rows))


def _run_feature_audit(test_seasons, scope, holdout, progress=None):
    test_seasons = sorted(set(int(s) for s in test_seasons))
    if not test_seasons:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Give the first selected test year a real prior training history.
    min_history = min(RESIDUAL_TRAIN_START, min(test_seasons) - 4)
    min_history = max(2014, min_history)
    max_history = max(test_seasons)

    if progress is not None:
        progress.progress(0.10, text="Building leakage-safe feature history…")
    feat = _feature_audit_history(min_history, max_history, scope)

    if progress is not None:
        progress.progress(0.45, text="Running spread leave-one-feature-group-out tests…")
    sp_seasons, sp_ablation, sp_preds = _walkforward_feature_audit(
        feat, test_seasons, "spread", holdout
    )

    if progress is not None:
        progress.progress(0.70, text="Running total feature audit (research-only)…")
    tot_seasons, tot_ablation, tot_preds = _walkforward_feature_audit(
        feat, test_seasons, "total", holdout
    )

    season_df = pd.concat([sp_seasons, tot_seasons], ignore_index=True)
    ablation_df = pd.concat([sp_ablation, tot_ablation], ignore_index=True)
    preds_df = pd.concat([sp_preds, tot_preds], ignore_index=True)
    summary_df = _feature_group_summary(ablation_df, holdout)

    if progress is not None:
        progress.progress(1.0, text="Feature audit complete.")

    return season_df, ablation_df, summary_df, preds_df


def _render_feature_audit_results(season_df, ablation_df, summary_df, preds_df, holdout):
    st.markdown("#### Walk-Forward Audit Results")
    st.caption(
        "Positive feature value means MAE got worse when that feature group was removed, "
        "so the group added out-of-sample value. Negative means removing it improved the model."
    )

    if season_df is None or season_df.empty:
        st.info("Run the feature audit to test which signals actually improve unseen-season residual accuracy.")
        return

    show = season_df.copy()
    for c in ["Market MAE", "Full Model MAE", "Improvement vs Market"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}" if c == "Improvement vs Market" else f"{v:.4f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("#### Feature group verdicts")
    summ = summary_df.copy()
    for c in ["Avg MAE Value", f"{holdout} Holdout Value"]:
        if c in summ.columns:
            summ[c] = summ[c].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "—")
    st.dataframe(summ, use_container_width=True, hide_index=True)

    spread_keep = summary_df[
        (summary_df["Market"] == "SPREAD") &
        (summary_df["Audit Verdict"] == "KEEP")
    ]["Feature Group"].tolist()

    spread_remove = summary_df[
        (summary_df["Market"] == "SPREAD") &
        (summary_df["Audit Verdict"] == "REMOVE CANDIDATE")
    ]["Feature Group"].tolist()

    if spread_keep:
        st.success("Spread groups that earned KEEP status: " + ", ".join(spread_keep))
    else:
        st.warning("No spread feature group earned automatic KEEP status yet.")

    if spread_remove:
        st.warning("Spread removal candidates: " + ", ".join(spread_remove))

    with st.expander("Detailed leave-one-group-out results", expanded=False):
        detail = ablation_df.copy()
        for c in ["Full Model MAE", "MAE Without Group", "Feature Value"]:
            detail[c] = detail[c].map(lambda v: f"{v:+.4f}" if c == "Feature Value" else f"{v:.4f}")
        st.dataframe(detail, use_container_width=True, hide_index=True)

    with st.expander("Feature Audit Downloads", expanded=False):
        feature_zip = _csv_download_bundle({
            "cfb_feature_audit_summary.csv": summary_df,
            "cfb_feature_audit_seasons.csv": season_df,
            "cfb_feature_audit_ablation.csv": ablation_df,
            "cfb_feature_audit_predictions.csv": preds_df,
        })
        st.download_button(
            "Download All 4 Feature Audit Files",
            data=feature_zip,
            file_name="cfb_feature_audit_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_feature_audit_all_zip",
        )
        st.download_button(
            "Download Feature Summary",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_feature_audit_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_feature_audit_summary",
        )
        st.download_button(
            "Download Season Results",
            data=season_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_feature_audit_seasons.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_feature_audit_seasons",
        )
        st.download_button(
            "Download Detailed Ablations",
            data=ablation_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_feature_audit_ablation.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_feature_audit_ablation",
        )
        st.download_button(
            "Download Walk-Forward Predictions",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_feature_audit_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_feature_audit_predictions",
        )


# ===== v2.7 sparse residual bake-off =====
SPARSE_BAKEOFF_VERSION = "v2.7.0-sparse-bakeoff"
SPARSE_ALPHA = 12.0
SPARSE_MIN_TRAIN = 300

# Locked candidate definitions. These are intentionally small and interpretable.
# The current full v2.6 audit set is retained as a benchmark only.
SPARSE_SPREAD_MODELS = {
    "A · Market Control": [],
    "B · Minimal": [
        "sp_hfa",
        "week_num",
        "early_week_flag",
        "confidence_num",
    ],
    "C · Minimal + Power": [
        "sp_hfa",
        "week_num",
        "early_week_flag",
        "confidence_num",
        "sp_rating_diff",
        "srs_adjustment_diff",
        "talent_adjustment_diff",
    ],
    "D · Full Residual": _flatten_feature_groups(SPREAD_FEATURE_GROUPS),
}

# Totals remain research-only. We include a sparse comparison for diagnostics,
# but no total candidate can be auto-promoted from this page.
SPARSE_TOTAL_MODELS = {
    "A · Market Control": [],
    "B · Minimal": [
        "week_num",
        "early_week_flag",
        "confidence_num",
        "market_total",
    ],
    "C · Minimal + Pace": [
        "week_num",
        "early_week_flag",
        "confidence_num",
        "market_total",
        "total_pace_adj",
        "avg_plays_per_drive",
    ],
    "D · Full Residual": _flatten_feature_groups(TOTAL_FEATURE_GROUPS),
}


def _sparse_model_predict(train, test, features, target):
    """
    Market-control candidate uses a zero residual correction.
    Other candidates use the same fixed-alpha standardized ridge and are fit
    only on seasons before the test season.
    """
    te = test.dropna(subset=[target]).copy()
    if te.empty:
        return None

    actual = te[target].to_numpy(dtype=float)

    if not features:
        preds = np.zeros(len(te), dtype=float)
        return {
            "n_train": int(len(train)),
            "n_test": int(len(te)),
            "mae": float(np.mean(np.abs(actual))),
            "preds": preds,
            "actual": actual,
            "index": te.index.to_numpy(),
        }

    usable = [f for f in features if f in train.columns and f in test.columns]
    if len(usable) != len(features):
        return None

    tr = train.dropna(subset=usable + [target]).copy()
    te = test.dropna(subset=usable + [target]).copy()
    if len(tr) < SPARSE_MIN_TRAIN or te.empty:
        return None

    fit = _ridge_fit_numpy(
        tr[usable].values,
        tr[target].values,
        SPARSE_ALPHA,
    )
    preds = np.array([
        _ridge_predict_numpy(fit, row)
        for row in te[usable].to_numpy(dtype=float)
    ])
    actual = te[target].to_numpy(dtype=float)

    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "mae": float(np.mean(np.abs(actual - preds))),
        "preds": preds,
        "actual": actual,
        "index": te.index.to_numpy(),
    }


def _run_sparse_market_bakeoff(feature_df, test_seasons, market_type, holdout):
    """
    Compare locked candidate architectures season-by-season.

    No model is allowed to select features, alpha, or thresholds using the
    test season. Every test season is scored exactly once.
    """
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if market_type == "spread":
        candidates = SPARSE_SPREAD_MODELS
        target = "spread_target_residual"
    else:
        candidates = SPARSE_TOTAL_MODELS
        target = "total_target_residual"

    season_rows = []
    pred_rows = []

    for season in sorted(set(int(s) for s in test_seasons)):
        train = feature_df[feature_df["season"].astype(int) < season].copy()
        test = feature_df[feature_df["season"].astype(int) == season].copy()

        for model_name, features in candidates.items():
            result = _sparse_model_predict(train, test, features, target)
            if result is None:
                continue

            season_rows.append({
                "Market": market_type.upper(),
                "Model": model_name,
                "Test Season": int(season),
                "Games": result["n_test"],
                "Train Games": result["n_train"],
                "MAE": result["mae"],
                "Holdout": "YES" if int(season) == int(holdout) else "NO",
            })

            for idx, actual, pred in zip(
                result["index"], result["actual"], result["preds"]
            ):
                pred_rows.append({
                    "market_type": market_type,
                    "model": model_name,
                    "season": int(season),
                    "row_index": int(idx),
                    "actual_market_residual": float(actual),
                    "predicted_residual": float(pred),
                    "abs_error": abs(float(actual) - float(pred)),
                })

    return pd.DataFrame(season_rows), pd.DataFrame(pred_rows)


def _sparse_bakeoff_summary(season_df, holdout):
    if season_df is None or season_df.empty:
        return pd.DataFrame()

    rows = []
    for (market, model_name), g in season_df.groupby(["Market", "Model"]):
        g = g.sort_values("Test Season")
        overall_mae = float(
            np.average(
                pd.to_numeric(g["MAE"], errors="coerce"),
                weights=pd.to_numeric(g["Games"], errors="coerce"),
            )
        )

        hold = g[g["Test Season"].astype(int) == int(holdout)]
        holdout_mae = float(hold.iloc[0]["MAE"]) if not hold.empty else np.nan

        market_rows = season_df[
            (season_df["Market"] == market) &
            (season_df["Model"] == "A · Market Control")
        ]
        market_overall = float(
            np.average(
                pd.to_numeric(market_rows["MAE"], errors="coerce"),
                weights=pd.to_numeric(market_rows["Games"], errors="coerce"),
            )
        ) if not market_rows.empty else np.nan

        market_hold = market_rows[
            market_rows["Test Season"].astype(int) == int(holdout)
        ]
        market_holdout = float(market_hold.iloc[0]["MAE"]) if not market_hold.empty else np.nan

        full_rows = season_df[
            (season_df["Market"] == market) &
            (season_df["Model"] == "D · Full Residual")
        ]
        full_overall = float(
            np.average(
                pd.to_numeric(full_rows["MAE"], errors="coerce"),
                weights=pd.to_numeric(full_rows["Games"], errors="coerce"),
            )
        ) if not full_rows.empty else np.nan

        full_hold = full_rows[
            full_rows["Test Season"].astype(int) == int(holdout)
        ]
        full_holdout = float(full_hold.iloc[0]["MAE"]) if not full_hold.empty else np.nan

        seasons_vs_market = 0
        tested_seasons = 0
        for _, r in g.iterrows():
            s = int(r["Test Season"])
            mr = market_rows[market_rows["Test Season"].astype(int) == s]
            if mr.empty:
                continue
            tested_seasons += 1
            if float(r["MAE"]) < float(mr.iloc[0]["MAE"]):
                seasons_vs_market += 1

        rows.append({
            "Market": market,
            "Model": model_name,
            "Overall MAE": overall_mae,
            "Holdout MAE": holdout_mae,
            "Overall vs Market": market_overall - overall_mae if pd.notna(market_overall) else np.nan,
            "Holdout vs Market": market_holdout - holdout_mae if pd.notna(market_holdout) else np.nan,
            "Overall vs Full": full_overall - overall_mae if pd.notna(full_overall) else np.nan,
            "Holdout vs Full": full_holdout - holdout_mae if pd.notna(full_holdout) else np.nan,
            "Seasons Beat Market": f"{seasons_vs_market}/{tested_seasons}",
        })

    out = pd.DataFrame(rows)

    # Locked candidate recommendation logic.
    # Spread only: candidate must beat market and current full model overall and
    # on holdout, and beat market in at least 3 of 4 tested seasons.
    verdicts = []
    for _, r in out.iterrows():
        if r["Model"] == "A · Market Control":
            verdicts.append("CONTROL")
            continue
        if r["Market"] == "TOTAL":
            verdicts.append("RESEARCH ONLY")
            continue

        try:
            season_wins = int(str(r["Seasons Beat Market"]).split("/")[0])
            season_n = int(str(r["Seasons Beat Market"]).split("/")[1])
        except Exception:
            season_wins, season_n = 0, 0

        promote = (
            pd.notna(r["Overall vs Market"]) and float(r["Overall vs Market"]) > 0
            and pd.notna(r["Holdout vs Market"]) and float(r["Holdout vs Market"]) > 0
            and pd.notna(r["Overall vs Full"]) and float(r["Overall vs Full"]) > 0
            and pd.notna(r["Holdout vs Full"]) and float(r["Holdout vs Full"]) > 0
            and season_n >= 3
            and season_wins >= max(2, season_n - 1)
        )

        verdicts.append("PROMOTION CANDIDATE" if promote else "KEEP IN RESEARCH")

    out["Locked Verdict"] = verdicts

    order = {
        "A · Market Control": 0,
        "B · Minimal": 1,
        "C · Minimal + Power": 2,
        "D · Full Residual": 3,
    }
    out["_order"] = out["Model"].map(order).fillna(99)
    return out.sort_values(["Market", "_order"]).drop(columns="_order").reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_sparse_bakeoff_cached(test_seasons_tuple, scope, holdout):
    test_seasons = sorted(set(int(s) for s in test_seasons_tuple))
    min_history = min(RESIDUAL_TRAIN_START, min(test_seasons) - 4)
    min_history = max(2014, min_history)
    max_history = max(test_seasons)

    feat = _feature_audit_history(min_history, max_history, scope)

    sp_seasons, sp_preds = _run_sparse_market_bakeoff(
        feat, test_seasons, "spread", holdout
    )
    tot_seasons, tot_preds = _run_sparse_market_bakeoff(
        feat, test_seasons, "total", holdout
    )

    season_df = pd.concat([sp_seasons, tot_seasons], ignore_index=True)
    pred_df = pd.concat([sp_preds, tot_preds], ignore_index=True)
    summary_df = _sparse_bakeoff_summary(season_df, holdout)

    return season_df, summary_df, pred_df


def _render_sparse_bakeoff_results(season_df, summary_df, pred_df, holdout):
    st.markdown("#### Sparse Residual Bake-Off Results")
    st.caption(
        "A = market only. B = Home Field + Week/Early Season + Confidence. "
        "C = B plus SP+/SRS/Talent. D = the current full residual feature set."
    )

    if summary_df is None or summary_df.empty:
        st.info("Run the sparse bake-off to compare the locked candidate models.")
        return

    summary_show = summary_df.copy()
    for c in [
        "Overall MAE",
        "Holdout MAE",
        "Overall vs Market",
        "Holdout vs Market",
        "Overall vs Full",
        "Holdout vs Full",
    ]:
        if c in summary_show.columns:
            if "vs" in c:
                summary_show[c] = summary_show[c].map(
                    lambda v: f"{v:+.4f}" if pd.notna(v) else "—"
                )
            else:
                summary_show[c] = summary_show[c].map(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                )

    st.dataframe(summary_show, use_container_width=True, hide_index=True)

    promoted = summary_df[
        (summary_df["Market"] == "SPREAD") &
        (summary_df["Locked Verdict"] == "PROMOTION CANDIDATE")
    ]

    if not promoted.empty:
        best = promoted.sort_values("Holdout MAE").iloc[0]
        st.success(
            f"{best['Model']} passed the locked sparse-model promotion screen. "
            "This does not yet change the live betting engine; it is eligible for a separate promotion build."
        )
    else:
        st.warning(
            "No sparse spread candidate passed the locked promotion screen. "
            "Do not loosen betting thresholds to force action."
        )

    with st.expander("Season-by-season sparse results", expanded=False):
        season_show = season_df.copy()
        season_show["MAE"] = season_show["MAE"].map(lambda v: f"{v:.4f}")
        st.dataframe(season_show, use_container_width=True, hide_index=True)

    with st.expander("Sparse Bake-Off Downloads", expanded=False):
        sparse_zip = _csv_download_bundle({
            "cfb_sparse_bakeoff_summary.csv": summary_df,
            "cfb_sparse_bakeoff_seasons.csv": season_df,
            "cfb_sparse_bakeoff_predictions.csv": pred_df,
        })
        st.download_button(
            "Download All 3 Sparse Files",
            data=sparse_zip,
            file_name="cfb_sparse_bakeoff_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_sparse_bakeoff_all_zip",
        )
        st.download_button(
            "Download Sparse Summary",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_sparse_bakeoff_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_sparse_bakeoff_summary",
        )
        st.download_button(
            "Download Sparse Seasons",
            data=season_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_sparse_bakeoff_seasons.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_sparse_bakeoff_seasons",
        )
        st.download_button(
            "Download Sparse Predictions",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_sparse_bakeoff_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_sparse_bakeoff_predictions",
        )



def _csv_download_bundle(files):
    """
    Build an in-memory ZIP from {filename: dataframe_or_bytes}.
    Keeps mobile users from having to leave/re-enter the page for each CSV.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, obj in files.items():
            if obj is None:
                continue
            if isinstance(obj, pd.DataFrame):
                payload = obj.to_csv(index=False).encode("utf-8")
            elif isinstance(obj, str):
                payload = obj.encode("utf-8")
            else:
                payload = bytes(obj)
            zf.writestr(filename, payload)
    return buf.getvalue()

# ===== v2.8 situational residual discovery =====
SITUATIONAL_DISCOVERY_VERSION = "v2.8.0-situational-discovery"

# Predeclared segments. These are diagnostic only and are not allowed to
# automatically create betting rules.
SITUATIONAL_SEGMENTS = [
    ("All Games", lambda d: pd.Series(True, index=d.index)),
    ("Weeks 0–3", lambda d: d["week_num"] <= 3),
    ("Weeks 4+", lambda d: d["week_num"] >= 4),
    ("Home Favorite", lambda d: d["market_margin"] > 0),
    ("Home Dog", lambda d: d["market_margin"] < 0),
    ("PK to 3", lambda d: d["abs_market_margin"] <= 3),
    ("3 to 7", lambda d: (d["abs_market_margin"] > 3) & (d["abs_market_margin"] <= 7)),
    ("7 to 14", lambda d: (d["abs_market_margin"] > 7) & (d["abs_market_margin"] <= 14)),
    ("14+", lambda d: d["abs_market_margin"] > 14),
    ("Confidence <70", lambda d: d["confidence_num"] < 70),
    ("Confidence 70–79", lambda d: (d["confidence_num"] >= 70) & (d["confidence_num"] < 80)),
    ("Confidence 80+", lambda d: d["confidence_num"] >= 80),
]


def _season_conference_flag(row):
    """
    Returns conference/non-conference when game rows include conference columns.
    Historical backtest payloads may not always expose them, so this is optional.
    """
    hc = row.get("home_conference")
    ac = row.get("away_conference")
    if hc is None or ac is None or pd.isna(hc) or pd.isna(ac):
        return None
    return "Conference" if str(hc) == str(ac) else "Non-Conference"


def _add_situational_columns(feature_df):
    d = feature_df.copy()

    if "week_num" not in d.columns:
        d["week_num"] = pd.to_numeric(d.get("week"), errors="coerce")
    if "market_margin" not in d.columns:
        d["market_margin"] = -pd.to_numeric(d.get("market_home_spread"), errors="coerce")
    if "abs_market_margin" not in d.columns:
        d["abs_market_margin"] = d["market_margin"].abs()
    if "confidence_num" not in d.columns:
        d["confidence_num"] = pd.to_numeric(d.get("confidence"), errors="coerce")

    if "conference_flag" not in d.columns:
        if "home_conference" in d.columns and "away_conference" in d.columns:
            d["conference_flag"] = d.apply(_season_conference_flag, axis=1)
        else:
            d["conference_flag"] = None
    return d


def _model_prediction_lookup(pred_df, market_type="spread"):
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    d = pred_df[pred_df["market_type"] == market_type].copy()
    if d.empty:
        return d
    return d[["model", "season", "row_index", "predicted_residual", "actual_market_residual", "abs_error"]]


def _situational_segment_rows(feature_df, sparse_pred_df, holdout):
    """
    Evaluate Market Control, Minimal, Minimal+Power and Full Residual inside
    predeclared situations. Each model prediction is already walk-forward.
    """
    feat = _add_situational_columns(feature_df)
    preds = _model_prediction_lookup(sparse_pred_df, "spread")
    if feat.empty or preds.empty:
        return pd.DataFrame()

    rows = []
    model_names = [
        "A · Market Control",
        "B · Minimal",
        "C · Minimal + Power",
        "D · Full Residual",
    ]

    for model_name in model_names:
        p = preds[preds["model"] == model_name].copy()
        if p.empty:
            continue

        merged = feat.merge(
            p,
            left_index=True,
            right_on="row_index",
            how="inner",
            suffixes=("", "_pred"),
        )
        if merged.empty:
            continue

        for seg_name, seg_fn in SITUATIONAL_SEGMENTS:
            try:
                mask = seg_fn(merged).fillna(False)
            except Exception:
                continue

            sg = merged[mask].copy()
            if sg.empty:
                continue

            for season, ss in sg.groupby("season"):
                season = int(season)
                mae = float(np.mean(np.abs(
                    pd.to_numeric(ss["actual_market_residual"], errors="coerce")
                    - pd.to_numeric(ss["predicted_residual"], errors="coerce")
                )))
                market_mae = float(np.mean(np.abs(
                    pd.to_numeric(ss["actual_market_residual"], errors="coerce")
                )))
                rows.append({
                    "Model": model_name,
                    "Segment": seg_name,
                    "Season": season,
                    "Games": int(len(ss)),
                    "Model MAE": mae,
                    "Market MAE": market_mae,
                    "Improvement vs Market": market_mae - mae,
                    "Holdout": "YES" if season == int(holdout) else "NO",
                })

        # Optional conference split when supported.
        if merged["conference_flag"].notna().any():
            for cf in ["Conference", "Non-Conference"]:
                sg = merged[merged["conference_flag"] == cf].copy()
                if sg.empty:
                    continue
                for season, ss in sg.groupby("season"):
                    season = int(season)
                    mae = float(np.mean(np.abs(
                        pd.to_numeric(ss["actual_market_residual"], errors="coerce")
                        - pd.to_numeric(ss["predicted_residual"], errors="coerce")
                    )))
                    market_mae = float(np.mean(np.abs(
                        pd.to_numeric(ss["actual_market_residual"], errors="coerce")
                    )))
                    rows.append({
                        "Model": model_name,
                        "Segment": cf,
                        "Season": season,
                        "Games": int(len(ss)),
                        "Model MAE": mae,
                        "Market MAE": market_mae,
                        "Improvement vs Market": market_mae - mae,
                        "Holdout": "YES" if season == int(holdout) else "NO",
                    })

    return pd.DataFrame(rows)


def _summarize_situational_segments(segment_df, holdout):
    if segment_df is None or segment_df.empty:
        return pd.DataFrame()

    rows = []
    for (model_name, segment), g in segment_df.groupby(["Model", "Segment"]):
        vals = pd.to_numeric(g["Improvement vs Market"], errors="coerce").dropna()
        total_games = int(pd.to_numeric(g["Games"], errors="coerce").sum())
        seasons = int(len(vals))
        seasons_won = int((vals > 0).sum())
        avg_impr = float(np.average(
            pd.to_numeric(g["Improvement vs Market"], errors="coerce"),
            weights=pd.to_numeric(g["Games"], errors="coerce"),
        )) if total_games else np.nan

        hold = g[g["Season"].astype(int) == int(holdout)]
        hold_games = int(hold.iloc[0]["Games"]) if not hold.empty else 0
        hold_impr = float(hold.iloc[0]["Improvement vs Market"]) if not hold.empty else np.nan

        # Conservative discovery labels; not automatic betting rules.
        if (
            total_games >= 250
            and seasons >= 3
            and seasons_won >= max(2, seasons - 1)
            and pd.notna(avg_impr) and avg_impr > 0
            and pd.notna(hold_impr) and hold_impr > 0
            and hold_games >= 50
        ):
            label = "STABLE SIGNAL"
        elif (
            total_games >= 150
            and seasons >= 3
            and seasons_won >= 2
            and pd.notna(avg_impr) and avg_impr > 0
        ):
            label = "LEAN"
        else:
            label = "NO SIGNAL"

        rows.append({
            "Model": model_name,
            "Segment": segment,
            "Games": total_games,
            "Seasons Helped": f"{seasons_won}/{seasons}",
            "Weighted Avg Improvement": avg_impr,
            f"{holdout} Holdout Improvement": hold_impr,
            f"{holdout} Holdout Games": hold_games,
            "Discovery Label": label,
        })

    order = {
        "A · Market Control": 0,
        "B · Minimal": 1,
        "C · Minimal + Power": 2,
        "D · Full Residual": 3,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["Model"].map(order).fillna(99)
    return out.sort_values(
        ["_order", "Discovery Label", "Weighted Avg Improvement"],
        ascending=[True, True, False],
    ).drop(columns="_order").reset_index(drop=True)


def _matched_sample_comparison(feature_df, sparse_pred_df, holdout):
    """
    Compare models only on games for which ALL four models have predictions.
    This removes the advanced-feature missingness advantage/disadvantage from
    the Full Residual benchmark.
    """
    if feature_df is None or feature_df.empty or sparse_pred_df is None or sparse_pred_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    p = sparse_pred_df[sparse_pred_df["market_type"] == "spread"].copy()
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot_pred = p.pivot_table(
        index=["season", "row_index", "actual_market_residual"],
        columns="model",
        values="predicted_residual",
        aggfunc="first",
    ).reset_index()

    required = [
        "A · Market Control",
        "B · Minimal",
        "C · Minimal + Power",
        "D · Full Residual",
    ]
    for c in required:
        if c not in pivot_pred.columns:
            return pd.DataFrame(), pd.DataFrame()

    matched = pivot_pred.dropna(subset=required).copy()
    if matched.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows = []
    season_rows = []

    for season, g in matched.groupby("season"):
        season = int(season)
        actual = pd.to_numeric(g["actual_market_residual"], errors="coerce").to_numpy(dtype=float)

        for model_name in required:
            pred = pd.to_numeric(g[model_name], errors="coerce").to_numpy(dtype=float)
            mae = float(np.mean(np.abs(actual - pred)))
            market_mae = float(np.mean(np.abs(actual)))

            season_rows.append({
                "Model": model_name,
                "Season": season,
                "Matched Games": int(len(g)),
                "MAE": mae,
                "Matched Market MAE": market_mae,
                "Improvement vs Matched Market": market_mae - mae,
                "Holdout": "YES" if season == int(holdout) else "NO",
            })

        for _, r in g.iterrows():
            rec = {
                "season": int(r["season"]),
                "row_index": int(r["row_index"]),
                "actual_market_residual": float(r["actual_market_residual"]),
            }
            for model_name in required:
                rec[model_name] = float(r[model_name])
            detail_rows.append(rec)

    return pd.DataFrame(season_rows), pd.DataFrame(detail_rows)


def _matched_sample_summary(matched_season_df, holdout):
    if matched_season_df is None or matched_season_df.empty:
        return pd.DataFrame()

    rows = []
    for model_name, g in matched_season_df.groupby("Model"):
        overall = float(np.average(
            pd.to_numeric(g["MAE"], errors="coerce"),
            weights=pd.to_numeric(g["Matched Games"], errors="coerce"),
        ))
        market_overall = float(np.average(
            pd.to_numeric(g["Matched Market MAE"], errors="coerce"),
            weights=pd.to_numeric(g["Matched Games"], errors="coerce"),
        ))
        hold = g[g["Season"].astype(int) == int(holdout)]
        hold_mae = float(hold.iloc[0]["MAE"]) if not hold.empty else np.nan
        hold_market = float(hold.iloc[0]["Matched Market MAE"]) if not hold.empty else np.nan

        rows.append({
            "Model": model_name,
            "Matched Overall MAE": overall,
            "Matched Market MAE": market_overall,
            "Overall Improvement": market_overall - overall,
            f"{holdout} Holdout MAE": hold_mae,
            f"{holdout} Holdout Market MAE": hold_market,
            f"{holdout} Holdout Improvement": hold_market - hold_mae if pd.notna(hold_market) and pd.notna(hold_mae) else np.nan,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _run_situational_discovery_cached(test_seasons_tuple, scope, holdout):
    test_seasons = sorted(set(int(s) for s in test_seasons_tuple))
    min_history = min(RESIDUAL_TRAIN_START, min(test_seasons) - 4)
    min_history = max(2014, min_history)
    max_history = max(test_seasons)

    feat = _feature_audit_history(min_history, max_history, scope)
    sparse_seasons, sparse_summary, sparse_preds = _run_sparse_bakeoff_cached(
        tuple(test_seasons), scope, int(holdout)
    )

    seg_season = _situational_segment_rows(feat, sparse_preds, holdout)
    seg_summary = _summarize_situational_segments(seg_season, holdout)

    matched_seasons, matched_detail = _matched_sample_comparison(
        feat, sparse_preds, holdout
    )
    matched_summary = _matched_sample_summary(matched_seasons, holdout)

    return (
        seg_season,
        seg_summary,
        matched_seasons,
        matched_summary,
        matched_detail,
    )


def _render_situational_discovery_results(
    seg_season,
    seg_summary,
    matched_seasons,
    matched_summary,
    matched_detail,
    holdout,
):
    st.markdown("#### Situational Residual Discovery Results")
    st.caption(
        "Segments are predeclared diagnostics. A STABLE SIGNAL is not automatically a betting rule; "
        "it identifies a segment eligible for a separate locked confirmation test."
    )

    if seg_summary is None or seg_summary.empty:
        st.info("Run situational discovery to test where the residual model may add value.")
        return

    focus = seg_summary[seg_summary["Model"].isin([
        "B · Minimal", "C · Minimal + Power"
    ])].copy()

    stable = focus[focus["Discovery Label"] == "STABLE SIGNAL"]
    watch = focus[focus["Discovery Label"] == "LEAN"]

    if not stable.empty:
        names = ", ".join(
            stable.apply(lambda r: f"{r['Model']} / {r['Segment']}", axis=1).tolist()
        )
        st.success("Stable situational candidates: " + names)
    elif not watch.empty:
        names = ", ".join(
            watch.head(6).apply(lambda r: f"{r['Model']} / {r['Segment']}", axis=1).tolist()
        )
        st.warning("No stable signal yet. Segments worth watching: " + names)
    else:
        st.warning("No stable situational spread signal found under the locked screen.")

    show = focus.copy()
    for c in ["Weighted Avg Improvement", f"{holdout} Holdout Improvement"]:
        show[c] = show[c].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "—")
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("#### Matched-Sample Comparison")
    st.caption(
        "All models below are scored on exactly the same games, so missing advanced-feature data cannot distort the Full Residual comparison."
    )
    if matched_summary is not None and not matched_summary.empty:
        mshow = matched_summary.copy()
        for c in mshow.columns:
            if c != "Model":
                mshow[c] = mshow[c].map(lambda v: f"{v:+.4f}" if "Improvement" in c else (f"{v:.4f}" if pd.notna(v) else "—"))
        st.dataframe(mshow, use_container_width=True, hide_index=True)

    with st.expander("Situational season detail", expanded=False):
        detail = seg_season.copy()
        for c in ["Model MAE", "Market MAE", "Improvement vs Market"]:
            detail[c] = detail[c].map(
                lambda v: f"{v:+.4f}" if c == "Improvement vs Market" else f"{v:.4f}"
            )
        st.dataframe(detail, use_container_width=True, hide_index=True)

    with st.expander("Matched season detail", expanded=False):
        md = matched_seasons.copy()
        for c in ["MAE", "Matched Market MAE", "Improvement vs Matched Market"]:
            md[c] = md[c].map(
                lambda v: f"{v:+.4f}" if c == "Improvement vs Matched Market" else f"{v:.4f}"
            )
        st.dataframe(md, use_container_width=True, hide_index=True)

    with st.expander("Situational Discovery Downloads", expanded=True):
        all_needed_zip = _csv_download_bundle({
            "cfb_situational_summary.csv": seg_summary,
            "cfb_situational_seasons.csv": seg_season,
            "cfb_matched_sample_summary.csv": matched_summary,
            "cfb_matched_sample_seasons.csv": matched_seasons,
            "cfb_matched_sample_predictions.csv": matched_detail,
        })

        st.download_button(
            "Download All 5 Files",
            data=all_needed_zip,
            file_name="cfb_v281_situational_validation_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_cfb_situational_all_zip",
        )
        st.caption("Recommended on mobile: one ZIP containing every file needed for review.")

        st.download_button(
            "Download Situational Summary",
            data=seg_summary.to_csv(index=False).encode("utf-8"),
            file_name="cfb_situational_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cfb_situational_summary",
        )
        st.download_button(
            "Download Situational Seasons",
            data=seg_season.to_csv(index=False).encode("utf-8"),
            file_name="cfb_situational_seasons.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cfb_situational_seasons",
        )
        st.download_button(
            "Download Matched Summary",
            data=matched_summary.to_csv(index=False).encode("utf-8"),
            file_name="cfb_matched_sample_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cfb_matched_sample_summary",
        )
        st.download_button(
            "Download Matched Seasons",
            data=matched_seasons.to_csv(index=False).encode("utf-8"),
            file_name="cfb_matched_sample_seasons.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cfb_matched_sample_seasons",
        )
        st.download_button(
            "Download Matched Predictions",
            data=matched_detail.to_csv(index=False).encode("utf-8"),
            file_name="cfb_matched_sample_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cfb_matched_sample_predictions",
        )

def _format_validation_bets(df):
    if df is None or df.empty:
        return df
    x = df.copy()
    if "Win %" in x.columns:
        x["Win %"] = x["Win %"].map(lambda v: f"{100*v:.1f}%" if pd.notna(v) else "—")
    if "ROI" in x.columns:
        x["ROI"] = x["ROI"].map(lambda v: f"{100*v:+.1f}%" if pd.notna(v) else "—")
    if "Units" in x.columns:
        x["Units"] = x["Units"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
    return x



V383_THRESHOLD_GRID = (0.78, 0.80, 0.82, 0.84, 0.86)

def _v383_slate_name_from_hour(hour):
    h = _v3_num(hour, np.nan)
    if not np.isfinite(h):
        return "Unknown"
    if h < 15.5:
        return "Early"
    if h < 19.0:
        return "Midday"
    return "Night"

def _v383_rebuild_slate_native(ranked):
    """
    Rebuild the live v3.8.2 selector cross-section inside each time slate.
    No threshold is applied here; this creates the score universe used for audit.
    """
    if ranked is None or ranked.empty:
        return pd.DataFrame()

    x = ranked.copy()
    required = [
        "season","game_date_et","kickoff_hour_et",
        "classifier_confidence","classifier_agreement","classifier_models",
        "reg_strength","direction_agreement","data_maturity","won"
    ]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Historical selector rows are missing: {', '.join(missing)}")

    x["slate_window"] = x["kickoff_hour_et"].apply(_v383_slate_name_from_hour)
    x = x[x["slate_window"].isin(["Early","Midday","Night"])].copy()
    if x.empty:
        return x

    rebuilt = []
    group_cols = ["season","game_date_et","slate_window"]
    for _, g in x.groupby(group_cols, dropna=False):
        z = g.copy()
        z["confidence_pct"] = pd.to_numeric(
            z["classifier_confidence"], errors="coerce"
        ).rank(method="average", pct=True)
        z["regression_pct"] = pd.to_numeric(
            z["reg_strength"], errors="coerce"
        ).rank(method="average", pct=True)
        z["agreement_rate"] = (
            pd.to_numeric(z["classifier_agreement"], errors="coerce")
            / pd.to_numeric(z["classifier_models"], errors="coerce").clip(lower=1)
        )
        z["selector_score_v383"] = (
            0.40 * z["confidence_pct"]
            + 0.20 * z["agreement_rate"]
            + 0.20 * z["regression_pct"]
            + 0.10 * pd.to_numeric(z["direction_agreement"], errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(z["data_maturity"], errors="coerce").fillna(0.0)
        )
        z["slate_rank"] = z["selector_score_v383"].rank(
            method="first", ascending=False
        ).astype(int)
        z["slate_size"] = int(len(z))
        rebuilt.append(z)

    return pd.concat(rebuilt, ignore_index=True) if rebuilt else pd.DataFrame()

def _v383_threshold_summary(rebuilt, holdout, thresholds=V383_THRESHOLD_GRID):
    if rebuilt is None or rebuilt.empty:
        return pd.DataFrame()

    rows = []
    for threshold in thresholds:
        official = rebuilt[
            pd.to_numeric(rebuilt["selector_score_v383"], errors="coerce") >= float(threshold)
        ].copy()

        all_m = _v34_metrics(official)
        h = official[official["season"].astype(int) == int(holdout)].copy()
        hm = _v34_metrics(h)

        slate_days = rebuilt[["season","game_date_et","slate_window"]].drop_duplicates()
        bet_days = official[["season","game_date_et","slate_window"]].drop_duplicates()
        total_slates = int(len(slate_days))
        played_slates = int(len(bet_days))
        no_bet_slates = max(total_slates - played_slates, 0)

        rows.append({
            "Threshold": float(threshold),
            "Bets": all_m["Bets"],
            "Win Rate": all_m["Win Rate"],
            "ROI": all_m["ROI"],
            "Units": all_m["Units"],
            "Max Drawdown": all_m["Max Drawdown"],
            "Slates": total_slates,
            "Slates With Bet": played_slates,
            "No-Bet Slates": no_bet_slates,
            "No-Bet Rate": no_bet_slates / total_slates if total_slates else np.nan,
            "Holdout Bets": hm["Bets"],
            "Holdout Win Rate": hm["Win Rate"],
            "Holdout ROI": hm["ROI"],
            "Holdout Units": hm["Units"],
            "Holdout Max Drawdown": hm["Max Drawdown"],
        })
    return pd.DataFrame(rows)

def _v383_threshold_by_slate(rebuilt, holdout, thresholds=V383_THRESHOLD_GRID):
    if rebuilt is None or rebuilt.empty:
        return pd.DataFrame()

    rows = []
    for slate_name in ["Early","Midday","Night"]:
        sg = rebuilt[rebuilt["slate_window"] == slate_name].copy()
        if sg.empty:
            continue
        for threshold in thresholds:
            official = sg[
                pd.to_numeric(sg["selector_score_v383"], errors="coerce") >= float(threshold)
            ].copy()
            m = _v34_metrics(official)
            h = official[official["season"].astype(int) == int(holdout)]
            hm = _v34_metrics(h)

            total_slates = sg[["season","game_date_et","slate_window"]].drop_duplicates().shape[0]
            played_slates = official[["season","game_date_et","slate_window"]].drop_duplicates().shape[0]
            rows.append({
                "Slate": slate_name,
                "Threshold": float(threshold),
                "Bets": m["Bets"],
                "Win Rate": m["Win Rate"],
                "ROI": m["ROI"],
                "Units": m["Units"],
                "Max Drawdown": m["Max Drawdown"],
                "No-Bet Rate": (
                    (total_slates - played_slates) / total_slates
                    if total_slates else np.nan
                ),
                "Holdout Bets": hm["Bets"],
                "Holdout Win Rate": hm["Win Rate"],
                "Holdout ROI": hm["ROI"],
            })
    return pd.DataFrame(rows)

def _v383_score_buckets(rebuilt, holdout):
    if rebuilt is None or rebuilt.empty:
        return pd.DataFrame()
    bins = [
        (0.00, 0.78, "<0.78"),
        (0.78, 0.80, "0.780–0.799"),
        (0.80, 0.82, "0.800–0.819"),
        (0.82, 0.84, "0.820–0.839"),
        (0.84, 0.86, "0.840–0.859"),
        (0.86, 1.01, "0.860+"),
    ]
    rows = []
    score = pd.to_numeric(rebuilt["selector_score_v383"], errors="coerce")
    for lo, hi, label in bins:
        g = rebuilt[(score >= lo) & (score < hi)].copy()
        if g.empty:
            continue
        m = _v34_metrics(g)
        h = g[g["season"].astype(int) == int(holdout)]
        hm = _v34_metrics(h)
        rows.append({
            "Score Bucket": label,
            "Games": m["Bets"],
            "Win Rate": m["Win Rate"],
            "ROI": m["ROI"],
            "Units": m["Units"],
            "Holdout Games": hm["Bets"],
            "Holdout Win Rate": hm["Win Rate"],
            "Holdout ROI": hm["ROI"],
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400, show_spinner=False)
def _run_v383_threshold_audit(test_seasons_tuple, scope, holdout, train_start):
    ranked, *_ = _run_v35_adaptive_daily_card(
        tuple(sorted(set(int(s) for s in test_seasons_tuple))),
        scope,
        int(holdout),
        int(train_start),
    )
    rebuilt = _v383_rebuild_slate_native(ranked)
    summary = _v383_threshold_summary(rebuilt, holdout)
    by_slate = _v383_threshold_by_slate(rebuilt, holdout)
    buckets = _v383_score_buckets(rebuilt, holdout)
    return rebuilt, summary, by_slate, buckets

def _v383_fmt_pct(v):
    try:
        return f"{float(v):.1%}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"

def _render_v383_threshold_audit_page():
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">VALIDATION LAB</div>'
        '<div class="mobile-page-title">Slate Threshold Audit</div>'
        '<div class="mobile-page-sub">Rebuild the historical selector exactly as Early / Midday / Night slate-native rankings, then test fixed score floors without forcing bets.</div></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This audit does not optimize a threshold in-sample. It shows how fixed candidate floors "
        "0.78 / 0.80 / 0.82 / 0.84 / 0.86 behave under the current slate-native workflow. "
        "Keep the untouched holdout season separate when judging whether a lower floor deserves promotion."
    )

    c1, c2 = st.columns(2)
    with c1:
        seasons = st.multiselect(
            "Historical seasons",
            [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            default=[2021, 2022, 2023, 2024, 2025],
            key="v383_audit_seasons",
        )
        scope = st.selectbox(
            "Game universe",
            ["Major FBS", "All FBS"],
            index=0,
            key="v383_audit_scope",
        )
    with c2:
        holdout_default = max(seasons) if seasons else 2025
        holdout = st.selectbox(
            "Untouched holdout",
            seasons if seasons else [2025],
            index=(len(seasons)-1 if seasons else 0),
            key="v383_audit_holdout",
        )
        train_start = st.selectbox(
            "Training history begins",
            [2015, 2016, 2017, 2018, 2019],
            index=0,
            key="v383_audit_train_start",
        )

    run = st.button(
        "Run Slate Threshold Audit",
        type="primary",
        use_container_width=True,
        key="run_v383_threshold_audit",
    )

    if run:
        if not seasons:
            st.error("Select at least one historical season.")
            st.stop()
        with st.spinner("Rebuilding leakage-safe historical slate rankings…"):
            rebuilt, summary, by_slate, buckets = _run_v383_threshold_audit(
                tuple(seasons), scope, int(holdout), int(train_start)
            )
        st.session_state["v383_audit_rebuilt"] = rebuilt
        st.session_state["v383_audit_summary"] = summary
        st.session_state["v383_audit_by_slate"] = by_slate
        st.session_state["v383_audit_buckets"] = buckets
        st.session_state["v383_audit_holdout_used"] = int(holdout)

    summary = st.session_state.get("v383_audit_summary", pd.DataFrame())
    by_slate = st.session_state.get("v383_audit_by_slate", pd.DataFrame())
    buckets = st.session_state.get("v383_audit_buckets", pd.DataFrame())
    rebuilt = st.session_state.get("v383_audit_rebuilt", pd.DataFrame())
    holdout_used = int(st.session_state.get("v383_audit_holdout_used", holdout))

    if summary is None or summary.empty:
        st.caption("Run the audit to compare fixed score floors.")
        return

    st.markdown("### Combined slate-native results")
    show = summary.copy()
    for c in ["Win Rate","ROI","No-Bet Rate","Holdout Win Rate","Holdout ROI"]:
        show[c] = show[c].map(_v383_fmt_pct)
    for c in ["Units","Max Drawdown","Holdout Units","Holdout Max Drawdown"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").map(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        )
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Explicit comparison against the current locked 0.84 floor.
    current = summary[np.isclose(summary["Threshold"], 0.84)]
    lower = summary[summary["Threshold"] < 0.84]
    if not current.empty:
        cur = current.iloc[0]
        st.markdown("### Former 0.84 benchmark")
        m1, m2, m3 = st.columns(3)
        m1.metric("Historical bets", int(cur["Bets"]))
        m2.metric("Historical ROI", _v383_fmt_pct(cur["ROI"]))
        m3.metric(f"{holdout_used} holdout ROI", _v383_fmt_pct(cur["Holdout ROI"]))

    st.markdown("### By slate")
    by_show = by_slate.copy()
    for c in ["Win Rate","ROI","No-Bet Rate","Holdout Win Rate","Holdout ROI"]:
        by_show[c] = by_show[c].map(_v383_fmt_pct)
    st.dataframe(by_show, use_container_width=True, hide_index=True)

    st.markdown("### Score buckets")
    bucket_show = buckets.copy()
    for c in ["Win Rate","ROI","Holdout Win Rate","Holdout ROI"]:
        bucket_show[c] = bucket_show[c].map(_v383_fmt_pct)
    st.dataframe(bucket_show, use_container_width=True, hide_index=True)

    st.warning(
        "The live production floor is 0.80. Use this table to monitor whether that choice remains stable. A lower threshold should only be promoted "
        "if it improves usable sample size while remaining profitable and stable in the untouched holdout, "
        "without materially worsening drawdown."
    )

    with st.expander("Downloads", expanded=False):
        st.download_button(
            "Download Threshold Summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="cfb_v383_threshold_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_v383_summary",
        )
        st.download_button(
            "Download By-Slate Results",
            data=by_slate.to_csv(index=False).encode("utf-8"),
            file_name="cfb_v383_threshold_by_slate.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_v383_by_slate",
        )
        st.download_button(
            "Download Score Buckets",
            data=buckets.to_csv(index=False).encode("utf-8"),
            file_name="cfb_v383_score_buckets.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_v383_buckets",
        )
        st.download_button(
            "Download Rebuilt Selector Rows",
            data=rebuilt.to_csv(index=False).encode("utf-8"),
            file_name="cfb_v383_slate_native_rows.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_v383_rows",
        )


def _render_model_validation_page():
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">MODEL VALIDATION</div>'
        '<div class="mobile-page-title">Spread + Total + Feature + Sparse + Situational + v3.1 ML</div>'
        '<div class="mobile-page-sub">Walk-forward validation now includes a v3 point-in-time rebuild using weekly pregame advanced stats, pregame Elo and richer preseason roster priors. Every test season uses only earlier seasons.</div></div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "This runner is diagnostic, not a threshold optimizer. The buckets and promotion gates are fixed "
        "before the results are displayed. Default mode uses leakage-safe preseason priors."
    )

    c1, c2 = st.columns(2)
    with c1:
        seasons = st.multiselect(
            "Seasons",
            [2018,2019,2020,2021,2022,2023,2024,2025],
            default=[2022,2023,2024,2025],
            key="validation_seasons",
        )
        scope = st.selectbox(
            "Game universe",
            ["Major FBS", "All FBS", "All college games"],
            index=0,
            key="validation_scope",
        )
    with c2:
        holdout_options = seasons if seasons else [2025]
        holdout = st.selectbox(
            "Untouched holdout",
            holdout_options,
            index=max(0, len(holdout_options)-1),
            key="validation_holdout",
        )
        st.caption("Recommended: latest selected season as holdout.")

    if st.button(
        "Run Spread + Total Validation",
        type="primary",
        use_container_width=True,
        key="run_spread_total_validation",
    ):
        if len(seasons) < 3:
            st.error("Select at least three seasons.")
            st.stop()

        progress = st.progress(0, text="Starting validation…")
        try:
            games_df, picks_df, projection_df, gate_df = _run_current_market_validation(
                seasons, scope, holdout, progress=progress
            )
        except Exception as e:
            progress.empty()
            st.error(f"Validation failed: {e}")
            st.exception(e)
            st.stop()
        progress.empty()

        st.session_state["cfb_validation_games"] = games_df
        st.session_state["cfb_validation_picks"] = picks_df
        st.session_state["cfb_validation_projection"] = projection_df
        st.session_state["cfb_validation_gate"] = gate_df
        st.session_state["cfb_validation_holdout"] = int(holdout)
        st.success("Validation complete.")

    games_df = st.session_state.get("cfb_validation_games", pd.DataFrame())
    picks_df = st.session_state.get("cfb_validation_picks", pd.DataFrame())
    projection_df = st.session_state.get("cfb_validation_projection", pd.DataFrame())
    gate_df = st.session_state.get("cfb_validation_gate", pd.DataFrame())
    holdout = int(st.session_state.get("cfb_validation_holdout", holdout))

    if not isinstance(projection_df, pd.DataFrame) or projection_df.empty:
        st.info("Run the validation to generate the spread and total diagnostics.")
        return

    st.markdown("### 1. Projection accuracy vs market")
    proj_show = projection_df.copy()
    for c in ["Market MAE", "Adjusted Model MAE", "Raw Model MAE", "Improvement vs Market"]:
        proj_show[c] = proj_show[c].map(lambda v: f"{v:+.3f}" if c == "Improvement vs Market" else f"{v:.3f}")
    st.dataframe(proj_show, use_container_width=True, hide_index=True)

    st.markdown("### 2. Official BET / BEST BET performance by season")
    official = picks_df[picks_df["verdict"].isin(["BET", "STRONG BET"])].copy()
    season_bets = _validation_betting_summary(official, ["market_type", "season"])
    if not season_bets.empty:
        st.dataframe(_format_validation_bets(season_bets), use_container_width=True, hide_index=True)
    else:
        st.info("No official BET / BEST BET rows under the current production grader.")

    st.markdown("### 3. Performance by projection disagreement")
    gap_perf = _validation_betting_summary(picks_df, ["market_type", "gap_bucket"])
    if not gap_perf.empty:
        st.dataframe(_format_validation_bets(gap_perf), use_container_width=True, hide_index=True)
    st.caption(
        "These are fixed point-gap buckets. Use them to see whether larger model/market disagreements actually improve results."
    )

    st.markdown("### 4. Performance by model edge")
    edge_perf = _validation_betting_summary(picks_df, ["market_type", "edge_bucket"])
    if not edge_perf.empty:
        st.dataframe(_format_validation_bets(edge_perf), use_container_width=True, hide_index=True)

    st.markdown("### 5. Early-season vs later-season stability")
    tmp = picks_df.copy()
    if not tmp.empty:
        tmp["week_group"] = tmp["week"].apply(lambda w: "Weeks 1–3" if int(w) <= 3 else "Week 4+")
        week_perf = _validation_betting_summary(tmp, ["market_type", "week_group"])
        st.dataframe(_format_validation_bets(week_perf), use_container_width=True, hide_index=True)

    st.markdown("### 6. Promotion gate")
    gate_show = gate_df.copy()
    for c in ["Official ROI", f"{holdout} ROI"]:
        if c in gate_show.columns:
            gate_show[c] = gate_show[c].map(lambda v: f"{100*v:+.1f}%")
    st.dataframe(gate_show, use_container_width=True, hide_index=True)

    spread_status = gate_df.loc[gate_df["Market"]=="SPREAD", "Status"]
    total_status = gate_df.loc[gate_df["Market"]=="TOTAL", "Status"]
    spread_status = spread_status.iloc[0] if not spread_status.empty else "NO RESULT"
    total_status = total_status.iloc[0] if not total_status.empty else "NO RESULT"

    if spread_status == "PASS TO PROMOTION REVIEW":
        st.success(f"Spread: {spread_status}")
    else:
        st.warning(f"Spread: {spread_status}")

    if total_status == "PASS TO PROMOTION REVIEW":
        st.success(f"Total: {total_status}")
    else:
        st.warning(f"Total: {total_status}")

    st.caption(
        "A PASS does not automatically alter the live grader. It means the market is eligible for a separate locked promotion review. "
        "A FAIL means we improve the projection/calibration before changing recommendation thresholds."
    )

    st.markdown("### 7. Feature Audit")
    st.caption(
        "This keeps the market-baseline architecture fixed and asks which football signals actually improve unseen-season residual MAE."
    )
    if st.button(
        "Run Residual Feature Audit",
        use_container_width=True,
        key="run_residual_feature_audit",
    ):
        audit_progress = st.progress(0, text="Starting feature audit…")
        try:
            a_seasons, a_ablation, a_summary, a_preds = _run_feature_audit(
                seasons, scope, holdout, progress=audit_progress
            )
        except Exception as e:
            audit_progress.empty()
            st.error(f"Feature audit failed: {e}")
            st.exception(e)
        else:
            audit_progress.empty()
            st.session_state["cfb_feature_audit_seasons"] = a_seasons
            st.session_state["cfb_feature_audit_ablation"] = a_ablation
            st.session_state["cfb_feature_audit_summary"] = a_summary
            st.session_state["cfb_feature_audit_predictions"] = a_preds
            st.success("Feature audit complete.")

    a_seasons = st.session_state.get("cfb_feature_audit_seasons", pd.DataFrame())
    a_ablation = st.session_state.get("cfb_feature_audit_ablation", pd.DataFrame())
    a_summary = st.session_state.get("cfb_feature_audit_summary", pd.DataFrame())
    a_preds = st.session_state.get("cfb_feature_audit_predictions", pd.DataFrame())
    _render_feature_audit_results(
        a_seasons, a_ablation, a_summary, a_preds, holdout
    )

    st.markdown("### 8. Sparse Residual Bake-Off")
    st.caption(
        "Locked comparison: Market Control vs Minimal vs Minimal + Power vs Full Residual. "
        "This tests simplification before any live-model promotion."
    )
    if st.button(
        "Run Sparse Residual Bake-Off",
        use_container_width=True,
        key="run_sparse_residual_bakeoff",
    ):
        sparse_progress = st.progress(0, text="Running sparse walk-forward models…")
        try:
            sparse_progress.progress(0.25, text="Building point-in-time history…")
            s_seasons, s_summary, s_preds = _run_sparse_bakeoff_cached(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
            )
            sparse_progress.progress(1.0, text="Sparse bake-off complete.")
        except Exception as e:
            sparse_progress.empty()
            st.error(f"Sparse bake-off failed: {e}")
            st.exception(e)
        else:
            sparse_progress.empty()
            st.session_state["cfb_sparse_bakeoff_seasons"] = s_seasons
            st.session_state["cfb_sparse_bakeoff_summary"] = s_summary
            st.session_state["cfb_sparse_bakeoff_predictions"] = s_preds
            st.success("Sparse residual bake-off complete.")

    s_seasons = st.session_state.get("cfb_sparse_bakeoff_seasons", pd.DataFrame())
    s_summary = st.session_state.get("cfb_sparse_bakeoff_summary", pd.DataFrame())
    s_preds = st.session_state.get("cfb_sparse_bakeoff_predictions", pd.DataFrame())
    _render_sparse_bakeoff_results(
        s_seasons, s_summary, s_preds, holdout
    )

    st.markdown("### 9. Situational Residual Discovery")
    st.caption(
        "Test the sparse residual models inside predeclared game situations and compare every model on a same-game matched sample."
    )

    if st.button(
        "Run Situational Residual Discovery",
        use_container_width=True,
        key="run_situational_residual_discovery",
    ):
        sit_progress = st.progress(0, text="Building situational discovery…")
        try:
            sit_progress.progress(0.25, text="Loading walk-forward sparse predictions…")
            (
                sit_seasons,
                sit_summary,
                matched_seasons,
                matched_summary,
                matched_detail,
            ) = _run_situational_discovery_cached(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
            )
            sit_progress.progress(1.0, text="Situational discovery complete.")
        except Exception as e:
            sit_progress.empty()
            st.error(f"Situational discovery failed: {e}")
            st.exception(e)
        else:
            sit_progress.empty()
            st.session_state["cfb_situational_seasons"] = sit_seasons
            st.session_state["cfb_situational_summary"] = sit_summary
            st.session_state["cfb_matched_sample_seasons"] = matched_seasons
            st.session_state["cfb_matched_sample_summary"] = matched_summary
            st.session_state["cfb_matched_sample_predictions"] = matched_detail
            st.success("Situational residual discovery complete.")

    sit_seasons = st.session_state.get("cfb_situational_seasons", pd.DataFrame())
    sit_summary = st.session_state.get("cfb_situational_summary", pd.DataFrame())
    matched_seasons = st.session_state.get("cfb_matched_sample_seasons", pd.DataFrame())
    matched_summary = st.session_state.get("cfb_matched_sample_summary", pd.DataFrame())
    matched_detail = st.session_state.get("cfb_matched_sample_predictions", pd.DataFrame())

    _render_situational_discovery_results(
        sit_seasons,
        sit_summary,
        matched_seasons,
        matched_summary,
        matched_detail,
        holdout,
    )

    st.markdown("### 10. v3.0 Point-in-Time Model Lab")
    st.caption(
        "Major rebuild: true pregame weekly advanced stats, pregame Elo, prior SP+/CORE/FPI, talent, returning production, recruiting and transfer portal."
    )
    v3_train_start = st.selectbox(
        "v3 training history start",
        options=[2016, 2017, 2018, 2019, 2021],
        index=2,
        key="v3_train_start",
        help="2018 is the recommended balance of sample size and modern-era relevance. 2020 remains in the history unless you start at 2021.",
    )
    st.caption(
        "The first run can use a meaningful number of CFBD API calls because advanced stats are rebuilt through Week N-1 for each season/week. Results are cached."
    )

    if st.button(
        "Run v3.0 Point-in-Time Lab",
        use_container_width=True,
        key="run_v300_point_in_time_lab",
    ):
        v3prog = st.progress(0, text="Building leakage-safe weekly history…")
        try:
            v3prog.progress(0.20, text="Loading preseason priors and historical markets…")
            (
                v3_history,
                v3_results,
                v3_summary,
                v3_preds,
                v3_readiness,
            ) = _run_v3_point_in_time_lab(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v3prog.progress(1.0, text="v3.0 point-in-time lab complete.")
        except Exception as e:
            v3prog.empty()
            st.error(f"v3.0 lab failed: {e}")
            st.exception(e)
        else:
            v3prog.empty()
            st.session_state["cfb_v300_history"] = v3_history
            st.session_state["cfb_v300_results"] = v3_results
            st.session_state["cfb_v300_summary"] = v3_summary
            st.session_state["cfb_v300_preds"] = v3_preds
            st.session_state["cfb_v300_readiness"] = v3_readiness
            st.success("v3.0 point-in-time rebuild complete.")

    v3_history = st.session_state.get("cfb_v300_history", pd.DataFrame())
    v3_results = st.session_state.get("cfb_v300_results", pd.DataFrame())
    v3_summary = st.session_state.get("cfb_v300_summary", pd.DataFrame())
    v3_preds = st.session_state.get("cfb_v300_preds", pd.DataFrame())
    v3_readiness = st.session_state.get("cfb_v300_readiness", pd.DataFrame())

    _render_v3_lab(
        v3_history,
        v3_results,
        v3_summary,
        v3_preds,
        v3_readiness,
        holdout,
    )

    st.markdown("### 11. v3.1 Nonlinear ML Bake-Off")
    st.caption(
        "Tests Ridge, Gradient Boosting, HistGradientBoosting, Extra Trees, Random Forest, "
        "and XGBoost when installed. It also runs direct ATS / O-U classification with point-in-time probability calibration."
    )
    st.caption(
        "Use 2022–2025 with 2025 holdout and 2018 training start. Betting thresholds are fixed at 54%, 56%, and 58%; "
        "they are not tuned after seeing the holdout."
    )

    if st.button(
        "Run v3.1 ML Bake-Off",
        use_container_width=True,
        key="run_v310_ml_bakeoff",
    ):
        v31prog = st.progress(0, text="Preparing point-in-time feature matrix…")
        try:
            v31prog.progress(0.15, text="Loading cached v3 history…")
            (
                v31_history,
                v31_reg,
                v31_reg_summary,
                v31_cls,
                v31_cls_summary,
                v31_bets_raw,
                v31_bet_summary,
                v31_preds,
                v31_gate,
            ) = _run_v31_ml_bakeoff(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v31prog.progress(1.0, text="v3.1 ML bake-off complete.")
        except Exception as e:
            v31prog.empty()
            st.error(f"v3.1 bake-off failed: {e}")
            st.exception(e)
        else:
            v31prog.empty()
            st.session_state["cfb_v310_history"] = v31_history
            st.session_state["cfb_v310_reg"] = v31_reg
            st.session_state["cfb_v310_reg_summary"] = v31_reg_summary
            st.session_state["cfb_v310_cls"] = v31_cls
            st.session_state["cfb_v310_cls_summary"] = v31_cls_summary
            st.session_state["cfb_v310_bets_raw"] = v31_bets_raw
            st.session_state["cfb_v310_bet_summary"] = v31_bet_summary
            st.session_state["cfb_v310_preds"] = v31_preds
            st.session_state["cfb_v310_gate"] = v31_gate
            st.success("v3.1 nonlinear ML bake-off complete.")

    _render_v31_ml_bakeoff(
        st.session_state.get("cfb_v310_history", pd.DataFrame()),
        st.session_state.get("cfb_v310_reg", pd.DataFrame()),
        st.session_state.get("cfb_v310_reg_summary", pd.DataFrame()),
        st.session_state.get("cfb_v310_cls", pd.DataFrame()),
        st.session_state.get("cfb_v310_cls_summary", pd.DataFrame()),
        st.session_state.get("cfb_v310_bets_raw", pd.DataFrame()),
        st.session_state.get("cfb_v310_bet_summary", pd.DataFrame()),
        st.session_state.get("cfb_v310_preds", pd.DataFrame()),
        st.session_state.get("cfb_v310_gate", pd.DataFrame()),
        holdout,
    )

    st.markdown("### 12. v3.2 Signal Stability + Ensemble")
    st.caption(
        "Audits the exact v3.1 predictions. Development discovery is limited to 2022–2024; "
        "2025 remains the locked confirmation season when selected as holdout."
    )

    if st.button(
        "Run v3.2 Signal Stability Audit",
        use_container_width=True,
        key="run_v320_signal_stability",
    ):
        v32prog = st.progress(0, text="Loading v3.1 walk-forward predictions…")
        try:
            v32prog.progress(0.20, text="Rebuilding corrected holdout gate…")
            (
                v32_history,
                v32_fixed_gate,
                v32_joined,
                v32_segment_summary,
                v32_segment_seasons,
                v32_prob_buckets,
                v32_monotonicity,
                v32_ensemble_detail,
                v32_ensemble_summary,
                v32_football_signals,
            ) = _run_v32_signal_stability(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v32prog.progress(1.0, text="v3.2 stability audit complete.")
        except Exception as e:
            v32prog.empty()
            st.error(f"v3.2 stability audit failed: {e}")
            st.exception(e)
        else:
            v32prog.empty()
            st.session_state["cfb_v320_fixed_gate"] = v32_fixed_gate
            st.session_state["cfb_v320_joined"] = v32_joined
            st.session_state["cfb_v320_segment_summary"] = v32_segment_summary
            st.session_state["cfb_v320_segment_seasons"] = v32_segment_seasons
            st.session_state["cfb_v320_prob_buckets"] = v32_prob_buckets
            st.session_state["cfb_v320_monotonicity"] = v32_monotonicity
            st.session_state["cfb_v320_ensemble_detail"] = v32_ensemble_detail
            st.session_state["cfb_v320_ensemble_summary"] = v32_ensemble_summary
            st.session_state["cfb_v320_football_signals"] = v32_football_signals
            st.success("v3.2 signal stability audit complete.")

    _render_v32_signal_stability(
        st.session_state.get("cfb_v320_fixed_gate", pd.DataFrame()),
        st.session_state.get("cfb_v320_joined", pd.DataFrame()),
        st.session_state.get("cfb_v320_segment_summary", pd.DataFrame()),
        st.session_state.get("cfb_v320_segment_seasons", pd.DataFrame()),
        st.session_state.get("cfb_v320_prob_buckets", pd.DataFrame()),
        st.session_state.get("cfb_v320_monotonicity", pd.DataFrame()),
        st.session_state.get("cfb_v320_ensemble_detail", pd.DataFrame()),
        st.session_state.get("cfb_v320_ensemble_summary", pd.DataFrame()),
        st.session_state.get("cfb_v320_football_signals", pd.DataFrame()),
        holdout,
    )

    st.markdown("### 13. v3.3 Game-Day Selector")
    st.caption(
        "Ranks every weekly slate and tests the exact workflow you want to use on Saturdays: "
        "take only the strongest 1, 3, 5, top 10%, or top 20% of games."
    )

    if st.button(
        "Run v3.3 Game-Day Selector",
        use_container_width=True,
        key="run_v330_selector",
    ):
        v33prog = st.progress(0, text="Rebuilding walk-forward slate ranks…")
        try:
            (
                v33_rank_frame,
                v33_detail,
                v33_summary,
                v33_seasons,
                v33_weeks,
                v33_phases,
                v33_gate,
            ) = _run_v33_gameday_selector(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v33prog.progress(1.0, text="v3.3 game-day selector complete.")
        except Exception as e:
            v33prog.empty()
            st.error(f"v3.3 selector failed: {e}")
            st.exception(e)
        else:
            v33prog.empty()
            st.session_state["cfb_v330_rank_frame"] = v33_rank_frame
            st.session_state["cfb_v330_detail"] = v33_detail
            st.session_state["cfb_v330_summary"] = v33_summary
            st.session_state["cfb_v330_seasons"] = v33_seasons
            st.session_state["cfb_v330_weeks"] = v33_weeks
            st.session_state["cfb_v330_phases"] = v33_phases
            st.session_state["cfb_v330_gate"] = v33_gate
            st.success("v3.3 game-day selector complete.")

    _render_v33_gameday_selector(
        st.session_state.get("cfb_v330_rank_frame", pd.DataFrame()),
        st.session_state.get("cfb_v330_detail", pd.DataFrame()),
        st.session_state.get("cfb_v330_summary", pd.DataFrame()),
        st.session_state.get("cfb_v330_seasons", pd.DataFrame()),
        st.session_state.get("cfb_v330_weeks", pd.DataFrame()),
        st.session_state.get("cfb_v330_phases", pd.DataFrame()),
        st.session_state.get("cfb_v330_gate", pd.DataFrame()),
        holdout,
    )

    st.markdown("### 14. v3.4 Slate-Aware Finalist")
    st.caption(
        "Ranks Early / Midday / Late independently and selects the final ranking architecture "
        "using development seasons only. This is the closest backtest to the intended live workflow."
    )

    if st.button(
        "Run v3.4 Slate-Aware Finalist",
        use_container_width=True,
        key="run_v340_finalist",
    ):
        v34prog = st.progress(0, text="Building historical slate windows…")
        try:
            (
                v34_base,
                v34_candidates,
                v34_gate,
                v34_winner_ranked,
                v34_winner_bets,
                v34_seasons,
                v34_slates,
                v34_weeks,
            ) = _run_v34_slate_finalist(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v34prog.progress(1.0, text="v3.4 slate-aware finalist complete.")
        except Exception as e:
            v34prog.empty()
            st.error(f"v3.4 finalist failed: {e}")
            st.exception(e)
        else:
            v34prog.empty()
            st.session_state["cfb_v340_base"] = v34_base
            st.session_state["cfb_v340_candidates"] = v34_candidates
            st.session_state["cfb_v340_gate"] = v34_gate
            st.session_state["cfb_v340_winner_ranked"] = v34_winner_ranked
            st.session_state["cfb_v340_winner_bets"] = v34_winner_bets
            st.session_state["cfb_v340_seasons"] = v34_seasons
            st.session_state["cfb_v340_slates"] = v34_slates
            st.session_state["cfb_v340_weeks"] = v34_weeks
            st.success("v3.4 slate-aware finalist complete.")

    _render_v34_slate_finalist(
        st.session_state.get("cfb_v340_base", pd.DataFrame()),
        st.session_state.get("cfb_v340_candidates", pd.DataFrame()),
        st.session_state.get("cfb_v340_gate", pd.DataFrame()),
        st.session_state.get("cfb_v340_winner_ranked", pd.DataFrame()),
        st.session_state.get("cfb_v340_winner_bets", pd.DataFrame()),
        st.session_state.get("cfb_v340_seasons", pd.DataFrame()),
        st.session_state.get("cfb_v340_slates", pd.DataFrame()),
        st.session_state.get("cfb_v340_weeks", pd.DataFrame()),
        holdout,
    )

    st.markdown("### 15. v3.5.5 Adaptive Daily Card")
    st.caption(
        "Best available bets on any day you run it. Friday night can be one slate; "
        "full Saturdays can be grouped for readability. The quality bar never drops to force action."
    )

    if st.button(
        "Run v3.5.5 Adaptive Daily Card",
        use_container_width=True,
        key="run_v350_daily_card",
    ):
        v35prog = st.progress(0, text="Ranking historical daily cards…")
        try:
            (
                v35_ranked,
                v35_tiered,
                v35_daily,
                v35_seasons,
                v35_groups,
                v35_holdout_tiers,
                v35_audit,
                v35_gate,
                v35_diag,
                v35_v33,
            ) = _run_v35_adaptive_daily_card(
                tuple(sorted(set(int(s) for s in seasons))),
                scope,
                int(holdout),
                int(v3_train_start),
            )
            v35prog.progress(1.0, text="v3.5.5 adaptive daily card complete.")
        except Exception as e:
            v35prog.empty()
            st.error(f"v3.5 daily card failed: {e}")
            st.exception(e)
        else:
            v35prog.empty()
            st.session_state["cfb_v350_ranked"] = v35_ranked
            st.session_state["cfb_v350_tiered"] = v35_tiered
            st.session_state["cfb_v350_daily"] = v35_daily
            st.session_state["cfb_v350_seasons"] = v35_seasons
            st.session_state["cfb_v350_groups"] = v35_groups
            st.session_state["cfb_v350_holdout_tiers"] = v35_holdout_tiers
            st.session_state["cfb_v350_audit"] = v35_audit
            st.session_state["cfb_v350_gate"] = v35_gate
            st.session_state["cfb_v350_diag"] = v35_diag
            st.session_state["cfb_v350_v33"] = v35_v33
            st.success("v3.5.5 adaptive daily card complete.")

    _render_v35_adaptive_daily_card(
        st.session_state.get("cfb_v350_ranked", pd.DataFrame()),
        st.session_state.get("cfb_v350_tiered", pd.DataFrame()),
        st.session_state.get("cfb_v350_daily", pd.DataFrame()),
        st.session_state.get("cfb_v350_seasons", pd.DataFrame()),
        st.session_state.get("cfb_v350_groups", pd.DataFrame()),
        st.session_state.get("cfb_v350_holdout_tiers", pd.DataFrame()),
        st.session_state.get("cfb_v350_audit", pd.DataFrame()),
        st.session_state.get("cfb_v350_gate", pd.DataFrame()),
        holdout,
        st.session_state.get("cfb_v350_diag", pd.DataFrame()),
        st.session_state.get("cfb_v350_v33", pd.DataFrame()),
    )

    with st.expander("Downloads", expanded=False):
        st.download_button(
            "Download Projection Errors",
            data=projection_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_validation_projection.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_validation_projection",
        )
        st.download_button(
            "Download Game Picks",
            data=picks_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_validation_picks.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_validation_picks",
        )
        st.download_button(
            "Download Promotion Gate",
            data=gate_df.to_csv(index=False).encode("utf-8"),
            file_name="cfb_validation_gate.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_validation_gate",
        )


app_section = "Research Lab" if st.session_state.get("cfb_research_mode", False) else "Betting Board"

if st.session_state.get("cfb_threshold_audit_mode", False):
    if st.button("Back to More", use_container_width=True, key="cfb_exit_threshold_audit"):
        st.session_state["cfb_threshold_audit_mode"] = False
        st.session_state["cfb_page"] = "More"
        st.rerun()
    _render_v383_threshold_audit_page()
    st.stop()

if st.session_state.get("cfb_validation_mode", False):
    if st.button("Back to More", use_container_width=True, key="cfb_exit_validation"):
        st.session_state["cfb_validation_mode"] = False
        st.session_state["cfb_page"] = "More"
        st.rerun()
    _render_model_validation_page()
    st.stop()

if app_section == "Research Lab":
    if st.button("Back to App", use_container_width=True, key="cfb_exit_research"):
        st.session_state["cfb_research_mode"] = False
        st.session_state["cfb_page"] = "More"
        st.rerun()
    st.markdown('<div class="section-kicker">Historical Backtest Lab</div>', unsafe_allow_html=True)
    st.markdown("### Model validation • v0.9.1 locked candidate")
    st.info(
        "Recommended mode is **Leakage-safe preseason prior**. It uses prior-season performance plus "
        "current-season talent/returning-production inputs, and does not use current-season SP+/SRS/PPA/advanced "
        "results. Historical travel/weather are excluded. This tests the betting layer conservatively; it is not "
        "an exact replay of every live-model input."
    )

    c1, c2 = st.columns(2)
    with c1:
        bt_seasons = st.multiselect(
            "Seasons",
            [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        )
        bt_scope = st.selectbox("Game universe", ["Major FBS", "All FBS", "All college games"], index=0)
    with c2:
        bt_method = st.selectbox(
            "Historical rating method",
            ["Leakage-safe preseason prior", "Retrospective full-season diagnostic"],
            index=0,
            help=(
                "Retrospective mode uses full-season historical rating files and therefore has look-ahead bias. "
                "Use it only to inspect mechanics, never as proof of historical ROI."
            ),
        )
        bt_policy = st.selectbox(
            "Signal policy",
            ["Best market per game", "All qualifying markets"],
            index=0,
            help="Best market per game prevents multiple official bets from the same matchup in the backtest.",
        )

        bt_run_mode = st.selectbox(
            "Validation workload",
            [
                "v0.9 candidate only (recommended)",
                "Full legacy research stack",
            ],
            index=0,
            help=(
                "Candidate-only runs the locked 56–57% validation without recalculating every old "
                "v0.5–v0.8 research layer. This is much faster and is the correct mode for v0.9."
            ),
        )

    holdout_default = max(bt_seasons) if bt_seasons else 2025
    bt_holdout = st.selectbox("Untouched holdout season", bt_seasons if bt_seasons else [2025],
                              index=(len(bt_seasons)-1 if bt_seasons else 0))

    if bt_method == "Retrospective full-season diagnostic":
        st.warning(
            "This mode contains look-ahead bias because CFBD's SP+, SRS, PPA and season-advanced historical "
            "endpoints are season-level, not archived week-by-week snapshots. Do not use its ROI as validation."
        )

    run_bt = st.button("Run Backtest", type="primary", use_container_width=True)
    if run_bt:
        if not bt_seasons:
            st.error("Select at least one season.")
            st.stop()

        all_rows = []
        game_rows = []
        progress = st.progress(0, text="Preparing historical data…")
        total_seasons = len(bt_seasons)

        for si, season in enumerate(sorted(bt_seasons)):
            progress.progress(si/total_seasons, text=f"Loading {season} games, lines and ratings…")
            try:
                games = get_backtest_games(season)
                line_payload = get_backtest_lines(season)
                data_full = get_backtest_model_data(season)
            except Exception as e:
                st.error(f"{season} data request failed: {e}")
                continue

            data = _bt_prior_only_data(data_full) if bt_method == "Leakage-safe preseason prior" else data_full

            # Build O(1) line lookup by game ID.
            line_index = {}
            for lr in line_payload or []:
                gid = lr.get("id")
                if gid is None:
                    continue
                try:
                    key = int(gid)
                except Exception:
                    key = gid
                line_index[key] = normalize_game_lines([lr], game_id=gid)

            season_games = [
                g for g in games or []
                if g.get("completed") is True
                and g.get("homePoints") is not None
                and g.get("awayPoints") is not None
                and _bt_game_scope(g, bt_scope)
            ]

            for gi, g in enumerate(season_games):
                gid = g.get("id")
                try:
                    lookup_gid = int(gid)
                except Exception:
                    lookup_gid = gid
                market = _bt_consensus_line(line_index.get(lookup_gid, []))
                if not market or (market.get("home_spread") is None and market.get("total") is None
                                  and market.get("home_ml") is None and market.get("away_ml") is None):
                    continue
                try:
                    p = _bt_project_game(g, data, hfa=DEFAULT_HFA)
                except Exception:
                    continue

                # One game-level row per version for MAE diagnostics.
                for version in ["v0.3.1", "v0.3.2"]:
                    if version == "v0.3.2":
                        adj_spread, _, _ = calibrated_market_projection(
                            p["model_home_spread"], market.get("home_spread"), p["week"], "side")
                        adj_total, _, _ = calibrated_market_projection(
                            p["model_total"], market.get("total"), p["week"], "total")
                    else:
                        adj_spread, adj_total = p["model_home_spread"], p["model_total"]
                    game_rows.append({
                        "version":version, "season":season, "week":p["week"], "game_id":gid,
                        "away_team":p["away"], "home_team":p["home"],
                        "away_points":float(g["awayPoints"]), "home_points":float(g["homePoints"]),
                        "raw_model_home_spread":float(p["model_home_spread"]),
                        "adjusted_model_home_spread":float(adj_spread),
                        "market_home_spread":market.get("home_spread"),
                        "raw_model_total":float(p["model_total"]),
                        "adjusted_model_total":float(adj_total),
                        "market_total":market.get("total"),
                        "market_away_ml":market.get("away_ml"),
                        "market_home_ml":market.get("home_ml"),
                        "confidence":float(p["confidence"]),
                        "base_power_margin":float(p["components"].get("base_power_margin", 0.0)),
                        "matchup_margin_adjustment":float(p["components"].get("matchup_margin_adjustment", 0.0)),
                        "hfa_adjustment":float(p["components"].get("hfa_adjustment", 0.0)),
                        "sp_total_base":float(p["components"].get("sp_total_base", 0.0)),
                        "efficiency_total_adjustment":float(p["components"].get("efficiency_total_adjustment", 0.0)),
                        "pace_total_adjustment":float(p["components"].get("pace_total_adjustment", 0.0)),
                        "away_sp_rating":float(p["away_rating"].get("sp_rating") or 0.0),
                        "home_sp_rating":float(p["home_rating"].get("sp_rating") or 0.0),
                        "away_srs_adjustment":float(p["away_rating"].get("srs_adjustment") or 0.0),
                        "home_srs_adjustment":float(p["home_rating"].get("srs_adjustment") or 0.0),
                        "away_talent_adjustment":float(p["away_rating"].get("talent_adjustment") or 0.0),
                        "home_talent_adjustment":float(p["home_rating"].get("talent_adjustment") or 0.0),
                        "away_returning_adjustment":float(p["away_rating"].get("returning_adjustment") or 0.0),
                        "home_returning_adjustment":float(p["home_rating"].get("returning_adjustment") or 0.0),

                        # v0.7 granular matchup inputs. In leakage-safe mode these
                        # are prior-season performance metrics plus current preseason
                        # talent/returning-production context.
                        "away_ppa_off_pass":p["away_rating"]["ppa"].get("off_pass"),
                        "home_ppa_off_pass":p["home_rating"]["ppa"].get("off_pass"),
                        "away_ppa_def_pass":p["away_rating"]["ppa"].get("def_pass"),
                        "home_ppa_def_pass":p["home_rating"]["ppa"].get("def_pass"),
                        "away_ppa_off_rush":p["away_rating"]["ppa"].get("off_rush"),
                        "home_ppa_off_rush":p["home_rating"]["ppa"].get("off_rush"),
                        "away_ppa_def_rush":p["away_rating"]["ppa"].get("def_rush"),
                        "home_ppa_def_rush":p["home_rating"]["ppa"].get("def_rush"),

                        "away_adv_off_success":p["away_rating"]["adv"].get("off_success"),
                        "home_adv_off_success":p["home_rating"]["adv"].get("off_success"),
                        "away_adv_def_success":p["away_rating"]["adv"].get("def_success"),
                        "home_adv_def_success":p["home_rating"]["adv"].get("def_success"),
                        "away_adv_off_expl":p["away_rating"]["adv"].get("off_expl"),
                        "home_adv_off_expl":p["home_rating"]["adv"].get("off_expl"),
                        "away_adv_def_expl":p["away_rating"]["adv"].get("def_expl"),
                        "home_adv_def_expl":p["home_rating"]["adv"].get("def_expl"),

                        "away_adv_off_pass_ppa":p["away_rating"]["adv"].get("off_pass_ppa"),
                        "home_adv_off_pass_ppa":p["home_rating"]["adv"].get("off_pass_ppa"),
                        "away_adv_def_pass_ppa":p["away_rating"]["adv"].get("def_pass_ppa"),
                        "home_adv_def_pass_ppa":p["home_rating"]["adv"].get("def_pass_ppa"),
                        "away_adv_off_rush_ppa":p["away_rating"]["adv"].get("off_rush_ppa"),
                        "home_adv_off_rush_ppa":p["home_rating"]["adv"].get("off_rush_ppa"),
                        "away_adv_def_rush_ppa":p["away_rating"]["adv"].get("def_rush_ppa"),
                        "home_adv_def_rush_ppa":p["home_rating"]["adv"].get("def_rush_ppa"),

                        "away_adv_off_ppo":p["away_rating"]["adv"].get("off_ppo"),
                        "home_adv_off_ppo":p["home_rating"]["adv"].get("off_ppo"),
                        "away_adv_def_ppo":p["away_rating"]["adv"].get("def_ppo"),
                        "home_adv_def_ppo":p["home_rating"]["adv"].get("def_ppo"),
                        "away_adv_def_havoc":p["away_rating"]["adv"].get("def_havoc"),
                        "home_adv_def_havoc":p["home_rating"]["adv"].get("def_havoc"),
                        "away_adv_off_plays":p["away_rating"]["adv"].get("off_plays"),
                        "home_adv_off_plays":p["home_rating"]["adv"].get("off_plays"),
                        "away_adv_off_drives":p["away_rating"]["adv"].get("off_drives"),
                        "home_adv_off_drives":p["home_rating"]["adv"].get("off_drives"),

                        "away_returning_pass":p["away_rating"].get("returning_pass"),
                        "home_returning_pass":p["home_rating"].get("returning_pass"),
                        "away_returning_usage":p["away_rating"].get("returning_usage"),
                        "home_returning_usage":p["home_rating"].get("returning_usage"),
                    })

                all_rows.extend(_bt_candidate_rows(g, p, market, season, "v0.3.1"))
                all_rows.extend(_bt_candidate_rows(g, p, market, season, "v0.3.2"))

            progress.progress((si+1)/total_seasons, text=f"Finished {season}")

        progress.empty()
        bt_df = pd.DataFrame(all_rows)
        bt_games_df = pd.DataFrame(game_rows)

        if bt_df.empty:
            st.error("No historical games with usable CFBD lines were returned for the selected sample.")
            st.stop()

        feature_df = _residual_feature_frame(bt_games_df)

        # v0.9.1: candidate validation should not be blocked by the entire legacy
        # research stack. The recommended fast path runs only what the locked
        # 56–57% rule needs.
        if bt_run_mode == "v0.9 candidate only (recommended)":
            progress = st.progress(0, text="Fitting rolling cover classifier…")

            residual_holdout = pd.DataFrame()
            residual_diag = {}
            walkforward_bets = pd.DataFrame()
            walkforward_holdouts = pd.DataFrame()
            walkforward_diag = {}
            signal_research = pd.DataFrame()
            signal_walkforward = pd.DataFrame()
            matchup_tests = pd.DataFrame()
            matchup_bets = pd.DataFrame()
            matchup_diag = {}
            matchup_holdout = pd.DataFrame()
            matchup_holdout_bets = pd.DataFrame()
            matchup_holdout_diag = {}
            audit_rows = pd.DataFrame()
            audit_tables = {}

            try:
                classifier_tests, classifier_rows, classifier_diag = _run_classifier_walkforward(feature_df)
                progress.progress(0.55, text="Running locked 56–57% candidate stress tests…")
                classifier_holdout, classifier_holdout_rows, classifier_holdout_diag = _fit_classifier_final_holdout(
                    feature_df, bt_holdout
                )
                candidate_validation = _candidate_validation_bundle(
                    classifier_rows, classifier_tests, feature_df
                )
                progress.progress(1.0, text="Candidate validation complete.")
            except Exception as e:
                progress.empty()
                st.error(f"v0.9 candidate validation failed: {e}")
                st.exception(e)
                st.stop()

            progress.empty()

        else:
            progress = st.progress(0, text="Running full legacy research stack…")

            # v0.5.0 RESIDUAL
            residual_holdout, residual_diag = _fit_residual_models(feature_df, bt_holdout)
            residual_rows = []
            for _, rr in residual_holdout.iterrows():
                residual_rows.extend(
                    _bt_residual_candidate_rows(
                        rr,
                        residual_diag["spread_sd"],
                        residual_diag["total_sd"],
                    )
                )
            if residual_rows:
                bt_df = pd.concat([bt_df, pd.DataFrame(residual_rows)], ignore_index=True, sort=False)

            progress.progress(0.15, text="Running residual walk-forward…")
            walkforward_bets, walkforward_holdouts, walkforward_diag = _run_walkforward_residual(feature_df)
            if not walkforward_bets.empty:
                bt_df = pd.concat([bt_df, walkforward_bets], ignore_index=True, sort=False)

            progress.progress(0.30, text="Running v0.6 signal research…")
            signal_research = _run_signal_research(feature_df, bt_holdout)
            signal_walkforward = _walkforward_signal_research(feature_df)

            progress.progress(0.45, text="Running v0.7 matchup research…")
            matchup_tests, matchup_bets, matchup_diag = _run_matchup_walkforward(feature_df)
            matchup_holdout, matchup_holdout_bets, matchup_holdout_diag = _fit_matchup_final_holdout(
                feature_df, bt_holdout
            )

            progress.progress(0.65, text="Running v0.8 classifier…")
            classifier_tests, classifier_rows, classifier_diag = _run_classifier_walkforward(feature_df)
            classifier_holdout, classifier_holdout_rows, classifier_holdout_diag = _fit_classifier_final_holdout(
                feature_df, bt_holdout
            )

            progress.progress(0.80, text="Running v0.8.1 audit…")
            audit_rows, audit_tables = _build_signal_audit_tables(classifier_rows, classifier_tests)

            progress.progress(0.90, text="Running locked v0.9 candidate stress test…")
            candidate_validation = _candidate_validation_bundle(
                classifier_rows, classifier_tests, feature_df
            )
            progress.progress(1.0, text="Full validation complete.")
            progress.empty()

        signal_df = _bt_best_per_game(bt_df) if bt_policy == "Best market per game" else bt_df.copy()
        official = signal_df[signal_df["verdict"].isin(["BET","STRONG BET"])].copy()

        st.session_state["cfb_backtest_df"] = bt_df
        st.session_state["cfb_backtest_signal_df"] = signal_df
        st.session_state["cfb_backtest_games_df"] = bt_games_df
        st.session_state["cfb_residual_holdout_df"] = residual_holdout
        st.session_state["cfb_residual_diag"] = residual_diag
        st.session_state["cfb_walkforward_bets_df"] = walkforward_bets
        st.session_state["cfb_walkforward_holdouts_df"] = walkforward_holdouts
        st.session_state["cfb_walkforward_diag"] = walkforward_diag
        st.session_state["cfb_signal_research_df"] = signal_research
        st.session_state["cfb_signal_walkforward_df"] = signal_walkforward
        st.session_state["cfb_matchup_tests_df"] = matchup_tests
        st.session_state["cfb_matchup_bets_df"] = matchup_bets
        st.session_state["cfb_matchup_diag"] = matchup_diag
        st.session_state["cfb_matchup_holdout_df"] = matchup_holdout
        st.session_state["cfb_matchup_holdout_bets_df"] = matchup_holdout_bets
        st.session_state["cfb_matchup_holdout_diag"] = matchup_holdout_diag

        st.session_state["cfb_classifier_tests_df"] = classifier_tests
        st.session_state["cfb_classifier_rows_df"] = classifier_rows
        st.session_state["cfb_classifier_diag"] = classifier_diag
        st.session_state["cfb_classifier_holdout_df"] = classifier_holdout
        st.session_state["cfb_classifier_holdout_rows_df"] = classifier_holdout_rows
        st.session_state["cfb_classifier_holdout_diag"] = classifier_holdout_diag

        st.session_state["cfb_signal_audit_rows_df"] = audit_rows
        st.session_state["cfb_signal_audit_tables"] = audit_tables
        st.session_state["cfb_candidate_validation"] = candidate_validation

        st.session_state["cfb_backtest_config"] = {
            "seasons": bt_seasons, "holdout": bt_holdout, "scope": bt_scope,
            "method": bt_method, "policy": bt_policy, "run_mode": bt_run_mode,
        }

        st.success(
            "Backtest complete. Results are shown below. "
            + ("Candidate-only mode skipped legacy research layers." if bt_run_mode.startswith("v0.9") else "")
        )

    if "cfb_backtest_signal_df" in st.session_state:
        bt_df = st.session_state["cfb_backtest_df"]
        signal_df = st.session_state["cfb_backtest_signal_df"]
        bt_games_df = st.session_state["cfb_backtest_games_df"]
        cfg = st.session_state.get("cfb_backtest_config", {})
        holdout = cfg.get("holdout", bt_holdout)

        if cfg.get("run_mode") == "v0.9 candidate only (recommended)":
            st.info(
                "Fast candidate mode was used. v0.5–v0.8 legacy research sections below may be empty; "
                "the v0.9 locked-candidate section is the intended output."
            )

        train = signal_df[signal_df["season"] != holdout]
        test = signal_df[signal_df["season"] == holdout]

        summaries=[]
        for version in ["v0.3.1","v0.3.2"]:
            summaries.append(_bt_summary(train[train["version"]==version], f"{version} • Development"))
            summaries.append(_bt_summary(test[test["version"]==version], f"{version} • Holdout {holdout}"))
            summaries.append(_bt_summary(signal_df[signal_df["version"]==version], f"{version} • All"))
        summaries.append(
            _bt_summary(
                test[test["version"]==RESIDUAL_VERSION],
                f"{RESIDUAL_VERSION} • Holdout {holdout}"
            )
        )
        summaries.append(
            _bt_summary(
                signal_df[
                    (signal_df["version"]==WALKFORWARD_VERSION)
                    & (signal_df["market_type"]=="spread")
                ],
                f"{WALKFORWARD_VERSION} • Spread only"
            )
        )
        summary_df = pd.DataFrame(summaries)
        for col in ["Win %","ROI"]:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        summary_df["Profit (u)"] = summary_df["Profit (u)"].map(lambda x: f"{x:+.2f}")

        st.markdown("### Betting results")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        err = _bt_error_table(bt_games_df.dropna(subset=["market_home_spread","market_total"], how="all"))
        if not err.empty:
            for c in [x for x in err.columns if "MAE" in x]:
                err[c] = err[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            st.markdown("### Projection error vs market")
            st.dataframe(err, use_container_width=True, hide_index=True)

        residual_holdout = st.session_state.get("cfb_residual_holdout_df", pd.DataFrame())
        residual_diag = st.session_state.get("cfb_residual_diag", {})
        if isinstance(residual_holdout, pd.DataFrame) and not residual_holdout.empty:
            st.markdown("### v0.5.0 residual holdout test")
            st.caption(
                "v0.5.0 starts from the market and predicts only the amount the market is wrong. "
                "Its coefficients are fit on development seasons; the selected holdout is not used in training."
            )
            resid_err = _residual_holdout_error_table(residual_holdout)
            if not resid_err.empty:
                for c in ["Market-only MAE", "Residual-model MAE", "Improvement"]:
                    resid_err[c] = resid_err[c].map(lambda x: f"{x:.3f}")
                st.dataframe(resid_err, use_container_width=True, hide_index=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Spread ridge α", f"{residual_diag.get('spread_alpha', float('nan')):g}")
            c2.metric("Total ridge α", f"{residual_diag.get('total_alpha', float('nan')):g}")
            c3.metric("Spread forecast σ", f"{residual_diag.get('spread_sd', float('nan')):.2f}")
            c4.metric("Total forecast σ", f"{residual_diag.get('total_sd', float('nan')):.2f}")

            with st.expander("Residual-model validation details", expanded=False):
                sp_cv = residual_diag.get("spread_cv", pd.DataFrame())
                tot_cv = residual_diag.get("total_cv", pd.DataFrame())
                if isinstance(sp_cv, pd.DataFrame) and not sp_cv.empty:
                    st.write("Spread regularization tuning")
                    st.dataframe(sp_cv, use_container_width=True, hide_index=True)
                if isinstance(tot_cv, pd.DataFrame) and not tot_cv.empty:
                    st.write("Total regularization tuning")
                    st.dataframe(tot_cv, use_container_width=True, hide_index=True)
                st.caption(
                    "Moneyline betting is intentionally disabled for v0.5.0 while the historical moneyline feed is audited."
                )

        wf_bets = st.session_state.get("cfb_walkforward_bets_df", pd.DataFrame())
        wf_holdouts = st.session_state.get("cfb_walkforward_holdouts_df", pd.DataFrame())

        if isinstance(wf_bets, pd.DataFrame) and not wf_bets.empty:
            st.markdown("### v0.5.1 rolling walk-forward test")
            st.caption(
                "Each season is predicted only from seasons that occurred before it. "
                "Official v0.5.1 bets are spreads only; totals are retained as research-only LEANs."
            )

            wf_signal = _bt_best_per_game(wf_bets) if cfg.get("policy") == "Best market per game" else wf_bets.copy()
            wf_season = _walkforward_season_summary(wf_signal)
            if not wf_season.empty:
                wf_show = wf_season.copy()
                wf_show["Win %"] = wf_show["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                wf_show["Units"] = wf_show["Units"].map(lambda x: f"{x:+.2f}")
                wf_show["ROI"] = wf_show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
                st.markdown("#### Unseen-season spread betting")
                st.dataframe(wf_show, use_container_width=True, hide_index=True)

            wf_err = _walkforward_error_summary(wf_holdouts)
            if not wf_err.empty:
                wf_err_show = wf_err.copy()
                for c in ["Market-only MAE", "Residual MAE", "Improvement"]:
                    wf_err_show[c] = wf_err_show[c].map(lambda x: f"{x:.3f}")
                st.markdown("#### Unseen-season prediction error")
                st.dataframe(wf_err_show, use_container_width=True, hide_index=True)

            official_wf_spreads = wf_signal[
                (wf_signal["version"] == WALKFORWARD_VERSION)
                & (wf_signal["market_type"] == "spread")
                & (wf_signal["verdict"].isin(["BET", "STRONG BET"]))
            ]
            if not official_wf_spreads.empty:
                wins = int((official_wf_spreads["result"]=="WIN").sum())
                losses = int((official_wf_spreads["result"]=="LOSS").sum())
                units = float(official_wf_spreads["profit_units"].sum())
                decided = wins + losses
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("WF spread bets", len(official_wf_spreads))
                c2.metric("WF spread record", f"{wins}-{losses}")
                c3.metric("WF spread win %", f"{100*wins/decided:.1f}%" if decided else "—")
                c4.metric("WF spread units", f"{units:+.2f}")

            st.info(
                "Promotion rule: do not move v0.5.1 to the live betting board unless spread signals "
                "are credible across multiple unseen seasons, not just the 2025 holdout."
            )

        candidate_validation = st.session_state.get("cfb_candidate_validation", {})

        st.markdown("### v0.9.0 Locked Candidate Validation • 56–57%")
        st.caption(
            "The candidate rule is frozen before this test: select only classifier probabilities "
            "56.0% ≤ P < 57.0%. No probability threshold or subgroup is optimized here. "
            "The primary aggregate excludes 2020, and a second stress test removes 2020 from "
            "training entirely."
        )

        if isinstance(candidate_validation, dict) and candidate_validation:
            std = candidate_validation.get("standard_candidate", pd.DataFrame())
            std_seasons = candidate_validation.get("standard_seasons", pd.DataFrame())
            std_eras = candidate_validation.get("standard_eras", pd.DataFrame())
            std_sides = candidate_validation.get("standard_sides", pd.DataFrame())
            std_loso = candidate_validation.get("standard_loso", pd.DataFrame())

            covid = candidate_validation.get("covid_excluded_candidate", pd.DataFrame())
            covid_seasons = candidate_validation.get("covid_excluded_seasons", pd.DataFrame())
            covid_eras = candidate_validation.get("covid_excluded_eras", pd.DataFrame())
            covid_sides = candidate_validation.get("covid_excluded_sides", pd.DataFrame())
            covid_loso = candidate_validation.get("covid_excluded_loso", pd.DataFrame())

            gate, gate_note = _candidate_pass_fail(covid if isinstance(covid, pd.DataFrame) else pd.DataFrame())

            if isinstance(std, pd.DataFrame) and not std.empty:
                primary = std[std["season"] != COVID_SEASON]
                s = _candidate_stats(primary, "Primary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ex-2020 bets", f"{s['Bets']}")
                c2.metric("Record", s["W-L-P"])
                c3.metric("Win %", f"{100*s['Win %']:.1f}%" if pd.notna(s["Win %"]) else "—")
                c4.metric("ROI", f"{100*s['ROI']:+.1f}%" if pd.notna(s["ROI"]) else "—")

            if gate == "PASS TO LIVE-CANDIDATE REVIEW":
                st.success(f"Candidate gate: {gate}. {gate_note}")
            else:
                st.warning(f"Candidate gate: {gate}. {gate_note}")

            def _format_candidate_table(df):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return df
                x = df.copy()
                for c in ["Win %","ROI","Wilson low","Wilson high","Edge over -110 BE"]:
                    if c in x.columns:
                        x[c] = x[c].map(lambda v: f"{100*v:.1f}%" if pd.notna(v) else "—")
                if "Units" in x.columns:
                    x["Units"] = x["Units"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
                return x

            st.markdown("#### Standard rolling walk-forward")
            st.caption("2020 is shown separately and excluded from the primary combined result.")
            if isinstance(std_seasons, pd.DataFrame) and not std_seasons.empty:
                st.dataframe(_format_candidate_table(std_seasons), use_container_width=True, hide_index=True)

            st.markdown("#### COVID-excluded-training stress test")
            st.caption(
                "2020 is never used for fitting and is never a test season. This checks whether "
                "the unusual 2020 season distorted parameters used in 2021 and later."
            )
            if isinstance(covid_seasons, pd.DataFrame) and not covid_seasons.empty:
                st.dataframe(_format_candidate_table(covid_seasons), use_container_width=True, hide_index=True)

            with st.expander("Era stability", expanded=False):
                if isinstance(std_eras, pd.DataFrame) and not std_eras.empty:
                    st.markdown("**Standard walk-forward**")
                    st.dataframe(_format_candidate_table(std_eras), use_container_width=True, hide_index=True)
                if isinstance(covid_eras, pd.DataFrame) and not covid_eras.empty:
                    st.markdown("**COVID-excluded training**")
                    st.dataframe(_format_candidate_table(covid_eras), use_container_width=True, hide_index=True)

            with st.expander("Home/Away and Favorite/Underdog diagnostics", expanded=False):
                if isinstance(covid_sides, pd.DataFrame) and not covid_sides.empty:
                    st.dataframe(_format_candidate_table(covid_sides), use_container_width=True, hide_index=True)

            with st.expander("Leave-one-season-out robustness", expanded=False):
                if isinstance(covid_loso, pd.DataFrame) and not covid_loso.empty:
                    st.dataframe(_format_candidate_table(covid_loso), use_container_width=True, hide_index=True)

            # Exports: the three files we actually need for promotion review.
            if isinstance(covid, pd.DataFrame) and not covid.empty:
                ios_save_button(
                    "Save v0.9 Candidate Bets CSV",
                    covid.to_csv(index=False),
                    f"cfb_v090_candidate_bets_{min(cfg.get('seasons',[2018]))}_{max(cfg.get('seasons',[2025]))}.csv"
                )

            if isinstance(covid_seasons, pd.DataFrame) and not covid_seasons.empty:
                ios_save_button(
                    "Save v0.9 Season Validation CSV",
                    covid_seasons.to_csv(index=False),
                    f"cfb_v090_candidate_seasons_{min(cfg.get('seasons',[2018]))}_{max(cfg.get('seasons',[2025]))}.csv"
                )

            stress_parts = []
            for section, frame in [
                ("standard_eras", std_eras),
                ("covid_excluded_eras", covid_eras),
                ("covid_excluded_sides", covid_sides),
                ("covid_excluded_leave_one_out", covid_loso),
            ]:
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    z = frame.copy()
                    z.insert(0, "Section", section)
                    stress_parts.append(z)
            stress_export = pd.concat(stress_parts, ignore_index=True, sort=False) if stress_parts else pd.DataFrame()
            if not stress_export.empty:
                ios_save_button(
                    "Save v0.9 Stress Tests CSV",
                    stress_export.to_csv(index=False),
                    f"cfb_v090_candidate_stress_{min(cfg.get('seasons',[2018]))}_{max(cfg.get('seasons',[2025]))}.csv"
                )

            st.info(
                "Primary promotion logic is intentionally conservative. The locked 56–57% rule "
                "must survive the longer history and the COVID-excluded-training test before it "
                "can be considered for the live board."
            )

        audit_rows = st.session_state.get("cfb_signal_audit_rows_df", pd.DataFrame())
        audit_tables = st.session_state.get("cfb_signal_audit_tables", {})

        st.markdown("### v0.8.1 Signal Audit • 56–58% Bucket")
        st.caption(
            "Audits only the pre-specified 56–58% classifier bucket. This does not create new bets "
            "or optimize a new threshold. The goal is to see whether the apparent edge survives "
            "across home/away, favorite/underdog, spread size, week, probability sub-band and "
            "matchup profiles."
        )

        if isinstance(audit_rows, pd.DataFrame) and not audit_rows.empty:
            c1, c2, c3, c4 = st.columns(4)
            w = int((audit_rows["result"] == "WIN").sum())
            l = int((audit_rows["result"] == "LOSS").sum())
            p = int((audit_rows["result"] == "PUSH").sum())
            units = float(pd.to_numeric(audit_rows["profit_units"], errors="coerce").fillna(0).sum())
            decided = w + l
            c1.metric("56–58% bets", f"{len(audit_rows)}")
            c2.metric("Record", f"{w}-{l}-{p}")
            c3.metric("Win %", f"{100*w/decided:.1f}%" if decided else "—")
            c4.metric("ROI", f"{100*units/len(audit_rows):+.1f}%")

            # Main one-way summaries.
            labels = [
                ("home_away","Home vs away"),
                ("fav_dog","Favorite vs underdog"),
                ("spread_bucket","Spread size"),
                ("week_bucket","Week bands"),
                ("prob_bucket","56–57 vs 57–58"),
                ("pass_profile","Pass matchup"),
                ("rush_profile","Rush matchup"),
                ("expl_profile","Explosiveness matchup"),
                ("havoc_profile","Havoc"),
                ("finishing_profile","Finishing drives"),
            ]

            for key, title in labels:
                df = audit_tables.get(key, pd.DataFrame()) if isinstance(audit_tables, dict) else pd.DataFrame()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    with st.expander(title, expanded=(key in ["fav_dog","spread_bucket","prob_bucket"])):
                        show = df.copy()
                        show["Win %"] = show["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                        show["Units"] = show["Units"].map(lambda x: f"{x:+.2f}")
                        show["ROI"] = show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
                        show["Avg model prob"] = show["Avg model prob"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                        st.dataframe(show, use_container_width=True, hide_index=True)

            # Two-way context tables.
            for key, title in [
                ("homeaway_favdog","Home/Away × Favorite/Underdog"),
                ("favdog_spread","Favorite/Underdog × Spread size"),
                ("homeaway_spread","Home/Away × Spread size"),
                ("prob_favdog","Probability sub-band × Favorite/Underdog"),
            ]:
                df = audit_tables.get(key, pd.DataFrame()) if isinstance(audit_tables, dict) else pd.DataFrame()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    with st.expander(title, expanded=False):
                        show = df.copy()
                        show["Win %"] = show["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                        show["Units"] = show["Units"].map(lambda x: f"{x:+.2f}")
                        show["ROI"] = show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
                        show["Avg model prob"] = show["Avg model prob"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                        st.dataframe(show, use_container_width=True, hide_index=True)

            survival = audit_tables.get("survival", pd.DataFrame()) if isinstance(audit_tables, dict) else pd.DataFrame()
            if isinstance(survival, pd.DataFrame) and not survival.empty:
                st.markdown("#### Multi-season survival summary")
                surv_show = survival.copy()
                surv_show["Positive-season rate"] = surv_show["Positive-season rate"].map(
                    lambda x: f"{100*x:.0f}%" if pd.notna(x) else "—"
                )
                surv_show["Combined units"] = surv_show["Combined units"].map(lambda x: f"{x:+.2f}")
                surv_show["Combined ROI"] = surv_show["Combined ROI"].map(
                    lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—"
                )
                st.dataframe(surv_show, use_container_width=True, hide_index=True)

            # Exports.
            ios_save_button(
                "Save v0.8.1 Audit Bets CSV",
                audit_rows.to_csv(index=False),
                f"cfb_v081_signal_audit_bets_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )

            # Combine all subgroup tables into one export with a Section field.
            audit_export_parts = []
            if isinstance(audit_tables, dict):
                for key, df in audit_tables.items():
                    if key == "survival" or not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    z = df.copy()
                    z.insert(0, "Section", key)
                    audit_export_parts.append(z)
            audit_export = pd.concat(audit_export_parts, ignore_index=True, sort=False) if audit_export_parts else pd.DataFrame()

            if not audit_export.empty:
                ios_save_button(
                    "Save v0.8.1 Audit Breakdown CSV",
                    audit_export.to_csv(index=False),
                    f"cfb_v081_signal_audit_breakdown_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
                )

            if isinstance(survival, pd.DataFrame) and not survival.empty:
                ios_save_button(
                    "Save v0.8.1 Survival CSV",
                    survival.to_csv(index=False),
                    f"cfb_v081_signal_audit_survival_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
                )

            st.info(
                "Audit rule: do not promote a subgroup because of the best combined ROI alone. "
                "A candidate should have meaningful sample size and survive across multiple unseen seasons."
            )

        classifier_tests = st.session_state.get("cfb_classifier_tests_df", pd.DataFrame())
        classifier_rows = st.session_state.get("cfb_classifier_rows_df", pd.DataFrame())
        classifier_diag = st.session_state.get("cfb_classifier_diag", {})
        classifier_holdout = st.session_state.get("cfb_classifier_holdout_df", pd.DataFrame())
        classifier_holdout_rows = st.session_state.get("cfb_classifier_holdout_rows_df", pd.DataFrame())
        classifier_holdout_diag = st.session_state.get("cfb_classifier_holdout_diag", {})

        st.markdown("### v0.8.0 Cover Classifier")
        st.caption(
            "Directly estimates P(home covers) and P(away covers) from the sportsbook line plus "
            "matchup features. It is evaluated by log loss, Brier score, calibration and fixed "
            "probability buckets. All outputs remain research-only."
        )

        if isinstance(classifier_tests, pd.DataFrame) and not classifier_tests.empty:
            score = _classifier_score_table(classifier_tests)
            if not score.empty:
                score_show = score.copy()
                for c in ["Classifier log loss","50/50 log loss","Log-loss improvement",
                          "Classifier Brier","50/50 Brier","Brier improvement"]:
                    score_show[c] = score_show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
                st.markdown("#### Rolling unseen-season probability benchmark")
                st.dataframe(score_show, use_container_width=True, hide_index=True)

            buckets = _classifier_bucket_table(classifier_rows)
            if not buckets.empty:
                bucket_show = buckets.copy()
                bucket_show["Win %"] = bucket_show["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                bucket_show["Units"] = bucket_show["Units"].map(lambda x: f"{x:+.2f}")
                bucket_show["ROI"] = bucket_show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
                bucket_show["Avg predicted prob"] = bucket_show["Avg predicted prob"].map(lambda x: f"{100*x:.1f}%")
                st.markdown("#### Probability-bucket ATS results")
                st.dataframe(bucket_show, use_container_width=True, hide_index=True)

            calibration = _classifier_calibration_table(classifier_tests)
            if not calibration.empty:
                cal_show = calibration.copy()
                for c in ["Avg predicted P(home cover)","Actual home-cover rate","Calibration gap"]:
                    cal_show[c] = cal_show[c].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                with st.expander("Calibration table", expanded=False):
                    st.dataframe(cal_show, use_container_width=True, hide_index=True)

            imp = classifier_holdout_diag.get("importance", pd.DataFrame()) if isinstance(classifier_holdout_diag, dict) else pd.DataFrame()
            if isinstance(imp, pd.DataFrame) and not imp.empty:
                with st.expander("v0.8 classifier feature importance", expanded=False):
                    imp_show = imp.copy()
                    imp_show["Standardized coefficient"] = imp_show["Standardized coefficient"].map(lambda x: f"{x:+.3f}")
                    imp_show["Absolute importance"] = imp_show["Absolute importance"].map(lambda x: f"{x:.3f}")
                    st.dataframe(imp_show, use_container_width=True, hide_index=True)

            # Exports
            ios_save_button(
                "Save v0.8 Classifier Walk-Forward CSV",
                classifier_tests.to_csv(index=False),
                f"cfb_v080_classifier_walkforward_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            ios_save_button(
                "Save v0.8 Probability Buckets CSV",
                buckets.to_csv(index=False),
                f"cfb_v080_classifier_buckets_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            ios_save_button(
                "Save v0.8 Classifier Picks CSV",
                classifier_rows.to_csv(index=False),
                f"cfb_v080_classifier_picks_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            ios_save_button(
                "Save v0.8 Calibration CSV",
                calibration.to_csv(index=False),
                f"cfb_v080_classifier_calibration_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            if isinstance(imp, pd.DataFrame) and not imp.empty:
                ios_save_button(
                    "Save v0.8 Feature Importance CSV",
                    imp.to_csv(index=False),
                    f"cfb_v080_classifier_importance_holdout_{cfg.get('holdout',2025)}.csv"
                )

            st.info(
                "Promotion gate: classifier probabilities must improve on the 50/50 benchmark "
                "in multiple unseen seasons, remain reasonably calibrated, and show credible "
                "ATS performance in fixed probability buckets. No bucket may be chosen because "
                "it happened to work best in the 2025 holdout."
            )

        matchup_tests = st.session_state.get("cfb_matchup_tests_df", pd.DataFrame())
        matchup_bets = st.session_state.get("cfb_matchup_bets_df", pd.DataFrame())
        matchup_diag = st.session_state.get("cfb_matchup_diag", {})
        matchup_holdout = st.session_state.get("cfb_matchup_holdout_df", pd.DataFrame())
        matchup_holdout_bets = st.session_state.get("cfb_matchup_holdout_bets_df", pd.DataFrame())
        matchup_holdout_diag = st.session_state.get("cfb_matchup_holdout_diag", {})

        st.markdown("### v0.7.0 Matchup Model")
        st.caption(
            "Market-first residual model using pass/rush PPA, success rate, explosiveness, "
            "finishing drives, havoc, pace, talent and returning-production matchup features. "
            "All v0.7 labels are research-only until unseen-season validation clears the benchmark."
        )

        if isinstance(matchup_tests, pd.DataFrame) and not matchup_tests.empty:
            mae = _matchup_mae_table(matchup_tests)
            if not mae.empty:
                mae_show = mae.copy()
                for c in ["Market-only MAE", "v0.7 matchup MAE", "Improvement"]:
                    mae_show[c] = mae_show[c].map(lambda x: f"{x:.3f}")
                st.markdown("#### Rolling unseen-season prediction benchmark")
                st.dataframe(mae_show, use_container_width=True, hide_index=True)

            btbl = _matchup_bet_table(matchup_bets)
            if not btbl.empty:
                bshow = btbl.copy()
                bshow["Win %"] = bshow["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
                bshow["Units"] = bshow["Units"].map(lambda x: f"{x:+.2f}")
                bshow["ROI"] = bshow["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
                st.markdown("#### Fixed-hurdle research bets")
                st.dataframe(bshow, use_container_width=True, hide_index=True)

            if isinstance(matchup_holdout, pd.DataFrame) and not matchup_holdout.empty:
                s = matchup_holdout.dropna(subset=["spread_target_residual","matchup_pred_residual"])
                if not s.empty:
                    market_mae = float(np.mean(np.abs(s["spread_target_residual"])))
                    model_mae = float(np.mean(np.abs(s["spread_target_residual"] - s["matchup_pred_residual"])))
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"{cfg.get('holdout')} market MAE", f"{market_mae:.3f}")
                    c2.metric(f"{cfg.get('holdout')} v0.7 MAE", f"{model_mae:.3f}")
                    c3.metric("Holdout improvement", f"{market_mae-model_mae:+.3f}")

            # Most recent holdout feature importance.
            imp = matchup_holdout_diag.get("importance", pd.DataFrame()) if isinstance(matchup_holdout_diag, dict) else pd.DataFrame()
            if isinstance(imp, pd.DataFrame) and not imp.empty:
                with st.expander("v0.7 feature importance", expanded=False):
                    imp_show = imp.copy()
                    imp_show["Standardized coefficient"] = imp_show["Standardized coefficient"].map(lambda x: f"{x:+.3f}")
                    imp_show["Absolute importance"] = imp_show["Absolute importance"].map(lambda x: f"{x:.3f}")
                    st.dataframe(imp_show, use_container_width=True, hide_index=True)

            # Exports
            ios_save_button(
                "Save v0.7 Matchup Walk-Forward CSV",
                matchup_tests.to_csv(index=False),
                f"cfb_v070_matchup_walkforward_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            ios_save_button(
                "Save v0.7 Matchup Bets CSV",
                matchup_bets.to_csv(index=False),
                f"cfb_v070_matchup_bets_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )
            if isinstance(matchup_holdout_diag, dict):
                imp = matchup_holdout_diag.get("importance", pd.DataFrame())
                if isinstance(imp, pd.DataFrame) and not imp.empty:
                    ios_save_button(
                        "Save v0.7 Feature Importance CSV",
                        imp.to_csv(index=False),
                        f"cfb_v070_matchup_importance_holdout_{cfg.get('holdout',2025)}.csv"
                    )

            st.info(
                "Promotion gate: v0.7 should not reach the live betting board unless its spread MAE "
                "beats market-only across multiple unseen seasons and its fixed-hurdle research bets "
                "show credible multi-season performance."
            )

        research_df = st.session_state.get("cfb_signal_research_df", pd.DataFrame())
        research_wf = st.session_state.get("cfb_signal_walkforward_df", pd.DataFrame())

        st.markdown("### v0.6.0 Signal Research")
        st.caption(
            "Research mode only. Signals are ranked using development seasons only. "
            "The holdout never affects signal direction, cutoff, or ranking."
        )

        if isinstance(research_df, pd.DataFrame) and not research_df.empty:
            show = research_df.drop(columns=["Dev score"], errors="ignore").copy()
            for c in ["Dev win %","Dev ROI","Holdout win %","Holdout ROI"]:
                show[c] = show[c].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
            show["Dev corr"] = show["Dev corr"].map(lambda x: f"{x:+.3f}")
            show["Frozen cutoff"] = show["Frozen cutoff"].map(lambda x: f"{x:.3f}")
            show["Holdout units"] = show["Holdout units"].map(lambda x: f"{x:+.2f}")
            st.markdown("#### Development-ranked signals → untouched holdout")
            st.dataframe(show, use_container_width=True, hide_index=True)

            research_csv = research_df.to_csv(index=False)
            ios_save_button(
                "Save Signal Research CSV",
                research_csv,
                f"cfb_v062_signal_research_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )

        if isinstance(research_wf, pd.DataFrame) and not research_wf.empty:
            wf_show = research_wf.copy()
            wf_show["Frozen cutoff"] = wf_show["Frozen cutoff"].map(lambda x: f"{x:.3f}")
            wf_show["Win %"] = wf_show["Win %"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "—")
            wf_show["Units"] = wf_show["Units"].map(lambda x: f"{x:+.2f}")
            wf_show["ROI"] = wf_show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
            st.markdown("#### Signal survival by unseen season")
            st.dataframe(wf_show, use_container_width=True, hide_index=True)

            wf_csv = research_wf.to_csv(index=False)
            ios_save_button(
                "Save Signal Walk-Forward CSV",
                wf_csv,
                f"cfb_v062_signal_walkforward_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )

            agg = research_wf.groupby("Signal", as_index=False).agg(
                Seasons=("Test season","nunique"),
                Bets=("Bets","sum"),
                Units=("Units","sum"),
            )
            profitable = research_wf.assign(Pos=lambda x: x["Units"] > 0).groupby("Signal")["Pos"].sum()
            agg["Profitable unseen seasons"] = agg["Signal"].map(profitable).fillna(0).astype(int)
            agg["ROI"] = agg["Units"] / agg["Bets"].replace(0, np.nan)
            agg = agg.sort_values(
                ["Profitable unseen seasons","ROI","Bets"],
                ascending=[False,False,False]
            )
            agg_show = agg.copy()
            agg_show["Units"] = agg_show["Units"].map(lambda x: f"{x:+.2f}")
            agg_show["ROI"] = agg_show["ROI"].map(lambda x: f"{100*x:+.1f}%" if pd.notna(x) else "—")
            st.markdown("#### Multi-season survival summary")
            st.dataframe(agg_show, use_container_width=True, hide_index=True)

            agg_csv = agg.to_csv(index=False)
            ios_save_button(
                "Save Signal Summary CSV",
                agg_csv,
                f"cfb_v062_signal_summary_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv"
            )

        st.warning(
            "v0.6.0 does not create new official bets. A signal should only be considered for the "
            "live model after it survives multiple unseen seasons with enough sample size."
        )

        official = signal_df[signal_df["verdict"].isin(["BET","STRONG BET"])].copy()
        if len(official):
            by_type=[]
            for (version, mtype), d in official.groupby(["version","market_type"]):
                s=_bt_summary(d, mtype)
                by_type.append({"Version":version,"Market":mtype.title(),"Bets":s["Bets"],
                                "W-L-P":s["W-L-P"],
                                "Win %":f"{(d['result'].eq('WIN').sum()/max(1,(d['result'].isin(['WIN','LOSS'])).sum())):.1%}",
                                "ROI":f"{(d['profit_units'].sum()/len(d)):.1%}",
                                "Profit (u)":f"{d['profit_units'].sum():+.2f}"})
            st.markdown("### Official bets by market")
            st.dataframe(pd.DataFrame(by_type), use_container_width=True, hide_index=True)

            by_week=[]
            for (version, week), d in official.groupby(["version","week"]):
                denom=(d["result"].isin(["WIN","LOSS"])).sum()
                by_week.append({"Version":version,"Week":int(week),"Bets":len(d),
                                "Win %":f"{(d['result'].eq('WIN').sum()/denom):.1%}" if denom else "—",
                                "ROI":f"{(d['profit_units'].sum()/len(d)):.1%}"})
            with st.expander("Results by week", expanded=False):
                st.dataframe(pd.DataFrame(by_week).sort_values(["Version","Week"]),
                             use_container_width=True, hide_index=True)

        st.markdown("### Historical bet log")
        show_cols=["version","season","week","away_team","home_team","market_type","market",
                   "odds","prob","edge","ev","verdict","result","profit_units"]
        log = signal_df[show_cols].copy()
        for c in ["prob","edge","ev"]:
            log[c] = pd.to_numeric(log[c], errors="coerce").round(4)
        log["profit_units"] = pd.to_numeric(log["profit_units"], errors="coerce").round(3)
        st.dataframe(log, use_container_width=True, hide_index=True, height=520)

        csv = signal_df.to_csv(index=False)
        ios_save_button("Save Backtest CSV", csv,
                        f"cfb_v091_candidate_fastfix_backtest_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv")

        st.caption(
            "Historical CFBD line records are treated as generic provider snapshots/consensus medians; this app does not "
            "label them opening or closing lines. Spread and total prices are standardized at -110 because the generic "
            "CFBD line structure does not reliably include side-specific juice. Absolute ROI from retrospective full-season "
            "mode is not valid because of look-ahead bias."
        )
    st.stop()

# ===== v3.8 primary navigation state =====
if "cfb_page" not in st.session_state:
    st.session_state["cfb_page"] = "Slate"
_v38_page_alias = {"Home": "Game", "Bets": "Slate", "Live": "Slate"}
_v38_main_view = _v38_page_alias.get(
    st.session_state.get("cfb_page", "Slate"),
    st.session_state.get("cfb_page", "Slate"),
)

if _v38_main_view == "Game":
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">GAME TERMINAL</div>'
        '<div class="mobile-page-title">Game</div>'
        '<div class="mobile-page-sub">Drill into one matchup after reviewing the daily slate.</div></div>',
        unsafe_allow_html=True,
    )

with st.container(key="ge433_filter_row"):
    top1, top2 = st.columns([1, 1], gap="small")
    with top1:
        selected_date = st.date_input(
            "Game date",
            value=date.today(),
            label_visibility="collapsed",
            key="v420_game_date",
        )
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
        return bool(_conference(g, side))

    def is_major_team(g, side):
        return (
            _conference(g, side) in MAJOR_CONFERENCES
            or _team(g, side) in MAJOR_INDEPENDENTS
        )

    with top2:
        slate_filter = st.selectbox(
            "Game level",
            ["Major FBS", "All FBS", "All college games"],
            index=0,
            label_visibility="collapsed",
            key="v420_game_level",
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
    # LIVE market integrity selector (v3.6.3).
    # Never averages different quoted spreads/totals into a synthetic line.
    if not rows:
        return {}

    clean = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        x = dict(r)
        for c in ("home_spread", "away_spread", "total", "away_ml", "home_ml"):
            try:
                v = x.get(c)
                x[c] = None if v is None else float(v)
            except Exception:
                x[c] = None
        clean.append(x)

    if not clean:
        return {}

    def _actual_consensus(field, max_unsupported_range):
        vals = [float(r[field]) for r in clean if r.get(field) is not None]
        if not vals:
            return None, 0, None, None, "NO LINE"

        rounded = [round(v, 4) for v in vals]
        counts = {}
        for v in rounded:
            counts[v] = counts.get(v, 0) + 1

        max_count = max(counts.values())
        leaders = [v for v, n in counts.items() if n == max_count]
        center = float(np.median(vals))
        selected = min(leaders, key=lambda v: (abs(v - center), abs(v), v))
        quoted_range = max(vals) - min(vals) if len(vals) > 1 else 0.0

        if max_count < 2 and len(vals) >= 2 and quoted_range > float(max_unsupported_range):
            return None, len(vals), min(vals), max(vals), "REJECTED SPLIT FEED"

        return float(selected), len(vals), min(vals), max(vals), (
            "MULTI-BOOK MODE" if max_count >= 2 else "ACTUAL QUOTE NEAREST MEDIAN"
        )

    home_spread, spread_books, spread_min, spread_max, spread_status = _actual_consensus(
        "home_spread", 4.0
    )
    total, total_books, total_min, total_max, total_status = _actual_consensus(
        "total", 3.0
    )

    chosen = None
    if home_spread is not None:
        matches = [
            r for r in clean
            if r.get("home_spread") is not None
            and abs(float(r["home_spread"]) - float(home_spread)) < 1e-9
        ]
        if matches:
            matches.sort(
                key=lambda r: (
                    r.get("away_ml") is None,
                    r.get("home_ml") is None,
                    str(r.get("provider") or "")
                )
            )
            chosen = matches[0]

    if chosen is None and total is not None:
        matches = [
            r for r in clean
            if r.get("total") is not None
            and abs(float(r["total"]) - float(total)) < 1e-9
        ]
        if matches:
            chosen = matches[0]

    if chosen is None:
        chosen = clean[0]

    def _ml(name):
        v = chosen.get(name)
        if v is None:
            vals = [r.get(name) for r in clean if r.get(name) is not None]
            if not vals:
                return None
            return int(round(float(np.median(vals))))
        return int(round(float(v)))

    provider_name = str(chosen.get("provider") or "Unknown")
    return {
        "provider": provider_name,
        "away_ml": _ml("away_ml"),
        "home_ml": _ml("home_ml"),
        "home_spread": home_spread,
        "away_spread": -home_spread if home_spread is not None else None,
        "total": total,
        "line_integrity": "OK" if home_spread is not None else spread_status,
        "spread_consensus_status": spread_status,
        "spread_provider_count": spread_books,
        "spread_min_quote": spread_min,
        "spread_max_quote": spread_max,
        "total_consensus_status": total_status,
        "total_provider_count": total_books,
        "total_min_quote": total_min,
        "total_max_quote": total_max,
    }



# ===== v3.6 locked production daily card =====
V36_VERSION = "v3.6.0-production-daily-card"
V36_SCORE_FLOOR = 0.80
V36_LEAN_FLOOR = 0.78
V36_TRAIN_START = 2018
V36_CORE_CLASSIFIERS = ("Gradient Boosting", "Extra Trees", "Logistic")
V36_CORE_REGRESSORS = ("Gradient Boosting", "Extra Trees", "Random Forest")


def _v36_live_prepare(train_df, live_df, features, min_coverage=0.35):
    """
    Training medians and feature coverage are learned from historical games only.
    Live rows never influence feature selection or imputation.
    """
    usable, medians = [], {}
    for f in features:
        if f not in train_df.columns or f not in live_df.columns:
            continue
        s = pd.to_numeric(train_df[f], errors="coerce")
        if len(s) == 0 or float(s.notna().mean()) < min_coverage:
            continue
        med = float(s.median()) if s.notna().any() else 0.0
        usable.append(f)
        medians[f] = med

    if len(usable) < 5 or live_df.empty:
        return None

    Xtr = np.column_stack([
        pd.to_numeric(train_df[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    Xlive = np.column_stack([
        pd.to_numeric(live_df[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    return Xtr, Xlive, usable


@st.cache_resource(show_spinner=False)
def _v36_fit_live_spread_models(current_season, scope="Major FBS"):
    """
    Fit the locked v3.6 spread ensemble using completed seasons only.
    For 2026 this means 2018-2025 history. 2026 outcomes never enter the fit.
    """
    current_season = int(current_season)
    hist = _v3_history_frame(V36_TRAIN_START, current_season - 1, scope)
    if hist is None or hist.empty:
        return None

    h = hist.copy()
    h["spread_residual_target"] = (
        pd.to_numeric(h["actual_margin"], errors="coerce")
        - pd.to_numeric(h["market_margin"], errors="coerce")
    )
    d = pd.to_numeric(h["spread_residual_target"], errors="coerce")
    h["home_cover_target"] = np.where(
        d > 0, 1.0, np.where(d < 0, 0.0, np.nan)
    )

    spread_groups, _ = _v31_feature_groups()
    features = _v31_flatten(spread_groups) + ["market_margin"]

    # Historical feature coverage/medians are stored with the fitted package.
    usable, medians = [], {}
    for f in features:
        if f not in h.columns:
            continue
        s = pd.to_numeric(h[f], errors="coerce")
        if float(s.notna().mean()) < 0.35:
            continue
        usable.append(f)
        medians[f] = float(s.median()) if s.notna().any() else 0.0

    if len(usable) < 5:
        return None

    class_train = h.dropna(subset=["home_cover_target"]).copy()
    reg_train = h.dropna(subset=["spread_residual_target"]).copy()

    Xc = np.column_stack([
        pd.to_numeric(class_train[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    yc = pd.to_numeric(class_train["home_cover_target"], errors="coerce").to_numpy(dtype=int)

    Xr = np.column_stack([
        pd.to_numeric(reg_train[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in usable
    ])
    yr = pd.to_numeric(reg_train["spread_residual_target"], errors="coerce").to_numpy(dtype=float)

    classifiers = {}
    for name in V36_CORE_CLASSIFIERS:
        est = _v31_classifiers().get(name)
        if est is None:
            continue
        try:
            # Same point-in-time Platt scheme used in v3.1+ research.
            cal = _v31_inner_calibration(
                class_train, usable, "home_cover_target", name
            )
            est.fit(Xc, yc)
            classifiers[name] = (est, cal)
        except Exception:
            continue

    regressors = {}
    for name in V36_CORE_REGRESSORS:
        est = _v31_regressors().get(name)
        if est is None:
            continue
        try:
            est.fit(Xr, yr)
            regressors[name] = est
        except Exception:
            continue

    if len(classifiers) < 2 or len(regressors) < 2:
        return None

    return {
        "features": usable,
        "medians": medians,
        "classifiers": classifiers,
        "regressors": regressors,
        "train_rows": int(len(h)),
        "train_through": current_season - 1,
    }


def _v36_live_feature_frame(games_today, slate_df):
    if not games_today or slate_df is None or slate_df.empty:
        return pd.DataFrame()

    game_map = {str(g.get("id")): g for g in games_today if g.get("id") is not None}
    current_season = None
    for g in games_today:
        if g.get("season") is not None:
            current_season = int(g.get("season"))
            break
    if current_season is None:
        return pd.DataFrame()

    preseason = _v3_preseason_data(current_season)
    adv_cache = {}
    rows = []

    for _, sr in slate_df.iterrows():
        gid = str(sr.get("game_id"))
        g = game_map.get(gid)
        if g is None:
            continue

        hs = sr.get("market_home_spread")
        if hs is None or (isinstance(hs, float) and pd.isna(hs)):
            continue

        week = int(g.get("week") or 1)
        if week not in adv_cache:
            adv_cache[week] = _v3_advanced_through(current_season, week - 1)

        market = {
            "home_spread": float(hs),
            "total": sr.get("market_total"),
        }
        try:
            row = _v3_game_feature_row(
                g, market, preseason, adv_cache.get(week, {})
            )
        except Exception:
            continue

        row["row_index"] = len(rows)
        rows.append(row)

    return pd.DataFrame(rows)


def _v36_live_daily_card(games_today, slate_df, scope="Major FBS"):
    """
    v3.9.2 production edge engine:
    fair spread chooses the betting side; classifier/regression consensus
    measures reliability; cross-sectional score controls qualification.
    """
    live = _v36_live_feature_frame(games_today, slate_df)
    if live is None or live.empty:
        return pd.DataFrame()

    season = int(pd.to_numeric(live["season"], errors="coerce").dropna().iloc[0])
    fitted = _v36_fit_live_spread_models(season, scope)
    if not fitted:
        return pd.DataFrame()

    features = fitted["features"]
    medians = fitted["medians"]
    X = np.column_stack([
        pd.to_numeric(live[f], errors="coerce").fillna(medians[f]).to_numpy(dtype=float)
        for f in features
    ])

    # Classification consensus.
    class_pred = {}
    for name, (est, cal) in fitted["classifiers"].items():
        try:
            raw = est.predict_proba(X)[:, 1]
            class_pred[name] = _v31_platt_apply(cal, raw)
        except Exception:
            continue

    # Residual-regression consensus.
    reg_pred = {}
    for name, est in fitted["regressors"].items():
        try:
            reg_pred[name] = np.clip(
                np.asarray(est.predict(X), dtype=float),
                -V31_SPREAD_CORRECTION_CAP,
                V31_SPREAD_CORRECTION_CAP,
            )
        except Exception:
            continue

    if len(class_pred) < 2 or len(reg_pred) < 2:
        return pd.DataFrame()

    rows = []
    _sl = slate_df.reset_index(drop=True) if slate_df is not None and not slate_df.empty else pd.DataFrame()

    for i, r in live.reset_index(drop=True).iterrows():
        row_index = int(r.get("row_index", i))

        # v4.0: INDEPENDENT FUNDAMENTAL FAIR LINE DETERMINES THE BETTING SIDE.
        # Negative home spread means the home team is favored.
        # If fair home spread is lower/more negative than market, HOME has value.
        # If fair home spread is higher/less negative than market, AWAY has value.
        if _sl.empty or row_index < 0 or row_index >= len(_sl):
            continue

        sr = _sl.iloc[row_index]
        market_home = _v3_num(sr.get("market_home_spread"), np.nan)
        fair_home = _v3_num(sr.get("fundamental_home_spread", sr.get("raw_model_home_spread")), np.nan)

        if not np.isfinite(market_home) or not np.isfinite(fair_home):
            continue

        signed_home_edge = float(market_home - fair_home)
        if abs(signed_home_edge) < 1e-9:
            # True coin-flip price: no directional edge to rank.
            continue

        pick_side = "HOME" if signed_home_edge > 0 else "AWAY"

        # Classifiers are now reliability evidence, not the side picker.
        votes = []
        chosen_side_probs = []
        for name, probs in class_pred.items():
            p_home = float(probs[i])
            classifier_side = "HOME" if p_home >= 0.5 else "AWAY"
            chosen_prob = p_home if pick_side == "HOME" else (1.0 - p_home)
            votes.append((name, classifier_side, chosen_prob, p_home))
            chosen_side_probs.append(chosen_prob)

        agreeing = [v for v in votes if v[1] == pick_side]
        classifier_agreement = len(agreeing)
        classifier_confidence = (
            float(np.mean(chosen_side_probs)) if chosen_side_probs else 0.5
        )

        # Regression models also become reliability evidence around the
        # fair-line-selected side.
        reg_votes = []
        for name, vals in reg_pred.items():
            corr = float(vals[i])
            reg_votes.append((name, "HOME" if corr >= 0 else "AWAY", abs(corr), corr))

        reg_agree = sum(1 for v in reg_votes if v[1] == pick_side)
        reg_strengths = [v[2] for v in reg_votes if v[1] == pick_side]
        reg_consensus_side = (
            "HOME"
            if sum(1 for v in reg_votes if v[1] == "HOME")
               >= sum(1 for v in reg_votes if v[1] == "AWAY")
            else "AWAY"
        )

        rows.append({
            "row_index": row_index,
            "game_id": r.get("game_id"),
            "season": season,
            "week": int(_v3_num(r.get("week"), 1)),
            "kickoff_et": r.get("kickoff_et", ""),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "market_margin": _v3_num(r.get("market_margin")),
            "market_home_spread": float(market_home),
            "fair_home_spread": float(fair_home),
            "signed_home_edge": float(signed_home_edge),
            "point_edge": abs(float(signed_home_edge)),
            "pick_side": pick_side,
            "classifier_models": len(votes),
            "classifier_agreement": int(classifier_agreement),
            "classifier_confidence": float(classifier_confidence),
            "reg_side": reg_consensus_side,
            "reg_agreement": int(reg_agree),
            "reg_strength": float(np.mean(reg_strengths)) if reg_strengths else 0.0,
            "direction_agreement": 1.0 if reg_consensus_side == pick_side else 0.0,
            "data_maturity": 1.0 if int(_v3_num(r.get("week"), 1)) >= 4 else 0.0,
            "train_rows": fitted["train_rows"],
            "train_through": fitted["train_through"],
        })

    card = pd.DataFrame(rows)
    if card.empty:
        return card

    # Exact daily cross-sectional ranking used in v3.5.
    card["confidence_pct"] = card["classifier_confidence"].rank(
        method="average", pct=True
    )
    card["regression_pct"] = card["reg_strength"].rank(
        method="average", pct=True
    )
    card["agreement_rate"] = (
        card["classifier_agreement"] / card["classifier_models"].clip(lower=1)
    )
    card["selector_score"] = (
        0.40 * card["confidence_pct"]
        + 0.20 * card["agreement_rate"]
        + 0.20 * card["regression_pct"]
        + 0.10 * card["direction_agreement"]
        + 0.10 * card["data_maturity"]
    )
    card["day_rank"] = card["selector_score"].rank(
        method="first", ascending=False
    ).astype(int)
    card["day_size"] = int(len(card))

    # v4.0 PRODUCTION VERDICT
    # Official spread status comes from the independent fair-line probability
    # and price-adjusted EV grade. Selector score is reliability context only.
    _fg = card.get("fundamental_grade", pd.Series("PASS", index=card.index)).astype(str)
    card["verdict"] = np.select(
        [
            _fg.eq("STRONG BET"),
            _fg.eq("BET"),
            _fg.eq("LEAN"),
        ],
        ["BET", "BET", "LEAN"],
        default="PASS",
    )

    official_idx = card.index[card["verdict"] == "BET"].tolist()
    if official_idx:
        # BEST BET = strongest expected value, then cover probability, then
        # reliability score. No forced official bet if none clears the grade.
        _rank = card.loc[official_idx].copy()
        _rank["_ev_rank"] = pd.to_numeric(_rank.get("expected_value"), errors="coerce").fillna(-999)
        _rank["_p_rank"] = pd.to_numeric(_rank.get("cover_probability"), errors="coerce").fillna(0)
        _rank["_r_rank"] = pd.to_numeric(_rank.get("selector_score"), errors="coerce").fillna(0)
        best_idx = _rank.sort_values(
            ["_ev_rank", "_p_rank", "_r_rank"],
            ascending=[False, False, False],
        ).index[0]
        card.loc[best_idx, "verdict"] = "BEST BET"

    card["suggested_units"] = np.where(
        card["verdict"].isin(["BEST BET", "BET"]), 1.0, 0.0
    )

    home_spread = pd.to_numeric(card["market_home_spread"], errors="coerce")
    card["selection"] = np.where(
        card["pick_side"] == "HOME",
        card["home_team"].astype(str) + " " + home_spread.map(lambda v: f"{v:+.1f}"),
        card["away_team"].astype(str) + " " + (-home_spread).map(lambda v: f"{v:+.1f}"),
    )

    # v3.9 transparency layer: attach the market/fair-line information used
    # elsewhere in the app. These fields are display/diagnostic only and do
    # not change the production selector verdict.
    if slate_df is not None and not slate_df.empty and "row_index" in card.columns:
        _lookup_cols = [
            "home_logo",
            "away_logo",
            "fundamental_home_cover_prob",
            "fundamental_away_cover_prob",
            "fundamental_cover_prob",
            "fundamental_grade",
            "fundamental_prob_edge",
            "fundamental_ev",
            "model_confidence",
            "data_completeness",
            "market_source",
            "raw_model_home_spread",
            "adjusted_model_home_spread",
            "spread_residual_correction",
        ]
        _sl = slate_df.reset_index(drop=True)
        for _c in _lookup_cols:
            if _c in _sl.columns:
                card[_c] = card["row_index"].map(
                    lambda _i: _sl.iloc[int(_i)].get(_c)
                    if pd.notna(_i) and 0 <= int(_i) < len(_sl)
                    else np.nan
                )

    card["fair_home_spread"] = pd.to_numeric(
        card.get("fair_home_spread"), errors="coerce"
    )
    card["market_home_spread_display"] = pd.to_numeric(
        card.get("market_home_spread"), errors="coerce"
    )
    card["point_edge"] = pd.to_numeric(
        card.get("point_edge"), errors="coerce"
    )

    card["cover_probability"] = pd.to_numeric(
        card.get("fundamental_cover_prob"), errors="coerce"
    )
    card["probability_edge"] = pd.to_numeric(
        card.get("fundamental_prob_edge"), errors="coerce"
    )
    card["expected_value"] = pd.to_numeric(
        card.get("fundamental_ev"), errors="coerce"
    )

    card["probability_direction_ok"] = (
        (pd.to_numeric(card["point_edge"], errors="coerce") <= 1e-9)
        | (pd.to_numeric(card["cover_probability"], errors="coerce") >= 0.50)
    )
    # Never promote a mathematically inconsistent row.
    _bad_prob_dir = ~card["probability_direction_ok"].fillna(False)
    if _bad_prob_dir.any():
        card.loc[_bad_prob_dir, "fundamental_grade"] = "PASS"
        card.loc[_bad_prob_dir, "fundamental_prob_edge"] = np.nan
        card.loc[_bad_prob_dir, "fundamental_ev"] = np.nan

    # v4.0 final official status after probability-direction sanity check.
    _fg_final = card.get("fundamental_grade", pd.Series("PASS", index=card.index)).astype(str)
    card["verdict"] = np.select(
        [_fg_final.eq("STRONG BET"), _fg_final.eq("BET"), _fg_final.eq("LEAN")],
        ["BET", "BET", "LEAN"],
        default="PASS",
    )
    _official_final = card.index[card["verdict"] == "BET"].tolist()
    if _official_final:
        _rank_final = card.loc[_official_final].copy()
        _rank_final["_ev"] = pd.to_numeric(_rank_final.get("expected_value"), errors="coerce").fillna(-999)
        _rank_final["_p"] = pd.to_numeric(_rank_final.get("cover_probability"), errors="coerce").fillna(0)
        _best_final = _rank_final.sort_values(["_ev","_p"], ascending=[False,False]).index[0]
        card.loc[_best_final, "verdict"] = "BEST BET"
    card["suggested_units"] = np.where(
        card["verdict"].isin(["BEST BET","BET"]), 1.0, 0.0
    )


    # Reliability is context, not a hard gate in v4.0.
    # It blends the old ensemble agreement signal with fundamental data quality.
    _sel = pd.to_numeric(card["selector_score"], errors="coerce").fillna(0.0)
    _mc = pd.to_numeric(card.get("model_confidence"), errors="coerce").fillna(0.0)
    _dc = pd.to_numeric(card.get("data_completeness"), errors="coerce").fillna(0.0)
    card["reliability"] = np.select(
        [
            (_sel >= 0.84) & (_mc >= 76) & (_dc >= 0.80),
            (_sel >= 0.72) & (_mc >= 70) & (_dc >= 0.70),
            (_mc >= 66) & (_dc >= 0.60),
        ],
        ["PRIME", "HIGH", "MODERATE"],
        default="LOW",
    )

    return card.sort_values(
        ["selector_score", "classifier_confidence"],
        ascending=[False, False]
    ).reset_index(drop=True)


# v3.6 tracker path must not depend on the legacy tracker constant being defined later.
V36_TRACKER_PATH = Path(".cfb_edge_tracker") / "cfb_official_bet_tracker.csv"
V36_LEGACY_TRACKER_PATH = Path(".cfb_edge_tracker") / "cfb_v36_forward_tracker.csv"

V401_TRACKER_COLUMNS = [
    "record_key","model_version","game_date","game_id","kickoff_et",
    "home_team","away_team","home_logo","away_logo","selection","market_type","pick_side","bet_line","odds",
    "verdict","fundamental_grade","fair_home_spread","market_home_spread",
    "fair_total","market_total",
    "point_edge","cover_probability","probability_edge","expected_value",
    "reliability","model_confidence","data_completeness","selector_score",
    "suggested_units","frozen_at_et","status","result","units_result",
    "final_home_score","final_away_score","graded_at"
]

def _v401_empty_tracker():
    return pd.DataFrame(columns=V401_TRACKER_COLUMNS)

def _v401_clean_tracker(df):
    if df is None or df.empty:
        return _v401_empty_tracker()
    x = df.copy()

    # Legacy v3.6 migration.
    legacy_map = {
        "game_date":"game_date","game_id":"game_id","kickoff_et":"kickoff_et",
        "home_team":"home_team","away_team":"away_team","selection":"selection",
        "verdict":"verdict","selector_score":"selector_score",
        "suggested_units":"suggested_units","status":"status","result":"result",
        "units_result":"units_result","final_home_score":"final_home_score",
        "final_away_score":"final_away_score","graded_at":"graded_at",
        "model_version":"model_version",
    }
    for c in V401_TRACKER_COLUMNS:
        if c not in x.columns:
            x[c] = None

    # Infer side/line from frozen selection for old rows.
    for idx, r in x.iterrows():
        sel = str(r.get("selection") or "")
        if not r.get("record_key"):
            _mt = str(r.get("market_type") or "SPREAD").upper()
            x.at[idx, "record_key"] = f"{r.get('game_date')}|{r.get('game_id')}|{_mt}"
        if not r.get("market_type"):
            x.at[idx, "market_type"] = "SPREAD"
        if not r.get("pick_side"):
            ht = str(r.get("home_team") or "")
            at = str(r.get("away_team") or "")
            if sel.startswith(ht + " "):
                x.at[idx, "pick_side"] = "HOME"
            elif sel.startswith(at + " "):
                x.at[idx, "pick_side"] = "AWAY"
        if pd.isna(pd.to_numeric(pd.Series([r.get("bet_line")]), errors="coerce").iloc[0]):
            m = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", sel)
            if m:
                x.at[idx, "bet_line"] = float(m.group(1))
        if pd.isna(pd.to_numeric(pd.Series([r.get("odds")]), errors="coerce").iloc[0]):
            x.at[idx, "odds"] = -110
        if not r.get("status"):
            x.at[idx, "status"] = "FROZEN"
    return x[V401_TRACKER_COLUMNS]

def _v401_load_tracker():
    # Prefer the new unified tracker. If absent, migrate the old forward tracker.
    try:
        if V36_TRACKER_PATH.exists():
            return _v401_clean_tracker(pd.read_csv(V36_TRACKER_PATH))
        if V36_LEGACY_TRACKER_PATH.exists():
            old = _v401_clean_tracker(pd.read_csv(V36_LEGACY_TRACKER_PATH))
            V36_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
            old.to_csv(V36_TRACKER_PATH, index=False)
            return old
    except Exception:
        pass
    return _v401_empty_tracker()

def _v401_save_tracker(df):
    try:
        x = _v401_clean_tracker(df)
        V36_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = V36_TRACKER_PATH.with_suffix(".tmp")
        x.to_csv(tmp, index=False)
        tmp.replace(V36_TRACKER_PATH)
        return True
    except Exception:
        return False

def _v401_american_profit(odds, stake=1.0):
    try:
        o = float(odds)
        s = float(stake)
    except Exception:
        return None
    if o > 0:
        return s * (o / 100.0)
    if o < 0:
        return s * (100.0 / abs(o))
    return None

def _v401_kickoff_has_started(kickoff_et, selected_date=None):
    """Best-effort pregame freeze guard. Unknown kickoff is treated as not started."""
    if kickoff_et is None or str(kickoff_et).strip() == "":
        return False
    try:
        k = pd.to_datetime(kickoff_et, errors="coerce")
        if pd.isna(k):
            # Time-only string fallback.
            if selected_date is not None:
                k = pd.to_datetime(f"{selected_date} {kickoff_et}", errors="coerce")
        if pd.isna(k):
            return False
        if getattr(k, "tzinfo", None) is None:
            k = k.tz_localize("America/New_York")
        else:
            k = k.tz_convert("America/New_York")
        now = pd.Timestamp.now(tz="America/New_York")
        return now >= k
    except Exception:
        return False

def _v401_track_daily_card(card, selected_date):
    """
    Freeze the FIRST official recommendation per game + market type.
    A game may therefore have one official spread and one official total.
    Later reruns never rewrite the frozen line.
    """
    if card is None or card.empty:
        return 0

    official = card[card["verdict"].isin(["BEST BET","BET"])].copy()
    if official.empty:
        return 0

    tracker = _v401_load_tracker()
    existing = set(tracker["record_key"].astype(str)) if not tracker.empty else set()

    rows = []
    now = pd.Timestamp.now(tz="America/New_York").isoformat()

    for _, r in official.iterrows():
        gid = str(r.get("game_id") or "")
        if not gid:
            continue
        market_type = str(r.get("market_type") or "SPREAD").upper()
        key = f"{selected_date}|{gid}|{market_type}"
        if key in existing:
            continue
        if _v401_kickoff_has_started(r.get("kickoff_et"), selected_date):
            continue

        def fnum(name):
            try:
                v = float(r.get(name))
                return v if np.isfinite(v) else None
            except Exception:
                return None

        if market_type == "TOTAL":
            bet_line = fnum("bet_line")
        else:
            try:
                bet_line = float(
                    r.get("market_home_spread_display")
                    if str(r.get("pick_side")) == "HOME"
                    else -float(r.get("market_home_spread_display"))
                )
            except Exception:
                m = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", str(r.get("selection") or ""))
                bet_line = float(m.group(1)) if m else None

        rows.append({
            "record_key": key,
            "model_version": MODEL_VERSION,
            "game_date": str(selected_date),
            "game_id": gid,
            "kickoff_et": r.get("kickoff_et"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "home_logo": r.get("home_logo", ""),
            "away_logo": r.get("away_logo", ""),
            "selection": r.get("selection"),
            "market_type": market_type,
            "pick_side": r.get("pick_side"),
            "bet_line": bet_line,
            "odds": fnum("odds") or -110,
            "verdict": r.get("verdict"),
            "fundamental_grade": r.get("fundamental_grade"),
            "fair_home_spread": fnum("fair_home_spread"),
            "market_home_spread": fnum("market_home_spread_display"),
            "fair_total": fnum("fair_total"),
            "market_total": fnum("market_total"),
            "point_edge": fnum("point_edge"),
            "cover_probability": fnum("cover_probability"),
            "probability_edge": fnum("probability_edge"),
            "expected_value": fnum("expected_value"),
            "reliability": r.get("reliability"),
            "model_confidence": fnum("model_confidence"),
            "data_completeness": fnum("data_completeness"),
            "selector_score": fnum("selector_score"),
            "suggested_units": 1.0,
            "frozen_at_et": now,
            "status": "FROZEN",
            "result": "",
            "units_result": None,
            "final_home_score": None,
            "final_away_score": None,
            "graded_at": None,
        })
        existing.add(key)

    if not rows:
        return 0

    out = pd.concat([tracker, pd.DataFrame(rows)], ignore_index=True)
    _v401_save_tracker(out)
    return len(rows)


@st.cache_data(ttl=900, show_spinner=False)
def _v401_fetch_finals(date_strings):
    out = {}
    years = set()
    dates = set()
    for d in date_strings:
        try:
            dt = pd.Timestamp(str(d))
            dates.add(dt.strftime("%Y-%m-%d"))
            years.add(int(dt.year))
        except Exception:
            pass

    for y in sorted(years):
        try:
            games = get_backtest_games(y)
        except Exception:
            games = []
        for g in games or []:
            if g.get("completed") is not True:
                continue
            gid = str(g.get("id") or "")
            if not gid:
                continue
            start = str(g.get("startDate") or g.get("start_date") or "")[:10]
            if dates and start and start not in dates:
                continue
            if g.get("homePoints") is None or g.get("awayPoints") is None:
                continue
            out[gid] = {
                "home_score": float(g.get("homePoints")),
                "away_score": float(g.get("awayPoints")),
            }
    return out

def _v401_result_from_final(row, final):
    try:
        line = float(row.get("bet_line"))
        hs = float(final["home_score"])
        aas = float(final["away_score"])
    except Exception:
        return None

    market_type = str(row.get("market_type") or "SPREAD").upper()
    side = str(row.get("pick_side") or "").upper()

    if market_type == "TOTAL":
        actual_total = hs + aas
        if side == "OVER":
            margin = actual_total - line
        elif side == "UNDER":
            margin = line - actual_total
        else:
            return None
    else:
        if side == "HOME":
            margin = hs + line - aas
        elif side == "AWAY":
            margin = aas + line - hs
        else:
            sel = str(row.get("selection") or "")
            if sel.startswith(str(row.get("home_team") or "") + " "):
                margin = hs + line - aas
            elif sel.startswith(str(row.get("away_team") or "") + " "):
                margin = aas + line - hs
            else:
                return None

    if margin > 0:
        return "WIN"
    if margin < 0:
        return "LOSS"
    return "PUSH"


def _v401_grade_tracker():
    df = _v401_load_tracker()
    if df.empty:
        return 0, df

    pending = ~df["result"].astype(str).str.upper().isin(["WIN","LOSS","PUSH"])
    if not pending.any():
        return 0, df

    finals = _v401_fetch_finals(df.loc[pending, "game_date"].tolist())
    if not finals:
        return 0, df

    n = 0
    now = pd.Timestamp.now(tz="America/New_York").isoformat()
    for idx, r in df.loc[pending].iterrows():
        final = finals.get(str(r.get("game_id") or ""))
        if not final:
            continue
        result = _v401_result_from_final(r, final)
        if result is None:
            continue

        stake = float(r.get("suggested_units") or 1.0)
        odds = float(r.get("odds") or -110)
        if result == "WIN":
            units = _v401_american_profit(odds, stake) or 0.0
        elif result == "LOSS":
            units = -stake
        else:
            units = 0.0

        df.at[idx, "result"] = result
        df.at[idx, "units_result"] = round(float(units), 6)
        df.at[idx, "status"] = "FINAL"
        df.at[idx, "final_home_score"] = final["home_score"]
        df.at[idx, "final_away_score"] = final["away_score"]
        df.at[idx, "graded_at"] = now
        n += 1

    if n:
        _v401_save_tracker(df)
    return n, df

def _v401_summary(df):
    if df is None or df.empty:
        return {
            "bets":0,"graded":0,"pending":0,"wins":0,"losses":0,"pushes":0,
            "win_rate":0.0,"units":0.0,"roi":0.0
        }
    x = df.copy()
    res = x["result"].astype(str).str.upper()
    graded = res.isin(["WIN","LOSS","PUSH"])
    wins = int((res=="WIN").sum())
    losses = int((res=="LOSS").sum())
    pushes = int((res=="PUSH").sum())
    units = pd.to_numeric(x.get("units_result"), errors="coerce").fillna(0.0).sum()
    risked = pd.to_numeric(
        x.loc[graded, "suggested_units"], errors="coerce"
    ).fillna(0.0).sum()
    return {
        "bets":int(len(x)),
        "graded":int(graded.sum()),
        "pending":int((~graded).sum()),
        "wins":wins,"losses":losses,"pushes":pushes,
        "win_rate":wins / max(wins+losses,1),
        "units":float(units),
        "roi":float(units)/risked if risked>0 else 0.0,
    }

def _v401_split_table(df, field, label):
    if df is None or df.empty or field not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(field, dropna=False):
        s = _v401_summary(g)
        rows.append({
            label: str(key),
            "Bets": s["bets"],
            "Record": f'{s["wins"]}-{s["losses"]}-{s["pushes"]}',
            "Win Rate": s["win_rate"],
            "Units": s["units"],
            "ROI": s["roi"],
        })
    return pd.DataFrame(rows)

def _v401_render_official_tracker():
    graded_now, df = _v401_grade_tracker()
    s = _v401_summary(df)

    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">OFFICIAL BET LEDGER</div>'
        '<div class="mobile-page-title">Tracker</div>'
        '<div class="mobile-page-sub">Official BET / BEST BET recommendations are frozen before kickoff and automatically graded after the final score.</div></div>',
        unsafe_allow_html=True,
    )

    if graded_now:
        st.success(f"Auto-graded {graded_now} completed official bet(s).")

    if df is None or df.empty:
        st.info("No official bets have been frozen yet. Run a slate; BET / BEST BET recommendations will be added automatically.")
        return

    # All-time headline.
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Record", f'{s["wins"]}-{s["losses"]}-{s["pushes"]}')
    c2.metric("Win Rate", f'{s["win_rate"]:.1%}')
    c3.metric("Units", f'{s["units"]:+.2f}u')
    c4.metric("ROI", f'{s["roi"]:+.1%}')

    st.caption(
        f'{s["bets"]} frozen official bets • {s["graded"]} graded • {s["pending"]} pending • '
        '1.0u flat risk • frozen line never rewritten'
    )

    # Current v4.0.1+ architecture performance separated from migrated legacy history.
    current = df[df["model_version"].astype(str).str.startswith("4.")].copy()
    if not current.empty:
        cs = _v401_summary(current)
        st.markdown('<div class="section-kicker">V4 FUNDAMENTAL ENGINE</div>', unsafe_allow_html=True)
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Record", f'{cs["wins"]}-{cs["losses"]}-{cs["pushes"]}')
        d2.metric("Bets", cs["bets"])
        d3.metric("Units", f'{cs["units"]:+.2f}u')
        d4.metric("ROI", f'{cs["roi"]:+.1%}')

    # Cumulative performance.
    graded = df[df["result"].astype(str).str.upper().isin(["WIN","LOSS","PUSH"])].copy()
    if not graded.empty:
        graded["_date"] = pd.to_datetime(graded["game_date"], errors="coerce")
        graded["_graded"] = pd.to_datetime(graded["graded_at"], errors="coerce")
        graded = graded.sort_values(["_date","_graded"], na_position="last")
        graded["_units"] = pd.to_numeric(graded["units_result"], errors="coerce").fillna(0)
        graded["Cumulative Units"] = graded["_units"].cumsum()
        chart = graded[["_date","Cumulative Units"]].dropna()
        if not chart.empty:
            st.line_chart(chart.set_index("_date"), use_container_width=True)

    # Pending.
    pending = df[~df["result"].astype(str).str.upper().isin(["WIN","LOSS","PUSH"])].copy()
    if not pending.empty:
        with st.expander(f"Pending official bets — {len(pending)}", expanded=True):
            for _, _pr in pending.head(8).iterrows():
                st.markdown(
                    f"""
                    <div class="ge440-pending-row">
                      {_ge440_pick_logo(_pr, 25)}
                      <div><b>{html.escape(str(_pr.get("selection","")))}</b>
                      <span>{html.escape(str(_pr.get("away_team","")))} @ {html.escape(str(_pr.get("home_team","")))}</span></div>
                      <em>{html.escape(str(_pr.get("market_type","SPREAD")))}</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            show = pending[[c for c in [
                "game_date","kickoff_et","market_type","selection","verdict","point_edge",
                "cover_probability","expected_value","reliability","model_version"
            ] if c in pending.columns]].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)

    # Performance diagnostics.
    with st.expander("Performance by model version", expanded=False):
        t = _v401_split_table(df, "model_version", "Model")
        if not t.empty:
            for c in ["Win Rate","ROI"]:
                t[c] = t[c].map(lambda v: f"{v:.1%}")
            t["Units"] = t["Units"].map(lambda v: f"{v:+.2f}")
            st.dataframe(t, use_container_width=True, hide_index=True)

    with st.expander("Performance by market", expanded=False):
        t = _v401_split_table(df, "market_type", "Market")
        if not t.empty:
            for c in ["Win Rate","ROI"]:
                t[c] = t[c].map(lambda v: f"{v:.1%}")
            t["Units"] = t["Units"].map(lambda v: f"{v:+.2f}")
            st.dataframe(t, use_container_width=True, hide_index=True)

    with st.expander("Performance by reliability", expanded=False):
        t = _v401_split_table(df, "reliability", "Reliability")
        if not t.empty:
            for c in ["Win Rate","ROI"]:
                t[c] = t[c].map(lambda v: f"{v:.1%}")
            t["Units"] = t["Units"].map(lambda v: f"{v:+.2f}")
            st.dataframe(t, use_container_width=True, hide_index=True)

    with st.expander("Full official bet history", expanded=False):
        show = df[[c for c in [
            "game_date","market_type","selection","odds","verdict","fair_home_spread","fair_total",
            "point_edge","cover_probability","expected_value","reliability",
            "result","units_result","final_away_score","final_home_score","model_version"
        ] if c in df.columns]].copy()
        st.dataframe(show.sort_values("game_date", ascending=False), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Official Bet Tracker",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cfb_official_bet_tracker.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_v401_official_tracker",
    )

    with st.expander("Tracker backup / restore", expanded=False):
        up = st.file_uploader("Restore official tracker CSV", type=["csv"], key="v401_tracker_restore")
        if st.button("Merge Tracker Backup", disabled=(up is None), use_container_width=True, key="v401_tracker_merge"):
            try:
                incoming = _v401_clean_tracker(pd.read_csv(up))
                merged = pd.concat([df,incoming], ignore_index=True)
                merged = merged.drop_duplicates("record_key", keep="first")
                _v401_save_tracker(merged)
                st.success(f"Tracker restored: {len(merged)} official records.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not restore tracker: {e}")


def _v410_total_card(slate_df):
    """Convert v4.1 independent total projections into production-card rows."""
    if slate_df is None or slate_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in slate_df.iterrows():
        side = str(r.get("fundamental_total_side") or "")
        market_total = pd.to_numeric(pd.Series([r.get("market_total")]), errors="coerce").iloc[0]
        fair_total = pd.to_numeric(pd.Series([r.get("fundamental_total")]), errors="coerce").iloc[0]
        prob = pd.to_numeric(pd.Series([r.get("fundamental_total_prob")]), errors="coerce").iloc[0]
        edge = pd.to_numeric(pd.Series([r.get("fundamental_total_edge")]), errors="coerce").iloc[0]
        ev = pd.to_numeric(pd.Series([r.get("fundamental_total_ev")]), errors="coerce").iloc[0]
        p_edge = pd.to_numeric(pd.Series([r.get("fundamental_total_prob_edge")]), errors="coerce").iloc[0]
        grade_name = str(r.get("fundamental_total_grade") or "PASS")

        if side not in {"OVER","UNDER"} or pd.isna(market_total):
            continue

        verdict = (
            "BET" if grade_name in {"STRONG BET","BET"}
            else "LEAN" if grade_name == "LEAN"
            else "PASS"
        )

        rows.append({
            "model_version": MODEL_VERSION,
            "market_type": "TOTAL",
            "game_date": r.get("game_date"),
            "game_id": r.get("game_id"),
            "kickoff_et": r.get("kickoff_et"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "home_logo": r.get("home_logo", ""),
            "away_logo": r.get("away_logo", ""),
            "selection": f"{side.title()} {float(market_total):g}",
            "pick_side": side,
            "bet_line": float(market_total),
            "market_display": float(market_total),
            "fair_display": float(fair_total) if pd.notna(fair_total) else None,
            "point_edge": float(edge) if pd.notna(edge) else None,
            "cover_probability": float(prob) if pd.notna(prob) else None,
            "probability_edge": float(p_edge) if pd.notna(p_edge) else None,
            "expected_value": float(ev) if pd.notna(ev) else None,
            "fundamental_grade": grade_name,
            "verdict": verdict,
            "suggested_units": 1.0 if verdict == "BET" else 0.0,
            "model_confidence": r.get("model_confidence"),
            "data_completeness": r.get("data_completeness"),
            "fcs_fallback_used": r.get("fcs_fallback_used"),
            "market_total": float(market_total),
            "fair_total": float(fair_total) if pd.notna(fair_total) else None,
            "odds": -110,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Highest-EV official total gets BEST BET only within totals. When combined
    # with spreads below, a single overall BEST BET is chosen.
    return out

def _v410_combine_cards(spread_card, total_card):
    """Normalize spread + total candidates into one simple production card."""
    frames = []

    if spread_card is not None and not spread_card.empty:
        s = spread_card.copy()
        s["market_type"] = "SPREAD"
        s["market_display"] = pd.to_numeric(s.get("market_home_spread_display"), errors="coerce")
        s["fair_display"] = pd.to_numeric(s.get("fair_home_spread"), errors="coerce")
        # Market/fair display must follow the selected side for readability.
        away_mask = s["pick_side"].astype(str).eq("AWAY")
        s.loc[away_mask, "market_display"] = -s.loc[away_mask, "market_display"]
        s.loc[away_mask, "fair_display"] = -s.loc[away_mask, "fair_display"]
        frames.append(s)

    if total_card is not None and not total_card.empty:
        frames.append(total_card.copy())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    # Reset previous per-market BEST BET labels, then choose ONE best official
    # wager across spreads and totals by modeled EV, then probability.
    out.loc[out["verdict"].eq("BEST BET"), "verdict"] = "BET"
    official = out[out["verdict"].eq("BET")].copy()
    if not official.empty:
        official["_ev"] = pd.to_numeric(official.get("expected_value"), errors="coerce").fillna(-999)
        official["_p"] = pd.to_numeric(official.get("cover_probability"), errors="coerce").fillna(0)
        best_idx = official.sort_values(["_ev","_p"], ascending=[False,False]).index[0]
        out.loc[best_idx, "verdict"] = "BEST BET"

    return out


def _ge440_team_logo(row, team, side=None, size=40):
    team = str(team or "")
    logo = ""
    if side == "HOME":
        logo = str(row.get("home_logo", "") or "")
    elif side == "AWAY":
        logo = str(row.get("away_logo", "") or "")
    else:
        if team == str(row.get("home_team", "")):
            logo = str(row.get("home_logo", "") or "")
        elif team == str(row.get("away_team", "")):
            logo = str(row.get("away_logo", "") or "")

    initials = "".join(w[0] for w in team.split()[:2] if w)[:2].upper() or "CF"
    if logo:
        return (
            f'<div class="ge440-logo-shell" style="width:{size}px;height:{size}px">'
            f'<img class="ge440-team-logo" src="{html.escape(logo, quote=True)}" '
            f'alt="{html.escape(team, quote=True)}" loading="lazy">'
            f'</div>'
        )
    return (
        f'<div class="ge440-logo-shell" style="width:{size}px;height:{size}px">'
        f'<span class="ge440-logo-fallback" style="display:flex">{html.escape(initials)}</span>'
        f'</div>'
    )


def _ge440_pick_logo(row, size=40):
    mtype = str(row.get("market_type", "")).upper()
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))

    if mtype == "TOTAL":
        return (
            '<div class="ge440-logo-pair">'
            + _ge440_team_logo(row, away, "AWAY", size)
            + _ge440_team_logo(row, home, "HOME", size)
            + '</div>'
        )

    side = str(row.get("pick_side", "")).upper()
    if side == "HOME":
        return _ge440_team_logo(row, home, "HOME", size)
    if side == "AWAY":
        return _ge440_team_logo(row, away, "AWAY", size)

    selection = str(row.get("selection", ""))
    if selection.startswith(home):
        return _ge440_team_logo(row, home, "HOME", size)
    return _ge440_team_logo(row, away, "AWAY", size)


def _ge440_matchup_logos(row, size=28):
    away = str(row.get("away_team", ""))
    home = str(row.get("home_team", ""))
    return (
        '<div class="ge440-matchup-logos">'
        + _ge440_team_logo(row, away, "AWAY", size)
        + _ge440_team_logo(row, home, "HOME", size)
        + '</div>'
    )


def _v410_line_text(row, field):
    try:
        v = float(row.get(field))
        if str(row.get("market_type")) == "TOTAL":
            return f"{v:.1f}"
        return f"{v:+.1f}"
    except Exception:
        return "—"


def _v390_spread_text(v):
    try:
        return f"{float(v):+.1f}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"

def _v390_prob_text(v):
    try:
        return f"{float(v):.1%}" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"

def _v390_edge_text(v):
    try:
        return f"+{float(v):.1f} pts" if np.isfinite(float(v)) else "—"
    except Exception:
        return "—"

def _render_v36_live_card(card, selected_date):
    """
    v4.3 Gridiron Edge presentation.
    Official bets are the primary card. Leans are compact secondary rows.
    Spreads and totals remain ranked together; model math is unchanged.
    """
    if card is None or card.empty:
        st.info("No qualifying spread or total candidates are available.")
        return

    ranked = card.copy()
    ranked["expected_value"] = pd.to_numeric(ranked.get("expected_value"), errors="coerce")
    ranked["cover_probability"] = pd.to_numeric(ranked.get("cover_probability"), errors="coerce")
    ranked["point_edge"] = pd.to_numeric(ranked.get("point_edge"), errors="coerce")

    tier_order = {"BEST BET": 0, "BET": 1, "LEAN": 2, "PASS": 9}
    ranked["_tier"] = ranked["verdict"].map(tier_order).fillna(9)
    ranked = ranked.sort_values(
        ["_tier","expected_value","cover_probability","point_edge"],
        ascending=[True,False,False,False],
        na_position="last",
    ).reset_index(drop=True)

    official = ranked[ranked["verdict"].isin(["BEST BET","BET"])].copy()
    leans = ranked[ranked["verdict"] == "LEAN"].copy()

    if official.empty:
        st.markdown(
            """
            <div class="ge-official-head">
              <div class="ge-check muted">—</div>
              <div><div class="ge-section-title">Official Bets</div>
              <div class="ge-section-sub">No official plays cleared the production criteria.</div></div>
              <div class="ge-count">0 PLAYS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="edge-empty ge-empty">
              <div class="edge-status-pass">NO OFFICIAL BETS</div>
              <div class="edge-empty-title">Pass the slate.</div>
              <div class="edge-empty-copy">The model found no spread or total strong enough to bet officially.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ge-official-head">
              <div class="ge-check">✓</div>
              <div><div class="ge-section-title">Official Bets</div>
              <div class="ge-section-sub">Highest-conviction plays from the current slate.</div></div>
              <div class="ge-count">{len(official)} {'PLAY' if len(official)==1 else 'PLAYS'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, (_, r) in enumerate(official.iterrows(), start=1):
            verdict = str(r.get("verdict") or "BET")
            market_type = str(r.get("market_type") or "SPREAD").upper()
            cls = "best" if verdict == "BEST BET" else "bet"
            st.markdown(
                f"""
                <div class="ge-official-card {cls}">
                  <div class="ge-card-top">
                    <div class="ge-rank">{i}</div>
                    <span class="ge-market">{html.escape(market_type)}</span>
                    <span class="ge-verdict {cls}">{'★ ' if cls == 'best' else ''}{html.escape(verdict)}</span>
                  </div>
                  <div class="ge431-pickline">
                    {_ge440_pick_logo(r, 42)}
                    <div class="ge431-pickcopy">
                      <div class="ge-pick">{html.escape(str(r.get("selection","")))}</div>
                      <div class="ge-game">{html.escape(str(r.get("away_team","")))} @ {html.escape(str(r.get("home_team","")))}</div>
                    </div>
                  </div>
                  <div class="ge-metric-grid">
                    <div><span>Market</span><b>{_v410_line_text(r, "market_display")}</b></div>
                    <div><span>Fair</span><b>{_v410_line_text(r, "fair_display")}</b></div>
                    <div><span>Edge</span><b>{_v390_edge_text(r.get("point_edge"))}</b></div>
                    <div><span>Cover</span><b>{_v390_prob_text(r.get("cover_probability"))}</b></div>
                    <div><span>EV</span><b>{_v390_prob_text(r.get("expected_value"))}</b></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not leans.empty:
        st.markdown(
            f"""
            <div class="ge-leans-head">
              <div><div class="ge-section-title">Next Best <span>(Leans)</span></div>
              <div class="ge-section-sub">Interesting value, but not official bets.</div></div>
              <div class="ge-count amber">{len(leans)} {'PLAY' if len(leans)==1 else 'PLAYS'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview = leans.head(3)
        for j, (_, r) in enumerate(preview.iterrows(), start=len(official)+1):
            market_type = str(r.get("market_type") or "SPREAD").upper()
            st.markdown(
                f"""
                <div class="ge-lean-row">
                  <div class="ge-lean-rank">{j}</div>
                  <div class="ge440-lean-logo">{_ge440_pick_logo(r, 27)}</div>
                  <div class="ge-lean-main">
                    <div><span class="ge-market small">{html.escape(market_type)}</span>
                    <b>{html.escape(str(r.get("selection","")))}</b></div>
                    <small>{html.escape(str(r.get("away_team","")))} @ {html.escape(str(r.get("home_team","")))}</small>
                  </div>
                  <div class="ge-lean-stats">
                    <span>{_v390_edge_text(r.get("point_edge"))}</span>
                    <span>{_v390_prob_text(r.get("cover_probability"))}</span>
                    <span>{_v390_prob_text(r.get("expected_value"))}</span>
                  </div>
                  <div class="ge-lean-pill">LEAN</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if len(leans) > 3:
            with st.expander(f"View all leans ({len(leans)})", expanded=False):
                extra = leans.iloc[3:]
                for j, (_, r) in enumerate(extra.iterrows(), start=len(official)+4):
                    st.markdown(
                        f"""
                        <div class="ge-lean-row compact">
                          <div class="ge-lean-rank">{j}</div>
                          <div class="ge440-lean-logo">{_ge440_pick_logo(r, 25)}</div>
                          <div class="ge-lean-main"><div><b>{html.escape(str(r.get("selection","")))}</b></div>
                          <small>{html.escape(str(r.get("away_team","")))} @ {html.escape(str(r.get("home_team","")))}</small></div>
                          <div class="ge-lean-stats"><span>{_v390_edge_text(r.get("point_edge"))}</span>
                          <span>{_v390_prob_text(r.get("cover_probability"))}</span>
                          <span>{_v390_prob_text(r.get("expected_value"))}</span></div>
                          <div class="ge-lean-pill">LEAN</div>
                        </div>
                        """, unsafe_allow_html=True)




# ===== CFB v2.1 forward recommendation tracker =====
CFB_TRACKER_DIR = Path(".cfb_edge_tracker")
CFB_TRACKER_PATH = CFB_TRACKER_DIR / "cfb_recommendation_tracker.csv"

CFB_TRACKER_COLUMNS = [
    "Record_Key","Tracked_At_ET","Slate_Date","Game_ID","Game","Kickoff_ET",
    "Market_Type","Pick","Side","Line","Odds","Grade","Edge","EV","Confidence",
    "Result","Units","Final_Away_Score","Final_Home_Score","Graded_At_ET",
]

def _empty_cfb_tracker():
    return pd.DataFrame(columns=CFB_TRACKER_COLUMNS)

def _clean_cfb_tracker(df):
    if df is None or df.empty:
        return _empty_cfb_tracker()
    out = df.copy()
    for c in CFB_TRACKER_COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[CFB_TRACKER_COLUMNS]

def _load_cfb_tracker():
    if "_cfb_tracker_df" in st.session_state:
        return _clean_cfb_tracker(st.session_state["_cfb_tracker_df"])
    try:
        df = pd.read_csv(CFB_TRACKER_PATH) if CFB_TRACKER_PATH.exists() else _empty_cfb_tracker()
    except Exception:
        df = _empty_cfb_tracker()
    st.session_state["_cfb_tracker_df"] = _clean_cfb_tracker(df)
    return _clean_cfb_tracker(df)

def _save_cfb_tracker(df):
    df = _clean_cfb_tracker(df)
    st.session_state["_cfb_tracker_df"] = df.copy()
    try:
        CFB_TRACKER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CFB_TRACKER_PATH.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(CFB_TRACKER_PATH)
        return True
    except Exception:
        return False

def _parse_cfb_market(market):
    market = str(market or "").strip()
    if market.endswith(" ML"):
        team = market[:-3].strip()
        return {"type":"MONEYLINE","pick":team,"side":team,"line":None}
    m = re.match(r"^(Over|Under)\s+(-?\d+(?:\.\d+)?)$", market, flags=re.IGNORECASE)
    if m:
        return {
            "type":"TOTAL",
            "pick":market,
            "side":m.group(1).upper(),
            "line":float(m.group(2)),
        }
    m = re.match(r"^(.*?)\s+([+-]\d+(?:\.\d+)?)$", market)
    if m:
        team = m.group(1).strip()
        return {
            "type":"SPREAD",
            "pick":market,
            "side":team,
            "line":float(m.group(2)),
        }
    return {"type":"OTHER","pick":market,"side":market,"line":None}

def _track_cfb_official_board(market_board, slate_df, slate_date):
    """Freeze the first Best Bet / Bet price for each official model market."""
    if market_board is None or market_board.empty:
        return 0

    official = market_board[market_board["grade"].isin(["A","B"])].copy()
    if official.empty:
        return 0

    tracker = _load_cfb_tracker()
    existing = set(tracker["Record_Key"].astype(str))
    new_rows = []

    for _, r in official.iterrows():
        gid = r.get("game_id")
        market = str(r.get("market") or "")
        if gid is None or market == "":
            continue
        key = f"{gid}|{market}"
        if key in existing:
            continue

        parsed = _parse_cfb_market(market)
        try:
            odds = int(round(float(r.get("odds"))))
        except Exception:
            odds = -110

        row = {
            "Record_Key": key,
            "Tracked_At_ET": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "Slate_Date": str(slate_date),
            "Game_ID": gid,
            "Game": r.get("game"),
            "Kickoff_ET": r.get("kickoff_et"),
            "Market_Type": parsed["type"],
            "Pick": parsed["pick"],
            "Side": parsed["side"],
            "Line": parsed["line"],
            "Odds": odds,
            "Grade": r.get("grade"),
            "Edge": r.get("edge"),
            "EV": r.get("ev"),
            "Confidence": r.get("confidence"),
            "Result": "PENDING",
            "Units": 0.0,
            "Final_Away_Score": None,
            "Final_Home_Score": None,
            "Graded_At_ET": None,
        }
        new_rows.append(row)
        existing.add(key)

    if new_rows:
        tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)
        _save_cfb_tracker(tracker)
    return len(new_rows)

def _american_win_units(odds):
    try:
        o = float(odds)
    except Exception:
        o = -110.0
    return o / 100.0 if o > 0 else 100.0 / abs(o)

def _score_for_pick(game, team):
    away = str(game.get("awayTeam") or "")
    home = str(game.get("homeTeam") or "")
    a = pd.to_numeric(pd.Series([game.get("awayPoints")]), errors="coerce").iloc[0]
    h = pd.to_numeric(pd.Series([game.get("homePoints")]), errors="coerce").iloc[0]
    if pd.isna(a) or pd.isna(h):
        return None, None
    a, h = float(a), float(h)
    if team_key(team) == team_key(away):
        return a, h
    if team_key(team) == team_key(home):
        return h, a
    return None, None

def _settle_cfb_tracker_row(rec, game):
    market_type = str(rec.get("Market_Type") or "").upper()
    side = str(rec.get("Side") or "")
    line = pd.to_numeric(pd.Series([rec.get("Line")]), errors="coerce").iloc[0]
    away_score = pd.to_numeric(pd.Series([game.get("awayPoints")]), errors="coerce").iloc[0]
    home_score = pd.to_numeric(pd.Series([game.get("homePoints")]), errors="coerce").iloc[0]

    if pd.isna(away_score) or pd.isna(home_score):
        return None
    away_score, home_score = float(away_score), float(home_score)

    if market_type == "MONEYLINE":
        picked, opp = _score_for_pick(game, side)
        if picked is None:
            return None
        if picked > opp:
            return "WIN"
        if picked < opp:
            return "LOSS"
        return "PUSH"

    if market_type == "SPREAD":
        picked, opp = _score_for_pick(game, side)
        if picked is None or pd.isna(line):
            return None
        covered = picked + float(line) - opp
        if covered > 0:
            return "WIN"
        if covered < 0:
            return "LOSS"
        return "PUSH"

    if market_type == "TOTAL":
        if pd.isna(line):
            return None
        total = away_score + home_score
        if side.upper() == "OVER":
            if total > float(line):
                return "WIN"
            if total < float(line):
                return "LOSS"
            return "PUSH"
        if side.upper() == "UNDER":
            if total < float(line):
                return "WIN"
            if total > float(line):
                return "LOSS"
            return "PUSH"

    return None

def _grade_cfb_tracker(games_today):
    tracker = _load_cfb_tracker()
    if tracker.empty:
        return 0
    game_map = {str(g.get("id")): g for g in games_today}
    changed = 0

    for idx, rec in tracker.iterrows():
        if str(rec.get("Result","PENDING")) != "PENDING":
            continue
        game = game_map.get(str(rec.get("Game_ID")))
        if not game or not bool(game.get("completed")):
            continue
        result = _settle_cfb_tracker_row(rec, game)
        if result is None:
            continue

        tracker.at[idx,"Result"] = result
        if result == "WIN":
            tracker.at[idx,"Units"] = _american_win_units(rec.get("Odds"))
        elif result == "LOSS":
            tracker.at[idx,"Units"] = -1.0
        else:
            tracker.at[idx,"Units"] = 0.0
        tracker.at[idx,"Final_Away_Score"] = game.get("awayPoints")
        tracker.at[idx,"Final_Home_Score"] = game.get("homePoints")
        tracker.at[idx,"Graded_At_ET"] = pd.Timestamp.now(tz="America/New_York").isoformat()
        changed += 1

    if changed:
        _save_cfb_tracker(tracker)
    return changed

def _cfb_live_market_status(rec, game):
    market_type = str(rec.get("Market_Type") or "").upper()
    side = str(rec.get("Side") or "")
    try:
        line = float(rec.get("Line"))
    except Exception:
        line = None

    a = pd.to_numeric(pd.Series([game.get("awayPoints")]), errors="coerce").iloc[0]
    h = pd.to_numeric(pd.Series([game.get("homePoints")]), errors="coerce").iloc[0]
    if pd.isna(a) or pd.isna(h):
        return "WAITING", "neutral", None
    a, h = float(a), float(h)

    if market_type == "MONEYLINE":
        picked, opp = _score_for_pick(game, side)
        if picked is None:
            return "WAITING","neutral",None
        diff = picked - opp
        return (
            ("LEADING","good",diff) if diff > 0
            else (("TRAILING","risk",diff) if diff < 0 else ("TIED","neutral",0))
        )

    if market_type == "SPREAD":
        picked, opp = _score_for_pick(game, side)
        if picked is None or line is None:
            return "WAITING","neutral",None
        cover_margin = picked + line - opp
        return (
            ("COVERING","good",cover_margin) if cover_margin > 0
            else (("NOT COVERING","risk",cover_margin) if cover_margin < 0 else ("PUSHING","neutral",0))
        )

    if market_type == "TOTAL":
        if line is None:
            return "WAITING","neutral",None
        current = a + h
        # This is deliberately descriptive, not a live win-probability estimate.
        if side.upper() == "OVER":
            if current > line:
                return "OVER LINE","good",current
            return "BELOW LINE","neutral",current
        if side.upper() == "UNDER":
            if current < line:
                return "BELOW LINE","good",current
            return "OVER LINE","risk",current

    return "WAITING","neutral",None

def _cfb_tracker_summary(tracker, games_today, slate_date):
    rows = tracker[tracker["Slate_Date"].astype(str) == str(slate_date)].copy() if not tracker.empty else tracker
    out = {
        "tracked":len(rows),"wins":0,"losses":0,"pushes":0,"live":0,
        "favorable":0,"unfavorable":0,"neutral":0,"units":0.0
    }
    if rows is None or rows.empty:
        return out
    game_map = {str(g.get("id")): g for g in games_today}
    for _, rec in rows.iterrows():
        result = str(rec.get("Result","PENDING"))
        if result == "WIN":
            out["wins"] += 1
            out["units"] += float(rec.get("Units") or 0)
        elif result == "LOSS":
            out["losses"] += 1
            out["units"] += float(rec.get("Units") or 0)
        elif result == "PUSH":
            out["pushes"] += 1
        else:
            game = game_map.get(str(rec.get("Game_ID")))
            if game and _cfb_game_state(game) == "LIVE":
                out["live"] += 1
                _, cls, _ = _cfb_live_market_status(rec, game)
                if cls == "good":
                    out["favorable"] += 1
                elif cls == "risk":
                    out["unfavorable"] += 1
                else:
                    out["neutral"] += 1
    return out

def _render_cfb_tracker_page(games_today, slate_date):
    _grade_cfb_tracker(games_today)
    tracker = _load_cfb_tracker()
    today = tracker[tracker["Slate_Date"].astype(str) == str(slate_date)].copy() if not tracker.empty else tracker
    summary = _cfb_tracker_summary(tracker, games_today, slate_date)
    game_map = {str(g.get("id")): g for g in games_today}

    st.markdown(
        f'<div class="mobile-page-head"><div class="mobile-page-kicker">LIVE MODEL MONITOR</div>'
        f'<div class="mobile-page-title">Tracker <span class="page-count">{summary["live"]}</span></div>'
        f'<div class="mobile-page-sub">Frozen Best Bet / Bet recommendations, live score progress, and final grading.</div></div>',
        unsafe_allow_html=True,
    )

    record = f'{summary["wins"]}-{summary["losses"]}' + (f'-{summary["pushes"]}P' if summary["pushes"] else "")
    pulse = "POSITIVE" if (summary["wins"] + summary["favorable"]) > (summary["losses"] + summary["unfavorable"]) else (
        "UNDER PRESSURE" if (summary["wins"] + summary["favorable"]) < (summary["losses"] + summary["unfavorable"]) else "MIXED"
    )
    st.markdown(
        f'<div class="cfb-slate-pulse"><div class="cfb-pulse-head"><div>'
        f'<span>TODAY\'S TRACKED CARD</span><b>{pulse}</b></div>'
        f'<div class="cfb-pulse-chip">{summary["tracked"]} TRACKED</div></div>'
        f'<div class="cfb-pulse-grid">'
        f'<div><span>FINAL</span><b>{record}</b></div>'
        f'<div><span>LIVE</span><b>{summary["live"]}</b></div>'
        f'<div><span>ON TRACK</span><b>{summary["favorable"]}</b></div>'
        f'<div><span>NEEDS HELP</span><b>{summary["unfavorable"]}</b></div>'
        f'<div><span>FINAL UNITS</span><b>{summary["units"]:+.2f}u</b></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    c1,c2 = st.columns(2)
    with c1:
        if st.button("Refresh Live Scores", use_container_width=True, key="cfb_tracker_refresh"):
            get_games.clear()
            st.rerun()
    with c2:
        st.download_button(
            "Download Tracker",
            data=tracker.to_csv(index=False).encode("utf-8"),
            file_name="cfb_model_tracker.csv",
            mime="text/csv",
            use_container_width=True,
            key="cfb_tracker_download",
        )

    if today is None or today.empty:
        st.info("No tracked recommendations for this date yet. Run the Bets slate and analyze the card first.")
    else:
        live_rows, upcoming_rows, final_rows = [], [], []
        for _, rec in today.iterrows():
            g = game_map.get(str(rec.get("Game_ID")))
            result = str(rec.get("Result","PENDING"))
            if result != "PENDING":
                final_rows.append((rec,g))
            elif g and _cfb_game_state(g) == "LIVE":
                live_rows.append((rec,g))
            else:
                upcoming_rows.append((rec,g))

        if live_rows:
            st.markdown('<div class="section-kicker">LIVE TRACKED BETS</div>', unsafe_allow_html=True)
            for rec,g in live_rows:
                status, cls, value = _cfb_live_market_status(rec,g)
                a = _cfb_score(g.get("awayPoints"))
                h = _cfb_score(g.get("homePoints"))
                market_type = str(rec.get("Market_Type"))
                line = rec.get("Line")
                try:
                    line_txt = f"{float(line):g}"
                except Exception:
                    line_txt = "—"

                visual = ""
                if market_type == "TOTAL":
                    current = float(g.get("awayPoints") or 0) + float(g.get("homePoints") or 0)
                    max_v = max(float(line or 1) * 1.5, current + 7, 50)
                    pct = max(0,min(100,current/max_v*100))
                    lpct = max(2,min(96,float(line or 0)/max_v*100))
                    visual = (
                        f'<div class="cfb-track-stats"><div><span>CURRENT POINTS</span><b>{current:g}</b></div>'
                        f'<div><span>BET LINE</span><b>{line_txt}</b></div></div>'
                        f'<div class="cfb-progress"><div class="cfb-progress-fill {cls}" style="width:{pct:.1f}%"></div>'
                        f'<i style="left:{lpct:.1f}%"></i></div>'
                    )
                elif market_type == "SPREAD":
                    try:
                        cover = float(value)
                        cover_text = f"{cover:+.1f}"
                    except Exception:
                        cover_text = "—"
                    visual = (
                        f'<div class="cfb-track-stats"><div><span>CURRENT COVER MARGIN</span><b>{cover_text}</b></div>'
                        f'<div><span>BET LINE</span><b>{float(line):+.1f}</b></div></div>'
                        f'<div class="cfb-spread-meter"><span>NOT COVERING</span><i class="{cls}"></i><span>COVERING</span></div>'
                    )
                else:
                    try:
                        lead = float(value)
                        lead_text = f"{lead:+.0f}"
                    except Exception:
                        lead_text = "—"
                    visual = (
                        f'<div class="cfb-track-stats"><div><span>PICK SCORE MARGIN</span><b>{lead_text}</b></div>'
                        f'<div><span>MARKET</span><b>ML</b></div></div>'
                    )

                st.markdown(
                    f'<div class="cfb-tracker-card">'
                    f'<div class="cfb-track-score"><div>'
                    f'<div class="cfb-score-row"><span>{html.escape(str(g.get("awayTeam")))}</span><b>{a}</b></div>'
                    f'<div class="cfb-score-row"><span>{html.escape(str(g.get("homeTeam")))}</span><b>{h}</b></div>'
                    f'</div><div class="cfb-live-chip">LIVE</div></div>'
                    f'<div class="cfb-track-divider"></div>'
                    f'<div class="cfb-track-head"><div><span>{market_type} • {_grade_label_from_grade(rec.get("Grade"))}</span><b>{html.escape(str(rec.get("Pick")))}</b></div>'
                    f'<div class="cfb-track-status {cls}">{status}</div></div>'
                    f'{visual}</div>',
                    unsafe_allow_html=True,
                )

        if upcoming_rows:
            with st.expander(f"Upcoming Tracked Bets — {len(upcoming_rows)}", expanded=False):
                for rec,g in upcoming_rows:
                    st.markdown(
                        f'<div class="cfb-upcoming-track"><b>{html.escape(str(rec.get("Pick")))}</b>'
                        f'<span>{html.escape(str(rec.get("Game")))} • {rec.get("Kickoff_ET")} • {_grade_label_from_grade(rec.get("Grade"))}</span></div>',
                        unsafe_allow_html=True,
                    )

        if final_rows:
            with st.expander(f"Completed Tracked Bets — {len(final_rows)}", expanded=False):
                for rec,g in final_rows:
                    result = str(rec.get("Result"))
                    cls = "good" if result=="WIN" else ("risk" if result=="LOSS" else "neutral")
                    st.markdown(
                        f'<div class="cfb-final-track"><div><b>{html.escape(str(rec.get("Pick")))}</b>'
                        f'<span>{html.escape(str(rec.get("Game")))}</span></div>'
                        f'<div class="cfb-track-status {cls}">{result} • {float(rec.get("Units") or 0):+.2f}u</div></div>',
                        unsafe_allow_html=True,
                    )

    with st.expander("Tracker backup / restore", expanded=False):
        restore = st.file_uploader("Restore tracker CSV", type=["csv"], key="cfb_tracker_restore")
        if st.button("Merge Tracker Backup", disabled=(restore is None), use_container_width=True, key="cfb_tracker_restore_btn"):
            try:
                incoming = _clean_cfb_tracker(pd.read_csv(restore))
                current = _load_cfb_tracker()
                merged = pd.concat([current,incoming],ignore_index=True).drop_duplicates("Record_Key",keep="first")
                _save_cfb_tracker(merged)
                st.success(f"Tracker restored: {len(merged)} records.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not restore tracker: {e}")

def _cfb_game_state(g):
    if bool(g.get("completed")):
        return "FINAL"
    k = kickoff_et(g)
    if k is not None and k <= pd.Timestamp.now(tz="America/New_York"):
        return "LIVE"
    return "UPCOMING"

def _cfb_score(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return str(int(float(v)))
    except Exception:
        return "—"

def _render_cfb_live_page(games_today):
    live = [g for g in games_today if _cfb_game_state(g) == "LIVE"]
    finals = [g for g in games_today if _cfb_game_state(g) == "FINAL"]
    st.markdown(
        f'<div class="mobile-page-head"><div class="mobile-page-kicker">SCOREBOARD</div>'
        f'<div class="mobile-page-title">Live <span class="page-count">{len(live)}</span></div>'
        f'<div class="mobile-page-sub">Current scores for the selected date. Refresh the app to update game state.</div></div>',
        unsafe_allow_html=True,
    )
    if live:
        for g in sorted(live, key=lambda x: kickoff_et(x) or pd.Timestamp.max.tz_localize("UTC")):
            away = str(g.get("awayTeam") or "Away")
            home = str(g.get("homeTeam") or "Home")
            st.markdown(
                f'<div class="cfb-live-card"><div class="cfb-live-top">'
                f'<div class="cfb-live-teams">'
                f'<div class="cfb-score-row"><span>{html.escape(away)}</span><b>{_cfb_score(g.get("awayPoints"))}</b></div>'
                f'<div class="cfb-score-row"><span>{html.escape(home)}</span><b>{_cfb_score(g.get("homePoints"))}</b></div>'
                f'</div><div class="cfb-game-state">LIVE</div></div>'
                f'<div class="cfb-live-meta">{html.escape(str(g.get("venue") or ""))}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No games are currently live on this date.")
    if finals:
        with st.expander(f"Final Games — {len(finals)}", expanded=False):
            for g in sorted(finals, key=lambda x: kickoff_et(x) or pd.Timestamp.max.tz_localize("UTC")):
                away = str(g.get("awayTeam") or "Away")
                home = str(g.get("homeTeam") or "Home")
                st.markdown(
                    f'<div class="cfb-live-card"><div class="cfb-live-top">'
                    f'<div class="cfb-live-teams">'
                    f'<div class="cfb-score-row"><span>{html.escape(away)}</span><b>{_cfb_score(g.get("awayPoints"))}</b></div>'
                    f'<div class="cfb-score-row"><span>{html.escape(home)}</span><b>{_cfb_score(g.get("homePoints"))}</b></div>'
                    f'</div><div class="cfb-game-state final">FINAL</div></div></div>',
                    unsafe_allow_html=True,
                )

def _render_saved_bets_page():
    board = st.session_state.get("cfb_latest_market_board")
    slate_name = st.session_state.get("cfb_latest_slate_name", "")
    slate_date_saved = st.session_state.get("cfb_latest_slate_date", "")
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">BETTING CARD</div>'
        '<div class="mobile-page-title">Bets</div>'
        '<div class="mobile-page-sub">Your most recently analyzed full-slate recommendations.</div></div>',
        unsafe_allow_html=True,
    )
    if not isinstance(board, pd.DataFrame) or board.empty:
        st.info("No saved slate yet. Open Slate and run the daily card first.")
        return
    official = board[board["grade"].isin(["A","B"])].copy().head(10)
    st.markdown(
        f'<div class="saved-bets-shell"><div class="saved-bets-title">{html.escape(str(slate_name or "Latest"))} Slate</div>'
        f'<div class="saved-bets-sub">{html.escape(str(slate_date_saved))} • {len(official)} Best Bet / Bet play(s)</div>',
        unsafe_allow_html=True,
    )
    if official.empty:
        st.caption("No Best Bet / Bet plays qualified on the most recently analyzed slate.")
    else:
        for i, (_, r) in enumerate(official.iterrows(), start=1):
            grade = str(r.get("grade","B"))
            st.markdown(
                f'<div class="saved-bet-row"><div class="saved-bet-rank">#{i}</div>'
                f'<div class="saved-bet-main"><div class="saved-bet-pick">{html.escape(str(r.get("market","")))}</div>'
                f'<div class="saved-bet-sub">{html.escape(str(r.get("game","")))} • '
                f'Edge {float(r.get("edge") or 0)*100:+.1f}% • EV {float(r.get("ev") or 0)*100:+.1f}%</div></div>'
                f'<div class="saved-grade {grade.lower()}">{_grade_label_from_grade(grade)}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
    st.download_button(
        "Download Latest Ranked Bets",
        data=board.to_csv(index=False).encode("utf-8"),
        file_name="cfb_latest_ranked_bets.csv",
        mime="text/csv",
        use_container_width=True,
        key="cfb_latest_bets_download",
    )

def _render_more_page():
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">CFB EDGE</div>'
        '<div class="mobile-page-title">More</div>'
        '<div class="mobile-page-sub">Research, advanced tools, downloads and model information.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="cfb-info-grid"><div><span>STATUS</span><b>MODEL LIVE</b></div>'
        f'<div><span>VERSION</span><b>{html.escape(MODEL_VERSION)}</b></div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Grade a custom market", expanded=False):
        cg1, cg2 = st.columns(2)
        with cg1:
            cg_prob = st.number_input("Model probability", min_value=0.0, max_value=1.0, value=0.55, step=0.005, format="%.3f", key="more_prob")
            cg_odds = st.number_input("American odds", value=-110, step=5, key="more_odds")
            cg_conf = st.number_input("Confidence", min_value=0, max_value=100, value=78, step=1, key="more_conf")
        with cg2:
            cg_type = st.selectbox("Market type", ["spread","moneyline","total"], index=0, key="more_type")
            cg_week = st.number_input("Week", min_value=1, max_value=20, value=1, step=1, key="more_week")
            cg_market = st.number_input("Market line", value=0.0, step=0.5, key="more_market")
            cg_model = st.number_input("Model fair line", value=0.0, step=0.5, key="more_model")
        gap = (cg_model - cg_market) if cg_type in {"spread","total"} else None
        verdict, edge, ev, _ = grade(
            cg_prob, cg_odds, cg_conf, market_type=cg_type,
            projection_gap=gap, week=cg_week
        )
        units = playable_stake(verdict, edge, cg_conf)
        st.metric("Recommendation", display_grade(verdict))
        st.caption(f"Edge {edge*100:+.1f}% • EV {ev*100:+.1f}% • Unit guide {units:.2f}u")

    st.markdown(
        """
        <div class="edge-method">
          <div class="edge-method-title">v4.0 production stack</div>
          <div class="edge-method-row"><b>1</b><span>Projection</span><em>SP+/SRS + talent + returning production + matchup + HFA build an independent fair line.</em></div>
          <div class="edge-method-row"><b>2</b><span>Edge</span><em>Sportsbook spread is compared with fair only after projection.</em></div>
          <div class="edge-method-row"><b>3</b><span>Decision</span><em>Cover probability + price EV grade the bet; ensemble agreement is reliability context.</em></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Open Slate Threshold Audit", type="primary", use_container_width=True, key="cfb_open_threshold_audit"):
        st.session_state["cfb_threshold_audit_mode"] = True
        st.rerun()

    if st.button("Open Model Validation", use_container_width=True, key="cfb_open_validation"):
        st.session_state["cfb_validation_mode"] = True
        st.rerun()

    if st.button("Open Research Lab", use_container_width=True, key="cfb_open_research"):
        st.session_state["cfb_research_mode"] = True
        st.rerun()

    latest_csv = st.session_state.get("cfb_latest_projection_csv")
    latest_filename = st.session_state.get("cfb_latest_projection_filename", "cfb_latest_projection.csv")
    if latest_csv:
        with st.expander("Latest single-game projection export", expanded=False):
            ios_save_button("Save Latest Projection CSV", latest_csv, latest_filename)


    latest_board = st.session_state.get("cfb_latest_market_board")
    if isinstance(latest_board, pd.DataFrame) and not latest_board.empty:
        with st.expander("Latest full-slate ranked export", expanded=False):
            ios_save_button(
                "Save Latest Ranked Bets CSV",
                latest_board.to_csv(index=False),
                "cfb_latest_ranked_bets.csv",
            )

    with st.expander("Model details & limitations", expanded=False):
        st.write(
            "Live views are for tracking only. Pregame projection, calibration, FCS protection, "
            "longshot moneyline guard and grading rules remain unchanged."
        )

main_view = _v38_main_view
if st.session_state.get("cfb_page") != main_view:
    st.session_state["cfb_page"] = main_view

def _cfb_nav_button(label, slug):
    active = main_view == label
    key = f"cfb_nav_{slug}_{'active' if active else 'idle'}"
    if st.button(label, key=key, use_container_width=True):
        st.session_state["cfb_page"] = label
        st.rerun()


if main_view == "Tracker":
    _v401_render_official_tracker()
    st.stop()
if main_view == "More":
    _render_more_page()
    st.stop()

run_mode = "Full Slate" if main_view == "Slate" else "Single Game"

if run_mode == "Full Slate":
    slate_choice = st.radio(
        "",
        ["Early", "Midday", "Night", "All Day"],
        index=3,
        horizontal=True,
        label_visibility="collapsed",
        key="v420_slate_segment",
        help=(
            "Early = before 3:30 PM ET • Midday = 3:30 PM–6:59 PM ET • "
            "Night = 7:00 PM ET or later."
        ),
    )

    # v3.8.2: the selected time window IS the production universe.
    # Each slate is ranked independently, while the No forced bets.
    production_games = sorted(
        list(daily),
        key=lambda g: kickoff_et(g) if kickoff_et(g) is not None else pd.Timestamp.max.tz_localize("UTC"),
    )
    if slate_choice == "All Day":
        slate_games = list(production_games)
    else:
        slate_games = [g for g in production_games if slate_bucket(g) == slate_choice]

    st.markdown(
        f'<div class="ge433-slate-meta"><span><b>{len(slate_games)} Games</b></span><span>No forced bets</span></div>',
        unsafe_allow_html=True,
    )


    if not slate_games:
        st.warning("No games are in this time window with the current game-level filter.")
        st.stop()

    with st.expander("Market Data", expanded=False):
        include_lines = st.checkbox(
            "Validated sportsbook lines",
            value=True,
            key="v420_market_lines",
            help="Uses actual CFBD provider quotes. Different provider lines are never averaged into a synthetic betting line."
        )
        st.caption("Split feeds are automatically rejected when the market is not trustworthy.")

    _build_clicked = st.button("Build Ranked Slate", type="primary", use_container_width=True, key="v420_run_slate")
    if _build_clicked:
        st.session_state["cfb_build_requested"] = True
    if st.session_state.get("cfb_build_requested"):
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

        # Fit/load the historical residual correction once per slate run,
        # not once for every matchup.
        residual_models = fit_live_residual_models(int(year), "Major FBS")

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

            # v0.4: sportsbook market is the baseline. Historical seasons prior
            # to the current year train a small regularized residual correction.
            raw_home_spread = gp["model_home_spread"]
            raw_total = gp["model_total"]

            # v4.0 FUNDAMENTAL SPREAD ENGINE
            # This projection is generated before the sportsbook line is introduced.
            # The market is comparison data only; it does not anchor the fair spread.
            fundamental_home_spread = float(raw_home_spread)
            fundamental_home_margin = -fundamental_home_spread
            fundamental_margin_sd = float(gp.get("margin_sd") or BASE_MARGIN_SD)

            fundamental_home_cover_prob = None
            fundamental_away_cover_prob = None
            fundamental_point_edge = None
            fundamental_pick_side = None
            fundamental_cover_prob = None
            fundamental_grade = "NO LINE"
            fundamental_prob_edge = None
            fundamental_ev = None

            # v4.1 FUNDAMENTAL TOTAL ENGINE
            # Like spreads, the projected total is built before the sportsbook
            # number is introduced. The market total is comparison data only.
            fundamental_total = float(raw_total)
            fundamental_total_sd = float(gp.get("total_sd") or BASE_TOTAL_SD)
            fundamental_total_edge = None
            fundamental_total_side = None
            fundamental_total_prob = None
            fundamental_over_prob = None
            fundamental_under_prob = None
            fundamental_total_grade = "NO LINE"
            fundamental_total_prob_edge = None
            fundamental_total_ev = None

            if market.get("total") is not None:
                _mkt_total = float(market["total"])
                _total_delta = fundamental_total - _mkt_total
                fundamental_total_edge = abs(_total_delta)

                if _total_delta > 1e-9:
                    fundamental_total_side = "OVER"
                elif _total_delta < -1e-9:
                    fundamental_total_side = "UNDER"

                fundamental_over_prob = 1.0 - NormalDist(
                    mu=fundamental_total,
                    sigma=max(fundamental_total_sd, 1e-6),
                ).cdf(_mkt_total)
                fundamental_under_prob = 1.0 - fundamental_over_prob

                if fundamental_total_side == "OVER":
                    fundamental_total_prob = fundamental_over_prob
                elif fundamental_total_side == "UNDER":
                    fundamental_total_prob = fundamental_under_prob

                if fundamental_total_prob is not None:
                    fundamental_total_grade, fundamental_total_prob_edge, fundamental_total_ev, _ = grade(
                        fundamental_total_prob,
                        -110,
                        gp["confidence"],
                        market_type="total",
                        projection_gap=fundamental_total_edge,
                        week=gp["week"],
                    )
                    # FCS fallback remains too uncertain for an official total.
                    fundamental_total_grade = apply_fcs_guard(
                        fundamental_total_grade,
                        gp.get("fcs_fallback_used", False),
                    )

            if market.get("home_spread") is not None:
                _mkt_hs = float(market["home_spread"])
                _signed_home_edge = _mkt_hs - fundamental_home_spread
                fundamental_point_edge = abs(_signed_home_edge)
                if abs(_signed_home_edge) > 1e-9:
                    fundamental_pick_side = "HOME" if _signed_home_edge > 0 else "AWAY"

                fundamental_home_cover_prob = cover_probability(
                    fundamental_home_margin,
                    _mkt_hs,
                    side="home",
                    sigma=fundamental_margin_sd,
                )
                fundamental_away_cover_prob = 1.0 - fundamental_home_cover_prob

                if fundamental_pick_side == "HOME":
                    fundamental_cover_prob = fundamental_home_cover_prob
                elif fundamental_pick_side == "AWAY":
                    fundamental_cover_prob = fundamental_away_cover_prob

                if fundamental_cover_prob is not None:
                    fundamental_grade, fundamental_prob_edge, fundamental_ev, _ = grade(
                        fundamental_cover_prob,
                        -110,
                        gp["confidence"],
                        market_type="spread",
                        projection_gap=fundamental_point_edge,
                        week=gp["week"],
                    )
                    fundamental_grade = apply_fcs_guard(
                        fundamental_grade,
                        gp.get("fcs_fallback_used", False),
                    )

            # Legacy market-residual layer is retained only for research and
            # calibration diagnostics. It no longer supplies the production fair line.
            residual_p = residual_market_projection(gp, market, residual_models)

            adjusted_home_spread = residual_p["adjusted_home_spread"]
            adjusted_home_margin = -adjusted_home_spread
            adjusted_total = residual_p["adjusted_total"]
            cal_margin_sd = residual_p["spread_sigma"]
            cal_total_sd = residual_p["total_sigma"]

            side_weight = 0.0
            side_shrink = residual_p["spread_correction"]
            total_weight = 0.0
            total_shrink = residual_p["total_correction"]

            # Moneyline is not the promoted v0.4 market; keep a conservative
            # market-calibrated probability rather than deriving ML from ATS residual.
            ml_spread, _, _ = calibrated_market_projection(
                raw_home_spread, market.get("home_spread"), gp["week"], "side"
            )
            ml_margin = -ml_spread
            adjusted_home_wp = 1.0 - NormalDist(
                mu=ml_margin, sigma=max(cal_margin_sd, 15.0)
            ).cdf(0)
            adjusted_away_wp = 1.0 - adjusted_home_wp

            if market.get("away_ml") is not None:
                v,e,ev,_ = grade(adjusted_away_wp, market["away_ml"], gp["confidence"],
                                 market_type="moneyline", projection_gap=None, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                v = apply_moneyline_guard(v, market["away_ml"], gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['away']} ML", market["away_ml"], e, ev))
            if market.get("home_ml") is not None:
                v,e,ev,_ = grade(adjusted_home_wp, market["home_ml"], gp["confidence"],
                                 market_type="moneyline", projection_gap=None, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                v = apply_moneyline_guard(v, market["home_ml"], gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['home']} ML", market["home_ml"], e, ev))

            if market.get("home_spread") is not None:
                spread_gap = residual_p["spread_correction"]
                hp = residual_p["home_cover_prob"]
                ap = residual_p["away_cover_prob"]
                v,e,ev,_ = grade(hp, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['home']} {market['home_spread']:+.1f}", -110, e, ev))
                v,e,ev,_ = grade(ap, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['away']} {-market['home_spread']:+.1f}", -110, e, ev))

            if market.get("total") is not None:
                total_gap = residual_p["total_correction"]
                op = residual_p["over_prob"]
                up = residual_p["under_prob"]
                v,e,ev,_ = grade(op, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
                v = cap_total_research_verdict(v)
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"Over {market['total']:g}", -110, e, ev))
                v,e,ev,_ = grade(up, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
                v = cap_total_research_verdict(v)
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"Under {market['total']:g}", -110, e, ev))

            if candidates:
                rank = {"STRONG BET":3, "BET":2, "LEAN":1, "PASS":0}
                candidates.sort(key=lambda x:(rank.get(x[0], -1), x[4]), reverse=True)
                b = candidates[0]
                best_verdict, best_market, best_odds, best_edge, best_ev = b

            market_grades = [
                {
                    "verdict": v,
                    "market": name,
                    "odds": odds,
                    "edge": e,
                    "ev": ev,
                }
                for v, name, odds, e, ev in candidates
            ]

            slate_rows.append({
                "model_version": MODEL_VERSION,
                "game_date": str(selected_date),
                "slate": slate_choice,
                "kickoff_et": k.strftime("%I:%M %p") if k is not None else "",
                "game_id": g.get("id"),
                "away_team": gp["away"],
                "home_team": gp["home"],
                "away_logo": _team_logo_url(model_data_s, gp["away"]),
                "home_logo": _team_logo_url(model_data_s, gp["home"]),
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
                "fcs_fallback_used": gp.get("fcs_fallback_used", False),
                "away_is_fbs": gp.get("away_is_fbs"),
                "home_is_fbs": gp.get("home_is_fbs"),
                "margin_sd": round(gp["margin_sd"], 3),
                "total_sd": round(gp["total_sd"], 3),
                "data_completeness": round(gp["data_completeness"], 4),
                "base_power_margin": round(gp["components"]["base_power_margin"], 4),
                "matchup_margin_adjustment": round(gp["components"]["matchup_margin_adjustment"], 4),
                "sp_total_base": round(gp["components"]["sp_total_base"], 4),
                "efficiency_total_adjustment": round(gp["components"]["efficiency_total_adjustment"], 4),
                "pace_total_adjustment": round(gp["components"]["pace_total_adjustment"], 4),
                "environment_margin_adjustment": round(gp["components"]["environment_margin_adjustment"], 4),
                "environment_total_adjustment": round(gp["components"]["environment_total_adjustment"], 4),
                "away_travel_miles": gp["environment"].get("away_travel_miles"),
                "home_travel_miles": gp["environment"].get("home_travel_miles"),
                "international_game": gp["environment"].get("international"),
                "venue_source": gp["environment"].get("venue_source"),
                "weather_source": gp["environment"].get("weather_source"),
                "wind_mph": gp["environment"].get("wind_mph"),
                "wind_gust_mph": gp["environment"].get("wind_gust_mph"),
                "temperature_f": gp["environment"].get("temperature_f"),
                "precip_probability": gp["environment"].get("precip_probability"),
                "environment_flags": "; ".join(gp["environment"].get("reasons") or []),
                "market_source": market.get("provider"),
                "market_away_ml": market.get("away_ml"),
                "market_home_ml": market.get("home_ml"),
                "market_home_spread": market.get("home_spread"),
                "market_total": market.get("total"),
                "raw_model_home_spread": round(raw_home_spread, 3),
                "fundamental_home_spread": round(fundamental_home_spread, 3),
                "fundamental_point_edge": round(fundamental_point_edge, 3) if fundamental_point_edge is not None else None,
                "fundamental_pick_side": fundamental_pick_side,
                "fundamental_home_cover_prob": round(float(fundamental_home_cover_prob), 6) if fundamental_home_cover_prob is not None else None,
                "fundamental_away_cover_prob": round(float(fundamental_away_cover_prob), 6) if fundamental_away_cover_prob is not None else None,
                "fundamental_cover_prob": round(float(fundamental_cover_prob), 6) if fundamental_cover_prob is not None else None,
                "fundamental_grade": fundamental_grade,
                "fundamental_prob_edge": round(float(fundamental_prob_edge), 6) if fundamental_prob_edge is not None else None,
                "fundamental_ev": round(float(fundamental_ev), 6) if fundamental_ev is not None else None,
                "fundamental_total": round(float(fundamental_total), 3),
                "fundamental_total_edge": round(float(fundamental_total_edge), 3) if fundamental_total_edge is not None else None,
                "fundamental_total_side": fundamental_total_side,
                "fundamental_total_prob": round(float(fundamental_total_prob), 6) if fundamental_total_prob is not None else None,
                "fundamental_over_prob": round(float(fundamental_over_prob), 6) if fundamental_over_prob is not None else None,
                "fundamental_under_prob": round(float(fundamental_under_prob), 6) if fundamental_under_prob is not None else None,
                "fundamental_total_grade": fundamental_total_grade,
                "fundamental_total_prob_edge": round(float(fundamental_total_prob_edge), 6) if fundamental_total_prob_edge is not None else None,
                "fundamental_total_ev": round(float(fundamental_total_ev), 6) if fundamental_total_ev is not None else None,
                "adjusted_model_home_spread": round(adjusted_home_spread, 3),
                "side_market_weight": round(side_weight, 3),
                "side_shrink_points": round(side_shrink, 3),
                "spread_residual_correction": round(residual_p["spread_correction"], 3),
                "spread_residual_train_n": residual_p["spread_model_n"],
                "raw_model_total": round(raw_total, 3),
                "adjusted_model_total": round(adjusted_total, 3),
                "total_market_weight": round(total_weight, 3),
                "total_shrink_points": round(total_shrink, 3),
                "total_residual_correction": round(residual_p["total_correction"], 3),
                "total_residual_train_n": residual_p["total_model_n"],
                "calibrated_margin_sd": round(cal_margin_sd, 3),
                "calibrated_total_sd": round(cal_total_sd, 3),
                "home_cover_prob": round(float(residual_p["home_cover_prob"]), 6) if residual_p.get("home_cover_prob") is not None else None,
                "away_cover_prob": round(float(residual_p["away_cover_prob"]), 6) if residual_p.get("away_cover_prob") is not None else None,
                "best_verdict": best_verdict,
                "best_market": best_market,
                "best_odds": best_odds,
                "best_edge": round(best_edge, 6) if best_edge is not None else None,
                "best_ev": round(best_ev, 6) if best_ev is not None else None,
                "market_grades_json": json.dumps(market_grades),
            })

        # Selected slate is both the analysis universe and the display universe.
        slate_df = pd.DataFrame(slate_rows)


        market_board = _ranked_market_board(slate_df)
        st.session_state["cfb_latest_market_board"] = market_board.copy()
        st.session_state["cfb_latest_slate_df"] = slate_df.copy()
        st.session_state["cfb_latest_slate_name"] = slate_choice
        st.session_state["cfb_latest_slate_date"] = str(selected_date)

        v36_card = pd.DataFrame()
        try:
            # v3.8.2: rank only the selected slate. Cross-sectional percentiles,
            # day_rank and BEST BET designation are therefore slate-native.
            v36_card = _v36_live_daily_card(slate_games, slate_df, scope="Major FBS")
        except Exception as _v36e:
            st.warning(f"Locked slate selector could not run: {_v36e}")
            v36_card = pd.DataFrame()

        total_card = _v410_total_card(slate_df)
        combined_card = _v410_combine_cards(v36_card, total_card)

        st.session_state["cfb_v36_latest_card"] = combined_card.copy()
        st.session_state["cfb_v36_latest_date"] = str(selected_date)
        st.session_state["cfb_v36_latest_slate"] = slate_choice

        # Freeze official spreads and totals independently before kickoff.
        _v36_added = _v401_track_daily_card(combined_card, selected_date)
        if _v36_added:
            st.toast(f"Official tracker froze {_v36_added} new bet(s) at the current line.")

        _render_v36_live_card(combined_card, selected_date)

        try:
            _v401_graded_now, _v401_df_now = _v401_grade_tracker()
            _v401_sum_now = _v401_summary(_v401_df_now)
            if _v401_sum_now["bets"] > 0:
                st.markdown(
                    f"""
                    <div class="ge-tracker-card">
                      <div class="ge-tracker-head"><span>▥</span><b>Today's Tracker</b></div>
                      <div class="ge-tracker-grid">
                        <div><b>{_v401_sum_now['wins']}-{_v401_sum_now['losses']}-{_v401_sum_now['pushes']}</b><span>W · L · P</span></div>
                        <div><b>{_v401_sum_now['units']:+.2f}u</b><span>Units</span></div>
                        <div><b>{_v401_sum_now['roi']:+.1%}</b><span>ROI</span></div>
                        <div><b>{_v401_sum_now['pending']}</b><span>Pending</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            pass


        with st.expander("Full slate data", expanded=False):
            if "market_board" in locals() and isinstance(market_board, pd.DataFrame) and len(market_board):
                st.markdown("**Ranked market-level board**")
                market_cols = [
                    "game","kickoff_et","market","market_type","odds","grade","verdict",
                    "prob","edge","ev","confidence","fcs_fallback_used"
                ]
                st.dataframe(market_board[market_cols], use_container_width=True, hide_index=True)
                ios_save_button(
                    f"Save {slate_choice} Ranked Bets CSV",
                    market_board[market_cols].to_csv(index=False),
                    f"cfb_v161_{selected_date}_{slate_choice.lower().replace(' ','_')}_ranked_bets.csv",
                )

            display_cols = [
                "kickoff_et","away_team","home_team","projected_away_score","projected_home_score",
                "model_home_spread","model_total","market_home_spread","market_total",
                "best_verdict","best_market","best_edge","best_ev","model_confidence"
            ]
            st.dataframe(slate_df[display_cols], use_container_width=True, hide_index=True)
            ios_save_button(
                f"Save {slate_choice} Slate CSV",
                slate_df.to_csv(index=False),
                f"cfb_v161_{selected_date}_{slate_choice.lower().replace(' ','_')}_slate.csv",
            )
            st.caption(
                "Slate markets use validated actual provider quotes; split feeds can be rejected. "
                "Early-season market shrinkage and wider uncertainty are applied before grading. "
                "Spread and total pricing are assumed at -110 in slate mode unless actual prices are available."
            )

    st.stop()

labels={}
for g in daily:
    label=f"{g.get('awayTeam','Away')} @ {g.get('homeTeam','Home')}"
    if g.get("neutralSite"): label += " (Neutral)"
    labels[label]=g

game=labels[st.selectbox("Matchup", list(labels.keys()))]

try:
    model_data = get_model_data(year)
except Exception as e:
    st.error(f"CFBD model-data request failed: {e}")
    st.stop()

hfa = 2.5
with st.expander("Advanced model settings", expanded=False):
    hfa=st.number_input(
        "Home-field advantage",
        min_value=0.0, max_value=6.0, value=2.5, step=.25,
        disabled=bool(game.get("neutralSite")),
        help="Leave this at the default unless you have a specific reason to override it."
    )
p=project_game(game,model_data,hfa=hfa)

st.markdown(
    '<div class="workflow-step"><div class="workflow-num">2</div><div>'
    '<div class="workflow-title">Model projection</div>'
    '<div class="workflow-sub">The model’s expected score, spread, total, and win probabilities.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="scoreboard">
      <div class="score-team">
        <div class="team-name">{html.escape(str(p['away']))}</div>
        <div class="team-score">{p['away_score']:.1f}</div>
      </div>
      <div class="score-center">
        <div class="score-at">Projected</div>
        <div class="score-total">TOTAL {p['model_total']:.1f}</div>
      </div>
      <div class="score-team right">
        <div class="team-name">{html.escape(str(p['home']))}</div>
        <div class="team-score">{p['home_score']:.1f}</div>
      </div>
    </div>
    <div class="edge-strip">
      <div class="edge-cell">
        <div class="edge-label">Fair Spread</div>
        <div class="edge-value">{html.escape(str(p['home']))} {p['model_home_spread']:+.1f}</div>
      </div>
      <div class="edge-cell">
        <div class="edge-label">Home Win Probability</div>
        <div class="edge-value">{p['home_win_prob']*100:.1f}%</div>
      </div>
      <div class="edge-cell">
        <div class="edge-label">Away Win Probability</div>
        <div class="edge-value">{p['away_win_prob']*100:.1f}%</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Rating sources • {p['away']}: {p['away_rating']['source']} • {p['home']}: {p['home_rating']['source']}")

d1,d2,d3=st.columns(3)
d1.metric("Model Confidence", f"{p['confidence']}/100")
d2.metric("Margin Volatility", f"{p['margin_sd']:.1f}")
d3.metric("Total Volatility", f"{p['total_sd']:.1f}")

with st.expander("Projection components"):
    c = p["components"]
    st.write(f"Base power margin: {c['base_power_margin']:+.2f}")
    st.write(f"Matchup adjustment: {c['matchup_margin_adjustment']:+.2f}")
    st.write(f"HFA adjustment: {c['hfa_adjustment']:+.2f}")
    st.write(f"Environment margin adjustment: {c['environment_margin_adjustment']:+.2f}")
    st.write(f"SP+ matchup total base: {c['sp_total_base']:.2f}")
    st.write(f"Efficiency total adjustment: {c['efficiency_total_adjustment']:+.2f}")
    st.write(f"Pace total adjustment: {c['pace_total_adjustment']:+.2f}")
    st.write(f"Environment total adjustment: {c['environment_total_adjustment']:+.2f}")

    e = p["environment"]
    st.write("---")
    st.write(f"Venue: {e.get('venue_name') or 'Unknown'}")
    st.write(f"Venue source: {e.get('venue_source') or 'unresolved'}")
    if e.get("venue_geocode_query"):
        st.write(f"Venue geocode query: {e['venue_geocode_query']}")
    st.write(f"Away travel: {e.get('away_travel_miles'):.0f} mi" if e.get('away_travel_miles') is not None else "Away travel: unavailable")
    st.write(f"Home travel: {e.get('home_travel_miles'):.0f} mi" if e.get('home_travel_miles') is not None else "Home travel: unavailable")
    st.write(f"Weather: {e.get('weather_description') or 'not available'}")
    st.write(f"Weather source: {e.get('weather_source') or 'not available'}")
    if e.get("forecast_hour"):
        st.write(f"Forecast hour: {e['forecast_hour']}")
    if e.get("wind_mph") is not None:
        st.write(f"Wind: {e['wind_mph']:.0f} mph")
    if e.get("wind_gust_mph") is not None:
        st.write(f"Gusts: {e['wind_gust_mph']:.0f} mph")
    if e.get("precip_probability") is not None:
        st.write(f"Precipitation probability: {e['precip_probability']:.0f}%")
    if e.get("temperature_f") is not None:
        st.write(f"Temperature: {e['temperature_f']:.0f}°F")
    if e.get("reasons"):
        st.write("Environment flags: " + ", ".join(e["reasons"]))



def add_result_fields(row, p, actual_away_score=None, actual_home_score=None,
                      market_home_spread=None, market_total=None):
    """
    Append postgame calibration fields. This does not change the projection.
    """
    if actual_away_score is None or actual_home_score is None:
        row.update({
            "game_final": False,
            "actual_away_score": None,
            "actual_home_score": None,
            "actual_total": None,
            "actual_home_margin": None,
            "spread_error_points": None,
            "total_error_points": None,
            "winner_correct": None,
            "model_abs_spread_error": None,
            "model_abs_total_error": None,
            "market_home_cover_result": None,
            "market_total_result": None,
        })
        return row

    aa = float(actual_away_score)
    ah = float(actual_home_score)
    actual_total = aa + ah
    actual_margin = ah - aa

    # Signed error: positive means model was too high.
    spread_error = float(p["home_margin"]) - actual_margin
    total_error = float(p["model_total"]) - actual_total

    pred_home_win = float(p["home_margin"]) > 0
    actual_home_win = actual_margin > 0
    winner_correct = (pred_home_win == actual_home_win) if actual_margin != 0 else None

    cover_result = None
    if market_home_spread is not None:
        ats_margin = actual_margin + float(market_home_spread)
        cover_result = "HOME_COVER" if ats_margin > 0 else ("AWAY_COVER" if ats_margin < 0 else "PUSH")

    total_result = None
    if market_total is not None:
        diff = actual_total - float(market_total)
        total_result = "OVER" if diff > 0 else ("UNDER" if diff < 0 else "PUSH")

    row.update({
        "game_final": True,
        "actual_away_score": aa,
        "actual_home_score": ah,
        "actual_total": actual_total,
        "actual_home_margin": actual_margin,
        "spread_error_points": round(spread_error, 3),
        "total_error_points": round(total_error, 3),
        "winner_correct": winner_correct,
        "model_abs_spread_error": round(abs(spread_error), 3),
        "model_abs_total_error": round(abs(total_error), 3),
        "market_home_cover_result": cover_result,
        "market_total_result": total_result,
    })
    return row


def build_export_row(p, game, selected_date, market=None):
    market = market or {}
    row = {
        "model_version": MODEL_VERSION,
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
        "environment_margin_adjustment": round(p["components"]["environment_margin_adjustment"], 4),
        "environment_total_adjustment": round(p["components"]["environment_total_adjustment"], 4),
        "venue_name": p["environment"].get("venue_name"),
        "venue_city": p["environment"].get("venue_city"),
        "venue_state": p["environment"].get("venue_state"),
        "venue_country": p["environment"].get("venue_country"),
        "venue_elevation": p["environment"].get("venue_elevation"),
        "venue_source": p["environment"].get("venue_source"),
        "venue_geocode_query": p["environment"].get("venue_geocode_query"),
        "international_game": p["environment"].get("international"),
        "away_travel_miles": p["environment"].get("away_travel_miles"),
        "home_travel_miles": p["environment"].get("home_travel_miles"),
        "weather_source": p["environment"].get("weather_source"),
        "weather_description": p["environment"].get("weather_description"),
        "forecast_hour": p["environment"].get("forecast_hour"),
        "wind_mph": p["environment"].get("wind_mph"),
        "wind_gust_mph": p["environment"].get("wind_gust_mph"),
        "temperature_f": p["environment"].get("temperature_f"),
        "precip_probability": p["environment"].get("precip_probability"),
        "precipitation_in": p["environment"].get("precipitation_in"),
        "environment_flags": "; ".join(p["environment"].get("reasons") or []),

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
    return add_result_fields(row, p)

st.divider()
st.markdown(
    '<div class="workflow-step"><div class="workflow-num">3</div><div>'
    '<div class="workflow-title">Load market odds</div>'
    '<div class="workflow-sub">Pull available lines, then edit them to match your sportsbook.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.caption("Use the automatic line pull as a starting point. Your sportsbook price should be the final input.")

line_rows = []
providers = []
game_id = game.get("id")

if st.button("Load Market Odds", use_container_width=True):
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
    st.info("Load market odds to prefill the betting inputs, or enter your book manually.")

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

tab_spread, tab_ml, tab_total = st.tabs(["Spread", "Moneyline", "Total"])

with tab_spread:
    s1,s2=st.columns(2)
    home_spread=s1.number_input(f"{p['home']} spread", value=default_home_spread, step=.5)
    home_spread_odds=s2.number_input("Home spread odds", value=-110, step=5)
    s3,s4=st.columns(2)
    away_spread=s3.number_input(f"{p['away']} spread", value=default_away_spread, step=.5)
    away_spread_odds=s4.number_input("Away spread odds", value=-110, step=5)

with tab_ml:
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

with tab_total:
    t1,t2,t3=st.columns(3)
    market_total=t1.number_input("Total", value=default_total, step=.5)
    over_odds=t2.number_input("Over odds", value=-110, step=5)
    under_odds=t3.number_input("Under odds", value=-110, step=5)


projection_only_df = pd.DataFrame([build_export_row(p, game, selected_date)])
st.session_state["cfb_latest_projection_csv"] = projection_only_df.to_csv(index=False)
st.session_state["cfb_latest_projection_filename"] = (
    f"cfb_projection_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv"
)

st.markdown(
    '<div class="workflow-step"><div class="workflow-num">4</div><div>'
    '<div class="workflow-title">Get your betting board</div>'
    '<div class="workflow-sub">Markets are ranked Best Bet, Bet, Lean, or Pass using probability, price, edge, EV, and model confidence.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="grade-legend-inline">'
    '<div class="grade-pill a"><b>Best Bet</b></div>'
    '<div class="grade-pill b"><b>Bet</b></div>'
    '<div class="grade-pill c"><b>Lean</b></div>'
    '<div class="grade-pill d"><b>Pass</b></div>'
    '</div>',
    unsafe_allow_html=True,
)

if st.button("Analyze Markets",type="primary",use_container_width=True):
    markets=[]

    # v0.4 market-baseline residual layer.
    single_market = {
        "home_spread": float(home_spread),
        "total": float(market_total),
        "away_ml": int(away_ml),
        "home_ml": int(home_ml),
    }
    residual_models = fit_live_residual_models(int(selected_date.year), "Major FBS")
    residual_p = residual_market_projection(p, single_market, residual_models)

    adj_home_spread = residual_p["adjusted_home_spread"]
    adj_home_margin = -adj_home_spread
    adj_total = residual_p["adjusted_total"]
    cal_margin_sd = residual_p["spread_sigma"]
    cal_total_sd = residual_p["total_sigma"]
    side_weight = 0.0
    side_shrink = residual_p["spread_correction"]
    total_weight = 0.0
    total_shrink = residual_p["total_correction"]

    ml_spread, _, _ = calibrated_market_projection(
        p["model_home_spread"], home_spread, p["week"], "side"
    )
    ml_margin = -ml_spread
    adj_home_wp = 1.0 - NormalDist(mu=ml_margin, sigma=max(cal_margin_sd, 15.0)).cdf(0)
    adj_away_wp = 1.0 - adj_home_wp

    for name,prob,odds in [(f"{p['away']} ML",adj_away_wp,away_ml),(f"{p['home']} ML",adj_home_wp,home_ml)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="moneyline",week=p["week"])
        v = apply_fcs_guard(v, p.get("fcs_fallback_used", False))
        v = apply_moneyline_guard(v, odds, p.get("fcs_fallback_used", False))
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    spread_gap = residual_p["spread_correction"]
    hc=residual_p["home_cover_prob"]
    ac=residual_p["away_cover_prob"]
    for name,prob,odds in [
        (f"{p['home']} {home_spread:+.1f}",hc,home_spread_odds),
        (f"{p['away']} {away_spread:+.1f}",ac,away_spread_odds)
    ]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="spread",
                         projection_gap=spread_gap,week=p["week"])
        v = apply_fcs_guard(v, p.get("fcs_fallback_used", False))
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    total_gap = residual_p["total_correction"]
    op=residual_p["over_prob"]
    up=residual_p["under_prob"]
    for name,prob,odds in [(f"Over {market_total:g}",op,over_odds),(f"Under {market_total:g}",up,under_odds)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="total",
                         projection_gap=total_gap,week=p["week"])
        v = cap_total_research_verdict(v)
        v = apply_fcs_guard(v, p.get("fcs_fallback_used", False))
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    if p.get("fcs_fallback_used", False):
        st.warning("FCS opponent uses a generic fallback rating. Confidence is reduced and Best Bet / Bet recommendations are capped at Lean until better team-specific data is available.")

    rank={"STRONG BET":3,"BET":2,"LEAN":1,"PASS":0}
    markets.sort(key=lambda x:(rank.get(x[0], -1),x[5]),reverse=True)
    best=markets[0]

    top_label = "NO PLAY" if best[0] == "PASS" else "TOP PLAY"
    st.markdown(f'<div class="section-kicker">{top_label}</div>', unsafe_allow_html=True)
    best_stake = playable_stake(best[0], best[4], p["confidence"])
    render_recommendation_card(
        best[0], best[1], best[2],
        prob=best[3], edge=best[4], ev=best[5],
        fair=best[6], stake=best_stake
    )

    st.markdown('<div class="section-kicker">MARKET BOARD</div>', unsafe_allow_html=True)
    primary_markets = [m for m in markets if m[0] in {"STRONG BET","BET","LEAN"}]
    pass_markets = [m for m in markets if m[0] == "PASS"]

    if primary_markets:
        st.markdown('<div class="market-board">', unsafe_allow_html=True)
        for v,name,odds,prob,e,ev,fair in primary_markets:
            render_market_row(v, name, odds, prob=prob, edge=e, ev=ev, fair=fair)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("No Best Bet, Bet, or Lean markets on this game.")

    if pass_markets:
        with st.expander(f"Other markets • {len(pass_markets)} pass", expanded=False):
            st.markdown('<div class="market-board">', unsafe_allow_html=True)
            for v,name,odds,prob,e,ev,fair in pass_markets:
                render_market_row(v, name, odds, prob=prob, edge=e, ev=ev, fair=fair)
            st.markdown('</div>', unsafe_allow_html=True)

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
        "raw_model_home_spread": round(p["model_home_spread"], 3),
        "adjusted_model_home_spread": round(adj_home_spread, 3),
        "side_market_weight": round(side_weight, 3),
        "side_shrink_points": round(side_shrink, 3),
        "spread_residual_correction": round(residual_p["spread_correction"], 3),
        "spread_residual_train_n": residual_p["spread_model_n"],
        "raw_model_total": round(p["model_total"], 3),
        "adjusted_model_total": round(adj_total, 3),
        "total_market_weight": round(total_weight, 3),
        "total_shrink_points": round(total_shrink, 3),
        "total_residual_correction": round(residual_p["total_correction"], 3),
        "total_residual_train_n": residual_p["total_model_n"],
        "calibrated_margin_sd": round(cal_margin_sd, 3),
        "calibrated_total_sd": round(cal_total_sd, 3),
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

    with st.expander("Export / Audit", expanded=False):
        st.caption("Download the full game audit CSV for review or upload back into ChatGPT.")
        ios_save_button(
            "Save Game CSV",
            export_df.to_csv(index=False),
            f"cfb_model_v141_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
        )

st.divider()
st.caption("Saturday Edge • v4.3 • Premium App UI • Spreads + totals ranked together.")
