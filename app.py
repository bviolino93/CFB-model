
import streamlit as st
import pandas as pd
import numpy as np
import base64
import html
import json
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
MODEL_VERSION = "2.2.1-TRACKER-REGEX-HOTFIX"

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

    if version == "v0.3.2":
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
    page_title="CFB Edge",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Sleek mobile-first app theme ----------

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

st.markdown(
    f"""
    <div class="cfb-hero">
      <div class="cfb-kicker">CFB BETTING MODEL</div>
      <div class="cfb-title">CFB Edge</div>
      <div class="cfb-subtitle">Pick a matchup, load the market, and get a ranked betting board in seconds.</div>
      <div class="version-pill">MODEL LIVE</div>
    </div>
    <div class="status-strip">
      <div class="status-live"><span class="status-dot"></span> Live model ready</div>
      <div>Best Bet &nbsp;•&nbsp; Bet &nbsp;•&nbsp; Lean &nbsp;•&nbsp; Pass</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

app_section = "Research Lab" if st.session_state.get("cfb_research_mode", False) else "Betting Board"

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

st.markdown(
    '<div class="mobile-page-head"><div class="mobile-page-kicker">SINGLE GAME</div>'
    '<div class="mobile-page-title">Home</div>'
    '<div class="mobile-page-sub">Choose one matchup, review the projection, then price the markets.</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="workflow-step"><div class="workflow-num">1</div><div>'
    '<div class="workflow-title">Choose your matchup</div>'
    '<div class="workflow-sub">Set the date, game level, and analysis mode.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

top1, top2 = st.columns([1, 1])
with top1:
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

with top2:
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

st.caption(f"{len(daily)} matchup(s) available • {selected_date:%A, %b %d}")


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
        st.info("No saved slate yet. Open the Bets tab and run Analyze Slate first.")
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

if "cfb_page" not in st.session_state:
    st.session_state["cfb_page"] = "Home"

main_view = st.session_state.get("cfb_page", "Home")

def _cfb_nav_button(label, slug):
    active = main_view == label
    key = f"cfb_nav_{slug}_{'active' if active else 'idle'}"
    if st.button(label, key=key, use_container_width=True):
        st.session_state["cfb_page"] = label
        st.rerun()

_cfb_nav_button("Home", "home")
_cfb_nav_button("Live", "live")
_cfb_nav_button("Tracker", "tracker")
_cfb_nav_button("Bets", "bets")
_cfb_nav_button("More", "more")

if main_view == "Live":
    _render_cfb_live_page(daily)
    st.stop()
if main_view == "Tracker":
    _render_cfb_tracker_page(daily, selected_date)
    st.stop()
if main_view == "More":
    _render_more_page()
    st.stop()

run_mode = "Full Slate" if main_view == "Bets" else "Single Game"

if run_mode == "Full Slate":
    st.markdown(
        '<div class="mobile-page-head"><div class="mobile-page-kicker">FULL SLATE</div>'
        '<div class="mobile-page-title">Slate</div>'
        '<div class="mobile-page-sub">Rank the strongest spread, moneyline and total opportunities across the selected slate.</div></div>',
        unsafe_allow_html=True,
    )
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

    if st.button("Analyze Slate", type="primary", use_container_width=True):
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

            # v0.3.2: calibrate projections to the market before converting to
            # probabilities. Raw projections remain in the export for audit.
            cal_margin_sd, cal_total_sd = calibrated_sigmas(gp["margin_sd"], gp["total_sd"], gp["week"])
            raw_home_spread = gp["model_home_spread"]
            raw_total = gp["model_total"]

            adjusted_home_spread = raw_home_spread
            side_weight = 1.0
            side_shrink = 0.0
            if market.get("home_spread") is not None:
                adjusted_home_spread, side_weight, side_shrink = calibrated_market_projection(
                    raw_home_spread, market["home_spread"], gp["week"], "side"
                )
            adjusted_home_margin = -adjusted_home_spread

            adjusted_total = raw_total
            total_weight = 1.0
            total_shrink = 0.0
            if market.get("total") is not None:
                adjusted_total, total_weight, total_shrink = calibrated_market_projection(
                    raw_total, market["total"], gp["week"], "total"
                )

            adjusted_home_wp = 1.0 - NormalDist(mu=adjusted_home_margin, sigma=cal_margin_sd).cdf(0)
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
                spread_gap = raw_home_spread - market["home_spread"]
                hp = cover_probability(adjusted_home_margin, market["home_spread"], "home", cal_margin_sd)
                ap = 1 - hp
                v,e,ev,_ = grade(hp, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['home']} {market['home_spread']:+.1f}", -110, e, ev))
                v,e,ev,_ = grade(ap, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"{gp['away']} {-market['home_spread']:+.1f}", -110, e, ev))

            if market.get("total") is not None:
                total_gap = raw_total - market["total"]
                op = total_probability(adjusted_total, market["total"], "over", cal_total_sd)
                up = 1 - op
                v,e,ev,_ = grade(op, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
                v = apply_fcs_guard(v, gp.get("fcs_fallback_used", False))
                candidates.append((v, f"Over {market['total']:g}", -110, e, ev))
                v,e,ev,_ = grade(up, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
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
                "adjusted_model_home_spread": round(adjusted_home_spread, 3),
                "side_market_weight": round(side_weight, 3),
                "side_shrink_points": round(side_shrink, 3),
                "raw_model_total": round(raw_total, 3),
                "adjusted_model_total": round(adjusted_total, 3),
                "total_market_weight": round(total_weight, 3),
                "total_shrink_points": round(total_shrink, 3),
                "calibrated_margin_sd": round(cal_margin_sd, 3),
                "calibrated_total_sd": round(cal_total_sd, 3),
                "best_verdict": best_verdict,
                "best_market": best_market,
                "best_odds": best_odds,
                "best_edge": round(best_edge, 6) if best_edge is not None else None,
                "best_ev": round(best_ev, 6) if best_ev is not None else None,
                "market_grades_json": json.dumps(market_grades),
            })

        slate_df = pd.DataFrame(slate_rows)

        st.markdown(f'<div class="section-kicker">{slate_choice} Slate</div>', unsafe_allow_html=True)

        market_board = _ranked_market_board(slate_df)
        st.session_state["cfb_latest_market_board"] = market_board.copy()
        st.session_state["cfb_latest_slate_df"] = slate_df.copy()
        st.session_state["cfb_latest_slate_name"] = slate_choice
        st.session_state["cfb_latest_slate_date"] = str(selected_date)
        _tracked_added = _track_cfb_official_board(market_board, slate_df, selected_date)
        if _tracked_added:
            st.toast(f"Tracker froze {_tracked_added} new Best Bet / Bet recommendation(s).")

        st.markdown(
            f"""
            <div class="app-shell-head">
              <div>
                <div class="app-eyebrow">CFB EDGE</div>
                <div class="app-title">{html.escape(str(slate_choice))} Slate</div>
                <div class="app-subtitle">{html.escape(str(selected_date))} • Spread • Moneyline • Totals</div>
              </div>
              <div class="app-live"><span></span> MODEL LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-kicker">SLATE BETTING CARD</div>', unsafe_allow_html=True)
        st.caption("Top Bets ranks Best Bet and Bet plays by the best combination of win probability, edge, EV and model confidence. Spreads, moneylines and totals all compete for the same list.")

        top_n = st.radio(
            "Ranked card size",
            options=[5, 10],
            index=0,
            horizontal=True,
            key="ranked_slate_size",
            format_func=lambda x: f"Top {x}",
            label_visibility="collapsed",
        )

        official_board = market_board[market_board["grade"].isin(["A","B"])].copy() if len(market_board) else pd.DataFrame()
        lean_board = market_board[market_board["grade"].eq("C")].copy() if len(market_board) else pd.DataFrame()

        show_n = int(top_n or 5)

        if len(official_board):
            top_bets = official_board.head(show_n)
            st.markdown(f'<div class="section-kicker">TOP {len(top_bets)} BETS</div>', unsafe_allow_html=True)
            for rank_num, (_, bet_row) in enumerate(top_bets.iterrows(), start=1):
                _render_top_slate_bet(bet_row, rank_num)

            st.caption("Ranking favors the most likely profitable bets—not simply the biggest projected EV.")

            if len(top_bets) < show_n:
                st.caption(f"Only {len(top_bets)} Best Bet / Bet plays qualify. The list is not padded with weaker Leans.")
        else:
            st.info("No Best Bet / Bet plays currently qualify on this slate.")

        if len(lean_board):
            lean_show_n = 5 if not len(official_board) else 3
            if not len(official_board):
                st.markdown('<div class="section-kicker">TOP LEANS</div>', unsafe_allow_html=True)
                st.caption("No Best Bet or Bet qualified, so these are the strongest secondary opinions—not official bets.")
                for rank_num, (_, bet_row) in enumerate(lean_board.head(lean_show_n).iterrows(), start=1):
                    _render_top_slate_bet(bet_row, rank_num)
            else:
                with st.expander(f"Top Leans — {min(lean_show_n, len(lean_board))}", expanded=False):
                    for rank_num, (_, bet_row) in enumerate(lean_board.head(lean_show_n).iterrows(), start=1):
                        _render_top_slate_bet(bet_row, rank_num)

        a_n = int((market_board["grade"] == "A").sum()) if len(market_board) else 0
        b_n = int((market_board["grade"] == "B").sum()) if len(market_board) else 0
        c_n = int((market_board["grade"] == "C").sum()) if len(market_board) else 0
        st.caption(f"Slate pool: {a_n} Best Bet • {b_n} Bet • {c_n} Lean")

        st.markdown('<div class="section-kicker">ALL GAMES</div>', unsafe_allow_html=True)
        st.caption("Games are listed in kickoff order. Open any matchup to see its #1 market first, followed by every other available spread, ML and total ranked underneath.")

        if len(market_board):
            game_groups = []
            for game_name, game_markets in market_board.groupby("game", sort=False):
                game_markets = game_markets.sort_values(
                    ["grade_rank","rank_score","ev","edge"],
                    ascending=[False,False,False,False],
                    na_position="last",
                ).reset_index(drop=True)
                best_game_market = game_markets.iloc[0]
                game_groups.append((
                    float(best_game_market.get("rank_score", 0)),
                    str(game_name),
                    game_markets,
                ))

            # Game dropdowns are chronological for easy slate navigation.
            # Top 5 / Top 10 remains ranked by betting quality above.
            def _kickoff_sort_key(item):
                _, _, gm = item
                if gm is None or len(gm) == 0:
                    return pd.Timestamp.max
                raw = gm.iloc[0].get("kickoff_et", "")
                try:
                    parsed = pd.to_datetime(raw, errors="coerce")
                    if pd.isna(parsed):
                        return pd.Timestamp.max
                    return parsed
                except Exception:
                    return pd.Timestamp.max

            game_groups.sort(key=_kickoff_sort_key)

            game_labels = []
            for _, game_name, game_markets in game_groups:
                best = game_markets.iloc[0]
                kickoff = str(best.get("kickoff_et", ""))
                game_labels.append(f"{kickoff} • {game_name}")

            selected_game_label = st.selectbox(
                "Choose matchup",
                game_labels,
                index=0,
                key="cfb_slate_game_picker",
                label_visibility="collapsed",
            )
            selected_idx = game_labels.index(selected_game_label)
            _, game_name, game_markets = game_groups[selected_idx]

            best = game_markets.iloc[0]
            grade = str(best.get("grade", "D"))
            best_name = str(best.get("market", ""))
            kickoff = str(best.get("kickoff_et", ""))
            first_row = game_markets.iloc[0]
            away = str(first_row.get("away_team", ""))
            home = str(first_row.get("home_team", ""))
            away_logo = str(first_row.get("away_logo", "") or "")
            home_logo = str(first_row.get("home_logo", "") or "")
            st.markdown(
                f"""
                <div class="game-detail-shell">
                  <div class="game-detail-head">
                    <div class="game-team">
                      {_logo_html(away_logo, away, 34)}
                      <div><span>Away</span><b>{html.escape(away)}</b></div>
                    </div>
                    <div class="game-at">@</div>
                    <div class="game-team home">
                      <div><span>Home</span><b>{html.escape(home)}</b></div>
                      {_logo_html(home_logo, home, 34)}
                    </div>
                  </div>
                  <div class="game-detail-sub">{html.escape(kickoff)} • Best market: {html.escape(best_name)} • {html.escape(_grade_label_from_grade(grade))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_game_market_stack(game_markets)
        else:
            st.caption("No market data available for these games.")

        with st.expander("Export / Full Slate Data", expanded=False):
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
                "Slate lines use the median across available CFBD providers. "
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

    # v0.3.2 guarded single-game calibration.
    cal_margin_sd, cal_total_sd = calibrated_sigmas(p["margin_sd"], p["total_sd"], p["week"])
    adj_home_spread, side_weight, side_shrink = calibrated_market_projection(
        p["model_home_spread"], home_spread, p["week"], "side"
    )
    adj_home_margin = -adj_home_spread
    adj_total, total_weight, total_shrink = calibrated_market_projection(
        p["model_total"], market_total, p["week"], "total"
    )
    adj_home_wp = 1.0 - NormalDist(mu=adj_home_margin, sigma=cal_margin_sd).cdf(0)
    adj_away_wp = 1.0 - adj_home_wp

    for name,prob,odds in [(f"{p['away']} ML",adj_away_wp,away_ml),(f"{p['home']} ML",adj_home_wp,home_ml)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="moneyline",week=p["week"])
        v = apply_fcs_guard(v, p.get("fcs_fallback_used", False))
        v = apply_moneyline_guard(v, odds, p.get("fcs_fallback_used", False))
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    spread_gap = p["model_home_spread"] - home_spread
    hc=cover_probability(adj_home_margin,home_spread,"home",cal_margin_sd)
    ac=1-hc
    for name,prob,odds in [
        (f"{p['home']} {home_spread:+.1f}",hc,home_spread_odds),
        (f"{p['away']} {away_spread:+.1f}",ac,away_spread_odds)
    ]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="spread",
                         projection_gap=spread_gap,week=p["week"])
        v = apply_fcs_guard(v, p.get("fcs_fallback_used", False))
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    total_gap = p["model_total"] - market_total
    op=total_probability(adj_total,market_total,"over",cal_total_sd)
    up=1-op
    for name,prob,odds in [(f"Over {market_total:g}",op,over_odds),(f"Under {market_total:g}",up,under_odds)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="total",
                         projection_gap=total_gap,week=p["week"])
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
        "raw_model_total": round(p["model_total"], 3),
        "adjusted_model_total": round(adj_total, 3),
        "total_market_weight": round(total_weight, 3),
        "total_shrink_points": round(total_shrink, 3),
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
st.caption("CFB Edge • v2.0.0 Mobile UI • Projection and betting logic preserved.")
