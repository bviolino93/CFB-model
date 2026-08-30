
import streamlit as st
import pandas as pd
import base64
import html
import json
import streamlit.components.v1 as components
from datetime import date

# ===== Embedded CFB v0.2.0 model engine =====

import math
import requests
from statistics import NormalDist, mean, pstdev
from functools import lru_cache
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api.collegefootballdata.com"
MODEL_VERSION = "0.4.1-BACKTEST-ODDS-FIX"

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

def grade(prob, odds, confidence=75, market_type="side", projection_gap=None, week=1):
    """
    v0.3.2 guarded betting grade.
    - Higher thresholds than v0.3.1.
    - Totals require more evidence.
    - Weeks 1-2 require extra edge/EV.
    - Very large raw model/market gaps are review-only until calibrated.
    """
    if not _valid_american_odds(odds):
        return "PASS", 0.0, 0.0, None
    imp = implied_prob(odds)
    edge = prob - imp
    ev = expected_value(prob, odds)
    me, mv = juice_thresholds(odds)

    # Base hurdle increase: the opening 1-5 card showed that the prior mapping
    # from projection -> probability -> BET was too aggressive.
    me += 0.015
    mv += 0.025

    if market_type == "total":
        me += 0.010
        mv += 0.015

    w = _week_num(week)
    if w <= 1:
        me += 0.010
        mv += 0.015
    elif w == 2:
        me += 0.005
        mv += 0.010

    # Extreme disagreements are not promoted to BET solely because the normal
    # distribution creates a large probability edge. Keep them review-only.
    review_only = False
    if projection_gap is not None:
        gap = abs(float(projection_gap))
        if market_type == "total" and gap >= 10.0:
            review_only = True
        elif market_type in {"side", "spread"} and gap >= 9.0:
            review_only = True

    if (not review_only) and confidence >= 80 and edge >= me + .025 and ev >= mv + .04:
        verdict = "STRONG BET"
    elif (not review_only) and confidence >= 72 and edge >= me and ev >= mv:
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

# ===== End v0.4.0 backtest engine =====

# ===== End embedded model engine =====


st.set_page_config(
    page_title="CFB Edge",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Professional app theme ----------
st.markdown("""
<style>
    /* App shell */
    .stApp {
        background:
          radial-gradient(circle at 15% 0%, rgba(37,99,235,.12), transparent 32%),
          radial-gradient(circle at 90% 8%, rgba(16,185,129,.08), transparent 26%),
          #07101f;
        color: #E8EEF8;
    }
    [data-testid="stHeader"] {
        background: rgba(7,16,31,.76);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148,163,184,.10);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Typography */
    h1, h2, h3 {
        letter-spacing: -0.025em;
        color: #F8FAFC !important;
    }
    p, label, .stCaption {
        color: #AFC0D6 !important;
    }

    /* Hero */
    .cfb-hero {
        padding: 22px 24px;
        margin-bottom: 22px;
        border: 1px solid rgba(148,163,184,.15);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15,30,54,.96), rgba(10,22,42,.92));
        box-shadow: 0 20px 60px rgba(0,0,0,.25);
    }
    .cfb-kicker {
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #60A5FA;
        margin-bottom: 7px;
    }
    .cfb-title {
        font-size: clamp(1.9rem, 4vw, 3rem);
        line-height: 1.02;
        font-weight: 850;
        color: #F8FAFC;
        letter-spacing: -.045em;
    }
    .cfb-subtitle {
        color: #AFC0D6;
        font-size: .98rem;
        margin-top: 8px;
    }
    .version-pill {
        display: inline-block;
        margin-top: 13px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(96,165,250,.10);
        border: 1px solid rgba(96,165,250,.25);
        color: #93C5FD;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .05em;
    }

    /* Section labels */
    .section-kicker {
        margin-top: 1.1rem;
        margin-bottom: .45rem;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #7DD3FC;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    [data-testid="stDateInput"] input,
    [data-testid="stNumberInput"] input {
        background: #0D1A2D !important;
        border-color: rgba(148,163,184,.18) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }

    /* Radio segmented control */
    div[role="radiogroup"] {
        gap: .45rem;
        background: rgba(13,26,45,.75);
        border: 1px solid rgba(148,163,184,.14);
        padding: 5px;
        border-radius: 14px;
    }
    div[role="radiogroup"] label {
        border-radius: 10px;
        padding: 3px 10px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 45px;
        font-weight: 750;
        border: 1px solid rgba(96,165,250,.30);
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        box-shadow: 0 10px 24px rgba(37,99,235,.18);
        transition: .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(147,197,253,.65);
        box-shadow: 0 12px 30px rgba(37,99,235,.28);
        color: white;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(16,31,53,.96), rgba(12,24,43,.96));
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 16px;
        padding: 15px 16px;
        box-shadow: 0 12px 30px rgba(0,0,0,.13);
    }
    [data-testid="stMetricLabel"] {
        color: #93A8C2 !important;
        font-weight: 700;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        letter-spacing: -.03em;
    }

    /* Scoreboard */
    .scoreboard {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 18px;
        padding: 22px;
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(15,31,54,.98), rgba(9,20,38,.98));
        border: 1px solid rgba(148,163,184,.16);
        box-shadow: 0 20px 55px rgba(0,0,0,.22);
        margin-bottom: 12px;
    }
    .score-team { min-width: 0; }
    .score-team.right { text-align: right; }
    .team-name {
        color: #C6D4E7;
        font-size: .84rem;
        font-weight: 750;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .team-score {
        color: #F8FAFC;
        font-weight: 900;
        font-size: clamp(2rem, 6vw, 3.3rem);
        letter-spacing: -.06em;
        line-height: 1;
        margin-top: 4px;
    }
    .score-center {
        text-align: center;
        padding: 0 9px;
    }
    .score-at {
        font-size: .72rem;
        color: #64748B;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .12em;
    }
    .score-total {
        margin-top: 5px;
        color: #7DD3FC;
        font-size: .78rem;
        font-weight: 800;
    }

    /* Edge strip */
    .edge-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 12px 0 18px 0;
    }
    .edge-cell {
        padding: 12px 14px;
        border-radius: 13px;
        background: rgba(13,26,45,.88);
        border: 1px solid rgba(148,163,184,.12);
    }
    .edge-label {
        color: #7890AC;
        font-size: .68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .10em;
    }
    .edge-value {
        color: #F8FAFC;
        margin-top: 3px;
        font-weight: 800;
        font-size: .98rem;
    }

    /* Expanders/data */
    [data-testid="stExpander"] {
        background: rgba(10,22,42,.76);
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 14px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(148,163,184,.12) !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,.14);
    }

    @media (max-width: 700px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .cfb-hero { padding: 18px; border-radius: 18px; }
        .scoreboard { padding: 17px 14px; gap: 8px; }
        .edge-strip { grid-template-columns: 1fr; }
    }

    /* Mobile-first slate cards */
    .slate-card {
        margin: 14px 0;
        padding: 17px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(15,31,54,.98), rgba(9,20,38,.98));
        border: 1px solid rgba(148,163,184,.14);
        box-shadow: 0 16px 40px rgba(0,0,0,.18);
    }
    .slate-card-top {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:14px;
        margin-bottom:14px;
    }
    .slate-time {
        color:#7890AC;
        font-size:.70rem;
        font-weight:800;
        letter-spacing:.10em;
        text-transform:uppercase;
    }
    .slate-matchup {
        margin-top:3px;
        color:#F8FAFC;
        font-size:1.08rem;
        font-weight:850;
        letter-spacing:-.02em;
    }
    .slate-matchup span { color:#64748B; font-weight:700; }
    .slate-badge {
        flex:0 0 auto;
        border-radius:999px;
        padding:5px 9px;
        font-size:.67rem;
        font-weight:900;
        letter-spacing:.07em;
        border:1px solid rgba(148,163,184,.16);
    }
    .slate-badge.strong {
        color:#BBF7D0; background:rgba(22,163,74,.16); border-color:rgba(34,197,94,.30);
    }
    .slate-badge.bet {
        color:#BAE6FD; background:rgba(2,132,199,.16); border-color:rgba(56,189,248,.30);
    }
    .slate-badge.lean {
        color:#FDE68A; background:rgba(202,138,4,.14); border-color:rgba(250,204,21,.25);
    }
    .slate-badge.pass, .slate-badge.noline {
        color:#CBD5E1; background:rgba(100,116,139,.12);
    }
    .slate-reco {
        padding:13px 14px;
        margin-bottom:12px;
        border-radius:13px;
        background:rgba(37,99,235,.09);
        border:1px solid rgba(96,165,250,.16);
    }
    .slate-reco-label, .slate-box-label {
        color:#7890AC;
        font-size:.66rem;
        font-weight:850;
        letter-spacing:.10em;
        text-transform:uppercase;
    }
    .slate-reco-value {
        margin-top:3px;
        color:#F8FAFC;
        font-size:1.12rem;
        font-weight:900;
    }
    .slate-reco-meta {
        margin-top:3px;
        color:#93C5FD;
        font-size:.76rem;
        font-weight:750;
    }
    .slate-grid {
        display:grid;
        grid-template-columns:repeat(4, 1fr);
        gap:8px;
    }
    .slate-box {
        padding:10px 11px;
        border-radius:11px;
        background:rgba(13,26,45,.82);
        border:1px solid rgba(148,163,184,.10);
    }
    .slate-box-value {
        margin-top:3px;
        color:#E8EEF8;
        font-size:.88rem;
        font-weight:800;
    }
    .slate-footer {
        margin-top:11px;
        color:#7890AC;
        font-size:.74rem;
        font-weight:650;
    }
    .slate-footer span { padding:0 5px; color:#475569; }

    @media (max-width: 700px) {
        .slate-grid { grid-template-columns:repeat(2, 1fr); }
        .slate-card { padding:15px; border-radius:16px; }
        .slate-reco-value { font-size:1.06rem; }
    }


    .market-row {
        margin: 7px 0;
        padding: 11px 12px;
        border-radius: 11px;
        background: rgba(13,26,45,.72);
        border: 1px solid rgba(148,163,184,.10);
    }
    .market-row-title {
        color: #E8EEF8;
        font-size: .88rem;
        font-weight: 800;
    }
    .market-row-sub {
        margin-top: 3px;
        color: #8EA4BE;
        font-size: .74rem;
        font-weight: 650;
    }

</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="cfb-hero">
      <div class="cfb-kicker">College Football Analytics</div>
      <div class="cfb-title">CFB Edge</div>
      <div class="cfb-subtitle">Market-aware projections powered by SP+, matchup efficiency, roster quality, travel and selective weather.</div>
      <div class="version-pill">{MODEL_VERSION}</div>
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

app_section = st.radio(
    "Workspace",
    ["Live Model", "Backtest"],
    horizontal=True,
    index=0,
)

if app_section == "Backtest":
    st.markdown('<div class="section-kicker">Historical Backtest Lab</div>', unsafe_allow_html=True)
    st.markdown("### v0.3.1 vs v0.3.2 betting-layer comparison")
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
            [2021, 2022, 2023, 2024, 2025],
            default=[2022, 2023, 2024, 2025],
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

        signal_df = _bt_best_per_game(bt_df) if bt_policy == "Best market per game" else bt_df.copy()
        official = signal_df[signal_df["verdict"].isin(["BET","STRONG BET"])].copy()

        st.session_state["cfb_backtest_df"] = bt_df
        st.session_state["cfb_backtest_signal_df"] = signal_df
        st.session_state["cfb_backtest_games_df"] = bt_games_df
        st.session_state["cfb_backtest_config"] = {
            "seasons": bt_seasons, "holdout": bt_holdout, "scope": bt_scope,
            "method": bt_method, "policy": bt_policy,
        }

    if "cfb_backtest_signal_df" in st.session_state:
        bt_df = st.session_state["cfb_backtest_df"]
        signal_df = st.session_state["cfb_backtest_signal_df"]
        bt_games_df = st.session_state["cfb_backtest_games_df"]
        cfg = st.session_state.get("cfb_backtest_config", {})
        holdout = cfg.get("holdout", bt_holdout)

        train = signal_df[signal_df["season"] != holdout]
        test = signal_df[signal_df["season"] == holdout]

        summaries=[]
        for version in ["v0.3.1","v0.3.2"]:
            summaries.append(_bt_summary(train[train["version"]==version], f"{version} • Train"))
            summaries.append(_bt_summary(test[test["version"]==version], f"{version} • Holdout {holdout}"))
            summaries.append(_bt_summary(signal_df[signal_df["version"]==version], f"{version} • All"))
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
                        f"cfb_v040_backtest_{min(cfg.get('seasons',[2022]))}_{max(cfg.get('seasons',[2025]))}.csv")

        st.caption(
            "Historical CFBD line records are treated as generic provider snapshots/consensus medians; this app does not "
            "label them opening or closing lines. Spread and total prices are standardized at -110 because the generic "
            "CFBD line structure does not reliably include side-specific juice. Absolute ROI from retrospective full-season "
            "mode is not valid because of look-ahead bias."
        )
    st.stop()

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
                                 market_type="side", projection_gap=None, week=gp["week"])
                candidates.append((v, f"{gp['away']} ML", market["away_ml"], e, ev))
            if market.get("home_ml") is not None:
                v,e,ev,_ = grade(adjusted_home_wp, market["home_ml"], gp["confidence"],
                                 market_type="side", projection_gap=None, week=gp["week"])
                candidates.append((v, f"{gp['home']} ML", market["home_ml"], e, ev))

            if market.get("home_spread") is not None:
                spread_gap = raw_home_spread - market["home_spread"]
                hp = cover_probability(adjusted_home_margin, market["home_spread"], "home", cal_margin_sd)
                ap = 1 - hp
                v,e,ev,_ = grade(hp, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                candidates.append((v, f"{gp['home']} {market['home_spread']:+.1f}", -110, e, ev))
                v,e,ev,_ = grade(ap, -110, gp["confidence"], market_type="spread",
                                 projection_gap=spread_gap, week=gp["week"])
                candidates.append((v, f"{gp['away']} {-market['home_spread']:+.1f}", -110, e, ev))

            if market.get("total") is not None:
                total_gap = raw_total - market["total"]
                op = total_probability(adjusted_total, market["total"], "over", cal_total_sd)
                up = 1 - op
                v,e,ev,_ = grade(op, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
                candidates.append((v, f"Over {market['total']:g}", -110, e, ev))
                v,e,ev,_ = grade(up, -110, gp["confidence"], market_type="total",
                                 projection_gap=total_gap, week=gp["week"])
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

        actionable = slate_df[slate_df["best_verdict"].isin(["BET","STRONG BET"])]
        leans = slate_df[slate_df["best_verdict"].eq("LEAN")]

        c1, c2, c3 = st.columns(3)
        c1.metric("Games", len(slate_df))
        c2.metric("Bet Signals", len(actionable))
        c3.metric("Leans", len(leans))

        if len(actionable):
            st.success(f"{len(actionable)} ranked BET signal(s) currently clear the threshold.")
        elif len(leans):
            st.info("No full BET signals right now. There are LEAN-level edges.")
        else:
            st.info("No games in this slate currently clear the LEAN threshold.")

        def _fmt_spread(team, value):
            if pd.isna(value):
                return "—"
            return f"{team} {float(value):+.1f}"

        def _fmt_num(value, digits=1):
            if pd.isna(value):
                return "—"
            return f"{float(value):.{digits}f}"

        def _fmt_pct(value):
            if pd.isna(value):
                return "—"
            return f"{float(value)*100:.1f}%"

        def _verdict_class(verdict):
            return {
                "STRONG BET": "strong",
                "BET": "bet",
                "LEAN": "lean",
                "PASS": "pass",
                "NO LINE": "noline",
            }.get(str(verdict), "pass")

        # Rank games by verdict first, then edge, then EV.
        verdict_rank = {
            "STRONG BET": 4,
            "BET": 3,
            "LEAN": 2,
            "PASS": 1,
            "NO LINE": 0,
        }
        ranked_df = slate_df.copy()
        ranked_df["_verdict_rank"] = ranked_df["best_verdict"].map(verdict_rank).fillna(0)
        ranked_df["_edge_sort"] = pd.to_numeric(ranked_df["best_edge"], errors="coerce").fillna(-999)
        ranked_df["_ev_sort"] = pd.to_numeric(ranked_df["best_ev"], errors="coerce").fillna(-999)
        ranked_df = ranked_df.sort_values(
            ["_verdict_rank", "_edge_sort", "_ev_sort"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        def _render_slate_card(r, rank=None):
            market_spread = _fmt_spread(r["home_team"], r["market_home_spread"])
            model_spread = _fmt_spread(r["home_team"], r["model_home_spread"])
            market_total = _fmt_num(r["market_total"])
            model_total = _fmt_num(r["model_total"])
            best_market = str(r["best_market"]) if pd.notna(r["best_market"]) and str(r["best_market"]).strip() else "No actionable market"
            verdict = str(r["best_verdict"])
            edge = _fmt_pct(r["best_edge"])
            ev = _fmt_pct(r["best_ev"])
            rank_text = f"#{rank} " if rank is not None else ""

            st.markdown(
                f"""
                <div class="slate-card">
                  <div class="slate-card-top">
                    <div>
                      <div class="slate-time">{rank_text}{html.escape(str(r['kickoff_et']))}</div>
                      <div class="slate-matchup">{html.escape(str(r['away_team']))} <span>@</span> {html.escape(str(r['home_team']))}</div>
                    </div>
                    <div class="slate-badge {_verdict_class(verdict)}">{html.escape(verdict)}</div>
                  </div>

                  <div class="slate-reco">
                    <div class="slate-reco-label">Top Model Call</div>
                    <div class="slate-reco-value">{html.escape(best_market)}</div>
                    <div class="slate-reco-meta">Edge {edge} &nbsp;•&nbsp; EV {ev}</div>
                  </div>

                  <div class="slate-grid">
                    <div class="slate-box">
                      <div class="slate-box-label">Model Spread</div>
                      <div class="slate-box-value">{html.escape(model_spread)}</div>
                    </div>
                    <div class="slate-box">
                      <div class="slate-box-label">Market Spread</div>
                      <div class="slate-box-value">{html.escape(market_spread)}</div>
                    </div>
                    <div class="slate-box">
                      <div class="slate-box-label">Model Total</div>
                      <div class="slate-box-value">{model_total}</div>
                    </div>
                    <div class="slate-box">
                      <div class="slate-box-label">Market Total</div>
                      <div class="slate-box-value">{market_total}</div>
                    </div>
                  </div>

                  <div class="slate-footer">
                    Confidence {int(r['model_confidence'])}/100
                    <span>•</span>
                    Projected {html.escape(str(r['away_team']))} {float(r['projected_away_score']):.1f} – {html.escape(str(r['home_team']))} {float(r['projected_home_score']):.1f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Full market audit for this game, ranked strongest to weakest.
            try:
                market_rows = json.loads(r.get("market_grades_json", "[]") or "[]")
            except Exception:
                market_rows = []

            with st.expander(f"All Markets • {r['away_team']} @ {r['home_team']}", expanded=False):
                if not market_rows:
                    st.caption("No market lines are available for this game.")
                else:
                    market_rank = {"STRONG BET": 4, "BET": 3, "LEAN": 2, "PASS": 1, "NO LINE": 0}
                    market_rows = sorted(
                        market_rows,
                        key=lambda x: (
                            market_rank.get(str(x.get("verdict")), 0),
                            float(x.get("edge") if x.get("edge") is not None else -999),
                            float(x.get("ev") if x.get("ev") is not None else -999),
                        ),
                        reverse=True,
                    )

                    for m in market_rows:
                        verdict_m = str(m.get("verdict", "PASS"))
                        market_m = str(m.get("market", ""))
                        odds_m = m.get("odds")
                        edge_m = m.get("edge")
                        ev_m = m.get("ev")

                        icon = "🟢" if verdict_m in {"BET", "STRONG BET"} else ("🟡" if verdict_m == "LEAN" else "⚪")
                        odds_txt = f"{int(float(odds_m)):+d}" if odds_m is not None else ""
                        edge_txt = f"{float(edge_m)*100:+.1f}%" if edge_m is not None else "—"
                        ev_txt = f"{float(ev_m)*100:+.1f}%" if ev_m is not None else "—"

                        st.markdown(
                            f"""
                            <div class="market-row">
                              <div class="market-row-title">{icon} {html.escape(verdict_m)} · {html.escape(market_m)} {html.escape(odds_txt)}</div>
                              <div class="market-row-sub">Edge {edge_txt} &nbsp;•&nbsp; EV {ev_txt}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Top bets shown immediately, ordered strongest to weakest.
        top_bets = ranked_df[ranked_df["best_verdict"].isin(["STRONG BET", "BET"])].copy()
        if len(top_bets):
            st.markdown('<div class="section-kicker">Top Bets</div>', unsafe_allow_html=True)
            for i, (_, r) in enumerate(top_bets.iterrows(), start=1):
                _render_slate_card(r, rank=i)
        else:
            st.info("No BET or STRONG BET signals currently qualify.")

        # Leans and passes stay out of the way until the user wants them.
        lean_df = ranked_df[ranked_df["best_verdict"].eq("LEAN")].copy()
        pass_df = ranked_df[ranked_df["best_verdict"].isin(["PASS", "NO LINE"])].copy()

        with st.expander(f"Leans ({len(lean_df)})", expanded=False):
            if len(lean_df):
                for i, (_, r) in enumerate(lean_df.iterrows(), start=1):
                    _render_slate_card(r, rank=i)
            else:
                st.caption("No lean-level signals on this slate.")

        with st.expander(f"Passes / No Line ({len(pass_df)})", expanded=False):
            if len(pass_df):
                for i, (_, r) in enumerate(pass_df.iterrows(), start=1):
                    _render_slate_card(r, rank=i)
            else:
                st.caption("No pass/no-line games on this slate.")

        with st.expander("View full slate data"):
            display_cols = [
                "kickoff_et","away_team","home_team","projected_away_score","projected_home_score",
                "model_home_spread","model_total","market_home_spread","market_total",
                "best_verdict","best_market","best_edge","best_ev","model_confidence"
            ]
            st.dataframe(slate_df[display_cols], use_container_width=True, hide_index=True)

        ios_save_button(
            f"Save {slate_choice} Slate CSV",
            slate_df.to_csv(index=False),
            f"cfb_v032_{selected_date}_{slate_choice.lower().replace(' ','_')}_slate.csv",
        )

        st.caption(
            "Slate lines use the median across available CFBD providers. "
            "v0.3.2 applies early-season market shrinkage and wider uncertainty before grading. "
            "Spread and total pricing are assumed at -110 in slate mode unless actual prices are available."
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

st.markdown('<div class="section-kicker">Model Projection</div>', unsafe_allow_html=True)

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
st.markdown('<div class="section-kicker">Market Comparison</div>', unsafe_allow_html=True)

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
    f"cfb_projection_v031_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
)
st.caption("Use this to save the projection file for audit/upload.")

if st.button("Should I Bet?",type="primary",use_container_width=True):
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
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="side",week=p["week"])
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
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    total_gap = p["model_total"] - market_total
    op=total_probability(adj_total,market_total,"over",cal_total_sd)
    up=1-op
    for name,prob,odds in [(f"Over {market_total:g}",op,over_odds),(f"Under {market_total:g}",up,under_odds)]:
        v,e,ev,imp=grade(prob,odds,p["confidence"],market_type="total",
                         projection_gap=total_gap,week=p["week"])
        markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    rank={"STRONG BET":3,"BET":2,"LEAN":1,"PASS":0}
    markets.sort(key=lambda x:(rank[x[0]],x[5]),reverse=True)
    best=markets[0]
    if best[0] in {"BET","STRONG BET"}:
        st.success(f"🟢 **{best[0]}: {best[1]} {int(best[2]):+d}**\n\nModel {best[3]*100:.1f}% • Edge {best[4]*100:+.1f}% • EV {best[5]*100:+.1f}%")
    elif best[0]=="LEAN":
        st.warning(f"🟡 **LEAN: {best[1]} {int(best[2]):+d}**")
    else:
        st.info("⚪ **PASS — no market clears the threshold.**")

    st.markdown('<div class="section-kicker">Market Grades</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-kicker">Audit Export</div>', unsafe_allow_html=True)
    st.caption("Download this CSV and upload it back into ChatGPT so the inputs, projection, market comparison, and betting call can be audited.")
    ios_save_button(
        "Save Game CSV",
        export_df.to_csv(index=False),
        f"cfb_model_v031_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
    )

st.divider()
st.caption("CFB Edge • v0.3.1-MARKET-DROPDOWNS • Projection logic unchanged from the calibrated v0.2.7.1 baseline.")
