"""
espn_injuries.py — ESPN injury lookup for Saturday Edge

Pulls injury reports from ESPN's public (undocumented) college-football
endpoints and flags picks where a key player is out.

Design notes:
  - This NEVER raises. If ESPN is down, changes their JSON, or renames
    something, every function returns empty and the app keeps working.
    You will see "injury data unavailable" instead of a crash on Saturday.
  - It only fetches teams that appear in your picks (usually 10-25 teams),
    not all 130+. Keeps it fast.
  - Results are cached so reloading the page doesn't re-hit ESPN.

ESPN is not an official API. It can change without warning. Treat a silent
empty result as "no data", not as "no injuries".
"""

import difflib
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

TIMEOUT = 6  # seconds per request — fail fast rather than hang the app

# Positions worth flagging. QB is the one that actually moves a line.
KEY_POSITIONS = {"QB"}

# Statuses that mean "probably not playing"
OUT_STATUSES = {"out", "doubtful", "suspension", "injured reserve"}

# CFBD name -> ESPN name, for the handful fuzzy matching gets wrong.
# Add entries here as you find them.
MANUAL_TEAM_MAP = {
    "Louisiana Monroe": "UL Monroe",
    "Massachusetts": "UMass",
    "Connecticut": "UConn",
    "Miami": "Miami",
    "San José State": "San Jose State",
}


# ---------------------------------------------------------------------------
# Low-level fetch
# ---------------------------------------------------------------------------

def _get(url, params=None):
    """GET a URL and return parsed JSON, or None on any failure."""
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Team directory
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_espn_teams():
    """
    Returns {espn_display_name: espn_team_id} for all college football teams.
    Cached for a day — this list basically never changes mid-season.
    """
    data = _get(f"{SITE}/teams", params={"limit": 1000})
    if not data:
        return {}

    teams = {}
    try:
        groups = data["sports"][0]["leagues"][0]["teams"]
    except (KeyError, IndexError, TypeError):
        return {}

    for entry in groups:
        team = entry.get("team", {})
        name = team.get("displayName") or team.get("name")
        tid = team.get("id")
        if name and tid:
            # Store both the full name ("Navy Midshipmen") and the short
            # location ("Navy"), since CFBD uses the short form.
            teams[name] = tid
            location = team.get("location")
            if location:
                teams[location] = tid
    return teams


def match_team(cfbd_name, espn_teams):
    """
    Map a CFBD team name to an ESPN team id.
    Returns the id, or None if no confident match.
    """
    if not cfbd_name or not espn_teams:
        return None

    name = MANUAL_TEAM_MAP.get(cfbd_name, cfbd_name)

    # Exact match first
    if name in espn_teams:
        return espn_teams[name]

    # Fuzzy fallback. cutoff=0.85 is deliberately strict — a wrong match
    # is worse than no match, because it silently flags the wrong game.
    candidates = difflib.get_close_matches(name, espn_teams.keys(), n=1, cutoff=0.85)
    if candidates:
        return espn_teams[candidates[0]]

    return None


# ---------------------------------------------------------------------------
# Roster (used to get player positions)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_roster(espn_team_id):
    """Returns {athlete_id: (full_name, position_abbrev)} for one team."""
    data = _get(f"{SITE}/teams/{espn_team_id}/roster")
    if not data:
        return {}

    roster = {}
    for group in data.get("athletes", []):
        # ESPN groups the roster by offense/defense/specialTeam
        items = group.get("items", []) if isinstance(group, dict) else []
        for athlete in items:
            aid = athlete.get("id")
            full_name = athlete.get("fullName") or athlete.get("displayName")
            pos = (athlete.get("position") or {}).get("abbreviation")
            if aid:
                roster[str(aid)] = (full_name, pos)
    return roster


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_team_injuries(espn_team_id):
    """
    Returns a list of dicts for one team:
        [{"player": str, "position": str, "status": str, "detail": str}, ...]

    Empty list means either no injuries OR no data. You cannot tell the
    difference from the return value — that's why the UI should say
    "no data" rather than "healthy".
    """
    index = _get(f"{CORE}/teams/{espn_team_id}/injuries", params={"limit": 100})
    if not index:
        return []

    roster = get_roster(espn_team_id)
    out = []

    for item in index.get("items", []):
        ref = item.get("$ref")
        if not ref:
            continue

        record = _get(ref)
        if not record:
            continue

        status = (record.get("status") or "").strip()
        detail = record.get("shortComment") or record.get("longComment") or ""

        # The athlete comes back as a $ref URL with the id as the last path
        # segment. Pull the id out and look it up in the roster rather than
        # making another request per player.
        athlete_ref = (record.get("athlete") or {}).get("$ref", "")
        athlete_id = athlete_ref.rstrip("/").split("/")[-1].split("?")[0]

        player, position = roster.get(str(athlete_id), (None, None))

        out.append({
            "player": player or "Unknown",
            "position": position or "?",
            "status": status or "Unknown",
            "detail": detail,
        })

    return out


# ---------------------------------------------------------------------------
# The bit you actually call
# ---------------------------------------------------------------------------

def check_team(cfbd_team_name):
    """
    For one team, returns (flagged, notes):
        flagged: True if a key-position player is listed out/doubtful
        notes:   list of human-readable strings to show in the UI
    """
    espn_teams = get_espn_teams()
    if not espn_teams:
        return False, ["ESPN injury data unavailable"]

    tid = match_team(cfbd_team_name, espn_teams)
    if not tid:
        return False, [f"No ESPN match for '{cfbd_team_name}'"]

    injuries = get_team_injuries(tid)
    if not injuries:
        return False, []

    notes = []
    flagged = False

    for inj in injuries:
        status_l = inj["status"].lower()
        is_out = any(s in status_l for s in OUT_STATUSES)

        if inj["position"] in KEY_POSITIONS and is_out:
            flagged = True
            notes.append(f"⚠️ {inj['position']} {inj['player']} — {inj['status']}")
        elif is_out:
            notes.append(f"{inj['position']} {inj['player']} — {inj['status']}")

    return flagged, notes


def check_matchup(home_team, away_team):
    """
    For one game, returns (flagged, notes) covering both sides.
    This is what you want in the picks table.
    """
    home_flag, home_notes = check_team(home_team)
    away_flag, away_notes = check_team(away_team)

    notes = []
    notes += [f"{home_team}: {n}" for n in home_notes]
    notes += [f"{away_team}: {n}" for n in away_notes]

    return (home_flag or away_flag), notes
