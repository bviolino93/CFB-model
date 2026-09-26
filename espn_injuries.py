"""
espn_injuries.py — ESPN injury lookup for Saturday Edge

Pulls injury reports from ESPN's public (undocumented) college-football
endpoints and flags picks where a key player is out.

Design notes:
  - This NEVER raises. Any failure downgrades to a status message.
  - It ALWAYS returns at least one note, so a card can never be silent.
    Silence was the old bug: you could not tell "no injuries" from
    "nothing worked". Now every card says which.
  - It only fetches teams that appear in your picks, not all 130+.
  - Results are cached so reloading doesn't re-hit ESPN.

Status strings you may see on a card:
  "No injuries listed"        -> working; ESPN has nothing for that team
  "ESPN unreachable"          -> request failed or timed out
  "No ESPN match for 'X'"     -> name mismatch; add X to MANUAL_TEAM_MAP
  "Injury module not loaded"  -> this file isn't being imported (see app.py)
"""

import difflib
import requests
import streamlit as st

# Bumped whenever this file changes. app.py prints it on each card so you
# can confirm which copy is actually running.
MODULE_VERSION = "v3"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

TIMEOUT = 6  # seconds per request — fail fast rather than hang the app

# Positions worth flagging prominently. QB is the one that moves a line.
KEY_POSITIONS = {"QB"}

# Statuses that mean "probably not playing"
OUT_STATUSES = {"out", "doubtful", "suspension", "injured reserve"}

# CFBD name -> ESPN name, for the handful fuzzy matching gets wrong.
# Add entries here as you see "No ESPN match for ..." on a card.
MANUAL_TEAM_MAP = {
    "Louisiana Monroe": "UL Monroe",
    "Massachusetts": "UMass",
    "Connecticut": "UConn",
    "San Jose State": "San Jose State",
    "Appalachian State": "App State",
    "Southern Mississippi": "Southern Miss",
}


# ---------------------------------------------------------------------------
# Low-level fetch
# ---------------------------------------------------------------------------

def _get(url, params=None):
    """GET a URL. Returns parsed JSON, or None if anything went wrong."""
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
    {espn_name: espn_team_id} for college football teams.
    Returns None (not {}) if every fetch failed, so callers can tell
    "no teams" from "couldn't ask".

    groups=80 is FBS, 81 is FCS. Both are requested — otherwise every FCS
    opponent (Lindenwood, Mercer, etc.) reads as a name mismatch.
    """
    teams = {}
    any_ok = False

    for group in (80, 81):
        data = _get(f"{SITE}/teams", params={"limit": 1000, "groups": group})
        if not data:
            continue
        try:
            entries = data["sports"][0]["leagues"][0]["teams"]
        except (KeyError, IndexError, TypeError):
            continue

        any_ok = True
        for entry in entries:
            team = entry.get("team", {})
            tid = team.get("id")
            if not tid:
                continue
            for key in (team.get("displayName"), team.get("name"),
                        team.get("location"), team.get("shortDisplayName")):
                if key:
                    teams.setdefault(key, tid)

    return teams if any_ok else None


def match_team(cfbd_name, espn_teams):
    """CFBD team name -> ESPN team id, or None if no confident match."""
    if not cfbd_name or not espn_teams:
        return None

    name = MANUAL_TEAM_MAP.get(cfbd_name, cfbd_name)
    if name in espn_teams:
        return espn_teams[name]

    # Fuzzy fallback. Strict on purpose — a wrong match silently flags the
    # wrong game, which is worse than no match at all.
    close = difflib.get_close_matches(name, list(espn_teams.keys()), n=1, cutoff=0.85)
    return espn_teams[close[0]] if close else None


# ---------------------------------------------------------------------------
# Roster (used to resolve player positions)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_roster(espn_team_id):
    """{athlete_id: (name, position)} for one team. {} on failure."""
    data = _get(f"{SITE}/teams/{espn_team_id}/roster")
    if not data:
        return {}

    roster = {}
    for group in data.get("athletes", []):
        items = group.get("items", []) if isinstance(group, dict) else []
        for athlete in items:
            aid = athlete.get("id")
            if not aid:
                continue
            name = athlete.get("fullName") or athlete.get("displayName")
            pos = (athlete.get("position") or {}).get("abbreviation")
            roster[str(aid)] = (name, pos)
    return roster


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_team_injuries(espn_team_id):
    """
    List of {"player","position","status"} for one team.
    Returns None if the request failed — distinct from [] meaning
    "ESPN answered and listed nobody".
    """
    index = _get(f"{CORE}/teams/{espn_team_id}/injuries", params={"limit": 100})
    if index is None:
        return None

    roster = get_roster(espn_team_id)
    out = []

    for item in index.get("items", []):
        ref = item.get("$ref")
        if not ref:
            continue
        record = _get(ref)
        if not record:
            continue

        # The athlete arrives as a $ref URL ending in the athlete id. Pull
        # the id and look it up in the roster rather than making another
        # request for every single player.
        athlete_ref = (record.get("athlete") or {}).get("$ref", "")
        athlete_id = athlete_ref.rstrip("/").split("/")[-1].split("?")[0]
        name, pos = roster.get(str(athlete_id), (None, None))

        out.append({
            "player": name or "Unknown",
            "position": pos or "?",
            "status": (record.get("status") or "Unknown").strip(),
        })

    return out


# ---------------------------------------------------------------------------
# What the app calls
# ---------------------------------------------------------------------------

def check_team(cfbd_team_name):
    """
    Returns (flagged, notes) for one team.
    notes is ALWAYS non-empty — a status line if nothing else.
    """
    label = cfbd_team_name or "?"

    espn_teams = get_espn_teams()
    if espn_teams is None:
        return False, ["ESPN unreachable"]
    if not espn_teams:
        return False, ["ESPN returned no teams"]

    tid = match_team(cfbd_team_name, espn_teams)
    if not tid:
        return False, [f"No ESPN match for '{label}'"]

    injuries = get_team_injuries(tid)
    if injuries is None:
        return False, ["ESPN unreachable"]
    if not injuries:
        return False, ["No injuries listed"]

    flagged = False
    key_notes = []
    other_out = 0

    for inj in injuries:
        is_out = any(s in inj["status"].lower() for s in OUT_STATUSES)
        if not is_out:
            continue
        if inj["position"] in KEY_POSITIONS:
            flagged = True
            key_notes.append(f"{inj['position']} {inj['player']} — {inj['status']}")
        else:
            other_out += 1

    notes = list(key_notes)
    if other_out:
        notes.append(f"{other_out} other listed out")
    if not notes:
        notes.append(f"{len(injuries)} listed, none out")

    return flagged, notes


def check_matchup(home_team, away_team):
    """
    Returns (flagged, notes) covering both sides of one game.
    This is what the app calls per pick card.
    """
    home_flag, home_notes = check_team(home_team)
    away_flag, away_notes = check_team(away_team)

    notes = [f"{away_team}: {n}" for n in away_notes]
    notes += [f"{home_team}: {n}" for n in home_notes]

    return (home_flag or away_flag), notes
