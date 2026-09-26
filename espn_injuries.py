"""
espn_injuries.py — ESPN injury lookup for Saturday Edge

v4: hard-capped. The previous version issued one HTTP request per injured
player with no overall limit, which could add minutes to a slate build.
This version can never cost more than TOTAL_BUDGET seconds per build,
no matter what ESPN does.

Safety properties:
  - ENABLED = False turns the whole thing off without touching app.py.
  - Every request has a short timeout.
  - A global time budget and request cap apply across the ENTIRE build,
    not per team. When either runs out, remaining teams return
    "injury check skipped" instantly.
  - Successful lookups are cached; skipped ones are NOT, so the next
    build retries them.
  - This NEVER raises to the caller and ALWAYS returns a note.

Status strings you may see on a card:
  "No injuries listed"        -> working; ESPN has nothing for that team
  "ESPN unreachable"          -> request failed or timed out
  "No ESPN match for 'X'"     -> name mismatch; add X to MANUAL_TEAM_MAP
  "Injury check skipped"      -> hit the time/request cap this build
  "Injury check off"          -> ENABLED is False
  "Injury module not loaded"  -> this file isn't being imported
"""

import difflib
import time

import requests
import streamlit as st

# Bumped whenever this file changes. app.py prints it on each card so you
# can confirm which copy is actually running.
MODULE_VERSION = "v4"

# ---------------------------------------------------------------------------
# Config — the knobs that matter
# ---------------------------------------------------------------------------

ENABLED = True          # set False to switch the feature off entirely

TIMEOUT = 2.5           # seconds per request
TOTAL_BUDGET = 12.0     # seconds of ESPN work allowed per build, all teams
MAX_CALLS = 60          # hard ceiling on requests per build
MAX_REFS_PER_TEAM = 8   # only resolve this many injured players per team
WINDOW_RESET = 45.0     # seconds of idle before the budget refills

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

KEY_POSITIONS = {"QB"}
OUT_STATUSES = {"out", "doubtful", "suspension", "injured reserve"}

# CFBD name -> ESPN name, for the handful fuzzy matching gets wrong.
MANUAL_TEAM_MAP = {
    "Louisiana Monroe": "UL Monroe",
    "Massachusetts": "UMass",
    "Connecticut": "UConn",
    "Appalachian State": "App State",
    "Southern Mississippi": "Southern Miss",
}


class _BudgetExhausted(Exception):
    """Raised internally so Streamlit does not cache a skipped result."""


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

_STATE = {"start": 0.0, "calls": 0}


def _budget_available():
    """True if there is time and request headroom left in this build."""
    now = time.monotonic()
    # A gap this long means a new build started; refill.
    if now - _STATE["start"] > WINDOW_RESET:
        _STATE["start"] = now
        _STATE["calls"] = 0
    if _STATE["calls"] >= MAX_CALLS:
        return False
    return (now - _STATE["start"]) < TOTAL_BUDGET


def _get(url, params=None):
    """
    GET a URL inside the budget. Returns parsed JSON, or None on failure.
    Raises _BudgetExhausted if there's no headroom left.
    """
    if not _budget_available():
        raise _BudgetExhausted()
    _STATE["calls"] += 1
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except _BudgetExhausted:
        raise
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Team directory
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_espn_teams():
    """
    {espn_name: espn_team_id}. None if every fetch failed.
    groups 80 (FBS) and 81 (FCS) so FCS opponents still match.
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

    # Strict on purpose — a wrong match silently flags the wrong game.
    close = difflib.get_close_matches(name, list(espn_teams.keys()), n=1, cutoff=0.85)
    return espn_teams[close[0]] if close else None


# ---------------------------------------------------------------------------
# Roster + injuries
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_roster(espn_team_id):
    """{athlete_id: (name, position)}. {} on failure."""
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


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_team_injuries(espn_team_id):
    """
    List of {"player","position","status"}, or None if the request failed.
    At most MAX_REFS_PER_TEAM players are resolved — enough to catch a QB
    without turning one team into thirty requests.
    """
    index = _get(f"{CORE}/teams/{espn_team_id}/injuries", params={"limit": 100})
    if index is None:
        return None

    items = [i for i in index.get("items", []) if i.get("$ref")]
    if not items:
        return []

    roster = get_roster(espn_team_id)
    out = []

    for item in items[:MAX_REFS_PER_TEAM]:
        record = _get(item["$ref"])
        if not record:
            continue

        athlete_ref = (record.get("athlete") or {}).get("$ref", "")
        athlete_id = athlete_ref.rstrip("/").split("/")[-1].split("?")[0]
        name, pos = roster.get(str(athlete_id), (None, None))

        out.append({
            "player": name or "Unknown",
            "position": pos or "?",
            "status": (record.get("status") or "Unknown").strip(),
        })

    if len(items) > MAX_REFS_PER_TEAM:
        out.append({
            "player": f"+{len(items) - MAX_REFS_PER_TEAM} not checked",
            "position": "-",
            "status": "unresolved",
        })

    return out


# ---------------------------------------------------------------------------
# What the app calls
# ---------------------------------------------------------------------------

def check_team(cfbd_team_name):
    """(flagged, notes) for one team. notes is always non-empty."""
    if not ENABLED:
        return False, ["Injury check off"]

    label = cfbd_team_name or "?"

    try:
        espn_teams = get_espn_teams()
        if espn_teams is None:
            return False, ["ESPN unreachable"]
        if not espn_teams:
            return False, ["ESPN returned no teams"]

        tid = match_team(cfbd_team_name, espn_teams)
        if not tid:
            return False, [f"No ESPN match for '{label}'"]

        injuries = get_team_injuries(tid)
    except _BudgetExhausted:
        return False, ["Injury check skipped"]
    except Exception:
        return False, ["Injury check error"]

    if injuries is None:
        return False, ["ESPN unreachable"]
    if not injuries:
        return False, ["No injuries listed"]

    flagged = False
    key_notes = []
    other_out = 0
    unresolved = 0

    for inj in injuries:
        if inj["status"] == "unresolved":
            unresolved += 1
            continue
        if not any(s in inj["status"].lower() for s in OUT_STATUSES):
            continue
        if inj["position"] in KEY_POSITIONS:
            flagged = True
            key_notes.append(f"{inj['position']} {inj['player']} — {inj['status']}")
        else:
            other_out += 1

    notes = list(key_notes)
    if other_out:
        notes.append(f"{other_out} other listed out")
    if unresolved:
        notes.append("some not checked")
    if not notes:
        notes.append("none listed out")

    return flagged, notes


def check_matchup(home_team, away_team):
    """(flagged, notes) covering both sides of one game."""
    home_flag, home_notes = check_team(home_team)
    away_flag, away_notes = check_team(away_team)

    notes = [f"{away_team}: {n}" for n in away_notes]
    notes += [f"{home_team}: {n}" for n in home_notes]

    return (home_flag or away_flag), notes
