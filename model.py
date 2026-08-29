
import requests
from statistics import NormalDist

BASE_URL = "https://api.collegefootballdata.com"
MODEL_VERSION = "0.1.0"
BASE_TOTAL = 55.0
DEFAULT_HFA = 2.5
MARGIN_SD = 16.0
TOTAL_SD = 13.5

def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}

def cfbd_get(path, api_key, params=None):
    r = requests.get(BASE_URL + path, headers=_headers(api_key), params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_games(api_key, year):
    return cfbd_get("/games", api_key, {"year": year, "seasonType": "regular"})

def fetch_sp(api_key, year):
    return cfbd_get("/ratings/sp", api_key, {"year": year})

def _num(x, default=0.0):
    try: return float(x)
    except: return float(default)

def _sp_map(rows):
    return {str(r.get("team")): r for r in rows if r.get("team")}

def load_rating_maps(api_key, year):
    current = _sp_map(fetch_sp(api_key, year))
    try:
        previous = _sp_map(fetch_sp(api_key, year-1))
    except:
        previous = {}
    return current, previous

def get_team_rating(team, current_map, previous_map):
    row = current_map.get(team)
    source = "Current-year SP+"
    if not row:
        row = previous_map.get(team)
        source = "Prior-year SP+ fallback"
    if not row:
        return {"team":team,"rating":0.0,"offense":0.0,"defense":0.0,"special":0.0,"pace":0.0,"source":"Average fallback"}
    off = row.get("offense") or {}
    deff = row.get("defense") or {}
    st = row.get("specialTeams") or {}
    return {
        "team": team,
        "rating": _num(row.get("rating")),
        "offense": _num(off.get("rating")),
        "defense": _num(deff.get("rating")),
        "special": _num(st.get("rating")),
        "pace": _num(off.get("pace")),
        "source": source,
    }

def project_game(game, current_map, previous_map, hfa=DEFAULT_HFA):
    away, home = game["awayTeam"], game["homeTeam"]
    neutral = bool(game.get("neutralSite"))
    ar = get_team_rating(away, current_map, previous_map)
    hr = get_team_rating(home, current_map, previous_map)
    applied_hfa = 0.0 if neutral else float(hfa)

    home_margin = (hr["rating"] - ar["rating"]) + applied_hfa

    offense_tilt = 0.20 * (hr["offense"] + ar["offense"])
    defense_tilt = -0.10 * (hr["defense"] + ar["defense"])
    pace_tilt = 0.04 * (hr["pace"] + ar["pace"])
    total = max(30.0, min(85.0, BASE_TOTAL + offense_tilt + defense_tilt + pace_tilt))

    home_score = max(7.0, (total + home_margin)/2)
    away_score = max(7.0, (total - home_margin)/2)
    total = home_score + away_score
    home_margin = home_score - away_score

    nd = NormalDist()
    home_wp = nd.cdf(home_margin / MARGIN_SD)
    return {
        "away":away,"home":home,"away_rating":ar,"home_rating":hr,
        "home_margin":home_margin,"model_home_spread":-home_margin,
        "model_total":total,"away_score":away_score,"home_score":home_score,
        "home_win_prob":home_wp,"away_win_prob":1-home_wp,
        "neutral":neutral,"hfa":applied_hfa
    }

def cover_probability(home_margin_mean, market_home_spread, side="home"):
    threshold = -float(market_home_spread)
    p_home = 1.0 - NormalDist(mu=home_margin_mean, sigma=MARGIN_SD).cdf(threshold)
    return p_home if side=="home" else 1-p_home

def total_probability(model_total, market_total, side="over"):
    p_over = 1.0 - NormalDist(mu=model_total, sigma=TOTAL_SD).cdf(float(market_total))
    return p_over if side=="over" else 1-p_over

def implied_prob(odds):
    odds=float(odds)
    return 100/(odds+100) if odds>0 else abs(odds)/(abs(odds)+100)

def fair_ml(prob):
    p=max(.001,min(.999,float(prob)))
    return -round(100*p/(1-p)) if p>=.5 else round(100*(1-p)/p)

def expected_value(prob, odds):
    odds=float(odds)
    profit=odds/100 if odds>0 else 100/abs(odds)
    return prob*profit-(1-prob)

def juice_thresholds(odds):
    odds=int(odds)
    if odds>=-149: return .025,.05
    if odds>=-179: return .03,.06
    if odds>=-199: return .04,.07
    if odds>=-249: return .05,.08
    return .07,.12

def grade(prob, odds, confidence=75):
    imp=implied_prob(odds); edge=prob-imp; ev=expected_value(prob, odds)
    me, mv = juice_thresholds(odds)
    if confidence>=80 and edge>=me+.02 and ev>=mv+.03: verdict="STRONG BET"
    elif confidence>=70 and edge>=me and ev>=mv: verdict="BET"
    elif edge>0 and ev>0: verdict="LEAN"
    else: verdict="PASS"
    return verdict,edge,ev,imp
