
import streamlit as st
import pandas as pd
from datetime import date
from model import MODEL_VERSION, fetch_games, load_rating_maps, project_game, cover_probability, total_probability, fair_ml, grade

st.set_page_config(page_title="CFB Model", page_icon="🏈", layout="centered")
st.title("🏈 CFB Model")
st.caption(f"Version {MODEL_VERSION} • Early-season prototype")

try:
    API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error("Missing CFBD_API_KEY in Streamlit Secrets.")
    st.stop()

@st.cache_data(ttl=1800)
def get_games(year):
    return fetch_games(API_KEY, year)

@st.cache_data(ttl=3600)
def get_ratings(year):
    return load_rating_maps(API_KEY, year)

def game_date_et(g):
    s=g.get("startDate")
    if not s: return None
    try: return pd.to_datetime(s, utc=True).tz_convert("America/New_York").date()
    except: return None

selected_date = st.date_input("Game date", value=date.today())
year = selected_date.year

try:
    games=get_games(year)
except Exception as e:
    st.error(f"CFBD games request failed: {e}")
    st.stop()

daily=[g for g in games if game_date_et(g)==selected_date]
if not daily:
    st.warning("No games found for that date.")
    st.stop()

labels={}
for g in daily:
    label=f"{g.get('awayTeam','Away')} @ {g.get('homeTeam','Home')}"
    if g.get("neutralSite"): label += " (Neutral)"
    labels[label]=g

game=labels[st.selectbox("Game", list(labels.keys()))]

try:
    current_map, previous_map = get_ratings(year)
except Exception as e:
    st.error(f"CFBD SP+ request failed: {e}")
    st.stop()

hfa=st.number_input("Home-field advantage", min_value=0.0, max_value=6.0, value=2.5, step=.25, disabled=bool(game.get("neutralSite")))
p=project_game(game,current_map,previous_map,hfa)

st.subheader("Model projection")
a,b,c=st.columns(3)
a.metric(p["away"],f"{p['away_score']:.1f}")
b.metric(p["home"],f"{p['home_score']:.1f}")
c.metric("Total",f"{p['model_total']:.1f}")
st.write(f"**Model spread:** {p['home']} {p['model_home_spread']:+.1f}")
st.write(f"**Win probability:** {p['home']} {p['home_win_prob']*100:.1f}% / {p['away']} {p['away_win_prob']*100:.1f}%")
st.caption(f"{p['away']} source: {p['away_rating']['source']} • {p['home']} source: {p['home_rating']['source']}")

st.divider()
st.subheader("Sportsbook lines")
m1,m2=st.columns(2)
away_ml=m1.number_input(f"{p['away']} ML",value=100,step=5)
home_ml=m2.number_input(f"{p['home']} ML",value=-110,step=5)

s1,s2=st.columns(2)
home_spread=s1.number_input(f"{p['home']} spread",value=float(round(p['model_home_spread']*2)/2),step=.5)
home_spread_odds=s2.number_input("Home spread odds",value=-110,step=5)
s3,s4=st.columns(2)
away_spread=s3.number_input(f"{p['away']} spread",value=float(-home_spread),step=.5)
away_spread_odds=s4.number_input("Away spread odds",value=-110,step=5)

t1,t2,t3=st.columns(3)
market_total=t1.number_input("Total",value=float(round(p["model_total"]*2)/2),step=.5)
over_odds=t2.number_input("Over odds",value=-110,step=5)
under_odds=t3.number_input("Under odds",value=-110,step=5)

if st.button("Should I Bet?",type="primary",use_container_width=True):
    markets=[]
    for name,prob,odds in [
        (f"{p['away']} ML",p["away_win_prob"],away_ml),
        (f"{p['home']} ML",p["home_win_prob"],home_ml)
    ]:
        v,e,ev,imp=grade(prob,odds,75); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    hc=cover_probability(p["home_margin"],home_spread,"home")
    ac=cover_probability(p["home_margin"],home_spread,"away")
    for name,prob,odds in [
        (f"{p['home']} {home_spread:+.1f}",hc,home_spread_odds),
        (f"{p['away']} {away_spread:+.1f}",ac,away_spread_odds)
    ]:
        v,e,ev,imp=grade(prob,odds,75); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

    op=total_probability(p["model_total"],market_total,"over")
    up=1-op
    for name,prob,odds in [(f"Over {market_total:g}",op,over_odds),(f"Under {market_total:g}",up,under_odds)]:
        v,e,ev,imp=grade(prob,odds,75); markets.append((v,name,odds,prob,e,ev,fair_ml(prob)))

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

st.divider()
st.caption("v0.1.0 uses SP+ as the anchor. Margin and total distributions are provisional and should be calibrated from tracked results.")
