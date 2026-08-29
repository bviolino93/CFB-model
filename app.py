
import streamlit as st
import pandas as pd
from datetime import date
from model import MODEL_VERSION, fetch_games, load_rating_maps, project_game, cover_probability, total_probability, fair_ml, grade, fetch_lines, normalize_game_lines

st.set_page_config(page_title="CFB Model", page_icon="🏈", layout="centered")
st.title("🏈 CFB Model")
st.caption("Version 0.1.6-IOS-DOWNLOAD • Early-season prototype")

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


@st.cache_data(ttl=300)
def get_market_lines(game_id, year):
    return fetch_lines(API_KEY, year=year, game_id=game_id)

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
            current_map_s, previous_map_s = get_ratings(year)
        except Exception as e:
            st.error(f"CFBD SP+ request failed: {e}")
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
            gp = project_game(g, current_map_s, previous_map_s, 2.5)
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
                v,e,ev,_ = grade(gp["away_win_prob"], market["away_ml"], 75)
                candidates.append((v, f"{gp['away']} ML", market["away_ml"], e, ev))
            if market.get("home_ml") is not None:
                v,e,ev,_ = grade(gp["home_win_prob"], market["home_ml"], 75)
                candidates.append((v, f"{gp['home']} ML", market["home_ml"], e, ev))

            if market.get("home_spread") is not None:
                hp = cover_probability(gp["home_margin"], market["home_spread"], "home")
                ap = 1 - hp
                v,e,ev,_ = grade(hp, -110, 75)
                candidates.append((v, f"{gp['home']} {market['home_spread']:+.1f}", -110, e, ev))
                v,e,ev,_ = grade(ap, -110, 75)
                candidates.append((v, f"{gp['away']} {-market['home_spread']:+.1f}", -110, e, ev))

            if market.get("total") is not None:
                op = total_probability(gp["model_total"], market["total"], "over")
                up = 1 - op
                v,e,ev,_ = grade(op, -110, 75)
                candidates.append((v, f"Over {market['total']:g}", -110, e, ev))
                v,e,ev,_ = grade(up, -110, 75)
                candidates.append((v, f"Under {market['total']:g}", -110, e, ev))

            if candidates:
                rank = {"STRONG BET":3, "BET":2, "LEAN":1, "PASS":0}
                candidates.sort(key=lambda x:(rank.get(x[0], -1), x[4]), reverse=True)
                b = candidates[0]
                best_verdict, best_market, best_odds, best_edge, best_ev = b

            slate_rows.append({
                "model_version": "0.1.6-IOS-DOWNLOAD",
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

        st.download_button(
            f"Download {slate_choice} Slate CSV",
            data=slate_df.to_csv(index=False).encode("utf-8"),
            file_name=f"cfb_v015_{selected_date}_{slate_choice.lower().replace(' ','_')}_slate.csv",
            mime="application/octet-stream",
            use_container_width=True,
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


def build_export_row(p, game, selected_date, market=None):
    market = market or {}
    row = {
        "model_version": "0.1.6-IOS-DOWNLOAD",
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
st.download_button(
    "Download Projection CSV",
    data=projection_only_df.to_csv(index=False).encode("utf-8"),
    file_name=f"cfb_projection_v014_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
    mime="application/octet-stream",
    use_container_width=True,
    help="Use this if you want to audit the model projection before entering or comparing sportsbook lines.",
)

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
    st.download_button(
        "Download Game CSV",
        data=csv_bytes,
        file_name=f"cfb_model_v014_{p['away'].replace(' ','_')}_at_{p['home'].replace(' ','_')}.csv",
        mime="application/octet-stream",
        use_container_width=True,
    )

st.divider()
st.caption("v0.1.0 uses SP+ as the anchor. Margin and total distributions are provisional and should be calibrated from tracked results.")
