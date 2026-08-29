
import streamlit as st
import pandas as pd
import re
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
from datetime import date
from model import MODEL_VERSION, fetch_games, load_rating_maps, project_game, cover_probability, total_probability, fair_ml, grade

st.set_page_config(page_title="CFB Model", page_icon="🏈", layout="centered")
st.title("🏈 CFB Model")
st.caption("Version 0.1.1-SCREENSHOT-LINES • Early-season prototype")

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


def _ocr_variants(img):
    """Lightweight multi-pass Tesseract; safe for Streamlit Community Cloud."""
    img = ImageOps.exif_transpose(img).convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.resize((img.width * 2, img.height * 2))
    variants = [img]
    for threshold in (135, 165, 195):
        variants.append(img.point(lambda p, t=threshold: 255 if p > t else 0))

    texts = []
    for v in variants:
        for psm in (6, 11):
            try:
                texts.append(pytesseract.image_to_string(v, config=f"--psm {psm}"))
            except Exception:
                pass
    return "\n".join(texts)

def _norm(s):
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("½", ".5")
    s = re.sub(r"(?<=\d)[%½](?=\s|$|\()", ".5", s)
    s = re.sub(r"(?<=\d)[¼](?=\s|$|\()", ".5", s)
    return s

def _team_tokens(team):
    words = re.findall(r"[A-Za-z0-9]+", team.lower())
    stop = {"university","college","state","the","of"}
    return [w for w in words if len(w) >= 3 and w not in stop]

def _line_has_team(line, team):
    low = line.lower()
    toks = _team_tokens(team)
    return any(t in low for t in toks)

def _american_odds(line):
    vals = []
    for x in re.findall(r"(?<![\d.])([+-]\d{3,4})(?![\d.])", line):
        try:
            n = int(x)
            if 100 <= abs(n) <= 5000:
                vals.append(n)
        except Exception:
            pass
    return vals

def _spread(line):
    # Spreads generally fall between 0.5 and 60.5 and include decimal/half notation.
    for x in re.findall(r"(?<!\d)([+-]\d{1,2}(?:\.5)?)(?!\d)", line):
        try:
            n = float(x)
            if 0.5 <= abs(n) <= 60.5 and abs(n) < 100:
                return n
        except Exception:
            pass
    return None

def _total(line):
    # College totals are usually 30-90. Require O/U context where possible.
    low = line.lower()
    patterns = [
        r"\b(?:o|over)\s*(\d{2}(?:\.5)?)",
        r"\b(?:u|under)\s*(\d{2}(?:\.5)?)",
        r"\btotal\s*(\d{2}(?:\.5)?)",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            v = float(m.group(1))
            if 20 <= v <= 100:
                return v
    return None

def parse_sportsbook_screenshots(files, away, home):
    detected = {}
    raw_parts = []

    for f in files:
        try:
            img = Image.open(f)
            text = _norm(_ocr_variants(img))
            raw_parts.append(text)
        except Exception:
            continue

        lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]

        for line in lines:
            odds = _american_odds(line)
            spr = _spread(line)

            if _line_has_team(line, away):
                if spr is not None and odds:
                    detected.setdefault("away_spread", spr)
                    detected.setdefault("away_spread_odds", odds[-1])
                elif odds:
                    detected.setdefault("away_ml", odds[0])

            if _line_has_team(line, home):
                if spr is not None and odds:
                    detected.setdefault("home_spread", spr)
                    detected.setdefault("home_spread_odds", odds[-1])
                elif odds:
                    detected.setdefault("home_ml", odds[0])

            low = line.lower()
            tot = _total(line)
            if tot is not None and odds:
                detected.setdefault("total", tot)
                if re.search(r"\b(o|over)\b", low):
                    detected.setdefault("over_odds", odds[-1])
                elif re.search(r"\b(u|under)\b", low):
                    detected.setdefault("under_odds", odds[-1])

        # Global total fallbacks for formats like O55.5 -110 / U55.5 -110.
        for m in re.finditer(r"\b[oO]\s*(\d{2}(?:\.5)?)\s*([+-]\d{3,4})", text):
            detected.setdefault("total", float(m.group(1)))
            detected.setdefault("over_odds", int(m.group(2)))
        for m in re.finditer(r"\b[uU]\s*(\d{2}(?:\.5)?)\s*([+-]\d{3,4})", text):
            detected.setdefault("total", float(m.group(1)))
            detected.setdefault("under_odds", int(m.group(2)))

    # If only one spread side was read, infer the opposite line (not the price).
    if "home_spread" in detected and "away_spread" not in detected:
        detected["away_spread"] = -detected["home_spread"]
    if "away_spread" in detected and "home_spread" not in detected:
        detected["home_spread"] = -detected["away_spread"]

    return detected, "\n\n".join(raw_parts)


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

uploaded = st.file_uploader(
    "Upload sportsbook screenshot(s)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Upload one or more screenshots. Detected lines remain fully editable."
)

detected = {}
raw_ocr = ""
if uploaded:
    with st.spinner("Reading sportsbook screenshot(s)..."):
        detected, raw_ocr = parse_sportsbook_screenshots(uploaded, p["away"], p["home"])

    if detected:
        st.success("Screenshot read. Review the detected lines below before betting.")
        summary = []
        if "away_ml" in detected: summary.append(f"{p['away']} ML {detected['away_ml']:+d}")
        if "home_ml" in detected: summary.append(f"{p['home']} ML {detected['home_ml']:+d}")
        if "away_spread" in detected:
            summary.append(f"{p['away']} {detected['away_spread']:+.1f} {detected.get('away_spread_odds', -110):+d}")
        if "home_spread" in detected:
            summary.append(f"{p['home']} {detected['home_spread']:+.1f} {detected.get('home_spread_odds', -110):+d}")
        if "total" in detected:
            summary.append(
                f"Total {detected['total']:g} • O {detected.get('over_odds', -110):+d} / "
                f"U {detected.get('under_odds', -110):+d}"
            )
        st.write(" • ".join(summary))
    else:
        st.warning("I couldn't confidently detect the lines. Enter them manually below.")

    with st.expander("OCR debug text"):
        st.text(raw_ocr[:12000] if raw_ocr else "No OCR text returned.")

st.caption("Screenshot values are only a convenience. Verify every detected line; all fields are editable.")

default_home_spread = float(detected.get("home_spread", round(p["model_home_spread"]*2)/2))
default_away_spread = float(detected.get("away_spread", -default_home_spread))

m1,m2=st.columns(2)
away_ml=m1.number_input(f"{p['away']} ML", value=int(detected.get("away_ml",100)), step=5)
home_ml=m2.number_input(f"{p['home']} ML", value=int(detected.get("home_ml",-110)), step=5)

s1,s2=st.columns(2)
home_spread=s1.number_input(f"{p['home']} spread", value=default_home_spread, step=.5)
home_spread_odds=s2.number_input("Home spread odds", value=int(detected.get("home_spread_odds",-110)), step=5)

s3,s4=st.columns(2)
away_spread=s3.number_input(f"{p['away']} spread", value=default_away_spread, step=.5)
away_spread_odds=s4.number_input("Away spread odds", value=int(detected.get("away_spread_odds",-110)), step=5)

t1,t2,t3=st.columns(3)
market_total=t1.number_input("Total", value=float(detected.get("total",round(p["model_total"]*2)/2)), step=.5)
over_odds=t2.number_input("Over odds", value=int(detected.get("over_odds",-110)), step=5)
under_odds=t3.number_input("Under odds", value=int(detected.get("under_odds",-110)), step=5)

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
