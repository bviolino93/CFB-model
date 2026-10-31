"""
Does game-time weather predict college totals beyond the closing line?

Runs on GitHub Actions (the weather archive isn't reachable from Claude's
sandbox): python weather_research.py. For FBS home games 2017-25 with a
closing total, it finds each stadium (OpenStreetMap search), pulls the
hourly wind, rain and temperature for the three hours from kickoff
(Open-Meteo archive, free), and compares the result with the closing
total: under rate and points vs the total by wind and rain, per era.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests

import cfb_data

GEO = "https://nominatim.openstreetmap.org/search"
WX = "https://archive-api.open-meteo.com/v1/archive"
UA = {"User-Agent": "cfb-model-research (github.com/bviolino93/CFB-model)"}
CACHE = os.path.join(cfb_data.CACHE_DIR, "weather")
DOMES = ("dome", "superdome", "ford field", "mercedes-benz stadium", "state farm stadium",
         "allegiant", "nrg stadium", "at&t stadium", "lucas oil", "u.s. bank stadium",
         "alamodome", "jma wireless", "edward jones", "georgia dome", "tropicana",
         "kibbie", "holt arena", "caesars superdome", "metrodome", "ud arena")


def _cached(name, fetch):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    out = fetch()
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def locate(venue: str, school: str):
    def fetch():
        for q in (venue, f"{venue} {school}", f"{school} university"):
            time.sleep(1.1)      # OpenStreetMap asks for at most one search a second
            try:
                r = requests.get(GEO, params={"q": q, "format": "json", "limit": 1},
                                 headers=UA, timeout=30)
                hits = r.json() if r.ok else []
            except (requests.RequestException, ValueError):
                hits = []
            if hits:
                return [float(hits[0]["lat"]), float(hits[0]["lon"]), q]
        return None
    key = "".join(c for c in f"{venue}_{school}" if c.isalnum() or c == "_")[:120]
    return _cached(f"geo_{key}.json", fetch)


def hourly(lat, lon, start, end):
    def fetch():
        for attempt in range(4):
            r = requests.get(WX, params={
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "hourly": "wind_speed_10m,precipitation,temperature_2m",
                "wind_speed_unit": "mph", "precipitation_unit": "inch",
                "temperature_unit": "fahrenheit", "timezone": "GMT"}, timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["hourly"]
        raise RuntimeError("weather archive kept rate-limiting")
    return _cached(f"wx_{lat:.3f}_{lon:.3f}_{start}_{end}.json", fetch)


def build() -> pd.DataFrame:
    g = cfb_data.load_games(range(2017, 2026))
    g = g[(g.home_division == "fbs") & ~g.neutral_site.astype(bool)
          & g.close_total.notna() & g.points.notna() & g.venue.notna()].copy()
    g["kick"] = pd.to_datetime(g.start_date, utc=True)
    g["dome"] = g.venue.str.lower().map(lambda v: any(d in v for d in DOMES))
    rows = []
    for (venue, school), vg in g[~g.dome].groupby(["venue", "home_team"]):
        loc = locate(venue, school)
        if not loc:
            print(f"no location: {venue} ({school})")
            continue
        lat, lon, _ = loc
        for season, sg in vg.groupby("season"):
            start = sg.kick.min().strftime("%Y-%m-%d")
            end = (sg.kick.max() + pd.Timedelta(hours=4)).strftime("%Y-%m-%d")
            try:
                h = hourly(lat, lon, start, end)
            except Exception as e:
                print(f"no weather: {venue} {season}: {e}")
                continue
            w = pd.DataFrame(h)
            w["time"] = pd.to_datetime(w.time, utc=True)
            w = w.set_index("time")
            for r in sg.itertuples():
                t0 = r.kick.floor("h")
                win = w.loc[t0:t0 + pd.Timedelta(hours=3)]
                if win.empty:
                    continue
                rows.append({"game_id": r.game_id, "wind": win.wind_speed_10m.mean(),
                             "rain": win.precipitation.sum(),
                             "temp": win.temperature_2m.mean()})
    wx = pd.DataFrame(rows)
    return g.merge(wx, on="game_id", how="left")


def report(d: pd.DataFrame) -> str:
    d = d.copy()
    d["resid"] = d.points - d.close_total
    d["era"] = np.where(d.season <= 2021, "2017-21", "2022-25")
    out = [f"# Weather vs college closing totals\n",
           f"{len(d)} FBS home games with a closing total; {int(d.dome.sum())} in domes, "
           f"{int(d.wind.notna().sum())} outdoor games with weather.\n"]

    def table(title, key, bins, labels):
        out.append(f"\n## {title}\n")
        out.append("| | era | games | under % | points vs total |\n|---|---|---|---|---|")
        x = d[d.wind.notna()].copy()
        x["bin"] = pd.cut(x[key], bins, labels=labels, right=False)
        for (b, era), s in x.groupby(["bin", "era"], observed=True):
            s = s[s.resid != 0]
            out.append(f"| {b} | {era} | {len(s)} | {(s.resid < 0).mean():.1%} | {s.resid.mean():+.2f} |")
        for b, s in x.groupby("bin", observed=True):
            s = s[s.resid != 0]
            out.append(f"| {b} | all | {len(s)} | {(s.resid < 0).mean():.1%} | {s.resid.mean():+.2f} |")

    table("Wind (mph, kickoff to +3h)", "wind", [0, 10, 15, 20, 100],
          ["under 10", "10-15", "15-20", "20+"])
    table("Rain (inches over the game window)", "rain", [0, 0.01, 0.1, 0.3, 10],
          ["dry", "trace", "0.1-0.3", "0.3+"])
    table("Temperature (F)", "temp", [-50, 32, 45, 90, 150],
          ["under 32", "32-45", "45-90", "90+"])
    out.append("\n## Fit: points vs total on wind over 10 mph, per era\n")
    x = d[d.wind.notna()]
    for era, s in x.groupby("era"):
        over = np.clip(s.wind - 10, 0, None)
        X = np.column_stack([np.ones(len(s)), over, (s.rain >= 0.1).astype(float)])
        beta, *_ = np.linalg.lstsq(X, s.resid.values, rcond=None)
        e = s.resid.values - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(s) - 3))
        out.append(f"- {era}: {beta[1]:+.3f} pts per mph over 10 (t {beta[1] / se[1]:+.2f}); "
                   f"rain 0.1in+: {beta[2]:+.2f} pts (t {beta[2] / se[2]:+.2f}); n={len(s)}")
    out.append("\n## Dome games\n")
    s = d[d.dome & (d.resid != 0)]
    out.append(f"{len(s)} games, under {(s.resid < 0).mean():.1%}, points vs total {s.resid.mean():+.2f}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    d = build()
    d.to_csv("weather_games.csv", index=False)
    md = report(d)
    print(md)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(md)
