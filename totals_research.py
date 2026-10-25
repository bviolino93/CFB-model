"""
What moves college totals beyond the betting line? A factor screen.

Each factor is tested against result-minus-line (points scored minus the
total) in three eras, 2009-16 / 2017-21 / 2022-25, against both the
opening and the closing total. A factor only counts if it points the same
way, by a usable amount, in every era: one good era is noise.

    python totals_research.py [features.parquet]

The features file is cfb_model.load_features() saved to disk (built fresh
when no path is given; the play-by-play download takes a few minutes).

Findings (2026-10-25), which became cfb_picks.TOTAL_RULES:

  * The over hits only ~49% overall (48.5% vs openers), and it falls as the
    total rises: 51% on totals under 50, 45% on totals over 65.
  * Totals of 60+ went under 51.9% / 54.9% / 56.2% vs the opener by era
    (50.9 / 54.1 / 53.6 vs the close); night conference games a bit more.
  * Big favorites with low totals go over, and the bigger the spread and
    the lower the total the stronger it gets: spread 21+ and total under
    55 hit 55.9 / 57.3 / 56.2% vs the opener (59.0 / 54.3 / 53.9 vs the
    close); spread 24+ and total under 50, about 60%.
  * Week, month, neutral site, pace, kickoff time alone, and the points
    rating's view of the total showed nothing that held in every era. The
    efficiency model's totals (cfb_model) stayed near break-even.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ERAS = {"2009-16": (2009, 2016), "2017-21": (2017, 2021), "2022-25": (2022, 2025)}
P5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}


def factors(g: pd.DataFrame, line: str) -> pd.DataFrame:
    t = g[f"{line}_total"]
    kick = pd.to_datetime(g.start_date, utc=True).dt.tz_convert("America/New_York")
    sp = g[f"{line}_spread"].abs()
    p5 = g.home_conference.isin(P5) & g.away_conference.isin(P5)
    f = pd.DataFrame(index=g.index)
    f["total 60+"] = t >= 60
    f["total 65+"] = t >= 65
    f["total under 45"] = t < 45
    f["spread 21+"] = sp >= 21
    f["spread 3 or less"] = sp <= 3
    f["weeks 1-4"] = g.week <= 4
    f["November+"] = kick.dt.month >= 11
    f["bowls/postseason"] = g.season_type == "postseason"
    f["neutral site"] = g.neutral_site.astype(bool)
    f["conference game"] = g.conference_game.astype(bool)
    f["both power 5"] = p5
    f["night (7pm ET+)"] = kick.dt.hour >= 19
    f["noon ET or earlier"] = kick.dt.hour <= 12
    f["fast pace (top 20%)"] = g.sum_plays >= g.groupby("season").sum_plays.transform(
        lambda x: x.quantile(0.8))
    f["slow pace (bottom 20%)"] = g.sum_plays <= g.groupby("season").sum_plays.transform(
        lambda x: x.quantile(0.2))
    f["points rating 3+ over"] = (g.pred_total - t) >= 3
    f["points rating 3+ under"] = (g.pred_total - t) <= -3
    if line == "close":
        mv = g.close_total - g.open_total
        f["total rose 2+ (open to close)"] = mv >= 2
        f["total fell 2+ (open to close)"] = mv <= -2
    return f


def screen(g: pd.DataFrame, line: str) -> pd.DataFrame:
    d = g.dropna(subset=[f"{line}_total", "points"])
    d = d[d.points != d[f"{line}_total"]]
    r = d.points - d[f"{line}_total"]
    F = factors(d, line)
    rows = []
    for name in F.columns:
        row = {"factor": name}
        sides = []
        for era, (a, b) in ERAS.items():
            m = F[name].fillna(False).astype(bool) & d.season.between(a, b)
            if m.sum() < 30:
                row[era] = "-"
                continue
            over = (r[m] > 0).mean()
            row[era] = f"{over:.3f}/{m.sum()}"
            sides.append(over)
        row["all"] = round(float((r[F[name].fillna(False).astype(bool)] > 0).mean()), 3)
        # same direction and at least 2 points from 50% in every era
        row["holds"] = (len(sides) == 3 and
                        (all(s >= 0.52 for s in sides) or all(s <= 0.48 for s in sides)))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        g = pd.read_parquet(sys.argv[1])
    else:
        import cfb_model
        g = cfb_model.load_features()
    pd.set_option("display.width", 200)
    for line in ("open", "close"):
        print(f"\nOver hit rate (n) by era, vs the {line.upper()} total")
        print(screen(g, line).to_string(index=False))
