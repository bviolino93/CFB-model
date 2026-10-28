"""
College underdog moneylines: is there anything to bet?

cfb_line_odds.csv.gz has every book's closing moneyline, so besides the
model this can test line shopping: the best dog price on the board against
the sharp-weighted de-vigged consensus, which is how the live picks price a
moneyline. Win chances from the model come from the same college margin
distribution the picks use (cfb_picks.outcome_models), and the model's
spread prediction is walk-forward (fitted on earlier seasons only) and
capped like the live one (cfb_picks.SPREAD_CAP).

  1. all dogs        every underdog at the median price, by price band
  2. best price      the best dog price across books when it beats the
                     de-vigged consensus by the EV bar (pure line shopping)
  3. DraftKings      the same, DraftKings' price only (2021 on)
  4. model           consensus chance moved by the model's spread lean;
                     bet the median price when that clears the bar

    python moneyline_research.py [features.parquet]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import cfb_data
import cfb_model
import selector as sel
from cfb_picks import SPREAD_CAP

SHARP = {"PINNACLE": 4.0, "BetCRIS & BOOKMAKER": 2.0, "MATCHBOOK": 2.0,
         "5Dimes & sportbet": 2.0}


def moneylines(g: pd.DataFrame) -> pd.DataFrame:
    """One row per game and book: home and away moneyline. Labels are
    mapped to team ids the way cfb_data maps spread rows."""
    lines = pd.read_csv(cfb_data._fetch(cfb_data.LINES_URL, "lines.csv.gz", 24 * 7))
    ml = lines[(lines.market_type == "money_line") & lines.odds.notna()
               & (lines.odds.abs() >= 100)]
    ids = g.set_index("game_id")[["home_id", "away_id"]]
    ml = ml.join(ids, on="game_id").dropna(subset=["home_id"])
    co = pd.concat([ml[["abbr", "game_id", "home_id"]].rename(columns={"home_id": "tid"}),
                    ml[["abbr", "game_id", "away_id"]].rename(columns={"away_id": "tid"})])
    co = co.drop_duplicates()
    cnt = co.groupby(["abbr", "tid"]).size().rename("n").reset_index()
    tot = co.groupby("abbr").game_id.nunique()
    best = cnt.sort_values("n").groupby("abbr").tail(1).set_index("abbr")
    best = best[best.n >= 0.8 * tot.reindex(best.index)]
    ml = ml.assign(tid=ml.abbr.map(best.tid))
    h = ml[ml.tid == ml.home_id].drop_duplicates(["game_id", "book"])
    a = ml[ml.tid == ml.away_id].drop_duplicates(["game_id", "book"])
    out = h[["game_id", "book", "odds"]].merge(
        a[["game_id", "book", "odds"]], on=["game_id", "book"], suffixes=("_h", "_a"))
    # Both sides the same sign (both +) is a data error.
    out = out[~((out.odds_h > 0) & (out.odds_a > 0))]
    ph = out.odds_h.map(sel.american_to_prob)
    pa = out.odds_a.map(sel.american_to_prob)
    out["p_home"] = ph / (ph + pa)
    out["hold"] = ph + pa - 1
    return out[(out.hold > 0) & (out.hold < 0.15)]


def per_game(ml: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, x in ml.groupby("game_id"):
        w = x.book.map(SHARP).fillna(1.0)
        p = sel._wmedian(x.p_home, w)
        dk = x[x.book == "DraftKings"]
        rows.append({"game_id": gid, "p_home": p, "books": len(x),
                     "med_h": x.odds_h.median(), "med_a": x.odds_a.median(),
                     "best_h": x.odds_h.max(), "best_a": x.odds_a.max(),
                     "dk_h": dk.odds_h.iloc[0] if len(dk) else np.nan,
                     "dk_a": dk.odds_a.iloc[0] if len(dk) else np.nan})
    return pd.DataFrame(rows)


def walk_forward_spread(g: pd.DataFrame, first_test=2017) -> pd.Series:
    """Model result-minus-close for each game, fitted on earlier seasons
    against openers (as cfb_picks fits it), capped like the live nudge."""
    d, mkt, y = cfb_model.training_rows(g, "spread", "close")
    dt, mt, yt = cfb_model.training_rows(g, "spread", "open")
    pred = pd.Series(np.nan, index=d.game_id.values)
    for S in range(first_test, int(d.season.max()) + 1):
        te = (d.season == S).values
        tr = (dt.season < S).values & (dt.season >= cfb_model.FIRST_TRAIN).values
        if te.sum() == 0 or tr.sum() < 500:
            continue
        m = cfb_model.LineModel("spread").fit(dt[tr], mt[tr], yt[tr])
        pred.iloc[np.where(te)[0]] = m.predict(d[te], mkt[te])
    return pred.clip(-SPREAD_CAP, SPREAD_CAP)


def record(x: pd.DataFrame, price_col: str, label: str) -> dict:
    pr = np.where(x.dog_result > 0, x[price_col] / 100.0,
                  np.where(x.dog_result < 0, -1.0, 0.0))
    n = len(x)
    return {"rule": label, "bets": n, "win%": (x.dog_result > 0).mean() if n else np.nan,
            "avg price": x[price_col].mean() if n else np.nan,
            "roi": pr.mean() if n else np.nan,
            "roi se": pr.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan}


def eras(d, mask, price_col, label, spans=((2012, 2016), (2017, 2021), (2022, 2025))):
    out = [record(d[mask & d.season.between(a, b)], price_col, f"{label} {a}-{str(b)[2:]}")
           for a, b in spans]
    return out + [record(d[mask], price_col, f"{label} all")]


def main(path: str | None):
    pd.set_option("display.width", 200)
    g = pd.read_parquet(path) if path else cfb_model.load_features(range(2006, 2026))
    g = g[g.margin.notna() & (g.season >= 2012)]
    cache = "/tmp/claude-0/sp/cfb_ml.parquet"
    try:
        pg = pd.read_parquet(cache)
    except Exception:
        pg = per_game(moneylines(g))
        pg.to_parquet(cache)
    d = g.merge(pg, on="game_id")
    d = d[d.books >= 3]
    home_dog = d.p_home < 0.5
    d["dog_home"] = home_dog
    for k in ("med", "best", "dk"):
        d[f"{k}_dog"] = np.where(home_dog, d[f"{k}_h"], d[f"{k}_a"])
    d = d[d.med_dog >= 100].copy()
    d["p_dog"] = np.where(d.dog_home, d.p_home, 1 - d.p_home)
    d["dog_result"] = np.sign(d.margin) * np.where(d.dog_home, 1, -1)
    print(f"{len(d)} games with a dog at +100 or longer, 3+ books\n")

    rows = []
    for lo, hi in ((100, 150), (150, 250), (250, 400), (400, 900), (900, 99999)):
        rows += eras(d, d.med_dog.between(lo, hi - 1), "med_dog", f"all dogs +{lo}..")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    def ev(p, price):
        return p * price / 100.0 - (1 - p)

    rows = []
    for cut in (0.0, 0.02, 0.03):
        rows += eras(d, ev(d.p_dog, d.best_dog) >= cut, "best_dog", f"best price ev>={cut:.2f}")
        rows += eras(d, (ev(d.p_dog, d.best_dog) >= cut) & (d.best_dog <= 300), "best_dog",
                     f"  ...and +300 or shorter")
    dk = d.dk_dog.notna()
    for cut in (0.0, 0.02):
        rows += eras(d, dk & (ev(d.p_dog, d.dk_dog) >= cut), "dk_dog", f"DraftKings ev>={cut:.2f}",
                     spans=((2021, 2025),))
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    pred = walk_forward_spread(g)
    d["lean"] = d.game_id.map(pred) * np.where(d.dog_home, 1, -1)   # + = toward dog
    d = d[d.lean.notna()].copy()
    margin, _ = cfb_outcome(g)
    mkt = -d.close_spread.to_numpy(float)
    shift = d.lean.to_numpy(float) * np.where(d.dog_home, 1, -1)
    dp = []
    for m0, s, hd in zip(mkt, shift, d.dog_home):
        p0, t0 = margin.probs(m0, 0.0)
        p1, t1 = margin.probs(m0 + s, 0.0)
        dp.append((p1 - p0) if hd else ((1 - p1 - t1) - (1 - p0 - t0)))
    d["p_model"] = d.p_dog + np.array(dp)
    rows = []
    spans = ((2017, 2021), (2022, 2025))
    for cut in (0.0, 0.03, 0.05):
        for price, name in (("med_dog", "median price"), ("best_dog", "best price")):
            m = ev(d.p_model, d[price]) >= cut
            rows += eras(d, m, price, f"model {name} ev>={cut:.2f}", spans)
    for lean in (1, 2, 3):
        rows += eras(d, d.lean >= lean, "med_dog", f"model leans dog {lean}+ pts", spans)
        rows += eras(d, (d.lean >= lean) & (d.med_dog <= 300), "med_dog", "  ...and +300 or shorter", spans)
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    d["won"] = (d.dog_result > 0).astype(float)
    d["band"] = pd.cut(d.med_dog, [99, 150, 250, 400, 900, 99999])
    print(d.groupby("band", observed=True).agg(
        games=("won", "size"), won=("won", "mean"), consensus=("p_dog", "mean"),
        model=("p_model", "mean")).round(3).to_string())


def cfb_outcome(g):
    from cfb_picks import outcome_models
    return outcome_models(g)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
