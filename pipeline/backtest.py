"""Stage 5: out-of-sample backtest, now a comparison of strategies.

Every strategy is fit on the training seasons only, predicts the test
season(s), flags bets with Stage 4 and settles them at the line it bet.

  * @close strategies bet and grade at the closing line (game-day betting)
  * @open strategies bet and grade at the opening line, and track closing
    line value (CLV): did the line then move toward our side, and by how
    many points?
  * "raw" models predict the score and compare with the line; "resid"
    models predict the market's error directly (margin + spread), using
    the line and its movement as features.

Sizing: each week's stakes are quarter-Kelly shares of the bankroll at the
start of that week. Units are flat 1-unit bets at the market price; units
at 734's usual -105 are shown too.

Opening prices aren't in the free data, so opening bets assume -110.

    python -m pipeline.backtest                 # uses pipeline/output/model_table.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import config as C
from .betting import american_to_decimal, pick_bets
from .model import MARKETS, Spec, eligible, feature_importance, fit, market_error

ODDS_COLS = {"spread": ("home_spread_odds", "away_spread_odds"),
             "total": ("over_odds", "under_odds")}


def strategies(market: str) -> list[Spec]:
    S = lambda **k: Spec(market=market, **k)  # noqa: E731
    return [
        S(kind="raw", line="close"),                                   # Stage 1-5 baseline
        S(kind="raw", line="close", groups=("priors", "qb")),
        S(kind="raw", line="open"),                                    # idea 1
        S(kind="raw", line="open", groups=("priors", "qb")),
        S(kind="resid", line="close", groups=("mkt_close",)),          # idea 3
        S(kind="resid", line="close", groups=("mkt_close", "priors", "qb")),
        S(kind="resid", line="open", groups=("mkt_open", "priors", "qb")),
    ]


# ---------------------------------------------------------------------------
def settle(games: pd.DataFrame, bets: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    g = games.loc[bets.index]
    res = market_error(g, spec)                     # > 0 home covered / over hit
    first = "home" if spec.market == "spread" else "over"
    won = np.where(bets.side == first, res > 0, res < 0)
    out = bets.copy()
    out["strategy"] = spec.label
    for c in ("season", "week", "kickoff", "home_team", "away_team", "close_spread",
              "open_spread", "close_total", "open_total", "margin", "points"):
        out[c] = g[c]
    out["result"] = np.where(res == 0, "push", np.where(won, "win", "loss"))
    dec = american_to_decimal(out.odds)
    out["units"] = np.select([out.result == "win", out.result == "loss"], [dec - 1, -1.0], 0.0)
    out["units_734"] = np.select([out.result == "win", out.result == "loss"],
                                 [100 / 105, -1.0], 0.0)
    out["clv"] = clv_points(g, out.side, spec)
    return out


def clv_points(g: pd.DataFrame, side: pd.Series, spec: Spec) -> np.ndarray:
    """Points the line moved in our favour from the open to the close
    (0 for bets made at the close). Spreads are from the home side, so a
    home bet at -3 that closes -5 gained 2 points."""
    if spec.line != "open":
        return np.zeros(len(g))
    if spec.market == "spread":
        move = (g.open_spread - g.close_spread).to_numpy(float)   # + = toward home
        return np.where(side == "home", move, -move)
    move = (g.close_total - g.open_total).to_numpy(float)         # + = toward over
    return np.where(side == "over", move, -move)


def run_bankroll(bets: pd.DataFrame, start=C.START_BANKROLL) -> pd.DataFrame:
    """Week-start Kelly sizing; returns bets with stake, profit, bankroll."""
    bets = bets.sort_values("kickoff").copy()
    bets["wk"] = (bets.kickoff.dt.tz_convert("America/New_York").dt.tz_localize(None)
                  .dt.to_period("W-MON"))
    bank = start
    rows = []
    for _, wk in bets.groupby("wk", sort=True):
        stakes = wk.kelly * bank
        dec = american_to_decimal(wk.odds)
        profit = np.select([wk.result == "win", wk.result == "loss"],
                           [stakes * (dec - 1), -stakes], 0.0)
        wk = wk.assign(stake=stakes, profit=profit)
        wk["bankroll"] = bank + wk.profit.cumsum()
        bank = float(wk.bankroll.iloc[-1])
        rows.append(wk)
    return pd.concat(rows)


def max_drawdown(path: pd.Series, start=C.START_BANKROLL) -> float:
    p = pd.concat([pd.Series([start]), path.reset_index(drop=True)])
    peak = p.cummax()
    return float(((p - peak) / peak).min())


def summarise(b: pd.DataFrame, label: str, start=C.START_BANKROLL) -> dict:
    if b is None or b.empty:
        return {"strategy": label, "bets": 0}
    w, l, p = (int((b.result == x).sum()) for x in ("win", "loss", "push"))
    staked = float(b.stake.sum())
    end = float(b.bankroll.iloc[-1])
    out = {
        "strategy": label, "bets": len(b), "W-L-P": f"{w}-{l}-{p}",
        "win%": w / max(w + l, 1), "units": float(b.units.sum()),
        "roi": float(b.units.sum()) / len(b),
        "roi_734": float(b.units_734.sum()) / len(b),
        "profit_$": end - start, "max_dd": max_drawdown(b.bankroll, start),
    }
    if (b.clv != 0).any():
        out["clv_pts"] = float(b.clv.mean())
        out["clv_beat%"] = float((b.clv > 0).mean())
    return out


def side_diagnostics(te: pd.DataFrame, probs: pd.DataFrame, spec: Spec) -> dict:
    """Every test game, no filter: does the model's side cover, and (for
    open strategies) does the line move toward the model's side?"""
    res = market_error(te, spec)
    g = probs.gap.to_numpy()
    nz = (res != 0) & (g != 0)
    d = {"games": int(len(te)),
         "side_cover%": float((np.sign(g[nz]) == np.sign(res[nz])).mean())}
    if spec.line == "open":
        side = np.where(g > 0, "home" if spec.market == "spread" else "over", "x")
        c = clv_points(te, pd.Series(side, index=te.index), spec)
        moved = c != 0
        d["line_moved_our_way%"] = float((c[moved] > 0).mean()) if moved.any() else np.nan
        d["avg_clv_pts"] = float(c.mean())
    return d


# ---------------------------------------------------------------------------
def _pct(x):
    return f"{x:+.1%}" if pd.notna(x) else ""


def _fmt(s: pd.DataFrame) -> str:
    f = s.copy()
    for c in ("win%", "roi", "roi_734", "max_dd", "clv_beat%", "side_cover%",
              "line_moved_our_way%"):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:.1%}" if c in ("win%", "clv_beat%",
                            "side_cover%", "line_moved_our_way%") and pd.notna(x)
                            else _pct(x))
    for c in ("units", "clv_pts", "avg_clv_pts"):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    if "profit_$" in f:
        f["profit_$"] = f["profit_$"].map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "")
    return f.fillna("").to_string(index=False)


def _bar_chart(imp: pd.DataFrame, width=30) -> str:
    top = imp.combined.max()
    return "\n".join(f"  {name:<28} {'#' * max(1, int(round(v / top * width)))} {v:.1%}"
                     for name, v in imp.combined.items())


def backtest(df: pd.DataFrame, train=C.TRAIN_SEASONS, test=C.TEST_SEASONS,
             markets=("spread", "total"), verbose=True):
    summaries, diags, early, all_bets, logs, models = [], [], [], [], [], {}
    for mk in markets:
        for spec in strategies(mk):
            fm = fit(df, train, spec, verbose)
            models[spec.label] = fm
            logs.append(fm.log)
            te = eligible(df, spec)
            te = te[te.season.isin(test)]
            probs = fm.predict(te)
            diags.append({"strategy": spec.label, **side_diagnostics(te, probs, spec)})
            priced = te
            if spec.line == "open":       # no opening prices in the data: -110
                priced = te.assign(**{c: np.nan for c in ODDS_COLS[mk]})
            bets = pick_bets(priced, probs, mk)
            b = None
            if not bets.empty:
                b = run_bankroll(settle(te, bets, spec).join(probs[["pred", "gap"]]))
                all_bets.append(b)
            summaries.append(summarise(b, spec.label))
            wk = te.week.le(4) & te.season_type.eq("regular")
            e = {"strategy": spec.label,
                 **{f"wk1-4 {k}": v for k, v in
                    side_diagnostics(te[wk], probs[wk], spec).items()
                    if k in ("games", "side_cover%")}}
            if b is not None:
                be = b[b.week.le(4)]
                e["wk1-4 bets"] = len(be)
                e["wk1-4 W-L"] = f"{(be.result == 'win').sum()}-{(be.result == 'loss').sum()}"
            early.append(e)

    out = ["", "=" * 96,
           f"OUT-OF-SAMPLE COMPARISON {', '.join(map(str, test))}  (train "
           f"{train[0]}-{train[-1]}, edge >= {C.MIN_EDGE:.1%}, {C.KELLY_FRACTION:g} Kelly, "
           f"${C.START_BANKROLL:,.0f} start each)", "=" * 96,
           "", "1) BETS THE STRATEGY WOULD HAVE MADE",
           _fmt(pd.DataFrame(summaries)),
           "   roi = units per bet at the market price (-110 at the open); roi_734 at -105;",
           "   clv_pts = avg points the line moved our way after an opening bet.",
           "", "2) EVERY TEST GAME, NO FILTER (is there any signal at all?)",
           _fmt(pd.DataFrame(diags)),
           "   side_cover% needs 52.4% at -110 (51.2% at -105). With ~800 games, +/-1.8% is noise.",
           "", "3) WEEKS 1-4 ONLY (where the preseason priors carry weight)",
           _fmt(pd.DataFrame(early)),
           "", "TRAINING LOG (2018-2024, leave-one-season-out)"]
    out += logs
    for lab in (f"spread resid @close [mkt_close+priors+qb]",
                f"spread resid @open [mkt_open+priors+qb]"):
        if lab in models:
            out.append(f"\nFeature importance, {lab}")
            out.append(_bar_chart(feature_importance(models[lab])))
    text = "\n".join(out)
    if verbose:
        print(text)
    bets = pd.concat(all_bets) if all_bets else pd.DataFrame()
    return models, pd.DataFrame(summaries), bets, text


def main(argv=None):
    path = os.path.join(C.OUTPUT_DIR, "model_table.parquet")
    if not os.path.exists(path):
        from .run_stage12 import main as stage12
        stage12([])
    df = pd.read_parquet(path)
    _, summ, bets, text = backtest(df)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUTPUT_DIR, "backtest_report.txt"), "w") as f:
        f.write(text)
    summ.to_csv(os.path.join(C.OUTPUT_DIR, "backtest_summary.csv"), index=False)
    if not bets.empty:
        cols = ["strategy", "season", "week", "kickoff", "home_team", "away_team",
                "side", "open_spread", "close_spread", "open_total", "close_total",
                "pred", "gap", "p_model", "p_fair", "edge", "odds", "kelly", "stake",
                "result", "units", "clv", "profit", "bankroll"]
        bets[[c for c in cols if c in bets]].to_csv(
            os.path.join(C.OUTPUT_DIR, "backtest_bets.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
