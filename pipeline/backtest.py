"""Stage 5: out-of-sample backtest.

Fits Stage 3 on the training seasons only, predicts the test season(s),
flags bets with Stage 4 and settles them against the final scores.

Sizing is realistic: each week's stakes are quarter-Kelly shares of the
bankroll at the start of that week (you can't reinvest Saturday's winnings
in Saturday's other games). Units are reported as flat 1-unit bets.

    python -m pipeline.backtest                 # uses pipeline/output/model_table.parquet
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import config as C
from .betting import american_to_decimal, pick_bets
from .model import MARKETS, eligible, feature_importance, fit


def settle(games: pd.DataFrame, bets: pd.DataFrame, market: str) -> pd.DataFrame:
    g = games.loc[bets.index]
    if market == "spread":
        res = g.margin + g.close_spread          # > 0 home covered
        won = np.where(bets.side == "home", res > 0, res < 0)
    else:
        res = g.points - g.close_total            # > 0 over hit
        won = np.where(bets.side == "over", res > 0, res < 0)
    out = bets.copy()
    out["market"] = market
    for c in ("season", "week", "kickoff", "home_team", "away_team",
              MARKETS[market]["line"], "margin", "points"):
        out[c] = g[c]
    out["result"] = np.where(res == 0, "push", np.where(won, "win", "loss"))
    dec = american_to_decimal(out.odds)
    out["units"] = np.select([out.result == "win", out.result == "loss"], [dec - 1, -1.0], 0.0)
    return out


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
    return pd.concat(rows) if rows else bets.assign(stake=[], profit=[], bankroll=[])


def max_drawdown(path: pd.Series, start=C.START_BANKROLL) -> float:
    p = pd.concat([pd.Series([start]), path.reset_index(drop=True)])
    peak = p.cummax()
    return float(((p - peak) / peak).min())


def summarise(b: pd.DataFrame, label: str, start=C.START_BANKROLL) -> dict:
    if b.empty:
        return {"strategy": label, "bets": 0}
    w, l, p = (int((b.result == x).sum()) for x in ("win", "loss", "push"))
    staked = float(b.stake.sum())
    end = float(b.bankroll.iloc[-1])
    return {
        "strategy": label, "bets": len(b), "W-L-P": f"{w}-{l}-{p}",
        "win%": w / max(w + l, 1), "units": float(b.units.sum()),
        "unit_roi": float(b.units.sum()) / len(b),
        "profit_$": end - start, "staked_$": staked,
        "roi_on_staked": (end - start) / staked if staked else 0.0,
        "end_bankroll": end, "max_drawdown": max_drawdown(b.bankroll, start),
        "avg_edge": float(b.edge.mean()),
    }


def _fmt(s: pd.DataFrame) -> str:
    f = s.copy()
    for c in ("win%", "unit_roi", "roi_on_staked", "max_drawdown", "avg_edge"):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    for c in ("units",):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
    for c in ("profit_$", "staked_$", "end_bankroll"):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    return f.to_string(index=False)


def _bar_chart(imp: pd.DataFrame, width=30) -> str:
    top = imp.combined.max()
    return "\n".join(f"  {name:<28} {'#' * max(1, int(round(v / top * width)))} {v:.1%}"
                     for name, v in imp.combined.items())


def backtest(df: pd.DataFrame, train=C.TRAIN_SEASONS, test=C.TEST_SEASONS,
             markets=("spread", "total"), verbose=True):
    print(f"Stage 3: training on {train[0]}-{train[-1]}")
    models = {mk: fit(df, train, mk, verbose) for mk in markets}

    print(f"Stage 4-5: betting {', '.join(map(str, test))} out of sample")
    summaries, all_bets, report = [], [], []
    for mk, fm in models.items():
        te = eligible(df, mk)
        te = te[te.season.isin(test)]
        probs = fm.predict(te)
        line = te[MARKETS[mk]["line"]]
        res = te[MARKETS[mk]["target"]] + (line if mk == "spread" else -line)
        mae_model = float((probs.pred - te[MARKETS[mk]["target"]]).abs().mean())
        report.append(f"{mk}: {len(te)} test games; model MAE {mae_model:.2f} vs "
                      f"closing line {float(res.abs().mean()):.2f}")
        # Did the model's side win more than half the time, all games?
        side_right = np.sign(probs.gap) == np.sign(res)
        nz = (res != 0) & (probs.gap != 0)
        report.append(f"  model side vs closing line, every game: "
                      f"{side_right[nz].mean():.1%} ({int(nz.sum())} games)")
        for prob_col, price, label in (
                ("p_platt", None, "calibrated, market price"),
                ("p_platt", C.YOUR_BOOK_ODDS, "calibrated, 734 at -105"),
                ("p_normal", None, "normal-dist, market price")):
            bets = pick_bets(te, probs, mk, prob_col=prob_col, price=price)
            if bets.empty:
                summaries.append({"strategy": f"{mk}: {label}", "bets": 0})
                continue
            b = run_bankroll(settle(te, bets, mk).join(probs[["pred", "gap"]]))
            b["strategy"] = f"{mk}: {label}"
            summaries.append(summarise(b, f"{mk}: {label}"))
            all_bets.append(b)

    # Main strategy: both markets together, calibrated, market price.
    main = [b for b in all_bets if b.strategy.iloc[0].endswith("calibrated, market price")]
    if main:
        comb = pd.concat(main).drop(columns=["bankroll", "stake", "profit", "wk"])
        comb = run_bankroll(comb)
        summaries.insert(0, summarise(comb, "MAIN: spreads + totals"))

    out = ["", "=" * 78, f"OUT-OF-SAMPLE BACKTEST {', '.join(map(str, test))}  "
           f"(train {train[0]}-{train[-1]}, edge >= {C.MIN_EDGE:.1%}, "
           f"{C.KELLY_FRACTION:g} Kelly, ${C.START_BANKROLL:,.0f} start)", "=" * 78]
    out += report
    out += ["", _fmt(pd.DataFrame(summaries))]
    out.append("\nunits = flat 1-unit bets; profit_$/drawdown = quarter-Kelly bankroll; "
               "break-even win% is 52.4% at -110, 51.2% at -105")
    for mk, fm in models.items():
        out.append(f"\nFeature importance, {mk} model (share of XGBoost gain and "
                   f"Ridge |coef|, averaged)")
        out.append(_bar_chart(feature_importance(fm)))
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
                "side", "close_spread", "close_total", "pred", "gap", "p_model",
                "p_fair", "edge", "odds", "kelly", "stake", "result", "units",
                "profit", "bankroll"]
        bets[[c for c in cols if c in bets]].to_csv(
            os.path.join(C.OUTPUT_DIR, "backtest_bets.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
