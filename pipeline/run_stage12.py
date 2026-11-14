"""Run Stages 1-2 and print a data check.

    python -m pipeline.run_stage12              # all seasons in config
    python -m pipeline.run_stage12 2023 2024    # just these (plus warm-up)

Writes pipeline/output/model_table.parquet (one row per game, every feature,
lines and targets), a CSV sample, and data_check.txt.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from . import config as C
from .features import METRICS, model_features, stage2
from .ingest import stage1


def data_check(df: pd.DataFrame, feats: list[str]) -> str:
    lines = []
    fbs = df[~df.non_fbs_game & df.margin.notna()]
    lines.append("Games per season (FBS vs FBS, played)")
    tab = fbs.groupby("season").agg(
        games=("game_id", "size"),
        with_close_spread=("close_spread", lambda s: s.notna().mean()),
        with_close_total=("close_total", lambda s: s.notna().mean()),
        with_epa_feats=("diff_ewm8_epa", lambda s: s.notna().mean()),
        with_travel=("travel_diff", lambda s: s.notna().mean()))
    lines.append(tab.to_string(float_format=lambda x: f"{x:.0%}"))

    lines.append("\nHow well each pre-game feature lines up with the result and the market")
    lines.append("(correlation on FBS-vs-FBS games, 2018+; a good feature correlates with")
    lines.append(" both; near 0 with the market but high with margin would be suspicious)")
    d = fbs[fbs.season >= 2018]
    rows = []
    for m in METRICS:
        for col in (f"diff_ewm4_{m}", f"diff_ewm8_{m}", f"diff_rating_{m}"):
            ok = d[[col, "margin", "close_spread"]].dropna()
            rows.append({"feature": col,
                         "corr_margin": ok[col].corr(ok.margin),
                         "corr_market": ok[col].corr(-ok.close_spread)})
    for col in ("home_field", "rest_diff", "travel_diff"):
        ok = d[[col, "margin", "close_spread"]].dropna()
        if len(ok) > 100:
            rows.append({"feature": col, "corr_margin": ok[col].corr(ok.margin),
                         "corr_market": ok[col].corr(-ok.close_spread)})
    lines.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    ok = d[["close_spread", "margin"]].dropna()
    lines.append(f"\nMarket benchmark: corr(-close_spread, margin) = "
                 f"{(-ok.close_spread).corr(ok.margin):.3f}; "
                 f"closing-line MAE = {(ok.margin + ok.close_spread).abs().mean():.2f} pts")
    tr = d[d.season.isin(C.TRAIN_SEASONS)]
    te = d[d.season.isin(C.TEST_SEASONS)]
    lines.append(f"Train games {len(tr):,} ({C.TRAIN_SEASONS[0]}-{C.TRAIN_SEASONS[-1]}), "
                 f"test games {len(te):,} ({', '.join(map(str, C.TEST_SEASONS))})")
    lines.append(f"Model feature columns available to Stage 3: {len(feats)}")
    return "\n".join(lines)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    seasons = sorted({int(a) for a in argv}) if argv else C.ALL_SEASONS
    if argv:
        seasons = sorted(set(seasons) | {min(seasons) - 1})   # warm-up year
    games, pbp, teams, venues = stage1(seasons)
    df, tg, audit = stage2(games, pbp, teams, venues)
    feats = model_features(df)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(C.OUTPUT_DIR, "model_table.parquet"))
    cols = ["season", "week", "kickoff", "home_team", "away_team", "close_spread",
            "close_total", "margin", "points", "diff_rating_epa", "diff_ewm4_epa",
            "diff_ewm8_epa", "rest_diff", "travel_diff", "home_cover"]
    df[df.season == df.season.max()][cols].head(200).to_csv(
        os.path.join(C.OUTPUT_DIR, "model_table_sample.csv"), index=False)
    report = data_check(df, feats) + f"\n\nLeakage audit: {audit}"
    with open(os.path.join(C.OUTPUT_DIR, "data_check.txt"), "w") as f:
        f.write(report)
    print("\n" + report)
    return df


if __name__ == "__main__":
    main()
