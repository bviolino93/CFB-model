# CFB modeling pipeline (research)

A standalone, reproducible pipeline for pricing college football games and
backtesting +EV bets. **It does not touch the live app** (`app.py` never
imports it). It reuses the same free data feed as the app (`cfb_data.py`,
cfbfastR files on GitHub), so no API key is needed.

| Stage | File | Status |
|---|---|---|
| 1. Ingestion and cleaning | `ingest.py` | done |
| 2. Features (strict anti-leakage) | `features.py` | done |
| 3. Model and calibration | `model.py` | done |
| 4. De-vig, edge filter, quarter Kelly | `betting.py` | done |
| 5. Out-of-sample backtest | `backtest.py` | done |

Settings live in `config.py`: train 2018-2024, test 2025 (2017 is warm-up
only), margin cap 28, min edge 3.5%, quarter Kelly, $1,000 bankroll.

## Run

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.run_stage12            # all seasons (~10 min first time, cached after)
python -m pipeline.backtest               # Stages 3-5 (~20 s once the table exists)
python -m pytest tests/test_pipeline_leakage.py tests/test_pipeline_betting.py
```

Outputs go to `pipeline/output/` (git-ignored): `model_table.parquet`
(one row per game: every feature, lines and targets), a CSV sample, and
`data_check.txt`.

Optional: set `CFBD_API_KEY` to pull team and venue coordinates from
CollegeFootballData.com for travel distance. Without it, travel is blank and
the model simply won't use it.

## Stage 1: what gets cleaned

* Games with no FBS team are dropped. FBS vs FCS games are kept but flagged
  `non_fbs_game`, and every non-FBS team shares one pooled rating, so beating
  an FCS team by 60 can't inflate a rating.
* Garbage time is removed from the play-by-play before any stat is computed:
  plays where the score gap at the snap is over 38 (Q2), 28 (Q3) or 22 (Q4).
* Point margins are capped at +/-28 for point-based features.
* Lines are the median across books of the closing (and opening) spread,
  total, spread prices, total prices and moneylines.

## Stage 2: features

Per offense per game (non-garbage scrimmage plays): EPA/play, success rate,
IsoPPP (EPA on successful plays), yards/play, havoc rate (TFL, sack, INT,
forced fumble, pass breakup) and capped points.

* **Schedule adjustment.** Each week a ridge regression
  `value = mu + offense_i + allowed_j + hfa*home` is fit on games played
  before that week's first kickoff (plus last season at a fading weight).
  The as-of-week ratings are features, and every past game is re-expressed
  relative to its opponent before averaging.
* **EWMAs.** 4- and 8-game exponentially weighted means of those adjusted
  numbers, shifted one game so a game never sees its own result.
* **Matchups.** `diff_*` = (home offense + away defense) - (away offense +
  home defense), for spreads; `sum_*` is the same added, for totals.
* **Situational.** Home field (0 on neutral sites), rest-day difference
  (openers count as 28 days), travel-mile difference.

**Anti-leakage.** `audit_leakage()` fails the run if any EWMA used a game at
or after kickoff or any weekly fit used a game after its cutoff, and
`tests/test_pipeline_leakage.py` changes one game's result and checks that no
feature of that game or any earlier game moves.

## Stages 3-5

* **Model.** Ridge (alpha by leave-one-season-out CV) and XGBoost (small,
  fixed settings), averaged. Spreads predict home margin from `diff_*`;
  totals predict points from `sum_*`.
* **Calibration.** Platt scaling on the model-minus-line gap, fit on
  out-of-fold training predictions, so a gap that doesn't predict covers
  gives ~50% and no bet. A negative slope is clamped to zero (never fade
  your own model). The naive Normal(pred, sigma) version is reported for
  comparison.
* **Betting.** De-vig both sides proportionally, bet only when model prob
  beats fair prob by 3.5% and EV > 0 at the price paid; quarter Kelly,
  capped at 5% a bet, sized off the bankroll at the start of each week.
* **Backtest.** Train 2018-2024, bet 2025 at closing lines; also re-priced
  at 734's usual -105. Writes `backtest_report.txt`, `backtest_summary.csv`
  and `backtest_bets.csv` to `pipeline/output/`.

### First result (2025, out of sample)

The closing line beat the model (MAE 11.8 vs 12.5 on spreads, 12.5 vs 13.0
on totals). In 2018-2024 out-of-fold, the model's side covered 47-53% at
every gap size, so the calibrated filter placed almost no bets (1 total).
Betting every "edge" the naive way would have meant ~600 bets per market:
spreads went 309-279-8 (+0.3% ROI, 58% drawdown) and totals lost 74% of the
bankroll. Public box-score stats are already in the closing line; an edge
would need information the market prices late (injuries, QB news, weather)
or a better price than the close.

## Round 2: opening lines, priors, market residual, injuries

`python -m pipeline.backtest` now compares 14 strategies (7 per market):

| Idea | What was built |
|---|---|
| Opening lines + CLV | `Spec(line="open")`: the same score model is calibrated, bet and graded at the opening line (at -110, since the free data has no opening prices). CLV = points the line moved our way from the open to the close. |
| Preseason priors | `features.add_preseason_priors`: last season's final schedule-adjusted ratings, plus returning production from the free rosters and last season's play-by-play (share of dropbacks, rushing yards, receiving yards and havoc credits still on the roster, and whether the top QB returns). All of it is multiplied by a weight that fades 1.0, 0.75, 0.5, 0.25, 0 over weeks 1-5. The 247 talent composite loads through `ingest.load_talent()` when CFBD is reachable (it is blocked from the cloud sandbox). |
| Market residual | `Spec(kind="resid")`: the target is margin + spread (the market's error), with the open, the close and the move as features. The `@open` variant only gets the open. |
| Injuries | `features.add_injury_feed(df, rows)` reads the app's injury rows (team, points, note, until, added_at: the sheet and `injuries_auto.json`) and only counts news added before kickoff and still in force. There's no historical college injury report, so the backtest uses `add_qb_change` (a different starting QB than last game, and the starter's starts) as a stand-in. |

### Result (2025, out of sample)

* **Market residual:** no signal. Out of fold, predicted vs actual market error correlates +0.02 to +0.03, and line movement gives no fade spots. It made 1 bet all year (lost).
* **Opening lines:** the model's opening side gets small, steady CLV. On spreads the line moved our way in 54-61% of moved games every season from 2018 to 2025, averaging +0.2-0.4 points. But the side still covered only ~50% in training, which isn't enough to beat -110 or -105. Totals at the open placed 51 bets, went 29-22 (+8.6% ROI) and beat the close by 2.1 points on average. That's the best result, but it's fragile: the priors+QB version went 47-50, and the training-era edge was in Unders while 2025's bets were mostly Overs.
* **Preseason priors:** they don't help. Spreads in weeks 1-4 at the open went 55.9% vs 54.9% for the model's side across 195 games, which is noise, and the calibration found no edge.
* **QB change:** already priced in. It adds nothing measurable.
