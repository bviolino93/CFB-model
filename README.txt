CFB Edge v0.5.1-WALKFORWARD-SPREAD

Changes from v0.5.0:
- Adds rolling walk-forward validation across unseen seasons.
- Example: train 2022 -> test 2023; train 2022-23 -> test 2024; train 2022-24 -> test 2025.
- Official v0.5.1 bets are spread-only.
- Residual total signals are retained as research-only LEANs and cannot become official BETs.
- Moneylines remain disabled for the residual model pending data-quality audit.
- Adds per-season W-L-P, win rate, ROI, units, and market-vs-residual MAE tables.
- No live-board promotion is made automatically; promotion should depend on multi-season unseen results.
