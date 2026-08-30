CFB Edge v0.5.0-RESIDUAL-LAB

This version adds an experimental market-first residual model to Backtest mode.
It trains only on development seasons and predicts the selected holdout once.
The residual model predicts market error rather than the full game result.
Spread and total residuals are fit separately with standardized ridge regression.
Regularization is selected inside the development sample only.
Moneyline bets are disabled for v0.5.0 pending a historical ML data-quality audit.
Do not promote v0.5.0 to the live betting board unless the untouched holdout improves on market-only MAE and produces credible betting results.
