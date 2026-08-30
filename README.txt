CFB Edge v0.7.0-MATCHUP-LAB

Purpose
-------
Rebuild the CFB edge layer around matchup-specific information instead of
re-weighting the old aggregate power projection.

Architecture
------------
Sportsbook consensus spread
    + predicted matchup-specific market residual
    = v0.7 fair spread

Matchup features
----------------
- Pass offense vs opponent pass defense PPA
- Rush offense vs opponent rush defense PPA
- Success-rate matchup
- Explosiveness matchup
- Advanced passing/rushing play PPA
- Finishing drives / points per opportunity
- Defensive havoc differential
- Pace / plays per drive
- SP+ rating differential
- Talent differential
- Returning-production differential
- Returning passing production and usage
- HFA, week, favorite size

Validation
----------
- Standardized ridge regression
- Alpha selected only inside development seasons
- Rolling unseen-season tests
- Example:
    train 2022 -> test 2023
    train 2022-23 -> test 2024
    train 2022-24 -> test 2025
- Direct comparison against market-only spread MAE
- Fixed research-bet hurdle; no historical threshold optimization
- Research labels only; no automatic live-board promotion

Leakage-safe mode
-----------------
When the Backtest selector is set to Leakage-safe preseason prior, current-season
SP+/SRS/PPA/advanced results are removed. The matchup statistics therefore come
from the prior season, with current preseason talent/returning-production inputs.

Exports
-------
- v0.7 Matchup Walk-Forward CSV
- v0.7 Matchup Bets CSV
- v0.7 Feature Importance CSV
- Existing backtest and v0.6 research exports remain available.

Promotion gate
--------------
Do not promote v0.7 to live betting unless it improves on market-only MAE across
multiple unseen seasons and the fixed-hurdle research-bet subset demonstrates
credible multi-season performance.
