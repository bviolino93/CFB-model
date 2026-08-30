CFB Edge v0.6.0-SIGNAL-RESEARCH

Purpose
-------
Stop tuning the same weak aggregate projection and instead test which individual
football inputs actually explain sportsbook spread errors.

What changed
------------
- Fixes v0.5.1 walk-forward display result labels (WIN/LOSS/PUSH).
- Preserves v0.5.1 residual walk-forward validation.
- Adds Signal Research using simple interpretable spread signals.
- Adds SP+, SRS, talent, and returning-production differential fields.
- Candidate signals are ranked using development data only.
- Signal direction and extreme-quartile cutoff are frozen before holdout testing.
- Adds rolling unseen-season signal survival tables.
- Adds multi-season survival summary.
- v0.6.0 Signal Research creates NO new official bets.
- Totals remain research-only; residual moneylines remain disabled.

Validation discipline
---------------------
For a 2022-2025 run with 2025 selected as holdout:
- Main research ranks signals using 2022-2024 only and reports 2025 afterward.
- Walk-forward signal research uses only prior seasons for each test year.
- Holdout results never determine the research ranking.

Promotion principle
-------------------
Do not promote a signal to the live betting board merely because it wins in one
season. Require credible sample size and survival across multiple unseen seasons.
