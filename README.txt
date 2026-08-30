CFB Edge v0.9.0-CANDIDATE-VALIDATION

Purpose
-------
Freeze the first promising classifier rule and stress-test it over a longer
historical window before any live-board promotion.

LOCKED RULE
-----------
56.0% <= model pick probability < 57.0%

The threshold is not re-optimized in this version.

Default history
---------------
2018-2025 selected by default.

Because rolling validation needs at least one prior season:
- 2018 is the initial training season
- 2019 is the first unseen test season

2020 handling
-------------
Two views are produced:

1. Standard rolling walk-forward
   - 2020 is reported separately
   - primary combined results exclude 2020

2. COVID-excluded-training stress test
   - 2020 is never used for fitting
   - 2020 is not used as a test season
   - 2021+ models are trained only on earlier non-2020 seasons

This avoids allowing the structurally unusual 2020 season to determine whether
the candidate survives.

Validation outputs
------------------
- Per-season W-L-P, win rate, units and ROI
- Wilson 95% interval for ATS win rate
- Edge over -110 breakeven
- Pre/post-COVID era stability
- Home vs away diagnostic
- Favorite vs underdog diagnostic
- Leave-one-season-out robustness
- Conservative research promotion gate

Exports
-------
1. cfb_v090_candidate_bets_2018_2025.csv
2. cfb_v090_candidate_seasons_2018_2025.csv
3. cfb_v090_candidate_stress_2018_2025.csv

Recommended run
---------------
Seasons: 2018-2025
Holdout: 2025
Game universe: Major FBS
Historical rating method: Leakage-safe preseason prior
Signal policy: Best market per game

Important
---------
v0.9 does not create a new model architecture and does not search for a better
threshold. It is a validation build for the already identified 56-57% candidate.
