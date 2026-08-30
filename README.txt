CFB Edge v0.8.1-SIGNAL-AUDIT

Purpose
-------
Audit the pre-specified 56-58% probability bucket from v0.8.0.

This version does NOT:
- change the classifier
- optimize a new probability threshold
- create new official bets
- promote anything to the live board

Audit dimensions
----------------
- Home vs Away
- Favorite vs Underdog
- Spread size: 0-3, 3.5-7, 7.5-14, 14+
- Week bands: W1-3, W4-6, W7-9, W10+
- Probability sub-band: 56-57% vs 57-58%
- Pass matchup edge/disadvantage
- Rush matchup edge/disadvantage
- Explosiveness matchup edge/disadvantage
- Havoc edge/disadvantage
- Finishing-drives edge/disadvantage

Two-way splits
--------------
- Home/Away x Favorite/Underdog
- Favorite/Underdog x Spread size
- Home/Away x Spread size
- Probability sub-band x Favorite/Underdog

Validation principle
--------------------
Do not select the subgroup with the highest combined historical ROI.
A candidate must have:
1. meaningful sample size
2. multiple qualifying unseen seasons
3. positive performance in more than one unseen season
4. no dependence on the 2025 holdout alone

Exports
-------
- cfb_v081_signal_audit_bets_2022_2025.csv
- cfb_v081_signal_audit_breakdown_2022_2025.csv
- cfb_v081_signal_audit_survival_2022_2025.csv

Recommended run
---------------
Seasons: 2022-2025
Holdout: 2025
Historical rating method: Leakage-safe preseason prior
