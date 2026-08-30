CFB Edge v0.8.0-COVER-CLASSIFIER

Purpose
-------
Change the modeling target from final-margin residual prediction to the betting
question we actually care about:

    P(home covers the sportsbook spread)
    P(away covers the sportsbook spread)

This version does NOT promote anything to the live betting board. It is a
research/validation layer.

Architecture
------------
Sportsbook consensus spread
    + matchup / personnel / pace features
    -> regularized logistic classifier
    -> ATS cover probability

Features
--------
- Market spread / favorite size
- Week and HFA
- SP+ differential
- Talent differential
- Returning-production differential
- Returning passing production / usage
- Pass offense vs opponent pass defense PPA
- Rush offense vs opponent rush defense PPA
- Success-rate matchup
- Explosiveness matchup
- Advanced pass/rush play PPA
- Finishing drives
- Defensive havoc
- Pace / plays per drive

Validation
----------
Rolling unseen-season validation:
- train 2022 -> test 2023
- train 2022-23 -> test 2024
- train 2022-24 -> test 2025

Regularization is selected only inside the development sample.

Primary metrics
---------------
- Log loss vs 50/50 benchmark
- Brier score vs 50/50 benchmark
- Calibration
- ATS performance by fixed predicted-probability bucket

Fixed probability buckets
-------------------------
- 50-52.4%
- 52.4-54%
- 54-56%
- 56-58%
- 58%+

The thresholds are fixed in advance. They are not optimized against 2025.

Exports
-------
- cfb_v080_classifier_walkforward_YYYY_YYYY.csv
- cfb_v080_classifier_buckets_YYYY_YYYY.csv
- cfb_v080_classifier_picks_YYYY_YYYY.csv
- cfb_v080_classifier_calibration_YYYY_YYYY.csv
- cfb_v080_classifier_importance_holdout_YYYY.csv

Promotion gate
--------------
Do not promote v0.8 to live betting unless:
1. Probability scores improve on the 50/50 baseline in multiple unseen seasons.
2. Probabilities are reasonably calibrated.
3. Fixed high-probability buckets show credible multi-season ATS performance.
4. No conclusion depends only on 2025.
