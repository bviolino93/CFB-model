CFB Edge v1.1.0-ALL-MARKET

Purpose
-------
Give the user a useful model estimate for every major betting market:
- spreads
- moneylines
- totals

while still incorporating what the historical validation taught us.

What stays
----------
- Existing CFB projection engine
- Market shrinkage / calibration
- Wider early-season uncertainty
- Model confidence
- Full Slate ranking
- All Markets dropdown
- CSV export
- Conservative staking
- Large-disagreement guardrails

What changes
------------
Totals and moneylines are no longer research-only.

Every available market receives:
- model probability
- implied probability
- edge
- EV
- verdict: STRONG BET / BET / LEAN / PASS

A LEAN is explicitly a lower-confidence model estimate.

Market-specific promotion bars
------------------------------
SPREAD
BET:
- confidence >= 78
- edge >= 4.5 percentage points
- EV >= +7.5%

STRONG BET:
- confidence >= 84
- edge >= 6.5 pts
- EV >= +11.0%

MONEYLINE
BET:
- confidence >= 80
- edge >= 5.0 pts
- EV >= +8.5%

STRONG BET:
- confidence >= 85
- edge >= 7.0 pts
- EV >= +12.5%

Adjustments:
- +200 or higher underdogs require more edge/EV
- heavy favorites <= -180 require more edge/EV
- +300 or higher dogs are capped at LEAN because historical ML pricing quality
  was not trustworthy enough to auto-promote them

TOTAL
BET:
- confidence >= 82
- edge >= 6.5 pts
- EV >= +10.0%

STRONG BET:
- confidence >= 87
- edge >= 8.5 pts
- EV >= +14.0%

Early season
------------
Weeks 1-3 add:
- +1.5 pts required edge
- +2.0% required EV

Totals receive an additional:
- +0.5 pts edge
- +1.0% EV

Disagreement guardrails
-----------------------
Spread raw model vs market gap >= 8 points:
- maximum verdict = LEAN

Total raw model vs market gap >= 9 points:
- maximum verdict = LEAN

This reflects the historical finding that larger model-market disagreement was
not evidence of a stronger edge.

Important
---------
This version does not claim that totals or moneylines have been historically
proven profitable. It provides the user's requested estimates while making
their promotion thresholds materially stricter than spreads.

Use the live slate to rank all available markets. Treat BET/STRONG BET as the
highest-conviction model recommendations and LEAN as an estimate worth tracking.
