CFB Edge v1.2.0-PLAYABLE

Goal
----
Keep the lessons from the historical research without making the live model
so conservative that almost nothing can be played.

User-facing grades
------------------
A — BEST BET
B — BET
C — LEAN
D — PASS

All three markets can qualify:
- spread
- moneyline
- total

Core change from v1.1
---------------------
Confidence is no longer a hard veto.

In v1.1, early-season confidence near 72 meant even a large modeled edge could
be automatically prevented from reaching BET status. In v1.2, confidence
modifies the amount of edge/EV required.

Lower confidence => requires more value.
Higher confidence => can qualify with slightly less value.
The model can therefore make actual bets in Week 1 without ignoring uncertainty.

Baseline playable hurdles before modifiers
------------------------------------------
SPREAD B:
edge >= 3.5 pts
EV >= +5.5%

TOTAL B:
edge >= 4.5 pts
EV >= +7.0%

MONEYLINE B:
edge >= 3.5 pts
EV >= +6.0%

Weeks 1-3 add modest edge/EV requirements rather than a confidence veto.
Totals remain somewhat stricter.
+200 or longer ML underdogs receive additional price penalties.
+300 or longer dogs are capped at C/LEAN because the old historical ML feed
showed suspicious long-underdog behavior.

Large projection disagreement
------------------------------
Very large raw model/market gaps no longer force PASS/LEAN. Instead they prevent
A/Best Bet status. A game can still become a B bet if the adjusted probability
and EV remain strong enough.

Suggested unit scale
--------------------
A — BEST BET: 1.00u
B — BET: 0.50u, or 0.75u for a stronger B
C — LEAN: 0.25u optional
D — PASS: 0u

This is intentionally not Kelly sizing and does not guarantee profitability.

Why this is the balance
-----------------------
Earlier versions were too aggressive.
v1.1 overcorrected and often produced no bets.
v1.2 keeps market calibration, early-season uncertainty, price-aware ML
protection, total-market caution, and extreme-gap protection while allowing
genuinely strong live edges to become playable.

Do not tune these thresholds game-by-game after seeing results. Freeze v1.2
and evaluate the forward 2026 sample by grade and market.
