CFB Edge v1.2.1-PRO-UI

This release is primarily a UI/UX upgrade over v1.2.0-PLAYABLE.

Model logic
-----------
No intentional betting-model threshold changes from v1.2.0.

Presentation changes
--------------------
- Professional dark sportsbook-style interface
- A / B / C / D grading language
- Cleaner recommendation hierarchy
- More polished spacing, borders, cards, buttons and metrics
- Market Board terminology instead of legacy Market Grades
- Grade legend:
    A = Best Bet
    B = Bet
    C = Lean
    D = Pass
- Cleaner app branding and version footer

Internal compatibility
----------------------
The underlying model still uses:
STRONG BET / BET / LEAN / PASS internally so existing exports and logic remain compatible.
The UI translates these to A / B / C / D.

Suggested unit scale remains:
A = 1.00u
B = 0.50-0.75u
C = 0.25u optional
D = 0u
