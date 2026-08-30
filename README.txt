CFB Edge v1.4.0-PREMIUM-UI

Purpose
-------
Make the front end feel like a polished consumer betting product rather than
a styled Streamlit research app.

Model logic
-----------
No betting-model or threshold changes from v1.2/v1.3.
This release is visual/UX only.

Major visual changes
--------------------
- Premium dark navy visual system
- Stronger typography and hierarchy
- Cleaner hero and status strip
- Smaller, denser A/B/C/D legend
- Premium top-play card
- Separate no-play state when every market is D
- Compact market-board cards
- Dense mobile layout so more markets fit on one screen
- Cleaner metric chips for Model / Edge / EV / Fair
- Refined buttons, tabs, inputs, metrics, expanders
- Better spacing and reduced visual clutter
- More professional mobile experience

Top result behavior
-------------------
If the top market is:
A/B/C -> header shows TOP PLAY and a premium result card.
D     -> header shows NO PLAY and a quieter no-play card.

Grades
------
A — Best Bet
B — Bet
C — Lean
D — Pass

Suggested units are unchanged:
A = 1.00u
B = 0.50–0.75u
C = 0.25u optional
D = 0u
