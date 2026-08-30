CFB Edge v0.6.1-SIGNAL-EXPORTS

Changes from v0.6.0
-------------------
- No modeling logic changes.
- Adds downloadable CSV for Development-ranked Signal Research.
- Adds downloadable CSV for rolling Signal Walk-Forward results.
- Adds downloadable CSV for Multi-season Signal Summary.
- Keeps the normal full backtest export.
- Fixes only the research export workflow so signal results can be audited outside Streamlit.

Recommended run
---------------
Run 2022-2025 with 2025 as the holdout, then export:
1) Signal Research CSV
2) Signal Walk-Forward CSV
3) Signal Summary CSV

Those three files contain the evidence needed to decide whether any signal should advance toward the live model.
