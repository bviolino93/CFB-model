CFB Edge v0.6.2-EXPORT-FIX

Fix
---
The v0.6.1 research CSV exports passed bytes into ios_save_button(), while the
helper expected a string and called .encode(). That caused the AttributeError
shown in Streamlit.

v0.6.2:
- Makes ios_save_button accept either str or bytes.
- Keeps new Signal Research exports as strings.
- No model, backtest, signal, threshold, or validation logic changed.
- Updates exported filenames to v0.6.2.

You do not need to reinterpret the prior model results. Re-run the same
2022-2025 / 2025-holdout backtest and use the three Signal Research download
buttons.
