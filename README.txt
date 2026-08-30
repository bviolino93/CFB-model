CFB Edge v1.6.1-LOGO-HOTFIX

Fix
===
The v1.6.0 slate logo feature referenced `data` inside Full Slate mode.
That scope actually uses `model_data_s`, causing the NameError shown in Streamlit.

v1.6.1 changes:
- away_logo uses _team_logo_url(model_data_s, ...)
- home_logo uses _team_logo_url(model_data_s, ...)
- removes duplicate FCS guard calls left from prior iterative patches
- preserves all v1.6.0 visual/app-polish features
- preserves Top 5/10 ranking, chronological game dropdowns, FCS protection, and longshot ML guard

Syntax check passed.
