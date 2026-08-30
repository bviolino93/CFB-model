CFB Edge v1.0.1-PIT-HOTFIX

Fix
---
v1.0.0 Point-in-Time Lab referenced `_bt_game_allowed`, but that helper name
does not exist in the inherited application. This caused:

NameError: name '_bt_game_allowed' is not defined

v1.0.1 adds a self-contained `_pit_game_allowed()` helper and changes the
Point-in-Time dataset builder to use it.

No modeling logic, point-in-time feature logic, market ingestion, or
availability ingestion was changed.

Run
---
Workspace: Point-in-Time Lab
Seasons: 2022-2025
Game universe: Major FBS
Leave optional uploads empty for the first run.
