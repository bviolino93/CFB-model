CFB Edge v1.0.0-POINT-IN-TIME-LAB

What changed
------------
v1.0 stops searching for a better threshold and rebuilds the historical data layer.

New Workspace
-------------
Point-in-Time Lab

The existing Live Model and Backtest workspaces remain available.

Pregame-only team form
----------------------
v1.0 pulls CFBD game-level team box scores by week and creates rolling team
features using ONLY games from earlier weeks.

Examples:
- points
- total/rush/pass yards
- plays
- yards per play
- first downs
- turnovers
- third/fourth down efficiency
- penalties
- sacks
- tackles for loss
- full-season-to-date averages
- last-3 averages when available

No target-game box score is included in its own pregame snapshot.

Prior-week CORE
---------------
v1.0 also attempts to ingest CFBD CORE rows and only permits a row where:
    throughWeek < target game week

Important: CFBD documents historical CORE as retrospective methodology.
Therefore v1.0 labels it as a prior-week snapshot feature but does NOT claim
that it is a literal archived rating that was published at that historical time.

Market movement
---------------
CFBD generic historical lines remain a fallback single snapshot.

For real opener/current/close-style movement, upload a timestamped CSV:
    game_id
    snapshot_time
    provider
    home_spread
    total
    home_ml
    away_ml

v1.0 filters snapshots to snapshot_time < kickoff and derives:
- opening consensus
- latest pregame consensus
- spread/total/ML movement
- snapshot count

It never labels a generic CFBD line as an opener or close.

QB / injury availability
------------------------
CFBD does not supply the historical point-in-time injury archive needed for
a clean backtest. v1.0 therefore accepts an optional availability CSV:
    game_id
    snapshot_time
    team
    player
    position
    status
    snap_share
    impact_rating

Only records timestamped before kickoff are used. v1.0 does not invent missing
injury values or player impact scores.

Recommended first run
---------------------
Workspace: Point-in-Time Lab
Seasons: 2022-2025
Game universe: Major FBS
No uploads required for the first pipeline test.

After the first clean run, expand the history backward.

Files to send back
------------------
1. cfb_v100_data_quality_2022_2025.csv
2. cfb_v100_point_in_time_2022_2025.csv

If failures are shown:
3. cfb_v100_failures_2022_2025.csv

Next step
---------
Do not fit v1.1 until we inspect coverage and leakage. If the point-in-time
dataset is clean, v1.1 can train a spread-cover model using only the new
pregame features and compare it to the sportsbook baseline.
