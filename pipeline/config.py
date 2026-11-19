"""Every setting the pipeline uses, in one place.

Change a number here, rerun, and everything downstream picks it up.
"""
from __future__ import annotations

import os

# ---- Seasons -------------------------------------------------------------
TRAIN_SEASONS = list(range(2018, 2025))   # 2018-2024 fit the model
TEST_SEASONS = [2025]                     # strictly out of sample
WARMUP_SEASONS = [2017]                   # only feeds early-2018 ratings/EWMAs
ALL_SEASONS = WARMUP_SEASONS + TRAIN_SEASONS + TEST_SEASONS

# ---- Cleaning ------------------------------------------------------------
MOV_CAP = 28              # margins beyond +/-28 count as 28 in point features
# Garbage time (Bill Connelly / CFBD definition): a play is dropped when the
# score gap at the snap exceeds this many points in the given quarter.
GARBAGE_GAP = {2: 38, 3: 28, 4: 22}
POOL_NON_FBS = True       # all FCS/lower teams share one rating ("tier" adj.)

# ---- Features ------------------------------------------------------------
EWMA_SPANS = (4, 8)       # 4-game and 8-game exponentially weighted means
# Opponent adjustment: a ridge fit of  value = mu + offense_i + allowed_j + hfa
# on games played before each week's first kickoff.
RIDGE_ALPHA = 4.0
PRIOR_SEASON_WEIGHT = 0.5     # weight on last season's games at week 1 ...
PRIOR_SEASON_DECAY_WEEKS = 3  # ... decaying by e^(-weeks/3) as the year goes on
REST_CAP_DAYS = 28            # openers and long layoffs count as 28 days

# ---- Betting (used in Stages 3-5) ----------------------------------------
MIN_EDGE = 0.035          # model prob minus de-vigged implied prob
KELLY_FRACTION = 0.25     # quarter Kelly
START_BANKROLL = 1000.0
DEFAULT_ODDS = -110       # when a book's price is missing
YOUR_BOOK_ODDS = -105     # 734 Games usually deals -105

# ---- Paths and keys ------------------------------------------------------
CACHE_DIR = os.environ.get("CFB_CACHE_DIR", ".cache_cfb")
PIPE_CACHE = os.path.join(CACHE_DIR, "pipeline")
OUTPUT_DIR = os.environ.get("CFB_PIPELINE_OUT", os.path.join("pipeline", "output"))
CFBD_API_KEY = os.environ.get("CFBD_API_KEY", "").strip()
CFBD_BASE = "https://api.collegefootballdata.com"
