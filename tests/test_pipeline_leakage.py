"""Stage 2 anti-leakage: changing a game's result must not move any feature
of that game or of any game that kicked off before it."""
import numpy as np
import pandas as pd

from pipeline import features as F


def _synthetic(seed=0, n_teams=12, weeks=10, seasons=(2023, 2024)):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    games, pbp, gid = [], [], 1
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for k in range(0, n_teams, 2):
                h, a = order[k], order[k + 1]
                hp, ap = rng.integers(0, 50, 2)
                games.append(dict(
                    game_id=gid, season=s, week=w, season_type="regular",
                    kickoff=pd.Timestamp(f"{s}-09-01", tz="UTC") + pd.Timedelta(weeks=w - 1),
                    completed=True, neutral_site=False, venue_id=1,
                    home_team=h, away_team=a, home_tier="fbs", away_tier="fbs",
                    home_points=float(hp), away_points=float(ap),
                    margin=float(hp - ap), points=float(hp + ap),
                    close_spread=-3.0, close_total=50.0, non_fbs_game=False))
                for t in (h, a):
                    pbp.append(dict(game_id=gid, team=t, epa=rng.normal(0, .2),
                                    sr=rng.uniform(.3, .55), isoppp=rng.normal(1, .2),
                                    ypp=rng.normal(5.5, 1), havoc=rng.uniform(.1, .2)))
                gid += 1
    return pd.DataFrame(games), pd.DataFrame(pbp)


def _features(games, pbp):
    df, _, _ = F.stage2(games, pbp, verbose=False)
    cols = F.model_features(df)
    return df.set_index("game_id")[cols]


def test_changing_a_game_never_moves_its_own_or_earlier_features():
    games, pbp = _synthetic()
    base = _features(games, pbp)
    target = games[(games.season == 2024) & (games.week == 5)].game_id.iloc[0]
    kick = games.set_index("game_id").kickoff[target]

    g2, p2 = games.copy(), pbp.copy()
    g2.loc[g2.game_id == target, ["home_points", "margin", "points"]] = [99.0, 99.0, 99.0]
    p2.loc[p2.game_id == target, ["epa", "sr", "ypp"]] = [3.0, 1.0, 30.0]
    new = _features(g2, p2)

    before = (games.set_index("game_id").kickoff <= kick).reindex(base.index)  # incl. itself
    pd.testing.assert_frame_equal(base[before], new[before])
    later = ~before
    assert not np.allclose(base[later].fillna(0), new[later].fillna(0)), \
        "later games should see the changed result"


def test_garbage_time_rule():
    from pipeline.ingest import garbage_mask
    period = pd.Series([1, 2, 2, 3, 4, 4, 5])
    gap = pd.Series([50, 38, 39, 29, 22, 23, 40])
    assert garbage_mask(period, gap).tolist() == [False, False, True, True, False, True, False]
