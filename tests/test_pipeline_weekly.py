import pandas as pd

from pipeline import weekly_picks as W


def test_match_book_names_to_schedule():
    keys = {W.team_key(t): t for t in ["Miami", "Miami (OH)", "Ohio State", "San José State",
                                       "Hawai'i", "App State", "Texas A&M"]}
    assert W.match_team("Miami Hurricanes", keys) == "Miami"
    assert W.match_team("Miami (OH) RedHawks", keys) == "Miami (OH)"
    assert W.match_team("Ohio State Buckeyes", keys) == "Ohio State"
    assert W.match_team("San Jose State Spartans", keys) == "San José State"
    assert W.match_team("Hawaii Rainbow Warriors", keys) == "Hawai'i"
    assert W.match_team("Appalachian State", keys) == "App State"
    assert W.match_team("Nowhere Tech", keys) is None


def test_center_slate_removes_median_lean():
    n = 12
    s = pd.DataFrame({"mkt_spread": [-3.0] * n, "fair_spread": [-4.0] * n,
                      "inj_h_pts": [0.0] * n, "inj_a_pts": [0.0] * n,
                      "mkt_total": [50.0] * n, "fair_total0": [53.0] * n,
                      "fair_total": [53.0] * n})
    out, note = W.center_slate(s)
    assert (out.fair_spread == -3.0).all() and (out.fair_total == 50.0).all()
    assert "totals +3.0" in note
