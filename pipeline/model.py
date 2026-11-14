"""Stage 3: model training and calibration.

Two targets, same machinery:
  * spread: predict home margin (home - away) from the diff_/h_/a_ features
  * total:  predict total points from the sum_/h_/a_ features

Two models, averaged into an ensemble:
  * Ridge regression (standardised features, alpha picked by
    leave-one-season-out CV on the training years)
  * XGBoost (small, shallow, fixed settings, so there's nothing to overfit
    by tuning)

Calibration: turning a predicted margin into a cover probability.
  The obvious way (margin ~ Normal(pred, sigma)) assumes the model's errors
  are independent of the market's, which is never true; it overstates edges.
  Instead we use Platt scaling on the *gap* between model and line:

      P(home covers) = logistic(a + b * (pred_margin + close_spread))

  fit on out-of-fold predictions for the training seasons (each season
  predicted by a model trained on the other training seasons). If the gap
  has no real information, b comes out near 0 and the probabilities stay
  near 50%, so few bets clear the edge bar. The normal-distribution version
  is also returned for comparison.

The test season is never used here: it is only predicted, in Stage 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import TEAM_FEATS

RIDGE_ALPHAS = [10, 30, 100, 300, 1000, 3000, 10000]
XGB_PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.6, min_child_weight=20,
                  reg_lambda=5.0, objective="reg:squarederror", n_jobs=4,
                  random_state=0)

MARKETS = {
    # target column, line column, feature prefix.
    # for spreads the home side covers when margin + spread > 0;
    # for totals the over hits when points - total > 0.
    "spread": dict(target="margin", line="close_spread", prefix="diff_"),
    "total": dict(target="points", line="close_total", prefix="sum_"),
}


def feature_columns(df: pd.DataFrame, market: str) -> list[str]:
    pre = MARKETS[market]["prefix"]
    cols = [c for c in df.columns if c.startswith(pre)]
    cols += [f"{p}_{f}" for p in ("h", "a") for f in TEAM_FEATS]
    cols += ["home_field", "rest_diff", "travel_diff"]
    # drop columns that are entirely blank (e.g. travel without a CFBD key)
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def eligible(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Played FBS-vs-FBS games with a line and features."""
    m = MARKETS[market]
    ok = (~df.non_fbs_game & df[m["target"]].notna() & df[m["line"]].notna()
          & df.diff_rating_epa.notna())
    return df[ok]


def gap(pred: np.ndarray, line: np.ndarray, market: str) -> np.ndarray:
    """Model minus market, oriented so positive = bet the home side / over."""
    return pred + line if market == "spread" else pred - line


@dataclass
class FittedModel:
    market: str
    features: list[str]
    ridge: object
    xgb: object
    platt: LogisticRegression
    sigma: float
    ridge_alpha: float
    oof: pd.DataFrame = field(repr=False, default=None)
    gap_table: pd.DataFrame = field(repr=False, default=None)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        pr = self.ridge.predict(X)
        px = self.xgb.predict(X)
        pe = 0.5 * (pr + px)
        line = df[MARKETS[self.market]["line"]].to_numpy(float)
        g = gap(pe, line, self.market)
        out = pd.DataFrame(index=df.index)
        out["pred_ridge"], out["pred_xgb"], out["pred"] = pr, px, pe
        out["gap"] = g
        # probability the home side covers / the over hits
        out["p_platt"] = self.platt.predict_proba(g.reshape(-1, 1))[:, 1]
        out["p_normal"] = norm.cdf(g / self.sigma)
        return out


def _ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=alpha))


def _xgb():
    return xgb.XGBRegressor(**XGB_PARAMS)


def _loso(train: pd.DataFrame, feats, target, make):
    """Leave-one-season-out predictions."""
    pred = pd.Series(np.nan, index=train.index)
    for s in sorted(train.season.unique()):
        tr, va = train[train.season != s], train[train.season == s]
        m = make().fit(tr[feats], tr[target])
        pred.loc[va.index] = m.predict(va[feats])
    return pred


def fit(df: pd.DataFrame, train_seasons, market="spread", verbose=True) -> FittedModel:
    m = MARKETS[market]
    train = eligible(df, market)
    train = train[train.season.isin(train_seasons)]
    feats = feature_columns(train, market)
    y = train[m["target"]]

    # Ridge alpha by leave-one-season-out MAE.
    scores = {}
    for a in RIDGE_ALPHAS:
        p = _loso(train, feats, m["target"], lambda a=a: _ridge(a))
        scores[a] = float((p - y).abs().mean())
    alpha = min(scores, key=scores.get)

    oof_r = _loso(train, feats, m["target"], lambda: _ridge(alpha))
    oof_x = _loso(train, feats, m["target"], _xgb)
    oof = 0.5 * (oof_r + oof_x)
    line = train[m["line"]].to_numpy(float)
    g = gap(oof.to_numpy(), line, market)

    # Outcome for calibration: 1 = home covered / over hit; pushes dropped.
    res = train[m["target"]].to_numpy() + (line if market == "spread" else -line)
    keep = res != 0
    platt = LogisticRegression(C=1.0).fit(g[keep].reshape(-1, 1), (res[keep] > 0).astype(int))
    # A negative slope would mean "bet against our own model". Never do
    # that: treat it as no information (probabilities stay at the base rate).
    if platt.coef_[0][0] < 0:
        platt.coef_[0][0] = 0.0
    sigma = float(np.std(res - g))  # spread of the result around the gap

    ridge = _ridge(alpha).fit(train[feats], y)
    xg = _xgb().fit(train[feats], y)
    oof_df = train[["game_id", "season", m["target"], m["line"]]].assign(
        oof_ridge=oof_r, oof_xgb=oof_x, oof=oof, gap=g)
    fm = FittedModel(market, feats, ridge, xg, platt, sigma, alpha, oof_df)
    if verbose:
        mae_line = float(np.abs(res).mean())
        print(f"  {market}: {len(train):,} training games, {len(feats)} features, "
              f"ridge alpha {alpha}")
        print(f"    out-of-fold MAE  ridge {float((oof_r - y).abs().mean()):.2f}  "
              f"xgb {float((oof_x - y).abs().mean()):.2f}  "
              f"ensemble {float((oof - y).abs().mean()):.2f}  "
              f"| closing line {mae_line:.2f}")
        bins = pd.cut(np.abs(g[keep]), [0, 2, 4, 6, 8, np.inf])
        hit = pd.Series((np.sign(g[keep]) == np.sign(res[keep])).astype(float)).groupby(
            np.asarray(bins), observed=True).agg(["mean", "size"])
        fm.gap_table = hit
        print("    out-of-fold: how often the model's side covered, by model-line gap")
        for b, r in hit.iterrows():
            print(f"      gap {str(b):<12} {r['mean']:.1%}  ({int(r['size'])} games)")
        print(f"    calibration: P(cover) = logistic({platt.intercept_[0]:+.3f} "
              f"{platt.coef_[0][0]:+.4f} x gap); normal sigma {sigma:.1f}")
    return fm


def feature_importance(fm: FittedModel, top=15) -> pd.DataFrame:
    """XGBoost gain share and Ridge standardised |coef| share per feature."""
    gain = fm.xgb.get_booster().get_score(importance_type="gain")
    g = pd.Series({fm.features[int(k[1:])] if k.startswith("f") and k[1:].isdigit()
                   else k: v for k, v in gain.items()})
    coef = pd.Series(np.abs(fm.ridge[-1].coef_), index=fm.features)
    imp = pd.DataFrame({"xgb_gain": g / g.sum(), "ridge_coef": coef / coef.sum()}).fillna(0)
    imp["combined"] = 0.5 * (imp.xgb_gain + imp.ridge_coef)
    return imp.sort_values("combined", ascending=False).head(top)
