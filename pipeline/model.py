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
    # target column, feature prefix, closing and opening line columns.
    # Spreads: the home side covers when margin + spread > 0.
    # Totals: the over hits when points - total > 0.
    "spread": dict(target="margin", prefix="diff_", close="close_spread",
                   open="open_spread", sign=+1),
    "total": dict(target="points", prefix="sum_", close="close_total",
                  open="open_total", sign=-1),
}

# Extra feature groups on top of the play-by-play base set.
GROUPS = {
    "priors": {"spread": lambda c: c.startswith("pri_") and not c.endswith("_sum"),
               "total": lambda c: c.startswith("pri_") and (c.endswith("_sum") or c == "pri_w")},
    "qb": {"spread": lambda c: c in ("qb_change_diff", "qb_starts_diff", "h_qb_change",
                                     "a_qb_change", "h_qb_starts", "a_qb_starts"),
           "total": lambda c: c in ("qb_change_sum", "h_qb_change", "a_qb_change",
                                    "h_qb_starts", "a_qb_starts")},
    "injury": {"spread": lambda c: c in ("inj_diff", "inj_qb_diff"),
               "total": lambda c: c in ("inj_sum",)},
    # what the market said at the open (known when you'd bet the open)
    "mkt_open": {"spread": lambda c: c in ("mkt_open_spread", "mkt_has_open_spread"),
                 "total": lambda c: c in ("mkt_open_total", "mkt_has_open_total")},
    # the close and the open-to-close move (only usable at the close)
    "mkt_close": {"spread": lambda c: c in ("mkt_open_spread", "mkt_has_open_spread",
                                            "mkt_close_spread", "mkt_spread_move"),
                  "total": lambda c: c in ("mkt_open_total", "mkt_has_open_total",
                                           "mkt_close_total", "mkt_total_move")},
}


@dataclass(frozen=True)
class Spec:
    """What to fit.

    kind  "raw":   predict the margin / total, then compare with the line
          "resid": predict the market's error directly
                   (margin + spread, or points - total), Brian's idea 3
    line  "close" or "open": which line the bet is priced and graded against
    groups  extra feature groups (see GROUPS)
    """
    market: str = "spread"
    kind: str = "raw"
    line: str = "close"
    groups: tuple = ()

    @property
    def line_col(self):
        return MARKETS[self.market][self.line]

    @property
    def label(self):
        g = "+".join(self.groups) if self.groups else "base"
        return f"{self.market} {self.kind} @{self.line} [{g}]"


def feature_columns(df: pd.DataFrame, spec: Spec) -> list[str]:
    pre = MARKETS[spec.market]["prefix"]
    cols = [c for c in df.columns if c.startswith(pre)]
    cols += [f"{p}_{f}" for p in ("h", "a") for f in TEAM_FEATS]
    cols += ["home_field", "rest_diff", "travel_diff"]
    for g in spec.groups:
        cols += [c for c in df.columns if GROUPS[g][spec.market](c)]
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    # drop blank or constant columns (e.g. travel without a CFBD key,
    # injury columns in seasons with no injury feed)
    return [c for c in cols if df[c].notna().any() and df[c].nunique() > 1]


def eligible(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Played FBS-vs-FBS games with the chosen line and features."""
    m = MARKETS[spec.market]
    ok = (~df.non_fbs_game & df[m["target"]].notna() & df[spec.line_col].notna()
          & df.diff_rating_epa.notna())
    return df[ok]


def market_error(df: pd.DataFrame, spec: Spec) -> np.ndarray:
    """> 0 when the home side covered / the over hit at `spec`'s line."""
    m = MARKETS[spec.market]
    return (df[m["target"]] + m["sign"] * df[spec.line_col]).to_numpy(float)


def target(df: pd.DataFrame, spec: Spec) -> pd.Series:
    if spec.kind == "resid":
        return pd.Series(market_error(df, spec), index=df.index)
    return df[MARKETS[spec.market]["target"]]


def gap(pred, df: pd.DataFrame, spec: Spec) -> np.ndarray:
    """Model's view minus the line, oriented so positive = home / over."""
    pred = np.asarray(pred, float)
    if spec.kind == "resid":
        return pred
    return pred + MARKETS[spec.market]["sign"] * df[spec.line_col].to_numpy(float)


@dataclass
class FittedModel:
    spec: Spec
    features: list[str]
    ridge: object
    xgb: object
    platt: LogisticRegression
    sigma: float
    ridge_alpha: float
    oof: pd.DataFrame = field(repr=False, default=None)
    gap_table: pd.DataFrame = field(repr=False, default=None)
    log: str = ""

    @property
    def market(self):
        return self.spec.market

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        pr = self.ridge.predict(X)
        px = self.xgb.predict(X)
        pe = 0.5 * (pr + px)
        g = gap(pe, df, self.spec)
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


def _loso(train: pd.DataFrame, feats, y: pd.Series, make):
    """Leave-one-season-out predictions."""
    pred = pd.Series(np.nan, index=train.index)
    for s in sorted(train.season.unique()):
        tr, va = train.season != s, train.season == s
        m = make().fit(train.loc[tr, feats], y[tr])
        pred.loc[va] = m.predict(train.loc[va, feats])
    return pred


def fit(df: pd.DataFrame, train_seasons, spec: Spec | str = "spread",
        verbose=True) -> FittedModel:
    if isinstance(spec, str):
        spec = Spec(market=spec)
    train = eligible(df, spec)
    train = train[train.season.isin(train_seasons)]
    feats = feature_columns(train, spec)
    y = target(train, spec)

    # Ridge alpha by leave-one-season-out MAE.
    scores = {}
    for a in RIDGE_ALPHAS:
        p = _loso(train, feats, y, lambda a=a: _ridge(a))
        scores[a] = float((p - y).abs().mean())
    alpha = min(scores, key=scores.get)

    oof_r = _loso(train, feats, y, lambda: _ridge(alpha))
    oof_x = _loso(train, feats, y, _xgb)
    oof = 0.5 * (oof_r + oof_x)
    g = gap(oof, train, spec)

    # Calibration on the out-of-fold gap; pushes dropped.
    res = market_error(train, spec)
    keep = res != 0
    platt = LogisticRegression(C=1.0).fit(g[keep].reshape(-1, 1), (res[keep] > 0).astype(int))
    # A negative slope would mean "bet against our own model". Never do
    # that: treat it as no information (probabilities stay at the base rate).
    if platt.coef_[0][0] < 0:
        platt.coef_[0][0] = 0.0
    sigma = float(np.std(res - g))

    ridge = _ridge(alpha).fit(train[feats], y)
    xg = _xgb().fit(train[feats], y)
    oof_df = train[["game_id", "season", "week"]].assign(
        oof_ridge=oof_r, oof_xgb=oof_x, oof=oof, gap=g, res=res)
    bins = pd.cut(np.abs(g[keep]), [0, 2, 4, 6, 8, np.inf])
    hit = pd.Series((np.sign(g[keep]) == np.sign(res[keep])).astype(float)).groupby(
        np.asarray(bins), observed=True).agg(["mean", "size"])
    lines = [f"  {spec.label}: {len(train):,} training games, {len(feats)} features, "
             f"ridge alpha {alpha}"]
    if spec.kind == "raw":
        yv = y.to_numpy()
        lines.append(f"    out-of-fold MAE {float(np.abs(oof - yv).mean()):.2f} "
                     f"vs the line {float(np.abs(res).mean()):.2f}")
    else:
        lines.append(f"    out-of-fold: predicted vs actual market error, corr "
                     f"{np.corrcoef(oof, res)[0, 1]:+.3f}")
    lines.append("    out-of-fold, model's side covered by |gap|: " + ", ".join(
        f"{str(b)} {r['mean']:.1%} (n={int(r['size'])})" for b, r in hit.iterrows()))
    lines.append(f"    calibration slope {platt.coef_[0][0]:+.4f} per point")
    log = "\n".join(lines)
    if verbose:
        print(log)
    return FittedModel(spec, feats, ridge, xg, platt, sigma, alpha, oof_df, hit, log)


def feature_importance(fm: FittedModel, top=15) -> pd.DataFrame:
    """XGBoost gain share and Ridge standardised |coef| share per feature."""
    gain = fm.xgb.get_booster().get_score(importance_type="gain")
    g = pd.Series({fm.features[int(k[1:])] if k.startswith("f") and k[1:].isdigit()
                   else k: v for k, v in gain.items()})
    coef = pd.Series(np.abs(fm.ridge[-1].coef_), index=fm.features)
    imp = pd.DataFrame({"xgb_gain": g / g.sum(), "ridge_coef": coef / coef.sum()}).fillna(0)
    imp["combined"] = 0.5 * (imp.xgb_gain + imp.ridge_coef)
    return imp.sort_values("combined", ascending=False).head(top)
