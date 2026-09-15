"""
Realised Volatility Forecasting (S&P 500)
=========================================

Forecasts tomorrow's log realised volatility for SPY from today's, the past
week's and the past month's realised volatility (the HAR structure), and
compares a linear benchmark against ridge and gradient boosting under
walk-forward validation.

Run top to bottom. Reproduces every number and every figure in the README.
Runtime is a few minutes; the gradient-boosting runs dominate it.

    python run_analysis.py
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --- config ---
DATA_PATH   = "data/spy.csv"
FIG_DIR     = "figures"
FEATURES    = ["vol_today", "vol_5", "vol_22"]

# Walk-forward protocol, fixed BEFORE any results were seen:
#   window        : expanding - train on all rows up to the prediction point
#   initial train : 1000 rows (~4 years)
#   refit         : every 21 trading days (~monthly)
#   prediction    : every day; between refits the model is frozen, features update
#   no shuffling, no k-fold, no tuning against test performance
START       = 1000
REFIT_EVERY = 21

os.makedirs(FIG_DIR, exist_ok=True)


# --- data ---
def load_spy(start="2010-01-01", path=DATA_PATH):
    """Load SPY daily OHLC, caching to disk so every run sees identical data."""
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"loaded {len(df)} rows from {path}")
    else:
        df = yf.download("SPY", start=start, auto_adjust=True)
        df.columns = df.columns.droplevel("Ticker")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"downloaded {len(df)} rows -> {path}")
    return df


df = load_spy()


# --- volatility estimators ---
# All three estimate DAILY RETURN VARIANCE.

# Close-to-close: unbiased but estimated from a single observation, and blind
# to everything that happens intraday.
df["ret"]      = np.log(df["Close"]).diff()
df["var_cc"]   = df["ret"] ** 2

# Parkinson (1980): uses the intraday range. 4*ln(2) normalises it so that a
# driftless random walk observed continuously gives an unbiased estimate.
df["var_park"] = np.log(df["High"] / df["Low"]) ** 2 / (4 * np.log(2))

# Garman-Klass (1980): range plus the open-to-close move.
df["var_gk"] = (0.5 * np.log(df["High"] / df["Low"]) ** 2
                - (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2)

print("\n--- estimator summary ---")
print(df[["var_cc", "var_park", "var_gk"]].describe())

print("\n--- data integrity ---")
bad = ((df["High"] < df[["Open", "Close"]].max(axis=1)) |
       (df["Low"]  > df[["Open", "Close"]].min(axis=1)))
print("inconsistent OHLC rows :", bad.sum())          # expect 0
print("negative Garman-Klass  :", (df["var_gk"] < 0).sum())   # expect 0
print("exact zeros in var_cc  :", (df["var_cc"] == 0).sum())

print("\n--- annualised mean volatility ---")
for c in ["var_cc", "var_park", "var_gk"]:
    print(f"{c:9s} {np.sqrt(252 * df[c].mean()):.4f}")
print("Parkinson / close-to-close variance :",
      round(df["var_park"].mean() / df["var_cc"].mean(), 3))

# Estimator choice, measured rather than cited. True volatility is persistent;
# measurement noise is not, so it dilutes autocorrelation. A less noisy
# estimator therefore shows higher lag-1 autocorrelation in logs.
print("\n--- lag-1 autocorrelation of log variance (estimator noise) ---")
for c in ["var_cc", "var_park", "var_gk"]:
    print(f"{c:9s} {np.log(df[c].replace(0, np.nan)).autocorr(lag=1):.4f}")

# Parkinson selected: ~5x the persistence of close-to-close, one term rather
# than two, and no zeros to poison the log.
df["log_vol"] = np.log(df["var_park"])
assert df["log_vol"].isna().sum() == 0
assert np.isinf(df["log_vol"]).sum() == 0


# ---- feature table ---
# Built as a separate frame so raw price columns physically cannot reach the
# model. Exactly one column here comes from the future, and it is the target.
d = pd.DataFrame(index=df.index)
d["vol_today"] = df["log_vol"]
d["vol_5"]     = df["log_vol"].rolling(5).mean()
d["vol_22"]    = df["log_vol"].rolling(22).mean()
d["target"]    = df["log_vol"].shift(-1)
d = d.dropna()

print("\n--- feature table ---")
print("rows:", len(d))                      # 4192 - 21 rolling - 1 shift

# Row t's target must equal row t+1's vol_today. Compared by POSITION, so the
# arrays go through .to_numpy() - as Series, pandas would realign them by date.
assert (d["target"].iloc[:-1].to_numpy() == d["vol_today"].iloc[1:].to_numpy()).all()

# The 5-day window must end ON the current row, not straddle the future.
_pos = df.index.get_loc(d.index[100])
assert np.isclose(d["vol_5"].iloc[100], df["log_vol"].iloc[_pos - 4:_pos + 1].mean())

assert d.isna().sum().sum() == 0
assert np.isinf(d).sum().sum() == 0
print("alignment and window checks passed")
print(d.corr().round(3))

X = d[FEATURES].to_numpy()
y = d["target"].to_numpy()
print("X", X.shape, " y", y.shape)


# --- in-sample wiring check ---
# NOT a result: both models are scored on the rows they were fitted on.
# Included because the HAR figure can be predicted from the correlation matrix
# in advance (0.477), which verifies X and y are aligned as intended.
har_is = LinearRegression().fit(X, y)
print("\n--- in-sample (wiring check, not a result) ---")
print("RW  R2 :", round(r2_score(y, d["vol_today"].to_numpy()), 4))
print("HAR R2 :", round(r2_score(y, har_is.predict(X)), 4))
print("intercept:", round(har_is.intercept_, 4))
print(dict(zip(FEATURES, har_is.coef_.round(4))))
print("coefficient sum:", round(har_is.coef_.sum(), 4))   # < 1 => mean reversion


# --- walk-forward harness ---
def walk_forward(model, X, y, start=START, refit_every=REFIT_EVERY):
    """Predict each row using only rows strictly before it.

    Row i's target is day i+1's log-vol. Standing at the close of day i, the
    most recent row whose answer is already known is row i-1, so X[:i] is the
    legitimate training set. X[:i+1] would hand the model the answer.
    """
    preds = np.full(len(y), np.nan)
    for i in range(start, len(y)):
        if (i - start) % refit_every == 0:
            model.fit(X[:i], y[:i])
        preds[i] = model.predict(X[i:i + 1])[0]
    return preds


def score(preds, y):
    mask = ~np.isnan(preds)
    return r2_score(y[mask], preds[mask])


def evaluate(preds, y, name):
    mask = ~np.isnan(preds)
    return {"model": name,
            "n":     mask.sum(),
            "RMSE":  np.sqrt(mean_squared_error(y[mask], preds[mask])),
            "R2":    r2_score(y[mask], preds[mask])}


# One prediction reproduced by hand. i must be a refit day, or the loop's
# prediction came from an earlier fit and the two will differ.
_i = START + REFIT_EVERY * 47          # 1987
_m = LinearRegression().fit(X[:_i], y[:_i])
_by_hand = _m.predict(X[_i:_i + 1])[0]
_by_loop = walk_forward(LinearRegression(), X, y)[_i]
assert np.isclose(_by_hand, _by_loop)
print("\nhand-verified prediction at row", _i, ":", round(_by_hand, 6))


# --- evaluation ---
MODELS = {
    "HAR":   LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),   # scaling fitted
    "GBM":   HistGradientBoostingRegressor(random_state=0),       # on train only
}
# GBM hyperparameters left at library defaults deliberately: tuning them
# against out-of-sample performance would be selecting on the test set.

pred = {"Random walk": np.where(np.arange(len(y)) < START, np.nan,
                                d["vol_today"].to_numpy())}
for _name, _model in MODELS.items():
    print(f"running walk-forward: {_name} ...")
    pred[_name] = walk_forward(_model, X, y)

mask = ~np.isnan(pred["HAR"])

results = pd.DataFrame([evaluate(p, y, n) for n, p in pred.items()]).set_index("model")
results["err_factor"] = np.exp(results["RMSE"])     # typical error, variance scale
print("\n--- out-of-sample results ---")
print(results.round(4))


# --- hindsight premium ---
# How much does a model gain from being allowed to see the whole sample?
# A rigid model gains almost nothing; a flexible one gains a lot, because it
# can memorise. The gap is why walk-forward validation exists.
def walk_forward_LEAKY(model, X, y, start=START, refit_every=REFIT_EVERY):
    """DELIBERATELY BROKEN. Trains on all rows, past and future. Diagnostic only."""
    preds = np.full(len(y), np.nan)
    for i in range(start, len(y)):
        if (i - start) % refit_every == 0:
            model.fit(X, y)
        preds[i] = model.predict(X[i:i + 1])[0]
    return preds


premium = pd.DataFrame({
    name: {"full_sample":  score(walk_forward_LEAKY(MODELS[name], X, y), y),
           "walk_forward": score(pred[name], y)}
    for name in ["HAR", "GBM"]
}).T
premium["premium"] = premium["full_sample"] - premium["walk_forward"]
print("\n--- hindsight premium ---")
print(premium.round(4))


# ---- LEAKAGE TEST ---
# Deliberately broken, for validation only. Hands the target to the model as a
# fourth feature; R2 must go to 1.0. Confirms the harness would surface a leak
# rather than silently absorbing one.
_p_leak = walk_forward(LinearRegression(), np.column_stack([X, y]), y)
print("\nleakage test (target as a feature), R2 =", round(score(_p_leak, y), 4))
assert score(_p_leak, y) > 0.999


# --- year-by-year R2 ---
res = pd.DataFrame({"y": y, **{k.lower().replace(" ", "_"): v for k, v in pred.items()}},
                   index=d.index)[mask]

by_year = res.groupby(res.index.year).apply(lambda g: pd.Series({
    "n":      len(g),
    "rw_r2":  r2_score(g["y"], g["random_walk"]),
    "har_r2": r2_score(g["y"], g["har"]),
    "gbm_r2": r2_score(g["y"], g["gbm"]),
}))

print("\n--- out-of-sample R2 by year ---")
print(by_year.round(3))
print("HAR beat RW  in", (by_year["har_r2"] > by_year["rw_r2"]).sum(),  "of", len(by_year), "years")
print("HAR beat GBM in", (by_year["har_r2"] > by_year["gbm_r2"]).sum(), "of", len(by_year), "years")
print("GBM beat RW  in", (by_year["gbm_r2"] > by_year["rw_r2"]).sum(),  "of", len(by_year), "years")
print("pooled HAR R2 :", round(results.loc["HAR", "R2"], 3),
      " median annual :", round(by_year["har_r2"].median(), 3))


# --- tail behaviour ---
# Neither model ever forecasts anything close to a crisis: both predict the
# conditional mean, and the largest realisations are partly unforecastable
# proxy noise.
def annualised(log_var):
    return 100 * np.sqrt(252 * np.exp(log_var))


print("\n--- extremes (annualised volatility, %) ---")
print("max realised     :", round(annualised(y[mask].max()), 1))
print("max HAR forecast :", round(annualised(pred["HAR"][mask].max()), 1))
print("max GBM forecast :", round(annualised(pred["GBM"][mask].max()), 1))

_hi = y[mask] > np.quantile(y[mask], 0.95)
print("HAR bias, all days      :", round((y[mask] - pred["HAR"][mask]).mean(), 3))
print("HAR bias, top 5% of days:", round((y[mask][_hi] - pred["HAR"][mask][_hi]).mean(), 3))


# --- figures ---
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 150,
})

# Figure 1 - hindsight premium. Values read from the computed table so the
fig, ax = plt.subplots(figsize=(6.5, 4))
xpos, w = np.arange(len(premium)), 0.36
b1 = ax.bar(xpos - w / 2 - 0.01, premium["full_sample"].values,  w,
            color=ORANGE, label="Fitted on all data")
b2 = ax.bar(xpos + w / 2 + 0.01, premium["walk_forward"].values, w,
            color=BLUE,   label="Walk-forward (honest)")

for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK2,
                    xytext=(0, 3), textcoords="offset points")

ax.set_xticks(xpos)
ax.set_xticklabels(["HAR", "Gradient boosting"])
ax.set_ylabel("$R^2$")
ax.set_ylim(0, max(premium["full_sample"]) * 1.15)
ax.set_title("Hindsight inflates the flexible model, not the linear one")
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/hindsight_premium.png", dpi=200)

# Figure 2 - why the target is modelled in logs.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
v = df["var_park"].dropna()
axes[0].hist(v * 1e4, bins=120, color=BLUE)
axes[0].set_xlabel("daily variance  ($\\times 10^{-4}$)")
axes[1].hist(np.log(v), bins=120, color=BLUE)
axes[1].set_xlabel("log daily variance")
for a, t in zip(axes, ["Before: positive and heavily right-skewed",
                       "After: roughly symmetric"]):
    a.set_title(t); a.set_ylabel("days"); a.yaxis.grid(True); a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/log_transform.png", dpi=200)

# Figure 3 - forecasts, with a crisis zoom. Both series are indexed by forecast
s = pd.Series(annualised(y), index=d.index)[mask]
h = pd.Series(annualised(pred["HAR"]), index=d.index)[mask]

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(s.index, s.values, lw=0.6, color=BLUE,   label="Realised")
axes[0].plot(h.index, h.values, lw=1.0, color=ORANGE, label="HAR forecast")
axes[0].set_title("Out-of-sample forecasts, 2014-2026")

lo, hi = "2020-02-01", "2020-06-30"
axes[1].plot(s.loc[lo:hi].index, s.loc[lo:hi].values, lw=1.3, color=BLUE,   label="Realised")
axes[1].plot(h.loc[lo:hi].index, h.loc[lo:hi].values, lw=2.0, color=ORANGE, label="HAR forecast")
axes[1].set_title("Feb-Jun 2020: the model tracks the regime but never reaches the peak")

for a in axes:
    a.set_ylabel("annualised volatility (%)")
    a.yaxis.grid(True); a.set_axisbelow(True); a.legend(loc="upper left")
fig.tight_layout(h_pad=2.5)
fig.savefig(f"{FIG_DIR}/forecasts.png", dpi=200)

print(f"\nfigures written to {FIG_DIR}/")
