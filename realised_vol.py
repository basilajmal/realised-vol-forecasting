#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:28:24 2026

@author: basil
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

#---Creating file---

DATA_PATH = "data/spy.csv"

def load_spy(start="2010-01-01", path=DATA_PATH):
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

#---estimators---

# close-to-close
df['ret']   = np.log(df['Close']).diff()
df['var_cc']  = df['ret'] ** 2

#parkinsons
df['var_park'] = (np.log(df['High'] / df['Low']))**2 / (4 * np.log(2))

#garmann klass
df['var_gk'] = (0.5 * (np.log(df['High'] / df['Low']))**2) - ((2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2)

print(df[['var_cc', 'var_park', 'var_gk']].describe())

#check
bad = ((df['High'] < df[['Open','Close']].max(axis=1)) |
       (df['Low']  > df[['Open','Close']].min(axis=1)))
print("inconsistent OHLC rows:", bad.sum())
print("negative GK:", (df['var_gk'] < 0).sum())

print((df['var_cc'] == 0).sum())
print(df['var_cc'].nsmallest(5))

for c in ['var_cc', 'var_park', 'var_gk']:
    print(c, np.log(df[c].replace(0, np.nan)).autocorr(lag=1))
    
#decided to use parkinsons estimator
df['log_vol'] = np.log(df['var_park'])

print(df['log_vol'].isna().sum())          # 0
print(np.isinf(df['log_vol']).sum())       # 0
print(df['log_vol'].describe())

#---table---

d = pd.DataFrame(index=df.index)
d['vol_today'] = df['log_vol']
d['vol_5']     = df['log_vol'].rolling(5).mean()
d['vol_22']    = df['log_vol'].rolling(22).mean()
d['target']    = df['log_vol'].shift(-1)
d = d.dropna()

#checks
print(len(d))
assert (d['target'].iloc[:-1].to_numpy() == d['vol_today'].iloc[1:].to_numpy()).all()
date = d.index[100]
pos  = df.index.get_loc(date)
assert np.isclose(d['vol_5'].iloc[100], df['log_vol'].iloc[pos-4:pos+1].mean())

print(d.isna().sum())
print(np.isinf(d).sum().sum())
print(d.describe())
print(d.corr().round(3))

#---basleines---

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

FEATURES = ['vol_today', 'vol_5', 'vol_22']
X = d[FEATURES].to_numpy()
y = d['target'].to_numpy()
print(X.shape, y.shape)          # (4170, 3) (4170,)

# random walk — no fitting
pred_rw = d['vol_today'].to_numpy()
print("RW   R2:", r2_score(y, pred_rw))
print("RW RMSE:", np.sqrt(mean_squared_error(y, pred_rw)))

# HAR
har = LinearRegression().fit(X, y)
pred_har = har.predict(X)
print("HAR   R2:", r2_score(y, pred_har))
print("HAR RMSE:", np.sqrt(mean_squared_error(y, pred_har)))

print("intercept:", har.intercept_)
print(dict(zip(FEATURES, har.coef_)))

#---walk forward validation---

#   window        : expanding — train on all rows up to the prediction point
#   initial train : 1000 rows (~4 years)
#   refit         : every 21 trading days (~monthly)
#   prediction    : every day; between refits the model is frozen, features update
#   no shuffling, no k-fold, no tuning against test performance

START       = 1000
REFIT_EVERY = 21

preds = np.full(len(y), np.nan)
model = LinearRegression()

for i in range(START, len(y)):
    if (i - START) % REFIT_EVERY == 0:
        model.fit(X[:i], y[:i])
    preds[i] = model.predict(X[i:i+1])[0]

d['pred_har'] = preds

print(np.isnan(preds).sum())
print((~np.isnan(preds)).sum())       

i = 2000
m = LinearRegression().fit(X[:i], y[:i])
print(m.predict(X[i:i+1])[0], preds[i])



def walk_forward(model, X, y, start=1000, refit_every=21):
    preds = np.full(len(y), np.nan)
    for i in range(start, len(y)):
        if (i - start) % refit_every == 0:
            model.fit(X[:i], y[:i])
        preds[i] = model.predict(X[i:i+1])[0]
    return preds

def score(preds, y):
    mask = ~np.isnan(preds)
    return r2_score(y[mask], preds[mask])

# 1. the honest result
p_har = walk_forward(LinearRegression(), X, y)
print("1. HAR walk-forward :", score(p_har, y))

# 2. random walk, same rows
p_rw = d['vol_today'].to_numpy().copy()
p_rw[:1000] = np.nan
print("2. random walk      :", score(p_rw, y))

# 3. leaked feature — target handed to the model as a 4th column
X_leak = np.column_stack([X, y])
p_leak = walk_forward(LinearRegression(), X_leak, y)
print("3. leaked feature   :", score(p_leak, y))

# 4. no time restriction — trains on all 16 years every refit
def walk_forward_LEAKY(model, X, y, start=1000, refit_every=21):
    preds = np.full(len(y), np.nan)
    for i in range(start, len(y)):
        if (i - start) % refit_every == 0:
            model.fit(X, y)
        preds[i] = model.predict(X[i:i+1])[0]
    return preds

p_full = walk_forward_LEAKY(LinearRegression(), X, y)
print("4. full-sample fit  :", score(p_full, y))

#---RMSE table---

def evaluate(preds, y, name):
    mask = ~np.isnan(preds)
    return {
        'model': name,
        'n':     mask.sum(),
        'RMSE':  np.sqrt(mean_squared_error(y[mask], preds[mask])),
        'R2':    r2_score(y[mask], preds[mask]),
    }

results = pd.DataFrame([
    evaluate(p_rw,  y, 'Random walk'),
    evaluate(p_har, y, 'HAR'),
]).set_index('model')

print(results.round(4))

print("typical error factor:", np.exp(results.loc['HAR', 'RMSE']))

#---ML layer---

#ridge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
p_ridge = walk_forward(ridge, X, y)

gbm = HistGradientBoostingRegressor(random_state=0)
p_gbm = walk_forward(gbm, X, y)

mask = ~np.isnan(p_gbm)
print("max actual   :", y[mask].max())
print("max HAR pred :", p_har[mask].max())
print("max GBM pred :", p_gbm[mask].max())

res = pd.DataFrame({'y': y, 'rw': p_rw, 'har': p_har, 'gbm': p_gbm}, index=d.index)
res = res[~np.isnan(p_har)]

by_year = res.groupby(res.index.year).apply(lambda g: pd.Series({
    'n':      len(g),
    'rw_r2':  r2_score(g['y'], g['rw']),
    'har_r2': r2_score(g['y'], g['har']),
    'gbm_r2': r2_score(g['y'], g['gbm']),
}))

print(by_year.round(3))
print("HAR beat RW  in", (by_year['har_r2'] > by_year['rw_r2']).sum(),  "of", len(by_year), "years")
print("HAR beat GBM in", (by_year['har_r2'] > by_year['gbm_r2']).sum(), "of", len(by_year), "years")
print("GBM beat RW  in", (by_year['gbm_r2'] > by_year['rw_r2']).sum(),  "of", len(by_year), "years")



