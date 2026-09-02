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



