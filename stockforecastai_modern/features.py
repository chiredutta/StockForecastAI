import numpy as np
import pandas as pd

def _rsi(s, n=14):
    d=s.diff()
    up=d.clip(lower=0).rolling(n).mean()
    dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def _atr(df,n=14):
    prev=df["Close"].shift(1)
    tr=pd.concat([
        (df["High"]-df["Low"]).abs(),
        (df["High"]-prev).abs(),
        (df["Low"]-prev).abs()
    ],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_technical_features(df):
    x=df.copy()
    c,h,l,o,v=x["Close"],x["High"],x["Low"],x["Open"],x["Volume"]
    x["ret_1"]=c.pct_change()
    x["ret_5"]=c.pct_change(5)
    x["ret_20"]=c.pct_change(20)
    x["gap"]=o/c.shift(1)-1
    x["range_pct"]=(h-l)/c.replace(0,np.nan)
    x["body_pct"]=(c-o)/o.replace(0,np.nan)
    for n in (5,10,20,50,100,200):
        x[f"sma_{n}"]=c.rolling(n).mean()
        x[f"close_sma_{n}"]=c/x[f"sma_{n}"]-1
    for n in (12,26,50):
        x[f"ema_{n}"]=c.ewm(span=n,adjust=False).mean()
    x["rsi_14"]=_rsi(c,14)
    x["atr_14"]=_atr(x,14)
    x["atr_pct"]=x["atr_14"]/c
    ema12=c.ewm(span=12,adjust=False).mean()
    ema26=c.ewm(span=26,adjust=False).mean()
    x["macd"]=ema12-ema26
    x["macd_signal"]=x["macd"].ewm(span=9,adjust=False).mean()
    x["macd_hist"]=x["macd"]-x["macd_signal"]
    mid=c.rolling(20).mean()
    sd=c.rolling(20).std()
    upper,lower=mid+2*sd,mid-2*sd
    x["bb_width"]=(upper-lower)/mid.replace(0,np.nan)
    x["bb_pos"]=(c-lower)/(upper-lower).replace(0,np.nan)
    lo14=l.rolling(14).min()
    hi14=h.rolling(14).max()
    x["stoch_k"]=100*(c-lo14)/(hi14-lo14).replace(0,np.nan)
    x["stoch_d"]=x["stoch_k"].rolling(3).mean()
    for n in (5,10,20,60):
        x[f"volatility_{n}"]=x["ret_1"].rolling(n).std()*np.sqrt(252)
        x[f"volume_ratio_{n}"]=v/v.rolling(n).mean()
    x["dow"]=x.index.dayofweek
    x["month"]=x.index.month
    return x.replace([np.inf,-np.inf],np.nan)
