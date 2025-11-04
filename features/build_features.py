# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, yaml, json
from pathlib import Path
cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
BARS = cfg['files']['bars_csv']; FEAT = cfg['files']['features_csv']

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def bbands(series, period=20, n_std=2.0):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + n_std*sd; lower = ma - n_std*sd
    width = (upper - lower) / (ma.replace(0,np.nan).abs() + 1e-12)
    return ma, upper, lower, width

def add_features(df):
    df = df.sort_values('time').reset_index(drop=True)
    c = df['close']
    df['ret1'] = np.log(c).diff()
    df['ema10'] = c.ewm(span=10).mean(); df['ema50']=c.ewm(span=50).mean(); df['ema200']=c.ewm(span=200).mean()
    d = c.diff(); up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=up/(down+1e-12)
    df['rsi14'] = 100 - (100/(1+rs))
    h,l,cl = df['high'], df['low'], df['close']
    tr = np.maximum(h-l, np.maximum((h-cl.shift()).abs(), (l-cl.shift()).abs()))
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    m,s,hst = macd(c); df['macd']=m; df['macd_signal']=s; df['macd_hist']=hst
    bb_ma, bb_up, bb_lo, bb_w = bbands(c); df['bb_ma']=bb_ma; df['bb_up']=bb_up; df['bb_lo']=bb_lo; df['bb_width']=bb_w
    df['minute']=df['time'].dt.hour*60+df['time'].dt.minute
    df['tod_sin']=np.sin(2*np.pi*df['minute']/1440); df['tod_cos']=np.cos(2*np.pi*df['minute']/1440); df.drop(columns=['minute'], inplace=True)
    df['dow']=df['time'].dt.dayofweek
    df['dow_sin']=np.sin(2*np.pi*df['dow']/7); df['dow_cos']=np.cos(2*np.pi*df['dow']/7); df.drop(columns=['dow'], inplace=True)
    df['close_log']=np.log(c.clip(lower=1e-12))
    mu = df['close_log'].rolling(2000, min_periods=200).mean()
    sd = df['close_log'].rolling(2000, min_periods=200).std().replace(0,np.nan)
    df['close_norm']=(df['close_log']-mu)/(sd+1e-8)
    feat_cols=['ret1','ema10','ema50','ema200','rsi14','atr14','macd','macd_signal','macd_hist','bb_ma','bb_up','bb_lo','bb_width','tod_sin','tod_cos','dow_sin','dow_cos']
    for col in feat_cols:
        mu = df[col].rolling(2000, min_periods=200).mean()
        sd = df[col].rolling(2000, min_periods=200).std().replace(0,np.nan)
        df[col]=(df[col]-mu)/(sd+1e-8)
    return df.dropna().reset_index(drop=True)

df = pd.read_csv(BARS, parse_dates=['time'])
df = add_features(df)
assert len(df) > int(cfg.get('window',128)) + 10, "Too few rows for features"
Path(FEAT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(FEAT, index=False)
base_cols = ['time','open','high','low','close','tick_volume','spread']
feature_columns = [c for c in df.columns if c not in base_cols]
spec = {"feature_columns": feature_columns, "price_column":"close_norm"}
Path('models').mkdir(parents=True, exist_ok=True)
Path('models/features_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
print("Saved features and features_spec.json")
