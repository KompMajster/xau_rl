# -*- coding: utf-8 -*-
import argparse, yaml, pandas as pd, numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true'); args=parser.parse_args()
cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
TICKS = cfg['files']['ticks_csv']; BARS = cfg['files']['bars_csv']

ticks = pd.read_csv(TICKS, parse_dates=['time'])
ticks['spread']=ticks['ask']-ticks['bid']; mid=(ticks['ask']+ticks['bid'])/2.0; dmid=mid.diff().abs()
spread_median=float(ticks['spread'].median()); spread_p95=float(ticks['spread'].quantile(0.95)); dmid_median=float(dmid.median())

bars=pd.read_csv(BARS, parse_dates=['time']); bar_move_med=float(bars['close'].diff().abs().median())
slippage_k=float(np.clip(dmid_median/max(bar_move_med,1e-12), 0.0, 0.5))

Path('reports').mkdir(parents=True, exist_ok=True)
Path('reports/costs_suggestion.yaml').write_text(
    yaml.safe_dump({'spread_abs_median':round(spread_median,5),'spread_abs_p95':round(spread_p95,5),'slippage_k_suggested':round(slipage_k if False else slippage_k,3)},
                   allow_unicode=True, sort_keys=False),
    encoding='utf-8')
print('_min', type=int, default=5)
args=parser.parse_args()
df=pd.read_csv(args.csv, parse_dates=['time']).sort_values('time').reset_index(drop=True)
print(f"Rows: {len(df)}\nFrom: {df['time'].iloc[0]} To: {df['time'].iloc[-1]}")
dt=(df['time'].diff().dt.total_seconds()/60).fillna(args.tf_min)
gaps=df.loc[dt>args.tf_min+0.1,['time']].copy(); gaps['gap_min']=dt[dt>args.tf_min+0.1].values
print(f"Gaps > {args.tf_min} min: {len(gaps)}")
if len(gaps): print(gaps.head(10))
