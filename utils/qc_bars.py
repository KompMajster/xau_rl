#!/usr/bin/env python
import pandas as pd, argparse
parser=argparse.ArgumentParser(); parser.add_argument('--csv', required=True); parser.add_argument('--tf_min', type=int, default=5)
args=parser.parse_args()
df=pd.read_csv(args.csv, parse_dates=['time']).sort_values('time').reset_index(drop=True)
print(f"Rows: {len(df)}\nFrom: {df['time'].iloc[0]} To: {df['time'].iloc[-1]}")
dt=(df['time'].diff().dt.total_seconds()/60).fillna(args.tf_min)
gaps=df.loc[dt>args.tf_min+0.1,['time']].copy(); gaps['gap_min']=dt[dt>args.tf_min+0.1].values
print(f"Gaps > {args.tf_min} min: {len(gaps)}");
if len(gaps): print(gaps.head(10))
