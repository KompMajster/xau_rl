# -*- coding: utf-8 -*-
import yaml, pandas as pd
from env_xau import XauTradingEnv
from stable_baselines3 import PPO
from pathlib import Path

cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
FEAT = cfg['files']['features_csv']
MODEL_PATH = cfg['files']['model_path']
WINDOW = int(cfg.get('window',128))
COSTS = cfg['costs']

df = pd.read_csv(FEAT, parse_dates=['time'])
cut = df['time'].max() - pd.Timedelta(days=120)
sim = df[df['time']>=cut].copy()

env = XauTradingEnv(sim, window=WINDOW, spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'])
obs, info = env.reset(); model = PPO.load(MODEL_PATH)
rows=[]; done=False
while not done:
    action,_=model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated) or bool(truncated)
    rows.append({'time': info.get('time',None), 'price': info.get('price',None), 'action': int(action), 'reward': float(reward), 'equity': float(env.equity)})
Path('reports').mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv('reports/trace_simulated.csv', index=False)
pd.Series([r['equity'] for r in rows]).to_csv('reports/equity_simulated.csv', index=False)
print('Saved simulated trace/equity.')
