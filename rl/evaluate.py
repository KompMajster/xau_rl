# -*- coding: utf-8 -*-
import argparse, yaml, json
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from env_xau import XauTradingEnv

parser = argparse.ArgumentParser()
parser.add_argument('--max_eval_steps', type=int, default=3000)
parser.add_argument('--out_dir', type=str, default='reports')
args = parser.parse_args()

OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
FEAT = cfg['files']['features_csv']
MODEL_PATH = cfg['files']['model_path']
VECNORM_PATH = cfg.get('files',{}).get('vecnorm_path','models/vecnorm_xauusd_m5.pkl')
WINDOW = int(cfg.get('window',128))
COSTS = cfg['costs']
ENV_CFG = cfg.get('env',{})
REWARD_MODE = ENV_CFG.get('reward_mode', cfg.get('reward_mode','pct'))
MIN_EQ = float(ENV_CFG.get('min_equity', 0.8))

df = pd.read_csv(FEAT, parse_dates=['time'])
cut_val = df['time'].max() - pd.Timedelta(days=120)
val = df[df['time']>=cut_val].copy()

features_spec = None
fspec = Path('models/features_spec.json')
if fspec.exists():
    features_spec = json.loads(fspec.read_text(encoding='utf-8')).get('feature_columns', None)

def make_eval():
    env = XauTradingEnv(val, window=WINDOW,
        spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'],
        reward_mode=REWARD_MODE, use_close_norm=True, features_spec=features_spec, min_equity=MIN_EQ)
    return TimeLimit(env, max_episode_steps=args.max_eval_steps)

venv = DummyVecEnv([make_eval]); venv = VecMonitor(venv)
if Path(VECNORM_PATH).exists():
    venv = VecNormalize.load(VECNORM_PATH, venv); venv.training=False; venv.norm_reward=False
model = PPO.load(MODEL_PATH, env=venv)

obs = venv.reset(); eq=[]; rows=[]; done=False
while not done:
    action,_ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = venv.step(action)
    equity = venv.get_attr('equity', indices=0)[0]
    eq.append(float(equity))
    info0 = infos[0] if isinstance(infos,(list,tuple)) else infos
    rows.append({'time': str(info0.get('time','')), 'price': info0.get('price',np.nan), 'pos': info0.get('pos',np.nan),
                 'equity': equity, 'action': int(action[0]) if hasattr(action,'__len__') else int(action)})
    done = bool(dones[0])

s = pd.Series(eq)
final_eq = float(s.iloc[-1]); min_eq=float(s.min()); peak=s.cummax(); max_dd=float((s/peak-1.0).min())
plt.figure(figsize=(10,4)); plt.plot(s.values); plt.title('Equity (validation)'); plt.tight_layout()
plt.savefig((OUT/'equity_val.png').as_posix(), dpi=150)
(OUT/'val_metrics.txt').write_text(
    f"Final equity: {final_eq:.6f}\nMin equity: {min_eq:.6f}\nMax drawdown: {max_dd:.6f}\n", encoding='utf-8')
pd.DataFrame(rows).to_csv(OUT/'eval_trace.csv', index=False)
print("Saved validation results.")
