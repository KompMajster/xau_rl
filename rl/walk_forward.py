# -*- coding: utf-8 -*-
import argparse, yaml, json
import pandas as pd, numpy as np
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from env_xau import XauTradingEnv

parser = argparse.ArgumentParser()
parser.add_argument('--segments', type=int, default=6)
parser.add_argument('--train_days', type=int, default=120)
parser.add_argument('--val_days', type=int, default=30)
parser.add_argument('--timesteps', type=int, default=500000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='reports_wf')
args = parser.parse_args()

cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
FEAT = cfg['files']['features_csv']
WINDOW = int(cfg.get('window',128))
COSTS = cfg['costs']
ENV_CFG = cfg.get('env',{})
REWARD_MODE = ENV_CFG.get('reward_mode', cfg.get('reward_mode','pct'))
FLIP_PENALTY = float(ENV_CFG.get('flip_penalty', cfg.get('flip_penalty',0.0)))
TRADE_HOURS = ENV_CFG.get('trade_hours_utc', cfg.get('trade_hours_utc', None))
MIN_EQ = float(ENV_CFG.get('min_equity', 0.8))

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(FEAT, parse_dates=['time']).sort_values('time').reset_index(drop=True)
end_time = df['time'].max()

features_spec=None
fspec = Path('models/features_spec.json')
if fspec.exists(): features_spec = json.loads(fspec.read_text(encoding='utf-8')).get('feature_columns', None)

rows=[]
for k in range(args.segments):
    val_end = end_time - pd.Timedelta(days=k*args.val_days)
    val_start = val_end - pd.Timedelta(days=args.val_days)
    train_end = val_start
    train_start = train_end - pd.Timedelta(days=args.train_days)
    tr = df[(df['time']>=train_start)&(df['time']<train_end)].copy()
    va = df[(df['time']>=val_start)&(df['time']<val_end)].copy()
    if len(tr)<=WINDOW or len(va)<=WINDOW: continue

    def make_train():
        env = XauTradingEnv(tr, window=WINDOW, spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'],
            reward_mode=REWARD_MODE, use_close_norm=True, flip_penalty=FLIP_PENALTY, trade_hours_utc=TRADE_HOURS,
            enforce_flat_outside_hours=True, features_spec=features_spec, min_equity=MIN_EQ)
        return TimeLimit(env, max_episode_steps=6000)

    def make_eval():
        env = XauTradingEnv(va, window=WINDOW, spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'],
            reward_mode=REWARD_MODE, use_close_norm=True, flip_penalty=0.0, trade_hours_utc=TRADE_HOURS,
            enforce_flat_outside_hours=True, features_spec=features_spec, min_equity=MIN_EQ)
        return TimeLimit(env, max_episode_steps=3000)

    vt = DummyVecEnv([make_train]); vt = VecMonitor(vt); vt = VecNormalize(vt, norm_obs=True, norm_reward=True)
    ve = DummyVecEnv([make_eval]);  ve = VecMonitor(ve)

    model = PPO('MlpPolicy', vt, n_steps=4096, batch_size=256, learning_rate=3e-4, ent_coef=0.02, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps)

    tmp = out/f'vecnorm_{k}.pkl'; vt.save(str(tmp))
    ve = VecNormalize.load(str(tmp), ve); ve.training=False; ve.norm_reward=False

    obs = ve.reset(); eq=[]; trace=[]; done=False
    while not done:
        action,_=model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = ve.step(action)
        equity = ve.get_attr('equity', indices=0)[0]
        eq.append(float(equity))
        info0=infos[0] if isinstance(infos,(list,tuple)) else infos
        trace.append({'k':k,'time':str(info0.get('time','')),'price':info0.get('price',np.nan),'pos':info0.get('pos',np.nan),'equity':equity,
                      'action':int(action[0]) if hasattr(action,'__len__') else int(action)})
        done = bool(dones[0])

    s = pd.Series(eq); final_eq=float(s.iloc[-1]); min_eq=float(s.min()); peak=s.cummax(); max_dd=float((s/peak-1.0).min())
    rows.append({'k':k,'train_start':train_start,'train_end':train_end,'val_start':val_start,'val_end':val_end,'final_eq':final_eq,'min_eq':min_eq,'max_dd':max_dd,'n_steps':int(len(s))})
    pd.DataFrame(trace).to_csv(out/f'fold_{k}_trace.csv', index=False)

res = pd.DataFrame(rows).sort_values('k'); res.to_csv(out/'wf_results.csv', index=False)
if res.empty:
    txt=['# Walk-Forward Report','No results (windows too short?).']
else:
    agg={'folds':len(res),'final_eq_med':float(res['final_eq'].median()),'max_dd_med':float(res['max_dd'].median()),'n_steps_sum':int(res['n_steps'].sum())}
    txt=['# Walk-Forward Report',f"Folds: {agg['folds']}",f"Median Final Equity: {agg['final_eq_med']:.4f}",f"Median MaxDD: {agg['max_dd_med']:.2%}",f"Total steps: {agg['n_steps_sum']}"]
(out/'wf_report.txt').write_text('\n'.join(txt), encoding='utf-8')
print("Saved walk-forward results.")
