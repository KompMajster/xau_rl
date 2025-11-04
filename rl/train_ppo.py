# -*- coding: utf-8 -*-
import argparse, yaml, json
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, sync_envs_normalization
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import TimeLimit
from env_xau import XauTradingEnv

parser = argparse.ArgumentParser()
parser.add_argument('--timesteps', type=int, default=1500000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_freq', type=int, default=100000)
parser.add_argument('--n_eval_episodes', type=int, default=1)
parser.add_argument('--max_train_steps', type=int, default=6000)
parser.add_argument('--max_eval_steps', type=int, default=3000)
args = parser.parse_args()

cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
FEAT = cfg['files']['features_csv']
MODEL_PATH = Path(cfg['files']['model_path'])
VECNORM_PATH = Path(cfg.get('files',{}).get('vecnorm_path','models/vecnorm_xauusd_m5.pkl'))
WINDOW = int(cfg.get('window',128))
COSTS = cfg['costs']
ENV_CFG = cfg.get('env',{})
REWARD_MODE = ENV_CFG.get('reward_mode', cfg.get('reward_mode','pct'))
FLIP_PENALTY = float(ENV_CFG.get('flip_penalty', cfg.get('flip_penalty',0.0)))
TRADE_HOURS = ENV_CFG.get('trade_hours_utc', cfg.get('trade_hours_utc', None))
MIN_EQ = float(ENV_CFG.get('min_equity', 0.8))

df = pd.read_csv(FEAT, parse_dates=['time'])
cut_val = df['time'].max() - pd.Timedelta(days=30)
train_df = df[df['time'] < cut_val].copy()
val_df   = df[df['time'] >= cut_val].copy()

features_spec = None
fspec = Path('models/features_spec.json')
if fspec.exists():
    features_spec = json.loads(fspec.read_text(encoding='utf-8')).get('feature_columns', None)

set_random_seed(args.seed)

def make_train():
    env = XauTradingEnv(train_df, window=WINDOW,
        spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'],
        reward_mode=REWARD_MODE, use_close_norm=True, flip_penalty=FLIP_PENALTY, trade_hours_utc=TRADE_HOURS,
        enforce_flat_outside_hours=True, features_spec=features_spec, min_equity=MIN_EQ)
    return TimeLimit(env, max_episode_steps=args.max_train_steps)

def make_eval():
    env = XauTradingEnv(val_df, window=WINDOW,
        spread_abs=COSTS['spread_abs'], commission_rate=COSTS['commission_rate'], slippage_k=COSTS['slippage_k'],
        reward_mode=REWARD_MODE, use_close_norm=True, flip_penalty=0.0, trade_hours_utc=TRADE_HOURS,
        enforce_flat_outside_hours=True, features_spec=features_spec, min_equity=MIN_EQ)
    return TimeLimit(env, max_episode_steps=args.max_eval_steps)

venv_train = DummyVecEnv([make_train]); venv_train = VecMonitor(venv_train); venv_train = VecNormalize(venv_train, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
venv_eval  = DummyVecEnv([make_eval]);  venv_eval  = VecMonitor(venv_eval);  venv_eval  = VecNormalize(venv_eval, training=False, norm_obs=True, norm_reward=False)

policy_kwargs = dict(net_arch=dict(pi=[128,128], vf=[128,128]))
try:
    import tensorboard as _tb
    tb_log_dir = 'logs/ppo_gold_m5'
except Exception:
    tb_log_dir = None

model = PPO('MlpPolicy', venv_train, n_steps=4096, batch_size=256, learning_rate=3e-4, ent_coef=0.02,
            policy_kwargs=policy_kwargs, seed=args.seed, verbose=1, tensorboard_log=tb_log_dir)

sync_envs_normalization(venv_eval, venv_train)
eval_cb = EvalCallback(venv_eval, best_model_save_path=str(MODEL_PATH.parent), log_path='logs/eval',
                       eval_freq=max(args.eval_freq,1), n_eval_episodes=max(args.n_eval_episodes,1),
                       deterministic=True, render=False)

model.learn(total_timesteps=args.timesteps, callback=eval_cb)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(str(MODEL_PATH))
venv_train.save(str(VECNORM_PATH))
print(f"Saved model: {MODEL_PATH}")
print(f"Saved VecNormalize: {VECNORM_PATH}")
