-- coding: utf-8 --
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import time as dtime
class XauTradingEnv(gym.Env):
metadata = {"render_modes": []}
def init(self, df: pd.DataFrame, window=128,
spread_abs=0.05, commission_rate=0.0001, slippage_k=0.10,
reward_mode: str = "pct", use_close_norm: bool = True,
flip_penalty: float = 0.0, trade_hours_utc=None,
enforce_flat_outside_hours: bool = True, features_spec: list | None = None,
min_equity: float = 0.8):
super().init()
self.df = df.reset_index(drop=True)
self.window = int(window)
self.spread_abs = float(spread_abs)
self.commission_rate = float(commission_rate)
self.slippage_k = float(slippage_k)
assert reward_mode in {"pct","points"}
self.reward_mode = reward_mode
self.use_close_norm = use_close_norm
self.flip_penalty = float(flip_penalty)
self.enforce_flat_outside = bool(enforce_flat_outside_hours)
self.min_equity = float(min_equity)
self.trade_hours = None
if trade_hours_utc and isinstance(trade_hours_utc,(list,tuple)) and len(trade_hours_utc)==2:
try:
s = [int(x) for x in str(trade_hours_utc[0]).split(":")]
e = [int(x) for x in str(trade_hours_utc[1]).split(":")]
self.trade_hours = (dtime(s[0], s[1] if len(s)>1 else 0), dtime(e[0], e[1] if len(e)>1 else 0))
except Exception:
self.trade_hours = None
base_cols = ['open','high','low','close','tick_volume','spread','time']
if features_spec is not None:
missing = [c for c in features_spec if c not in self.df.columns]
if missing: raise ValueError(f"Missing feature columns: {missing}")
self.feat_cols = list(features_spec)
else:
self.feat_cols = [c for c in self.df.columns if c not in base_cols]
self.price_col = 'close_norm' if (self.use_close_norm and 'close_norm' in self.df.columns) else 'close'
if len(self.df) <= self.window:
raise ValueError(f"Not enough rows: {len(self.df)} <= window {self.window}")
obs_dim = self.window * (1 + len(self.feat_cols)) + 2
self.action_space = spaces.Discrete(3)
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
self._start = self.window
self._i = None
self.pos = 0
self.entry = None
self.equity = 1.0
self.prev_eq = 1.0
def _in_trade_hours(self, ts)->bool:
if self.trade_hours is None: return True
try: t = ts.to_pydatetime().time()
except Exception: t = ts
start, end = self.trade_hours
if start <= end: return (t>=start) and (t<=end)
return (t>=start) or (t<=end)
def _obs(self):
sl = slice(self._i - self.window, self.i)
block_df = self.df.iloc[sl][[self.price_col] + self.feat_cols]
block = block_df.to_numpy(dtype=np.float32)
expected = 1 + len(self.feat_cols)
if block.shape != (self.window, expected):
raise RuntimeError(f"Bad window shape: {block.shape} vs {(self.window, expected)}")
flat = block.flatten()
price = float(self.df.iloc[self.i]['close'])
unreal = 0.0
if self.pos != 0 and self.entry is not None:
dir = 1 if self.pos>0 else -1
unreal = dir * (price - self.entry)
import numpy as np
return np.concatenate([flat, np.array([self.pos, unreal], dtype=np.float32)])
def reset(self, seed=None, options=None):
super().reset(seed=seed)
self._i = self._start
self.pos = 0; self.entry = None
self.equity = 1.0; self.prev_eq = 1.0
return self._obs(), {}
def step(self, action):
import numpy as np
if isinstance(action,(np.ndarray,list,tuple)): action = int(action[0])
else: action = int(action)
if not self.action_space.contains(action): raise ValueError("Invalid action")
info = {}
ts = self.df.iloc[self._i]['time']
inside = self._in_trade_hours(ts)
price = float(self.df.iloc[self.i]['close'])
prev = float(self.df.iloc[self.i-1]['close'])
slip = self.slippage_k * abs(price - prev)
desired = [-1,0,1][action]
if not inside and self.enforce_flat_outside: desired = 0
if desired != self.pos:
if self.pos != 0 and desired != 0 and np.sign(self.pos) != np.sign(desired) and self.flip_penalty>0:
if self.reward_mode == 'pct': self.equity *= max(1.0 - self.flip_penalty, 1e-6)
else: self.equity -= self.flip_penalty
info['flip_penalty'] = float(self.flip_penalty)
if self.pos != 0 and self.entry is not None:
cost = (self.spread_abs + slip)/max(price,1e-12) + self.commission_rate
if self.reward_mode == 'pct': self.equity *= max(1.0 - cost, 1e-6)
else: self.equity -= cost
self.pos = desired
if self.pos != 0:
cost = (self.spread_abs + slip)/max(price,1e-12) + self.commission_rate
if self.reward_mode == 'pct': self.equity *= max(1.0 - cost, 1e-6)
else: self.equity -= cost
self.entry = price
else:
self.entry = None
if self.reward_mode == 'pct':
step_ret = 0.0
if self.pos != 0:
dir = 1 if self.pos>0 else -1
step_ret = dir * ((price/max(prev,1e-12)) - 1.0)
self.equity *= (1.0 + step_ret)
reward = float(self.equity - self.prev_eq)
else:
reward = float(self.equity - self.prev_eq)
self.prev_eq = self.equity
self._i += 1
terminated = bool(self._i >= len(self.df) - 1)
truncated = False
if self.equity <= self.min_equity:
truncated = True
info['early_stop'] = True
info.update({'time': ts, 'price': price, 'pos': int(self.pos), 'equity': float(self.equity), 'inside_hours': bool(inside)})
return self._obs(), reward, terminated, truncated, info
