# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import pandas as pd, numpy as np, yaml, time, logging, json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.mt5_health import ensure_mt5_ready

logging.basicConfig(filename='paper_demo/paper_trading.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

cfg = yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
SYMBOL = cfg['symbol']; WINDOW=int(cfg.get('window',128))
MODEL_PATH = cfg['files']['model_path']; VECNORM_PATH = cfg.get('files',{}).get('vecnorm_path','models/vecnorm_xauusd_m5.pkl')
TF_MAP={'M1': mt5.TIMEFRAME_M1,'M5': mt5.TIMEFRAME_M5,'M15': mt5.TIMEFRAME_M15}; TF_NAME = cfg.get('timeframe','M5'); TF = TF_MAP.get(TF_NAME, mt5.TIMEFRAME_M5)

features_spec=None; fs=Path('models/features_spec.json')
if fs.exists(): features_spec=json.loads(fs.read_text(encoding='utf-8')).get('feature_columns', None)

def add_features_incremental(df: pd.DataFrame)->pd.DataFrame:
    df=df.sort_values('time').reset_index(drop=True)
    df['ret1']=np.log(df['close']).diff()
    df['ema10']=df['close'].ewm(span=10).mean(); df['ema50']=df['close'].ewm(span=50).mean(); df['ema200']=df['close'].ewm(span=200).mean()
    d=df['close'].diff(); up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=up/(down+1e-12)
    df['rsi14']=100-(100/(1+rs))
    h,l,c=df['high'],df['low'],df['close']; tr=np.maximum(h-l, np.maximum(abs(h-c.shift()), abs(l-c.shift())))
    df['atr14']=tr.ewm(alpha=1/14,adjust=False).mean()
    ema_fast=df['close'].ewm(span=12,adjust=False).mean(); ema_slow=df['close'].ewm(span=26,adjust=False).mean(); macd_line=ema_fast-ema_slow; signal_line=macd_line.ewm(span=9,adjust=False).mean()
    df['macd']=macd_line; df['macd_signal']=signal_line; df['macd_hist']=macd_line-signal_line
    ma=df['close'].rolling(20).mean(); sd=df['close'].rolling(20).std()
    df['bb_ma']=ma; df['bb_up']=ma+2*sd; df['bb_lo']=ma-2*sd; df['bb_width']=(df['bb_up']-df['bb_lo'])/(ma.replace(0,np.nan).abs()+1e-12)
    df['minute']=df['time'].dt.hour*60+df['time'].dt.minute; df['tod_sin']=np.sin(2*np.pi*df['minute']/1440); df['tod_cos']=np.cos(2*np.pi*df['minute']/1440); df.drop(columns=['minute'], inplace=True)
    df['dow']=df['time'].dt.dayofweek; df['dow_sin']=np.sin(2*np.pi*df['dow']/7); df['dow_cos']=np.cos(2*np.pi*df['dow']/7); df.drop(columns=['dow'], inplace=True)
    df['close_log']=np.log(df['close'].clip(lower=1e-12)); mu=df['close_log'].rolling(2000,min_periods=200).mean(); s=df['close_log'].rolling(2000,min_periods=200).std().replace(0,np.nan)
    df['close_norm']=(df['close_log']-mu)/(s+1e-8)
    norm=['ret1','ema10','ema50','ema200','rsi14','atr14','macd','macd_signal','macd_hist','bb_ma','bb_up','bb_lo','bb_width','tod_sin','tod_cos','dow_sin','dow_cos']
    for c in norm:
        mu=df[c].rolling(2000,min_periods=200).mean(); s=df[c].rolling(2000,min_periods=200).std().replace(0,np.nan); df[c]=(df[c]-mu)/(s+1e-8)
    return df.dropna().reset_index(drop=True)

def get_last_bars(symbol, timeframe, n:int):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates)<n: return None
    df=pd.DataFrame(rates); df['time']=pd.to_datetime(df['time'], unit='s', utc=True); df.rename(columns={'real_volume':'tick_volume'}, inplace=True)
    return df[['time','open','high','low','close','tick_volume','spread']]

def build_obs(df_feat: pd.DataFrame, model_obs_dim: int)->np.ndarray:
    per_step=(model_obs_dim-2)//WINDOW
    price_col='close_norm' if 'close_norm' in df_feat.columns else 'close'
    if features_spec is None:
        base=['open','high','low','close','tick_volume','spread','time']; feat_cols=[c for c in df_feat.columns if c not in base]
    else:
        feat_cols=[c for c in features_spec if c in df_feat.columns]
    tail=df_feat.tail(WINDOW); block=tail[[price_col]+feat_cols].to_numpy(dtype=np.float32)
    if block.shape!=(WINDOW, per_step): raise RuntimeError(f"Bad block {block.shape} vs {(WINDOW, per_step)}")
    flat=block.flatten(); import numpy as np
    return np.concatenate([flat, np.array([0.0,0.0],dtype=np.float32)])

def main():
    if not ensure_mt5_ready(): raise RuntimeError("MT5 not ready (terminal/account). Start MT5 and login to a demo account.")
    try:
        model = PPO.load(MODEL_PATH); expected_dim=int(model.observation_space.shape[0]); assert (expected_dim-2)%WINDOW==0
        vecnorm=None
        if Path(VECNORM_PATH).exists():
            from gymnasium import spaces
            class _ObsOnlyEnv:
                def __init__(self, obs_dim):
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                    self.action_space = spaces.Discrete(3)
                def reset(self, *, seed=None, options=None):
                    import numpy as np; return np.zeros(self.observation_space.shape, dtype=np.float32), {}
                def step(self, action):
                    import numpy as np; return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
            dummy=DummyVecEnv([lambda: _ObsOnlyEnv(expected_dim)])
            vecnorm=VecNormalize.load(VECNORM_PATH, dummy); vecnorm.training=False; vecnorm.norm_reward=False

        Path('paper_demo').mkdir(parents=True, exist_ok=True)
        csv=Path('paper_demo/decisions.csv'); if not csv.exists(): csv.write_text('time_utc,price,action_id,action_label\n', encoding='utf-8')

        last_ts=None; hb_t=time.time()
        while True:
            bars=get_last_bars(SYMBOL, TF, n=WINDOW+800)
            if bars is None or len(bars)<WINDOW+200: time.sleep(5); continue
            feat=add_features_incremental(bars)
            if len(feat)<WINDOW: time.sleep(5); continue
            cur_ts=feat['time'].iloc[-1]
            if last_ts is not None and cur_ts==last_ts: time.sleep(2); continue
            obs=build_obs(feat, expected_dim)
            if vecnorm is not None:
                import numpy as np; obs=vecnorm.normalize_obs(obs.reshape(1,-1)).reshape(-1)
            action,_=model.predict(obs, deterministic=True)
            decision={0:'SHORT',1:'FLAT',2:'LONG'}[int(action)]
            price=float(feat['close'].iloc[-1]); msg=f"DECISION {decision} @ {price:.2f} (ts={cur_ts})"
            print(msg); logging.info(msg)
            with open(csv,'a',encoding='utf-8') as f: f.write(f"{cur_ts},{price:.5f},{int(action)},{decision}\n")
            last_ts=cur_ts
            if time.time()-hb_t>600: logging.info(f"[HB] {SYMBOL} {TF_NAME} last_ts={cur_ts} price={price:.2f}"); hb_t=time.time()
            time.sleep(30)
    finally:
        mt5.shutdown()

if __name__ == '__main__': main()
