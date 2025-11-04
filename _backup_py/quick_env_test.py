import yaml, pandas as pd
from env_xau import XauTradingEnv
cfg=yaml.safe_load(open('config.yaml','r',encoding='utf-8'))
df=pd.read_csv(cfg['files']['features_csv'], parse_dates=['time']).sort_values('time').reset_index(drop=True)
df=df.tail(max(int(cfg.get('window',128))+256,512))
env=XauTradingEnv(df, window=int(cfg.get('window',128)),
spread_abs=cfg['costs']['spread_abs'], commission_rate=cfg['costs']['commission_rate'], slippage_k=cfg['costs']['slippage_k'],
reward_mode=cfg.get('env',{}).get('reward_mode', cfg.get('reward_mode','pct')), use_close_norm=True,
min_equity=float(cfg.get('env',{}).get('min_equity',0.8)))
obs,info=env.reset(); print('reset OK', len(obs))
obs,r,term,trunc,info=env.step(1); print('step OK', r, term, trunc)
