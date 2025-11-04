import json, shutil
from pathlib import Path
ROOT = Path(file).resolve().parents[1]
REG = ROOT / 'models' / 'registry'; REG.mkdir(parents=True, exist_ok=True)
CUR_TXT = REG / 'current.txt'
MODEL_DST = ROOT / 'models' / 'ppo_xauusd_m5.zip'
VEC_DST   = ROOT / 'models' / 'vecnorm_xauusd_m5.pkl'
def load_metrics(d: Path):
if not d: return None
m = d / 'metrics.json'
return json.loads(m.read_text('utf-8')) if m.exists() else None
def pick_latest_dir():
dirs = [p for p in REG.iterdir() if p.is_dir()]
return sorted(dirs, key=lambda p: p.name)[-1] if dirs else None
def better(m_new, m_old):
if m_old is None: return True
s_new, s_old = m_new.get('sharpe', 0.0) or 0.0, m_old.get('sharpe', 0.0) or 0.0
dd_new, dd_old = m_new.get('max_dd', -1.0) or -1.0, m_old.get('max_dd', -1.0) or -1.0
dd_ok = (dd_new >= dd_old * 1.2)
s_ok  = (s_new >= s_old * 1.10) or (s_new >= 0.10 and s_new > s_old)
fe_new, fe_old = m_new.get('final_equity', 0.0) or 0.0, m_old.get('final_equity', 0.0) or 0.0
if (s_new == 0.0 and s_old == 0.0):
return (fe_new > fe_old) and dd_ok
return s_ok and dd_ok
def promote(new_dir: Path):
REG.mkdir(parents=True, exist_ok=True)
(REG/'current.txt').write_text(new_dir.name, encoding='utf-8')
shutil.copy2(new_dir/'model.zip', MODEL_DST)
if (new_dir/'vecnorm.pkl').exists(): shutil.copy2(new_dir/'vecnorm.pkl', VEC_DST)
print(f"[promote] Champion -> {new_dir.name}")
def main():
latest = pick_latest_dir()
if latest is None: print('[promote] No candidates.'); return
new_m = load_metrics(latest)
if new_m is None: print('[promote] Latest has no metrics.json'); return
cur_dir = REG / CUR_TXT.read_text('utf-8').strip() if CUR_TXT.exists() else None
old_m = load_metrics(cur_dir) if cur_dir and cur_dir.exists() else None
if better(new_m, old_m): promote(latest)
else: print('[promote] Challenger not better. No promotion.')
if name == 'main': main()
