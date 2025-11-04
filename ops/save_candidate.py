import json, shutil
from pathlib import Path
ROOT = Path(file).resolve().parents[1]
REG = ROOT / 'models' / 'registry'; REG.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / 'models' / 'ppo_xauusd_m5.zip'
VEC   = ROOT / 'models' / 'vecnorm_xauusd_m5.pkl'
METR  = ROOT / 'reports' / 'val_metrics.txt'
EQP   = ROOT / 'reports' / 'equity_val.png'
def parse_metrics_txt(p: Path):
d = {}
if p.exists():
txt = p.read_text(encoding='utf-8')
for line in txt.splitlines():
if ':' in line:
k, v = line.split(':', 1)
k = k.strip().lower().replace(' ', '')
try: d[k] = float(v.strip())
except: pass
return d
def main():
m = parse_metrics_txt(METR)
if not m:
print('[save_candidate] No metrics.'); return
tag = 'cand' + Path.cwd().name + '_'
n=0
while (REG / f"{tag}{n:02d}").exists(): n+=1
dst = REG / f"{tag}{n:02d}"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(MODEL, dst/'model.zip')
if VEC.exists(): shutil.copy2(VEC, dst/'vecnorm.pkl')
if EQP.exists(): shutil.copy2(EQP, dst/'equity.png')
(dst/'metrics.json').write_text(json.dumps({
'final_equity': m.get('final_equity'),
'max_dd': m.get('max_drawdown'),
'sharpe': m.get('sharpe_daily'),
}, indent=2), encoding='utf-8')
print(f'[save_candidate] Saved candidate -> {dst}')
if name == 'main': main()
