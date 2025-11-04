# XAUUSD RL PPO (M5) – Starter (Demo/Edu)
**Uwaga**: tylko do celów edukacyjnych, backtestów i *paper tradingu* (demo). Brak kodu wysyłającego realne zlecenia.

## Szybki start
```powershell
python -m venv .venv
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Na końcu zainstaluj TORCH odpowiedni dla CPU/GPU z pytorch.org (wheel).
python fetch__mt5_data.py
python utils\calibration.py
python utils\calibration.py --apply
python features\build_features.py
python rl\train_ppo.py --timesteps 1500000
python rl\evaluate.py --out_dir reports
python utils\make_report.py

24/7 na Windows
Użyj skryptów w ops\ oraz NSSM (patrz ops\install_services.ps1).
Dane i historia
Trzymamy 720 dni historii świec. Zbieracz działa inkrementalnie: dociąga ostatnie update_days (domyślnie 60), scala z istniejacym CSV i przycina do 720 dni. Zapisy CSV są atomowe.
