# -*- coding: utf-8 -*-
import argparse, base64
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--equity_csv', type=str, default='reports/eval_trace.csv')
parser.add_argument('--metrics', type=str, default='reports/val_metrics.txt')
parser.add_argument('--figure', type=str, default='reports/equity_val.png')
parser.add_argument('--out_html', type=str, default='reports/report.html')
parser.add_argument('--out_txt', type=str, default='reports/report.txt')
parser.add_argument('--bars_per_day', type=int, default=288)
parser.add_argument('--trading_days', type=int, default=252)
args = parser.parse_args()

out_html = Path(args.out_html)
out_txt  = Path(args.out_txt)
out_html.parent.mkdir(parents=True, exist_ok=True)
out_txt.parent.mkdir(parents=True, exist_ok=True)

eq = None
p = Path(args.equity_csv)
if p.exists():
    df = pd.read_csv(p)
    if 'equity' in df.columns:
        eq = df['equity'].astype(float).to_numpy()
    elif df.shape[1] == 1:
        eq = df.iloc[:, 0].astype(float).to_numpy()

metrics_text = Path(args.metrics).read_text(encoding='utf-8') if Path(args.metrics).exists() else ''

img_b64 = ''
if Path(args.figure).exists():
    img_b64 = base64.b64encode(Path(args.figure).read_bytes()).decode('ascii')

# Jeśli nie ma obrazka, wygeneruj go z danych equity
if not img_b64 and eq is not None and len(eq) > 1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eq, color='#0057B8', lw=1.5)
    ax.set_title('Equity (validation)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Equity')
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=144)
    plt.close(fig)
    img_b64 = base64.b64encode(Path(args.figure).read_bytes()).decode('ascii')

# Dodatkowe metryki
extra = {}
if eq is not None and len(eq) > 2:
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    mu = float(np.nanmean(ret))
    sigma = float(np.nanstd(ret) + 1e-12)
    downside = ret[ret < 0]
    ds = float(np.nanstd(downside) + 1e-12)

    daily_mu = mu * args.bars_per_day
    daily_sigma = sigma * (args.bars_per_day ** 0.5)
    sharpe = daily_mu / (daily_sigma + 1e-12)
    sortino = daily_mu / (((args.bars_per_day ** 0.5) * ds) + 1e-12)

    n = len(eq)
    years = max(n / args.bars_per_day / args.trading_days, 1e-6)
    cagr = (eq[-1] / max(eq[0], 1e-12)) ** (1 / years) - 1 if years > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1.0).min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float('inf')

    extra = dict(
        n_steps=n,
        sharpe_daily=sharpe,
        sortino_daily=sortino,
        vol_daily=daily_sigma,
        cagr=cagr,
        max_dd=max_dd,
        calmar=calmar
    )

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

md = [
    '# RL Validation Report (XAUUSD M5)',
    f'Date: {now}',
    '',
    '## Base metrics',
    '',
    metrics_text.strip() if metrics_text else '(no val_metrics.txt)',
    '',
    ''
]

if extra:
    md += [
        '## Extra metrics from equity',
        f"- Steps: {extra['n_steps']}",
        f"- Sharpe (daily): {extra['sharpe_daily']:.3f}",
        f"- Sortino (daily): {extra['sortino_daily']:.3f}",
        f"- Daily vol: {extra['vol_daily']:.4f}",
        f"- CAGR (est.): {extra['cagr']:.3%}",
        f"- MaxDD: {extra['max_dd']:.2%}",
        f"- Calmar: {extra['calmar']:.3f}",
    ]
else:
    md += ['No equity data.']

out_txt.write_text('\n'.join(md) + '\n', encoding='utf-8')

html = f"""RL Validation
RL Validation Report (XAUUSD M5)
Base metrics
{metrics_text if metrics_text else '(no val_metrics.txt)'}
Extra metrics
{('\n'.join(md)) if extra else 'No equity data.'}
Equity
{('data:image/png;base64,' + img_b64) if img_b64 else 'No plot.'}
Generated: {now}
"""
out_html.write_text(html, encoding='utf-8')
print('Saved report.')