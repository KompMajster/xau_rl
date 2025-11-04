# -*- coding: utf-8 -*-
"""
Fetch M5 bars & 24h tick sample from MT5 (demo/real).
- Utrzymuje ~keep_days (np. 720) dni historii (inkrementalne dobijanie + przycinanie).
- Atomiczne zapisy CSV.
- Weekend-safe ticki (ostrzeżenie zamiast błędu).
- Obejścia dla brokerów z 'Invalid params' na copy_rates_range:
  * daty 'naive' (bez tzinfo),
  * fallback do copy_rates_from_pos (tail od końca),
  * chunkowanie przy pierwszym pobraniu.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import time
import yaml

# nasze utilsy
from utils.atomic import atomic_write_csv
from utils.mt5_health import ensure_mt5_ready


# -------------------------
# Konfiguracja / walidacja
# -------------------------
CFG_PATH = 'config.yaml'
cfg = yaml.safe_load(open(CFG_PATH, 'r', encoding='utf-8'))
if not isinstance(cfg, dict) or 'files' not in cfg:
    raise RuntimeError(f"{CFG_PATH} jest pusty lub ma złą strukturę (brak sekcji 'files').")

SYMBOL     = cfg.get('symbol', 'XAUUSD')
BARS_CSV   = cfg['files']['bars_csv']
TICKS_CSV  = cfg['files']['ticks_csv']
HISTORY_DAYS = int(cfg.get('history_days', 720))

TF_MAP  = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15}
TF_NAME = cfg.get('timeframe', 'M5')
TF      = TF_MAP.get(TF_NAME, mt5.TIMEFRAME_M5)

MT5_CFG     = cfg.get('mt5', {}) or {}
CHUNK       = int(MT5_CFG.get('bars_chunk', 20000))   # 20k ~ 70–90 dni na M5 (zależnie od brokera)
UPDATE_DAYS = int(MT5_CFG.get('update_days', 60))


# -------------------------
# MT5 init / symbol utils
# -------------------------
def init_mt5():
    """Initialize MT5 and print short info about terminal/account."""
    if not ensure_mt5_ready():
        last_err = mt5.last_error()
        raise RuntimeError(f"MT5 not ready (terminal/account). Start MT5 and login. last_error={last_err}")

    term = mt5.terminal_info()
    acc  = mt5.account_info()
    parts = []
    if getattr(term, "company", None): parts.append(f"Company={term.company}")
    if getattr(term, "name", None):    parts.append(f"TerminalName={term.name}")
    if getattr(acc, "login", 0):       parts.append(f"Login={acc.login}")
    if getattr(acc, "server", None):   parts.append(f"Server={acc.server}")
    if getattr(acc, "name", None):     parts.append(f"AccountName={acc.name}")
    print("[MT5] " + " | ".join(parts) if parts else "[MT5] initialized")


def ensure_symbol(symbol: str) -> bool:
    """Upewnij się, że symbol jest widoczny w Market Watch."""
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    if info is None:
        return False
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            return False
    return True


def suggest_similar(symbol: str, limit=12):
    """Podpowiedz nazwy symboli powiązanych z GOLD/XAU (serwer‑specyficzne)."""
    out = []
    for s in mt5.symbols_get():
        nm = s.name.upper()
        if ("GOLD" in nm) or ("XAU" in nm):
            out.append(s.name)
    if not out:
        out = [s.name for s in mt5.symbols_get()[:limit]]
    return sorted(set(out))[:limit]


# -------------------------
# Pomocnicze
# -------------------------
def with_retries(fn, attempts=3, sleep_s=2, *a, **kw):
    last = None
    for i in range(attempts):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    if last:
        raise last


def _to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    if 'real_volume' in df.columns:
        df.rename(columns={'real_volume': 'tick_volume'}, inplace=True)
    cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
    return df[cols].sort_values('time').drop_duplicates('time')


def _make_naive(dt: datetime) -> datetime:
    """Zwróć 'naive' datetime (bez tzinfo) – część brokerów tego wymaga dla copy_rates_range."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


# -------------------------
# Pobieranie danych
# -------------------------
def fetch_range(symbol, timeframe, utc_from: datetime, utc_to: datetime) -> pd.DataFrame:
    """
    Range z obejściami:
      - daty 'naive' (bez tzinfo),
      - fallback: tail od końca jeśli range zwróci pustkę/Invalid params.
    """
    frm_n = _make_naive(utc_from)
    to_n  = _make_naive(utc_to)

    rates = mt5.copy_rates_range(symbol, timeframe, frm_n, to_n)
    if rates is None or len(rates) == 0:
        # fallback: tail od końca
        tail = mt5.copy_rates_from_pos(symbol, timeframe, 0, CHUNK)
        if tail is None or len(tail) == 0:
            last_err = mt5.last_error()
            raise RuntimeError(f"copy_rates_range empty for '{symbol}'. last_error={last_err}")
        return _to_df(tail)
    return _to_df(rates)


def fetch_from_pos_tail(symbol, timeframe, count) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        last_err = mt5.last_error()
        raise RuntimeError(f"copy_rates_from_pos empty for '{symbol}'. last_error={last_err}")
    return _to_df(rates)


def load_existing_bars(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['time'])
        if df.empty:
            return None
        return df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    except Exception:
        return None


def merge_clip(existing: pd.DataFrame | None, new_df: pd.DataFrame, keep_days: int) -> pd.DataFrame:
    if existing is None or existing.empty:
        base = new_df.copy()
    else:
        base = (
            pd.concat([existing, new_df], ignore_index=True)
              .drop_duplicates('time')
              .sort_values('time')
        )
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days + 1)
    base = base[base['time'] >= pd.Timestamp(cutoff)]
    return base.reset_index(drop=True)


def fetch_bars_incremental(symbol, timeframe, keep_days: int, update_days: int) -> pd.DataFrame:
    """
    Strategia:
      * Gdy brak pliku – zbierzemy historię 'od końca' (copy_rates_from_pos) w 1–4 chunkach,
        aż pokryjemy ~keep_days (z marginesem). Jeśli to się nie uda – spróbujemy krótkiego range.
      * Gdy plik istnieje – dociągniemy ostatnie update_days (mały range z fallbackiem).
    """
    existing = load_existing_bars(BARS_CSV)

    if existing is None:
        wanted_days = keep_days + 2
        parts_df = None

        # 1–4 podejścia z narastającą liczbą świec (od końca)
        for i in range(1, 5):
            count = CHUNK * i
            try:
                dft = with_retries(lambda: fetch_from_pos_tail(symbol, timeframe, count))
            except Exception:
                # jeżeli tail zwróci pustkę (czasem zaraz po zalogowaniu) – krótka pauza i spróbuj dalej
                time.sleep(1.0)
                continue

            if dft is not None and len(dft) > 0:
                parts_df = dft
                span_days = (parts_df['time'].max() - parts_df['time'].min()).days
                if span_days >= wanted_days:
                    break

        if parts_df is None or parts_df.empty:
            # Ostatnia próba – krótki range (np. 60 dni)
            to  = datetime.utcnow()
            frm = to - timedelta(days=min(60, wanted_days))
            parts_df = fetch_range(symbol, timeframe, frm, to)

        return merge_clip(None, parts_df, keep_days)

    # inkrementalnie: dociągnij update_days (mały range z fallbackiem)
    to  = datetime.utcnow()
    frm = to - timedelta(days=max(2, update_days))
    fresh = fetch_range(symbol, timeframe, frm, to)
    merged = merge_clip(existing, fresh, keep_days)
    return merged


def fetch_ticks(symbol: str, hours: int = 24) -> pd.DataFrame:
    utc_to   = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(hours=hours)
    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        last_err = mt5.last_error()
        raise RuntimeError(f"No MT5 ticks for '{symbol}'. last_error={last_err}")
    tdf = pd.DataFrame(ticks)
    tdf['time'] = pd.to_datetime(tdf['time'], unit='s', utc=True)
    return tdf[['time', 'bid', 'ask', 'last', 'volume']]


def suggest_costs(tdf: pd.DataFrame) -> dict:
    spr = (tdf['ask'] - tdf['bid']).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    med_spread = float(np.median(spr)) if len(spr) else 0.0
    p75_spread = float(np.percentile(spr, 75)) if len(spr) else 0.0

    mid = (tdf['ask'] + tdf['bid']) / 2.0
    dm  = (mid.diff().abs()).replace([np.inf, -np.inf], np.nan).dropna()
    med_dm    = float(np.median(dm)) if len(dm) else 0.0
    med_price = float(np.nanmedian(mid)) if len(mid) else 1.0
    slippage_k = float(min(max(med_dm / max(med_price, 1e-12), 0.0), 0.01))
    return {
        'spread_abs_median':       round(med_spread, 5),
        'spread_abs_p75':          round(p75_spread, 5),
        'slippage_k_suggested':    round(slippage_k, 4)
    }


# -------------------------
# Main
# -------------------------
def main():
    init_mt5()
    try:
        print(f"[CFG] symbol={SYMBOL} tf={TF_NAME} keep_days={HISTORY_DAYS} update_days={UPDATE_DAYS}")

        if not ensure_symbol(SYMBOL):
            similar = suggest_similar(SYMBOL)
            raise SystemExit(
                "Symbol not found or not visible in Market Watch: '{}'\n"
                "Try one of these (server-specific): {}\n"
                "Also open Symbols window and SHOW the instrument."
                .format(SYMBOL, ", ".join(similar))
            )

        # ŚWIECE
        bars = fetch_bars_incremental(SYMBOL, TF, keep_days=HISTORY_DAYS, update_days=UPDATE_DAYS)
        assert {'time', 'open', 'high', 'low', 'close'}.issubset(bars.columns), "Bars frame malformed"
        assert len(bars) > 100, "Too few bars"
        atomic_write_csv(bars, BARS_CSV)
        print(f"Saved bars: {BARS_CSV} ({len(bars)})")

        # TICKI (opcjonalnie; weekend-safe)
        try:
            ticks = with_retries(lambda: fetch_ticks(SYMBOL, hours=24))
            atomic_write_csv(ticks, TICKS_CSV)
            print(f"Saved ticks: {TICKS_CSV} ({len(ticks)})")

            sugg = suggest_costs(ticks)
            Path('reports').mkdir(parents=True, exist_ok=True)
            Path('reports/costs_suggestion.yaml').write_text(
                yaml.safe_dump(sugg, allow_unicode=True, sort_keys=False),
                encoding='utf-8'
            )
            print("Cost suggestions -> reports/costs_suggestion.yaml")

        except Exception as e:
            # np. weekend: brak ticków (OK)
            print(f"[warn] ticks unavailable ({e}); keeping previous {TICKS_CSV}")

    finally:
        mt5.shutdown()


if __name__ == '__main__':
    main()