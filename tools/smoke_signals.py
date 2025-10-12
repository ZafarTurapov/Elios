# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, os, argparse

# ensure project root on sys.path
if "/root/stockbot" not in sys.path:
    sys.path.insert(0, "/root/stockbot")

from core.trading.signals.config import SignalsCfg
from core.trading.signals.providers import get_default_providers
from core.trading.signals.features import (
    compute_daily_features,
    long_upper_wick,
)
from core.trading.signals.filters import (
    tradable_guard,
    vwap_orh_reclaim,
)
# Берём _fetch_history из движка, чтобы совпадали источники
from core.trading.signal_engine import _fetch_history

def main():
    ap = argparse.ArgumentParser(description="Elios Smoke Test — providers + features + filters")
    ap.add_argument("--symbols", "-s", required=True, help="Список тикеров через запятую: AAPL,TSLA,AMD")
    ap.add_argument("--days", type=int, default=45, help="Сколько дней истории тянуть (default: 45)")
    args = ap.parse_args()

    cfg = SignalsCfg.from_env()
    providers = get_default_providers(cfg)

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        print("Нет тикеров. Пример: --symbols AAPL,TSLA")
        sys.exit(1)

    print(f"⚙️  Yahoo enabled: {providers.yahoo.enabled} | Alpaca feed: {providers.alpaca.cfg.feed}")
    print(f"🧪 Smoke на {len(syms)} тикеров: {', '.join(syms)}")
    print("-" * 80)
    print(f"ℹ️  Если внешние данные недоступны, положи CSV с колонками Open,High,Low,Close,Volume сюда: {os.getenv('ELIOS_LOCAL_HISTORY_DIR','data/history')}/<TICKER>.csv")

    for sym in syms:
        print(f"\n🔎 {sym}")
        # 1) tradable?
        td = tradable_guard(providers, sym)
        print(f"  • Tradable: {td.ok}" + (f" ({td.reason})" if td.reason else ""))

        # 2) history
        df, src = _fetch_history(sym, days=args.days)
        if df is None or getattr(df, "empty", True):
            print("  • История: нет данных")
            continue
        rows = getattr(df, "shape", (0, 0))[0]
        print(f"  • Источник истории: {src} | строк: {rows}")

        # 3) daily features
        try:
            feats = compute_daily_features(df)
        except Exception as e:
            print(f"  • Ошибка фич: {e}")
            continue

        print(f"  • RSI={feats['rsi']:.2f} | pct={feats['percent_change']:.2f}% | vol_ratio={feats['volume_ratio']:.2f}×")
        print(f"  • ATR%={feats['atr_pct']:.2f}% | vol%={feats['volatility']:.2f}% | body={feats['bullish_body']:.2f}% | gap={feats['gap_up']:.2f}%")

        # 4) wick flag
        try:
            wick = long_upper_wick(
                feats["today_open_hist"], float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), feats["today_close_hist"]
            )
        except Exception:
            wick = False
        print(f"  • Long upper wick: {wick}")

        # 5) live intraday / reclaim
        dec = vwap_orh_reclaim(providers, sym, vwap_tol=0.001, orh_tol=0.001)
        if dec.live:
            v = dec.live
            print(f"  • Live: VWAP={v.get('vwap',0):.2f} ORH={v.get('orh',0):.2f} last={v.get('last',0):.2f} n={v.get('n','?')}")
        print(f"  • Reclaim allow: {dec.allow_reclaim}")

if __name__ == "__main__":
    main()
