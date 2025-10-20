from __future__ import annotations

import json
import math

# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.utils.alpaca_headers import alpaca_headers

CACHE_FP = Path("/root/stockbot/data/cache/rs_cache.json")
TTL_HOURS = float(os.getenv("RS_TTL_HOURS", "6"))


def _alpaca_base():
    return (
        "https://api.alpaca.markets"
        if os.getenv("ALPACA_LIVE", "0") == "1"
        else "https://paper-api.alpaca.markets"
    )


def _hdr():
    return alpaca_headers()


def _load_cache():
    try:
        if CACHE_FP.exists():
            return json.loads(CACHE_FP.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(data):
    try:
        CACHE_FP.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _daily_bars(sym: str, days: int = 120):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = f"{_alpaca_base()}/v2/stocks/bars"
    params = {
        "symbols": sym,
        "timeframe": "1Day",
        "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "end": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "limit": 1000,
        "adjustment": "all",
    }
    r = requests.get(url, headers=_hdr(), params=params, timeout=15)
    r.raise_for_status()
    arr = r.json().get("bars", {}).get(sym, [])
    return arr


def _cum_ret(bars, lookback=60):
    # кум. доходность за lookback дней (Close/Close_{-lookback} - 1)
    if not bars or len(bars) < lookback + 1:
        return float("nan")
    c0 = bars[-(lookback + 1)]["c"]
    c1 = bars[-1]["c"]
    try:
        return (c1 / c0 - 1.0) * 100.0
    except Exception:
        return float("nan")


def get_rs(symbol: str, bench: str = "SPY", lookback: int = 60) -> float:
    cache = _load_cache()
    key = f"{symbol.upper()}__{bench}__{lookback}"
    now = time.time()
    hit = cache.get(key)
    if hit and now - hit.get("ts", 0) < TTL_HOURS * 3600:
        return float(hit.get("rs", float("nan")))
    try:
        sb = _daily_bars(symbol.upper(), days=max(lookback + 10, 90))
        bb = _daily_bars(bench.upper(), days=max(lookback + 10, 90))
        if not sb or not bb:
            raise RuntimeError("no bars")
        r_sym = _cum_ret(sb, lookback)
        r_ben = _cum_ret(bb, lookback)
        if any((v != v or math.isinf(v)) for v in (r_sym, r_ben)):
            rs = float("nan")
        else:
            rs = r_sym - r_ben  # RS > 0 — лучше бенчмарка
        cache[key] = {"ts": now, "rs": rs}
        _save_cache(cache)
        return rs
    except Exception:
        # не кэшируем ошибку — в след.запуске попробуем снова
        return float("nan")
