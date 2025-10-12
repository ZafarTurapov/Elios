# /root/stockbot/core/utils/trade_log.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

CANON_PATH = LOGS / "trade_log.json"  # NDJSON: одна запись JSON в строке

def _iso_utc(dt: Optional[datetime]=None) -> str:
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def append_trade_event(*, side: str, symbol: str, qty: float,
                       entry_price: Optional[float]=None,
                       exit_price: Optional[float]=None,
                       pnl: Optional[float]=None,
                       reason: str="",
                       source: str="",
                       timestamp: Optional[str]=None,
                       extra: Optional[Dict[str, Any]]=None) -> None:
    """
    Пишет атомарно событие сделки в NDJSON-журнал (/logs/trade_log.json).
    side: "BUY" | "SELL" | "CLOSE" ...
    """
    rec = {
        "timestamp": timestamp or _iso_utc(),
        "side": (side or "").upper(),
        "symbol": (symbol or "").upper(),
        "qty": qty,
        "entry": entry_price,
        "exit": exit_price,
        "pnl": pnl,
        "reason": reason or "",
        "source": source or "engine",
    }
    if extra and isinstance(extra, dict):
        rec.update(extra)

    # атомарная дозапись (NDJSON): на каждую запись — отдельная строка
    with open(CANON_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
