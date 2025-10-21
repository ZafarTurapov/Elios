# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

CACHE = Path("/root/stockbot/data/metrics/short_interest.json")


def load_short_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE.exists():
        return {}
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # нормализуем ключи-тикеры в UPPER
            return {str(k).upper(): v for k, v in data.items()}
    except Exception:
        pass
    return {}


def get_squeeze_row(symbol: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    row = cache.get(symbol.upper(), {}) or {}
    to_float = lambda x: (
        float(x)
        if isinstance(x, (int, float))
        else float(str(x)) if x not in (None, "", "null") else float("nan")
    )
    out = {
        "short_float_pct": (
            to_float(row.get("short_float_pct"))
            if "short_float_pct" in row
            else float("nan")
        ),
        "days_to_cover": (
            to_float(row.get("days_to_cover"))
            if "days_to_cover" in row
            else float("nan")
        ),
        "borrow_fee_pct": (
            to_float(row.get("borrow_fee_pct"))
            if "borrow_fee_pct" in row
            else float("nan")
        ),
    }
    return out
