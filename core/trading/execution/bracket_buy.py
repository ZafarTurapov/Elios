from core.utils.alpaca_headers import alpaca_headers
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math, time, json
from datetime import datetime, timezone
import requests

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY  = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC  = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()

# Параметры риска / брекета (можно переопределить в .env)
SL_PCT      = float(os.getenv("ELIOS_BRACKET_SL_PCT", "0.03"))   # 3%
TP_PCT      = float(os.getenv("ELIOS_BRACKET_TP_PCT", "0.05"))   # 5%
MAX_RISK_PCT= float(os.getenv("ELIOS_MAX_RISK_PCT", "0.01"))     # 1% от equity на риск
MAX_NOTION  = float(os.getenv("ELIOS_MAX_ORDER_NOTIONAL", "2500"))

def _get(u, params=None):
    r = requests.get(f"{BASE}{u}", headers=H, params=params or {}, timeout=20)
    r.raise_for_status(); return r.json()

def _post(u, payload: dict):
    r = requests.post(f"{BASE}{u}", headers=H, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {u} -> {r.status_code} {r.text}")
    return r.json()

def nowz():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def calc_qty_by_risk(symbol: str, entry_price: float) -> int:
    """Размер позы: risk = equity * MAX_RISK_PCT, SL = entry*SL_PCT → qty = risk / (entry*SL_PCT)."""
    try:
        acc = _get("/v2/account")
        equity = float(acc.get("equity") or 0)
    except Exception:
        equity = 0.0
    risk_usd = max(0.0, equity * MAX_RISK_PCT)
    denom = entry_price * SL_PCT
    if entry_price <= 0 or denom <= 0:
        return 0
    qty = math.floor(risk_usd / denom)
    # дополнительно ограничим нотионом
    if entry_price > 0 and MAX_NOTION > 0:
        cap_qty = math.floor(MAX_NOTION / entry_price)
        qty = min(qty, cap_qty) if cap_qty > 0 else qty
    return int(max(qty, 0))

def place_limit_bracket(symbol: str, limit_price: float, client_tag: str = "ELIOS"):
    """BUY limit c order_class='bracket' (TP/SL как OCO)."""
    limit_price = float(limit_price)
    if limit_price <= 0:
        raise ValueError("limit_price must be > 0")

    qty = calc_qty_by_risk(symbol, limit_price)
    if qty <= 0:
        # fallback: минимальная 1 акция, но тоже под notional-cap
        if limit_price <= MAX_NOTION:
            qty = 1
        else:
            raise RuntimeError(f"Qty=0 by risk and limit_price>{MAX_NOTION} — refuse order")

    tp = round(limit_price * (1.0 + TP_PCT), 4)
    sl = round(limit_price * (1.0 - SL_PCT), 4)

    coid = f"{client_tag}:{symbol}:{nowz()}"
    payload = {
        "symbol": symbol,
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": round(limit_price, 4),
        "qty": str(qty),
        "order_class": "bracket",
        "take_profit": {"limit_price": tp},
        "stop_loss": {"stop_price": sl},
        "extended_hours": True,
        "client_order_id": coid,
    }
    res = _post("/v2/orders", payload)
    print(f"[BracketBuy] Placed {symbol} qty={qty} LMT={limit_price:.4f} TP={tp:.4f} SL={sl:.4f} id={res.get('id','?')}")
    return res

if __name__ == "__main__":
    # Пример: python -m core.trading.execution.bracket_buy AAPL 210.5
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m core.trading.execution.bracket_buy SYMBOL LIMIT_PRICE"); sys.exit(1)
    place_limit_bracket(sys.argv[1].upper(), float(sys.argv[2]))
