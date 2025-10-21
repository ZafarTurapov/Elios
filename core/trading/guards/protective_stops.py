from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
"""
Protective Stops Guard:
- Для КАЖДОЙ открытой long-позиции проверяет, есть ли привязанные защитные ордера (SL/TP)
- Если нет, выставляет хотя бы защитный стоп (SELL stop) на уровне avg_entry*(1-abs(SL_PCT))
- Опционально добавляет TP (limit) на avg_entry*(1+TP_PCT), если разрешено
- Никаких overnight допущений: просто следит каждые N минут и навешивает «страховку», если её нет
ENV:
  ELIOS_SL_PCT=0.03       # 3% стоп (для long)
  ELIOS_TP_PCT=0.05       # 5% take-profit (опционально)
  ELIOS_ADD_TP=0          # 1 = также добавлять TP, 0 = только стоп
  ELIOS_TZ=Asia/Tashkent
Примечание: если используешь полноценные bracket-ордера при входе — этот страж просто ничего не сделает.
"""
import os
import sys
from datetime import datetime, timezone, timedelta
import requests

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()

SL_PCT = float(os.getenv("ELIOS_SL_PCT", "0.03"))
TP_PCT = float(os.getenv("ELIOS_TP_PCT", "0.05"))
ADD_TP = str(os.getenv("ELIOS_ADD_TP", "0")).strip() in ("1", "true", "yes", "on")


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def jpost(path, payload: dict):
    r = requests.post(f"{BASE}{path}", headers=H, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {path} -> {r.status_code} {r.text}")
    return r.json()


def sym_open_orders_map():
    # активные ордера (open) сгруппируем по символам
    after = (
        (datetime.now(timezone.utc) - timedelta(days=3))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    arr = jget(
        "/v2/orders",
        {"status": "open", "direction": "asc", "limit": 500, "after": after},
    )
    mp = {}
    for o in arr:
        s = o.get("symbol")
        if not s:
            continue
        mp.setdefault(s, []).append(o)
    return mp


def has_protective_for_long(orders):
    # ищем SELL stop/stop_limit или order_class=bracket с дочерними ногами
    for o in orders or []:
        side = o.get("side")
        typ = (o.get("type") or "").lower()
        cls = (o.get("order_class") or "").lower()
        if cls in ("bracket", "oco"):
            return True
        if side == "sell" and ("stop" in typ):
            return True
    return False


def place_stop(symbol: str, qty: float, stop_price: float):
    payload = {
        "symbol": symbol,
        "side": "sell",
        "type": "stop",
        "time_in_force": "day",
        "qty": f"{qty:g}",
        "stop_price": round(
            stop_price, 4
        ),  # Alpaca требует строку/число без лишних знаков
        "extended_hours": True,
    }
    res = jpost("/v2/orders", payload)
    print(
        f"[Protective] STOP placed {symbol} qty={qty:g} stop={stop_price:.4f} id={res.get('id','?')}"
    )


def place_takeprofit(symbol: str, qty: float, limit_price: float):
    payload = {
        "symbol": symbol,
        "side": "sell",
        "type": "limit",
        "time_in_force": "day",
        "qty": f"{qty:g}",
        "limit_price": round(limit_price, 4),
        "extended_hours": True,
    }
    res = jpost("/v2/orders", payload)
    print(
        f"[Protective] TP   placed {symbol} qty={qty:g} limit={limit_price:.4f} id={res.get('id','?')}"
    )


def main():
    if not (KEY and SEC):
        print("[ERR] Alpaca keys missing", file=sys.stderr)
        sys.exit(2)

    pos = jget("/v2/positions")
    if not isinstance(pos, list) or not pos:
        print("[Protective] no open positions")
        return

    mp = sym_open_orders_map()

    for p in pos:
        symbol = p.get("symbol")
        side = p.get("side")
        if side != "long":
            # Для short-логики можно добавить зеркальные стопы выше цены входа (пока игнорируем)
            continue

        qty_str = p.get("qty") or "0"
        try:
            qty = float(qty_str)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue

        avg_entry = float(p.get("avg_entry_price") or 0)
        if avg_entry <= 0:
            continue

        orders = mp.get(symbol, [])
        if has_protective_for_long(orders):
            print(f"[Protective] {symbol}: protective order exists")
            continue

        # Считаем уровни
        stop_price = avg_entry * (1.0 - abs(SL_PCT))
        take_price = avg_entry * (1.0 + abs(TP_PCT))

        # Ставим хотя бы стоп
        try:
            place_stop(symbol, qty, stop_price)
        except Exception as e:
            print(f"[ERR] place_stop {symbol}: {e}", file=sys.stderr)

        # Опционально ставим take-profit
        if ADD_TP:
            try:
                place_takeprofit(symbol, qty, take_price)
            except Exception as e:
                print(f"[ERR] place_takeprofit {symbol}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
