from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
"""
Daily Circuit Breaker:
- Считает дневную просадку от equity на открытии дня (локально TZ).
- Если достигнут лимит (порог в $ или %), мгновенно:
  * отменяет все открытые ордера
  * закрывает все позиции
  * кладёт temp/TRADE_HALT.until (до следующего локального дня 00:01)
- Если активен HALT, печатает статус и выходит.
ENV:
  ELIOS_TZ=Asia/Tashkent
  ELIOS_DD_LIMIT_USD=400          # сработает при просадке <= -400$
  ELIOS_DD_LIMIT_PCT=0.01         # и/или при просадке <= -1% (берётся тот, что раньше)
"""
import os
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
import requests

ROOT = Path("/root/stockbot")
TMP = ROOT / "temp"
TMP.mkdir(parents=True, exist_ok=True)
HALT_FILE = TMP / "TRADE_HALT.until"

TZ = os.getenv("ELIOS_TZ", "Asia/Tashkent")
Z = ZoneInfo(TZ)
BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()

DD_USD = float(os.getenv("ELIOS_DD_LIMIT_USD", "400"))
DD_PCT = float(os.getenv("ELIOS_DD_LIMIT_PCT", "0.01"))


def jget(p, params=None):
    r = requests.get(f"{BASE}{p}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {p} -> {r.status_code} {r.text}")
    return r.json()


def jdel(p, params=None):
    r = requests.delete(f"{BASE}{p}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"DELETE {p} -> {r.status_code} {r.text}")
    return r.status_code


def today_local():
    return datetime.now(Z).date()


def halt_active() -> bool:
    if not HALT_FILE.exists():
        return False
    try:
        until = datetime.fromisoformat(HALT_FILE.read_text().strip())
    except Exception:
        return True
    return datetime.now(Z) < until


def set_halt_until_next_day():
    nxt = datetime.combine(today_local() + timedelta(days=1), dtime(0, 1), tzinfo=Z)
    HALT_FILE.write_text(nxt.isoformat())


def start_of_day_equity():
    # берём portfolio/history 2D 1D и находим equity на текущий локальный день и пред. день → Δ по 1D
    ph = jget(
        "/v2/account/portfolio/history",
        {"period": "2D", "timeframe": "1D", "extended_hours": "true"},
    )
    ts = ph.get("timestamp", []) or []
    eq = ph.get("equity", []) or []
    rows = []
    for t, e in zip(ts, eq):
        d = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(Z).date()
        rows.append((d, float(e)))
    rows.sort()
    # equity на начало текущего дня = предыдущее значение (вчера 1D)
    if len(rows) >= 2 and rows[-1][0] == today_local():
        return rows[-2][1], rows[-1][1]
    # fallback: вернём (текущее, текущее)
    acc = jget("/v2/account")
    cur = float(acc.get("equity") or 0.0)
    return cur, cur


def flatten_all():
    try:
        jdel("/v2/orders")  # cancel all open orders
    except Exception:
        pass
    try:
        jdel("/v2/positions")  # close all positions
    except Exception:
        pass


def main():
    if halt_active():
        print("[Breaker] HALT активен — торговля остановлена до завтра.")
        return

    acc = jget("/v2/account")
    cur_eq = float(acc.get("equity") or 0.0)
    sod, today_eq_1d = start_of_day_equity()
    dd_usd = cur_eq - sod
    dd_pct = (dd_usd / sod) if sod > 0 else 0.0

    print(
        f"[Breaker] SOD={sod:.2f}  NOW={cur_eq:.2f}  Δ={dd_usd:.2f} ({dd_pct*100:.2f}%)  Limits: {-(DD_USD)}$ / {-(DD_PCT*100):.2f}%"
    )

    trigger = (dd_usd <= -abs(DD_USD)) or (dd_pct <= -abs(DD_PCT))
    if trigger:
        print("[Breaker] LIMIT REACHED -> FLATTEN + HALT until tomorrow")
        flatten_all()
        set_halt_until_next_day()
    else:
        print("[Breaker] OK — работать можно")


if __name__ == "__main__":
    main()
