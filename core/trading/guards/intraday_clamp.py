from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
"""
Intraday Clamp Guard:
- Быстрый стоп: UPL% <= FAST_STOP_PCT (по умолчанию -0.02 => -2%)
- Тайм-стоп: если прошло HOLDING_MAX_MIN (45 мин) и UPL% <= TIME_STOP_PCT (0.5%) — закрыть
- Брейк-ивен: если HWM >= IMPULSE_PCT (1.2%), а текущее UPL% <= BREAKEVEN_PCT (0.2%) — закрыть
- Работает только в торговое окно по локальному TZ
ENV (опционально):
  ELIOS_TZ=Asia/Tashkent
  ELIOS_TRADING_FROM=18:30
  ELIOS_TRADING_TILL=23:59
  ELIOS_FAST_STOP_PCT=-0.02
  ELIOS_TIME_STOP_MIN=45
  ELIOS_TIME_STOP_PCT=0.005
  ELIOS_IMPULSE_PCT=0.012
  ELIOS_BREAKEVEN_PCT=0.002
"""
import os
import sys
import json
from datetime import datetime, timedelta, time as dtime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests

ROOT = Path("/root/stockbot")
TMP = ROOT / "temp"
TMP.mkdir(parents=True, exist_ok=True)
HWM_PATH = TMP / "hwm.json"

TZ = os.environ.get("ELIOS_TZ", "Asia/Tashkent")
Z = ZoneInfo(TZ)
T_FROM = os.environ.get("ELIOS_TRADING_FROM", "18:30")
T_TILL = os.environ.get("ELIOS_TRADING_TILL", "23:59")

FAST_STOP = float(os.environ.get("ELIOS_FAST_STOP_PCT", "-0.02"))
TIME_STOP_MIN = int(os.environ.get("ELIOS_TIME_STOP_MIN", "45"))
TIME_STOP_PCT = float(os.environ.get("ELIOS_TIME_STOP_PCT", "0.005"))
IMPULSE_PCT = float(os.environ.get("ELIOS_IMPULSE_PCT", "0.012"))
BREAKEVEN_PCT = float(os.environ.get("ELIOS_BREAKEVEN_PCT", "0.002"))

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def jdel(path, params=None):
    r = requests.delete(f"{BASE}{path}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"DELETE {path} -> {r.status_code} {r.text}")
    return (r.status_code, r.text)


def now_local():
    return datetime.now(Z)


def in_window(t: datetime) -> bool:
    def p(s):
        hh, mm = s.split(":")
        return dtime(int(hh), int(mm))

    tt = dtime(t.hour, t.minute)
    return p(T_FROM) <= tt <= p(T_TILL)


def load_hwm() -> dict:
    try:
        return json.loads(HWM_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_hwm(d: dict):
    try:
        HWM_PATH.write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def last_buy_time(symbol: str):
    """Пытаемся получить время последней покупки по символу из /v2/orders?status=all&symbols=..."""
    try:
        # берём за 3 дня, чтобы покрыть любые входы
        now = datetime.now(timezone.utc)
        after = (
            (now - timedelta(days=3))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        params = {
            "status": "all",
            "direction": "desc",
            "limit": 200,
            "after": after,
            "symbols": symbol,
        }
        arr = jget("/v2/orders", params)
        for o in arr:
            try:
                if (
                    o.get("symbol") == symbol
                    and o.get("side") == "buy"
                    and o.get("filled_at")
                ):
                    # возвращаем первую найденную заполненную покупку (по убыванию)
                    dt = o["filled_at"].replace("Z", "+00:00")
                    return datetime.fromisoformat(dt)
            except Exception:
                continue
    except Exception:
        pass
    return None


def close_symbol(sym: str):
    code, txt = jdel(f"/v2/positions/{sym}")
    print(f"[Clamp] Close {sym} -> {code}")


def main():
    if not (KEY and SEC):
        print("[ERR] Alpaca keys missing", file=sys.stderr)
        sys.exit(2)

    now = now_local()
    if not in_window(now):
        print(f"[Clamp] {now.isoformat()} — вне торгового окна")
        return

    pos = jget("/v2/positions")
    if not isinstance(pos, list) or not pos:
        print(f"[Clamp] {now.isoformat()} — позиций нет")
        return

    hwm = load_hwm()
    touched = False

    for p in pos:
        sym = str(p.get("symbol", "")).upper()
        uplpc_raw = p.get("unrealized_plpc")
        try:
            uplpc = float(uplpc_raw) if uplpc_raw is not None else None
        except Exception:
            uplpc = None

        # обновляем HWM
        cur_hwm = float(hwm.get(sym, {}).get("hwm_plpc", 0.0))
        if uplpc is not None and uplpc > cur_hwm:
            hwm[sym] = {"hwm_plpc": uplpc, "ts": now.isoformat()}
            touched = True
            cur_hwm = uplpc

        # быстрый стоп
        if uplpc is not None and uplpc <= FAST_STOP:
            print(f"[Clamp] {sym} UPL {uplpc:.4f} <= {FAST_STOP:.4f} -> close")
            try:
                close_symbol(sym)
            except Exception as e:
                print(f"[ERR] close {sym}: {e}", file=sys.stderr)
            continue

        # тайм-стоп
        lb = last_buy_time(sym)
        if lb is not None:
            held_min = (datetime.now(timezone.utc) - lb).total_seconds() / 60.0
            if held_min >= TIME_STOP_MIN and (uplpc is None or uplpc <= TIME_STOP_PCT):
                print(
                    f"[Clamp] {sym} held {held_min:.1f}m, UPL {uplpc if uplpc is not None else 'NA'} <= {TIME_STOP_PCT:.4f} -> close"
                )
                try:
                    close_symbol(sym)
                except Exception as e:
                    print(f"[ERR] close {sym}: {e}", file=sys.stderr)
                continue

        # брейк-ивен после импульса
        if cur_hwm >= IMPULSE_PCT and uplpc is not None and uplpc <= BREAKEVEN_PCT:
            print(
                f"[Clamp] {sym} HWM {cur_hwm:.4f} -> now {uplpc:.4f} <= {BREAKEVEN_PCT:.4f} -> close"
            )
            try:
                close_symbol(sym)
            except Exception as e:
                print(f"[ERR] close {sym}: {e}", file=sys.stderr)
            continue

    if touched:
        save_hwm(hwm)


if __name__ == "__main__":
    main()
