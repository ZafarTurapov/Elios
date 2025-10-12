from core.utils.alpaca_headers import alpaca_headers
# /root/stockbot/core/trading/eod_flatten.py
# -*- coding: utf-8 -*-
"""
EOD-флаттенер: закрывает все открытые позиции в конце дня.
- Отменяет связанные открытые ордера (TP/SL) при закрытии позиции.
- Делает по-символьно (чтобы дать понятный отчёт в Telegram).
- Имеет DRY-RUN режим.

Запуск:
PYTHONPATH=/root/stockbot python3 -m core.trading.eod_flatten

Опции окружения:
  ELIOS_EOD_DRY_RUN=1     — только отчёт, без фактического закрытия
  ELIOS_EOD_MAX_WAIT=30   — сколько секунд ждать подтверждения закрытия по тикеру
"""

import os
import time
import json
from datetime import datetime, timezone

import requests
from core.utils.telegram import send_telegram_message, escape_markdown

# === Alpaca Keys (жёстко, как просил) ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

HEADERS = alpaca_headers()

DRY_RUN   = os.getenv("ELIOS_EOD_DRY_RUN", "0") == "1"
MAX_WAIT  = float(os.getenv("ELIOS_EOD_MAX_WAIT", "30"))  # сек

# ---------- HTTP helpers ----------
def _req_get(url, params=None, timeout=12, retries=3, backoff=0.6):
    for a in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            if r.status_code in (429,) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(backoff * (2 ** a))

def _req_delete(url, params=None, timeout=12, retries=3, backoff=0.6):
    for a in range(retries):
        try:
            r = requests.delete(url, headers=HEADERS, params=params, timeout=timeout)
            if r.status_code in (429,) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(backoff * (2 ** a))

# ---------- Alpaca helpers ----------
def list_positions():
    r = _req_get(f"{ALPACA_BASE_URL}/v2/positions")
    if r.status_code == 200:
        return r.json() or []
    return []

def fetch_position(symbol: str):
    r = _req_get(f"{ALPACA_BASE_URL}/v2/positions/{symbol}")
    if r.status_code == 200:
        return r.json()
    return None  # 404 → позиции нет

def flatten_symbol(symbol: str) -> (bool, str):
    """
    Закрыть позицию по тикеру:
    DELETE /v2/positions/{symbol}?cancel_orders=true
    """
    r = _req_delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", params={"cancel_orders": "true"})
    ok = r.status_code < 300
    return ok, (r.text or "")

def wait_closed(symbol: str, max_wait: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        pos = fetch_position(symbol)
        if pos is None:  # 404
            return True
        time.sleep(1.2)
    return False

def fmt_money(x):
    try:
        return f"{float(x):.2f}"
    except:
        return str(x)

def main():
    # 1) Заберём активные позиции
    try:
        positions = list_positions()
    except Exception as e:
        send_telegram_message(f"⚠️ *EOD*: не удалось получить позиции: {escape_markdown(str(e))}")
        return

    if not positions:
        send_telegram_message("🌙 *EOD*: позиций нет — ничего закрывать.")
        return

    # 2) Подготовим стартовый отчёт
    lines = ["🌙 *EOD*: начинаю закрытие позиций", ""]  # шапка
    for p in positions:
        sym = p.get("symbol")
        side = p.get("side")  # 'long' | 'short'
        qty  = p.get("qty")
        aep  = p.get("avg_entry_price")
        upl  = p.get("unrealized_pl")       # $ PnL (float в JSON-строке)
        uplpc= p.get("unrealized_plpc")     # доля (напр. 0.025)
        lines.append(f"• {escape_markdown(sym)} {escape_markdown(side or '')} x{qty} @ ${escape_markdown(fmt_money(aep))}  "
                     f"PnL: ${escape_markdown(fmt_money(upl))} ({escape_markdown(f'{float(uplpc)*100:.2f}%') if uplpc is not None else '—'})")
    if DRY_RUN:
        lines.append("")
        lines.append("_DRY-RUN: только отчёт, без закрытия._")
    try:
        send_telegram_message("\n".join(lines))
    except Exception:
        pass

    if DRY_RUN:
        return

    # 3) Закрываем по-символьно и ждём подтверждения
    results = []
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        if not sym:
            continue
        ok, body = False, ""
        err_text = ""
        try:
            ok, body = flatten_symbol(sym)
            if not ok:
                err_text = f"HTTP {body[:180]}"
            else:
                # подождём подтверждения, что позиции больше нет
                closed = wait_closed(sym, MAX_WAIT)
                if not closed:
                    ok = False
                    err_text = "timeout while waiting for close"
        except Exception as e:
            ok = False
            err_text = str(e)
        results.append((sym, ok, err_text))

    # 4) Итоговый отчёт
    ok_cnt = sum(1 for _,ok,_ in results if ok)
    fail_cnt = len(results) - ok_cnt
    out = ["✅ *EOD завершён*: позиции закрыты." if fail_cnt==0 else "⚠️ *EOD завершён с ошибками*"]
    for sym, ok, err in results:
        if ok:
            out.append(f"  • {escape_markdown(sym)} — закрыт")
        else:
            out.append(f"  • {escape_markdown(sym)} — _ошибка_: {escape_markdown(err or 'unknown')}")
    try:
        send_telegram_message("\n".join(out))
    except Exception:
        pass

if __name__ == "__main__":
    main()
