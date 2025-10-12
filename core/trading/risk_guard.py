from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers
# -*- coding: utf-8 -*-
"""
Risk Guard (kill-switch) для Elios:
- Блокирует новые покупки на день, если дневная просадка ниже порога (equity day change)
- Или если подряд K убыточных закрытий (по trade_log.json)
- Пишет флаг temp/kill_switch.json с дедлайном и причиной
- Отправляет уведомление в Telegram (если настроено)

Запуск (проверка):  python -m core.trading.risk_guard --check
Код выхода: 0 = можно торговать; 100 = БЛОК; !=0 = ошибка
"""
import os, sys, json, argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import requests

# --- ENV / константы ---
TZ = os.environ.get("ELIOS_TZ", "Asia/Tashkent")

ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY      = os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
ALPACA_SECRET   = os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")

DAILY_MAX_LOSS_PCT = float(os.environ.get("ELIOS_DAILY_MAX_LOSS_PCT", "-2.0"))   # напр. -2.0 (%)
MAX_CONSEC_LOSSES  = int(os.environ.get("ELIOS_MAX_CONSEC_LOSSES", "3"))
COOLDOWN_HOURS     = int(os.environ.get("ELIOS_KILL_COOLDOWN_HOURS", "20"))

TELEGRAM_ENABLED = os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID_MAIN")

ROOT = Path("/root/stockbot")
TMP_DIR = ROOT / "temp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
KILL_SWITCH_PATH = TMP_DIR / "kill_switch.json"

# возможные пути трейд-логов
TRADE_LOG_CANDIDATES = [
    ROOT / "data" / "trades" / "trade_log.json",
    ROOT / "core" / "trading" / "trade_log.json",
    ROOT / "logs" / "trade_log.json",
]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _tg_send(msg: str) -> None:
    if not TELEGRAM_ENABLED:
        return
    try:
        from core.utils.telegram import send_telegram_message
        send_telegram_message(msg)
    except Exception as e:
        print(f"[TG] send failed: {e}", file=sys.stderr)

# --- Alpaca helpers ---
def alpaca_get(path: str):
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca API keys not set")
    url = f"{ALPACA_BASE_URL.rstrip('/')}{path}"
    headers = alpaca_headers()
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca GET {path} -> {r.status_code} {r.text}")
    return r.json()

def get_day_change_pct() -> Optional[float]:
    """
    Используем /v2/account: там есть equity и last_equity (или equity_prev_day).
    day_change_pct = (equity - last_equity) / last_equity * 100
    """
    try:
        acc = alpaca_get("/v2/account")
        equity = float(acc.get("equity"))
        last_eq = acc.get("last_equity") or acc.get("equity_prev_day")
        last_eq = float(last_eq) if last_eq is not None else None
        if last_eq and last_eq > 0:
            return (equity - last_eq) / last_eq * 100.0
    except Exception as e:
        print(f"[RiskGuard] get_day_change_pct error: {e}", file=sys.stderr)
    return None

# --- Trade log helpers ---
def _load_trade_log() -> List[dict]:
    for p in TRADE_LOG_CANDIDATES:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and isinstance(data.get("trades"), list):
                    return data["trades"]
        except Exception:
            continue
    return []

def get_consecutive_losses(limit:int=20) -> int:
    """
    Считаем подряд идущие убыточные закрытия с конца журнала (MAX последних limit).
    Ожидаем поля profit/pnl (в $) или pnl_pct (в %). Важен знак.
    """
    trades = _load_trade_log()
    if not trades:
        return 0
    losses = 0
    for rec in reversed(trades[-limit:]):
        # пропускаем открытые
        if str(rec.get("status","")).lower() not in ("closed","sold","closed_full","closed_partial","exit"):
            continue
        pnl = None
        for key in ("profit", "pnl", "pnl_usd", "profit_usd"):
            if key in rec:
                try:
                    pnl = float(rec[key]); break
                except Exception:
                    pass
        if pnl is None:
            for key in ("pnl_pct", "profit_pct"):
                if key in rec:
                    try:
                        pct = float(rec[key])
                        pnl = -1.0 if pct < 0 else 1.0
                        break
                    except Exception:
                        pass
        if pnl is None:
            break
        if pnl < 0:
            losses += 1
        else:
            break
    return losses

# --- Kill switch state ---
def read_kill_switch() -> Optional[dict]:
    data = _read_json(KILL_SWITCH_PATH)
    if not data:
        return None
    until = data.get("until_utc")
    try:
        if until and _now_utc() < datetime.fromisoformat(until):
            return data
    except Exception:
        pass
    try:
        KILL_SWITCH_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    return None

def set_kill_switch(reason:str, hours:int=COOLDOWN_HOURS) -> dict:
    until = _now_utc() + timedelta(hours=hours)
    payload = {
        "active": True,
        "reason": reason,
        "created_utc": _now_utc().isoformat(),
        "until_utc": until.isoformat(),
        "cooldown_hours": hours,
        "limits": {
            "DAILY_MAX_LOSS_PCT": DAILY_MAX_LOSS_PCT,
            "MAX_CONSEC_LOSSES": MAX_CONSEC_LOSSES,
        }
    }
    _write_json(KILL_SWITCH_PATH, payload)
    return payload

# --- Core check ---
def check_guard() -> Tuple[bool, str]:
    ks = read_kill_switch()
    if ks:
        return False, f"Kill-switch активен до {ks['until_utc']} (причина: {ks.get('reason','')})"

    day_pct = get_day_change_pct()
    if day_pct is not None and day_pct <= DAILY_MAX_LOSS_PCT:
        msg = f"Дневная просадка {day_pct:.2f}% <= лимита {DAILY_MAX_LOSS_PCT:.2f}%"
        set_kill_switch(msg, COOLDOWN_HOURS)
        return False, msg

    consec = get_consecutive_losses(limit=30)
    if consec >= MAX_CONSEC_LOSSES:
        msg = f"Серия убыточных закрытий: {consec} подряд >= лимита {MAX_CONSEC_LOSSES}"
        set_kill_switch(msg, COOLDOWN_HOURS)
        return False, msg

    return True, "ОК для торговли"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Проверить и вывести статус; код 0/100")
    args = ap.parse_args()

    ok, reason = check_guard()
    if args.check:
        if ok:
            print(f"[RiskGuard] ALLOW — {reason}")
            sys.exit(0)
        else:
            print(f"[RiskGuard] BLOCK — {reason}")
            _tg_send(f"🛑 Elios Kill-Switch: {reason}")
            sys.exit(100)

if __name__ == "__main__":
    main()
