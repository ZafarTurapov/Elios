from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
"""
core/trading/account_sync.py — сводка по счёту Alpaca + отчёт в Telegram

Что делает:
  • Тянет /v2/account и /v2/positions из Alpaca (paper/live — по ALPACA_BASE_URL)
  • Считает дневной P/L, суммарный нереализованный/интрадей P/L, топ-двигатели
  • Смотрит наш трейд-лог (TRADE_LOG_PATH) и оценивает realized P/L за сегодня
  • Считает прогресс обучения (кол-во примеров в training_data.json)
  • Печатает отчёт и (опционально) отправляет в Telegram

Запуск:
    PYTHONPATH=/root/stockbot python3 -m core.trading.account_sync --send --limit 5 --tz Asia/Tashkent
"""

import os
import json
import argparse


# --- .env autoload (simple) ---
def _load_env_from_files(paths=("/root/stockbot/.env", ".env")):
    import os

    for fp in paths:
        try:
            p = Path(fp)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ or not os.environ.get(k)):
                    os.environ[k] = v
        except Exception:
            pass


from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Iterable, Union

import requests

from core.utils.paths import TRADE_LOG_PATH
from core.utils.telegram import (
    send_telegram_message,
)  # должен уметь отправлять plain text/Markdown

# ensure .env
_load_env_from_files()
# === НАСТРОЙКИ / КЛЮЧИ =========================================================
# Берём из окружения, но сохраняем ваши дефолты — чтобы ничего не сломать.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

HEADERS = alpaca_headers(content_json=True)

# Пути артефактов
ROOT = Path("/root/stockbot")
TRAINING_DATA_PATH = ROOT / "core" / "trading" / "training_data.json"
SUMMARY_PATH = ROOT / "core" / "trading" / "account_summary.json"
JSONL_LOG_PATH = ROOT / "logs" / "account_sync.jsonl"


# === УТИЛИТЫ ===================================================================
def _to_f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _alpaca_get(path: str, timeout: int = 20) -> Any:
    """GET к Alpaca, поднимает исключение при !=200."""
    url = ALPACA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code == 404:
        # для /v2/positions при пустом портфеле это может быть нормальным (иногда 200 с [] у новых API)
        return []
    r.raise_for_status()
    return r.json()


def _now_str(tz: str = "Asia/Tashkent") -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M %Z")


# === ТРЕЙД-ЛОГ: устойчивый парсер ==============================================
def _load_trade_log_any(path: Path) -> Union[dict, list]:
    """
    Поддерживаем форматы:
      1) dict {SYM: [trades, ...], ...}
      2) list [trade, ...]
      3) JSONL (по строкам) -> приведём к list
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    # попытка обычного JSON
    try:
        return json.loads(raw)
    except Exception:
        pass
    # попытка JSONL
    items = []
    for line in raw.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            items.append(obj)
        except Exception:
            continue
    return items


def _iter_trades(trade_log: Union[dict, list]) -> Iterable[dict]:
    """
    Нормализованный итератор по сделкам.
    Допускаем:
      • dict -> values(): могут быть списки/одиночные объекты
      • list -> элементы — сделки либо списки сделок
    """
    if isinstance(trade_log, dict):
        for v in trade_log.values():
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, dict):
                        yield t
            elif isinstance(v, dict):
                yield v
    elif isinstance(trade_log, list):
        for v in trade_log:
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, dict):
                        yield t
            elif isinstance(v, dict):
                yield v


# === ОСНОВНАЯ ЛОГИКА ===========================================================
def calculate_today_profit_from_tradelog() -> float:
    """Грубая оценка realized P/L за сегодня из нашего трейд-лога."""
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0

    try:
        trade_log = _load_trade_log_any(TRADE_LOG_PATH)
    except Exception:
        return 0.0

    for trade in _iter_trades(trade_log):
        # считаем закрытые сделки, закрытые сегодня
        ts_exit = str(trade.get("timestamp_exit", "") or trade.get("closed_at", ""))
        if not ts_exit.startswith(today):
            continue
        qty = _to_f(trade.get("qty", trade.get("quantity", 0)))
        entry = _to_f(trade.get("entry_price", trade.get("price", 0)))
        exit_ = _to_f(trade.get("exit_price", trade.get("close_price", 0)))
        if qty and exit_ and entry:
            total += (exit_ - entry) * qty

    return round(total, 2)


def count_training_examples() -> int:
    try:
        data = (
            json.loads(TRAINING_DATA_PATH.read_text())
            if TRAINING_DATA_PATH.exists()
            else []
        )
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def fetch_account_snapshot() -> Dict[str, Any]:
    account = _alpaca_get("/v2/account")
    positions = _alpaca_get("/v2/positions")

    # Метрики счета
    equity = _to_f(account.get("equity"))
    last_equity = _to_f(account.get("last_equity"), equity)
    day_pl = equity - last_equity
    portfolio_value = _to_f(account.get("portfolio_value"), equity)
    cash = _to_f(account.get("cash"))
    buying_power = _to_f(account.get("buying_power"))
    status = account.get("status", "unknown")
    blocked = bool(account.get("trading_blocked", False))
    currency = account.get("currency", "USD")
    multiplier = account.get("multiplier", "N/A")

    # Позиции и их агрегаты
    pos_list: List[Dict[str, Any]] = []
    unreal_pl_sum = 0.0
    intraday_pl_sum = 0.0

    if isinstance(positions, list):
        for p in positions:
            sym = p.get("symbol")
            qty = _to_f(p.get("qty"))
            avg = _to_f(p.get("avg_entry_price"))
            price = _to_f(p.get("current_price"))
            upl = _to_f(p.get("unrealized_pl"))
            uplpc = _to_f(p.get("unrealized_plpc")) * 100.0
            intr = _to_f(p.get("unrealized_intraday_pl"))
            intrp = _to_f(p.get("unrealized_intraday_plpc")) * 100.0

            unreal_pl_sum += upl
            intraday_pl_sum += intr

            pos_list.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg": avg,
                    "price": price,
                    "unreal_pl": upl,
                    "unreal_pl_pct": uplpc,
                    "intraday_pl": intr,
                    "intraday_pl_pct": intrp,
                }
            )

    pos_list.sort(key=lambda x: abs(x.get("unreal_pl", 0.0)), reverse=True)

    # Реализованный за сегодня из нашего лога
    realized_today = calculate_today_profit_from_tradelog()

    # Прогресс обучения
    train_count = count_training_examples()

    summary = {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "equity": equity,
        "last_equity": last_equity,
        "day_pl": day_pl,
        "buying_power": buying_power,
        "currency": currency,
        "status": status,
        "trading_blocked": blocked,
        "multiplier": multiplier,
        "realized_profit_today": realized_today,
        "training_progress": train_count,
    }
    # сохраняем аккуратный JSON для возможной интеграции
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "account": account,
        "positions": pos_list,
        "metrics": summary,
    }


def format_report(
    snap: Dict[str, Any],
    limit: int = 5,
    tz: str = "Asia/Tashkent",
    no_emoji: bool = False,
) -> str:
    m = snap["metrics"]
    acc = snap["account"]
    ts = _now_str(tz)

    em = (lambda e, t: t) if no_emoji else (lambda e, t: f"{e} {t}")

    lines: List[str] = []
    lines.append("📦 Elios — Account Sync Report")
    lines.append(em("🕒", ts))
    lines.append(
        em(
            "💼",
            f"Equity: ${m['equity']:.2f} | Cash: ${m['cash']:.2f} | BP: ${m['buying_power']:.2f}",
        )
    )
    lines.append(
        em(
            "📊",
            f"Portfolio: ${m['portfolio_value']:.2f} | Day P/L: ${m['day_pl']:+.2f}",
        )
    )
    lines.append(em("💰", f"Realized today: ${m['realized_profit_today']:+.2f}"))
    lines.append(em("🧠", f"Training examples: {m['training_progress']}"))
    lines.append(
        em(
            "🏦",
            f"Status: {m['status']} | Currency: {m['currency']} | Multiplier: {m['multiplier']}",
        )
    )
    lines.append(
        em("🔒", "Trading BLOCKED") if m["trading_blocked"] else em("✅", "Trading OK")
    )

    # Позиции
    pos = snap["positions"]
    lines.append(em("📦", f"Positions: {len(pos)}"))
    if pos:
        lines.append("")
        lines.append(em("🏁", f"Top movers (abs P/L), {min(limit, len(pos))}:"))
        for p in pos[: max(0, limit)]:
            lines.append(
                f"• ${p['symbol']}: qty={p['qty']:.0f} @ {p['avg']:.2f} → {p['price']:.2f} "
                f"| P/L={p['unreal_pl']:+.2f} ({p['unreal_pl_pct']:+.2f}%) "
                f"| intraday={p['intraday_pl']:+.2f} ({p['intraday_pl_pct']:+.2f}%)"
            )

        # Агрегаты по позициям
        agg_unr = sum(_to_f(x.get("unreal_pl")) for x in pos)
        agg_intr = sum(_to_f(x.get("intraday_pl")) for x in pos)
        lines.append("")
        lines.append(
            em("Σ", f"Unrealized sum: ${agg_unr:+.2f} | Intraday sum: ${agg_intr:+.2f}")
        )

    return "\n".join(lines)


def log_jsonl(data: Dict[str, Any]) -> None:
    JSONL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": data.get("metrics", {}),
        "positions_count": len(data.get("positions", [])),
    }
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--send",
        action="store_true",
        help="Отправить отчёт в Telegram (иначе только печать)",
    )
    ap.add_argument(
        "--limit", type=int, default=5, help="Сколько позиций показать в топе"
    )
    ap.add_argument(
        "--tz", type=str, default="Asia/Tashkent", help="Часовой пояс для таймстемпа"
    )
    ap.add_argument("--no-emoji", action="store_true", help="Вывести без эмодзи")
    args = ap.parse_args()

    try:
        snap = fetch_account_snapshot()
        msg = format_report(snap, limit=args.limit, tz=args.tz, no_emoji=args.no_emoji)
        print(msg)
        log_jsonl(snap)
        if args.send:
            try:
                send_telegram_message(msg)
            except Exception as te:
                print(f"⚠️ Ошибка отправки в Telegram: {te}")
    except requests.HTTPError as http_err:
        err = f"❌ HTTP {http_err.response.status_code if http_err.response else '??'}: {http_err}"
        print(err)
        if args.send:
            try:
                send_telegram_message(err)
            except Exception:
                pass
        raise
    except Exception as e:
        err = f"❌ AccountSync fatal error: {e}"
        print(err)
        if args.send:
            try:
                send_telegram_message(err)
            except Exception:
                pass
        raise
    try:
        # lightweight summary -> Telegram
        import json
        import os
        from datetime import datetime, timezone

        summ_path = "core/trading/account_summary.json"
        txt = None
        if os.path.exists(summ_path):
            try:
                with open(summ_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                eq = j.get("equity")
                cash = j.get("cash")
                bp = j.get("buying_power")
                cur = j.get("currency", "USD")
                pos = j.get("positions_count", 0)
                pl_d = j.get("pl_day", 0.0)
                txt = (
                    f"📦 Elios — Account Sync\\n"
                    f"🕒 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}\\n"
                    f"💼 Equity: ${eq:,.2f} | Cash: ${cash:,.2f} | BP: ${bp:,.2f}\\n"
                    f"📊 P/L day: ${pl_d:+,.2f} | Positions: {pos}"
                )
            except Exception:
                pass
        if not txt:
            txt = "📦 Elios — Account Sync: done."
        txt = txt.replace("\\n", "\n")
        send_telegram_message(txt)
    except Exception:
        pass


if __name__ == "__main__":
    main()
