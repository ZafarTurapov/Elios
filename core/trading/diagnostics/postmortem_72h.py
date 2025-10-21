# -*- coding: utf-8 -*-
"""
Postmortem 72h — анализ убыточных дней:
- Берём trades из известных путей, фильтруем закрытые за последние 72 часа
- Считаем PnL, hit-rate, средний PnL, серию подряд убыточных, max intraday drawdown по журналу
- Атрибуция: по тикерам, по часу входа, по "стратегическим тегам" (если есть), по типу выхода (TP/SL/Force/EOD)
- Отдельно печатаем ТОП-5 источников убытка и сохраняем CSV-отчёты в logs/diagnostics
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

ROOT = Path("/root/stockbot")
OUT_DIR = ROOT / "logs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_CANDIDATES = [
    ROOT / "data" / "trades" / "trade_log.json",
    ROOT / "core" / "trading" / "trade_log.json",
    ROOT / "logs" / "trade_log.json",
]
SIGNALS_CANDIDATES = [
    ROOT / "core" / "trading" / "signals.json",
    ROOT / "logs" / "signals.json",
]
REJECTED_CANDIDATES = [
    ROOT / "core" / "trading" / "rejected.json",
    ROOT / "logs" / "rejected.json",
]


def load_json_any(paths):
    for p in paths:
        try:
            if Path(p).exists():
                return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def parse_dt(s):
    if not s:
        return None
    try:
        # поддержка ISO / 'YYYY-mm-dd HH:MM:SS' / '...Z'
        s = str(s).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except Exception:
            return None


def is_closed(rec):
    st = str(rec.get("status", "")).lower()
    return st in ("closed", "sold", "closed_full", "closed_partial", "exit", "flatten")


def as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def get_exit_type(rec):
    # пытаемся классифицировать выход
    r = (
        rec.get("exit_reason") or rec.get("reason_exit") or rec.get("reason") or ""
    ).lower()
    if "take" in r or "tp" in r or "profit" in r:
        return "TP"
    if "stop" in r or "sl" in r or "loss" in r:
        return "SL"
    if "force" in r or "flatten" in r or "eod" in r or "close_all" in r:
        return "FORCE/EOD"
    # эвристика по величине PnL%
    pnl_pct = as_float(rec.get("pnl_pct") or rec.get("profit_pct"))
    if pnl_pct is not None:
        if pnl_pct >= 4.5:
            return "TP~"
        if pnl_pct <= -2.8:
            return "SL~"
    return "OTHER"


def hour_bucket(dt):
    if not dt:
        return "NA"
    # Бакеты по часу локально не знаем, оставим по UTC
    return f"{dt.hour:02d}:00"


def get_strategy_tag(rec):
    # собираем всё, что может описывать «источник» сигнала
    for k in (
        "strategy",
        "pass",
        "source",
        "tag",
        "signal_type",
        "gate",
        "phase",
        "selector",
        "model",
        "gpt",
    ):
        if rec.get(k):
            return str(rec[k])
    # иногда в info можно хранить pass/phase
    info = rec.get("info") or {}
    for k in ("pass", "phase", "gate", "selector", "model", "gpt_decision"):
        if isinstance(info, dict) and info.get(k):
            return str(info[k])
    return "NA"


def main():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=72)

    trades = load_json_any(TRADE_LOG_CANDIDATES)
    if trades is None:
        print("[ERR] trade_log.json не найден", file=sys.stderr)
        sys.exit(2)

    if isinstance(trades, dict) and isinstance(trades.get("trades"), list):
        trades = trades["trades"]
    if not isinstance(trades, list):
        print("[ERR] trade_log.json имеет неожиданный формат", file=sys.stderr)
        sys.exit(2)

    closed = []
    for t in trades:
        if not is_closed(t):
            continue
        # дата выхода (или дата входа, если нет)
        dt_exit = parse_dt(
            t.get("exit_time") or t.get("closed_at") or t.get("sell_time")
        )
        dt_entry = parse_dt(t.get("entry_time") or t.get("buy_time"))
        dt_ref = dt_exit or dt_entry
        if not dt_ref:
            continue
        if dt_ref < since:
            continue

        pnl = as_float(
            t.get("pnl") or t.get("profit") or t.get("pnl_usd") or t.get("profit_usd"),
            0.0,
        )
        pnl_pct = as_float(t.get("pnl_pct") or t.get("profit_pct"))
        qty = as_float(t.get("qty") or t.get("quantity"), 0.0)
        sym = str(t.get("symbol") or t.get("ticker") or "?").upper()
        entry_h = hour_bucket(dt_entry)
        exit_t = get_exit_type(t)
        tag = get_strategy_tag(t)

        closed.append(
            {
                "symbol": sym,
                "exit_time": dt_exit.isoformat() if dt_exit else "",
                "entry_time": dt_entry.isoformat() if dt_entry else "",
                "entry_hour_bucket_utc": entry_h,
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct if pnl_pct is not None else "",
                "qty": qty,
                "exit_type": exit_t,
                "tag": tag,
            }
        )

    if not closed:
        print("[INFO] За последние 72 часа закрытых сделок не найдено.")
        sys.exit(0)

    # агрегаты
    total_pnl = sum(
        x["pnl_usd"] for x in closed if isinstance(x["pnl_usd"], (int, float))
    )
    wins = sum(
        1 for x in closed if isinstance(x["pnl_usd"], (int, float)) and x["pnl_usd"] > 0
    )
    losses = sum(
        1 for x in closed if isinstance(x["pnl_usd"], (int, float)) and x["pnl_usd"] < 0
    )
    hit_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    # последовательность PnL для подряд убыточных
    seq = [x["pnl_usd"] for x in closed if isinstance(x["pnl_usd"], (int, float))]
    consec = 0
    max_consec = 0
    for v in seq:
        if v < 0:
            consec += 1
        else:
            max_consec = max(max_consec, consec)
            consec = 0
    max_consec = max(max_consec, consec)

    # атрибуция
    def group_sum(key):
        acc = defaultdict(float)
        for x in closed:
            v = x["pnl_usd"]
            if isinstance(v, (int, float)):
                acc[x[key]] += v
        return sorted(acc.items(), key=lambda kv: kv[1])

    by_symbol = group_sum("symbol")
    by_hour = group_sum("entry_hour_bucket_utc")
    by_exit = group_sum("exit_type")
    by_tag = group_sum("tag")

    # сохраняем CSV
    def save_csv(name, rows, headers):
        p = OUT_DIR / name
        with p.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
        return p

    p_trades = save_csv(
        "postmortem_72h_trades.csv",
        closed,
        [
            "symbol",
            "entry_time",
            "exit_time",
            "entry_hour_bucket_utc",
            "pnl_usd",
            "pnl_pct",
            "qty",
            "exit_type",
            "tag",
        ],
    )

    def save_pairs(name, pairs):
        p = OUT_DIR / name
        with p.open("w", encoding="utf-8") as f:
            f.write("key,pnl_usd\n")
            for k, v in pairs:
                f.write(f"{k},{v}\n")
        return p

    p_sym = save_pairs("postmortem_72h_by_symbol.csv", by_symbol)
    p_hr = save_pairs("postmortem_72h_by_hour.csv", by_hour)
    p_exit = save_pairs("postmortem_72h_by_exit_type.csv", by_exit)
    p_tag = save_pairs("postmortem_72h_by_tag.csv", by_tag)

    # печать краткой сводки
    print("=== Postmortem 72h — краткая сводка ===")
    print(f"Всего закрытых сделок: {len(closed)}")
    print(f"Суммарный PnL (USD): {total_pnl:.2f}")
    print(f"Hit-Rate: {hit_rate:.1f}%  (W={wins} / L={losses})")
    print(f"Макс. серия убыточных подряд: {max_consec}")

    def head5(label, pairs):
        losers = [x for x in pairs if x[1] < 0]
        worst = sorted(losers, key=lambda kv: kv[1])[:5]
        if worst:
            print(f"\nТОП убытков по {label}:")
            for k, v in worst:
                print(f"  • {k:20s} {v:10.2f} USD")

    head5("тикерам", by_symbol)
    head5("часам входа (UTC)", by_hour)
    head5("типу выхода", by_exit)
    head5("стратегическим тегам", by_tag)

    print("\nФайлы отчётов сохранены в:", OUT_DIR)
    print(" - postmortem_72h_trades.csv")
    print(" - postmortem_72h_by_symbol.csv")
    print(" - postmortem_72h_by_hour.csv")
    print(" - postmortem_72h_by_exit_type.csv")
    print(" - postmortem_72h_by_tag.csv")


if __name__ == "__main__":
    main()
