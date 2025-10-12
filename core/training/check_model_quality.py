from core.utils.alpaca_headers import alpaca_headers
from core.utils.paths import TRADE_LOG_PATH
# /root/stockbot/core/training/check_model_quality.py
# -*- coding: utf-8 -*-
"""
Ночной контроль качества модели на реальных сделках.

Что делает:
- Собирает закрытые сделки из core/trading/trade_log.json (BUY/SELL пары)
- Считает win-rate, средний/суммарный PnL за последние N дней (по умолч. 7)
- Считает отдельно нереализованный PnL по открытым позициям (по текущей цене)
- Подтягивает последнюю F1 из training_metrics.csv
- Сравнивает с порогами и шлёт отчёт/предупреждение в Telegram
- Сохраняет подробный отчёт в core/training/model_quality_report.json
- Пишет лог в /root/stockbot/logs/model_quality.log

Важно: мы НЕ принимаем торговых решений — только мониторинг качества.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import yfinance as yf
import csv

ROOT = Path("/root/stockbot")
LOG_PATH = ROOT / "logs" / "model_quality.log"
REPORT_PATH = ROOT / "core" / "training" / "model_quality_report.json"

OPEN_POSITIONS_PATH = ROOT / "core" / "trading" / "open_positions.json"
TRAINING_METRICS_CSV = ROOT / "core" / "training" / "training_metrics.csv"

ALPACA_BASE_URL = "https://data.alpaca.markets/v2"
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

from core.utils.telegram import send_telegram_message

# --- параметры окна и пороги ---
DAYS_LOOKBACK = int(os.getenv("QUALITY_LOOKBACK_DAYS", "7"))
MIN_WIN_RATE = float(os.getenv("QUALITY_MIN_WINRATE", "0.55"))     # 55%+
MIN_F1 = float(os.getenv("QUALITY_MIN_F1", "0.65"))                # 0.65+
MIN_AVG_PNL = float(os.getenv("QUALITY_MIN_AVG_PNL", "0.0"))       # >= 0

def log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts} {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def _alpaca_headers():
    return alpaca_headers()

def latest_price(symbol: str) -> float:
    # quotes -> trades -> bars -> yfinance close
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/stocks/{symbol}/quotes/latest",
                         headers=_alpaca_headers(), timeout=8)
        if r.status_code == 200:
            q = (r.json() or {}).get("quote") or {}
            ap, bp = q.get("ap") or 0.0, q.get("bp") or 0.0
            if ap or bp:
                return float(ap or bp)
    except Exception:
        pass
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/stocks/{symbol}/trades/latest",
                         headers=_alpaca_headers(), timeout=8)
        if r.status_code == 200:
            p = ((r.json() or {}).get("trade") or {}).get("p") or 0.0
            if p:
                return float(p)
    except Exception:
        pass
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/stocks/{symbol}/bars/latest",
                         headers=_alpaca_headers(), timeout=8)
        if r.status_code == 200:
            c = ((r.json() or {}).get("bar") or {}).get("c") or 0.0
            if c:
                return float(c)
    except Exception:
        pass
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

def read_trade_log():
    if not TRADE_LOG_PATH.exists():
        return {}
    try:
        return json.loads(TRADE_LOG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[WARN] read_trade_log: {e}")
        return {}

def read_open_positions():
    if not OPEN_POSITIONS_PATH.exists():
        return {}
    try:
        return json.loads(OPEN_POSITIONS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[WARN] read_open_positions: {e}")
        return {}

def parse_iso(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def last_f1_from_csv() -> float | None:
    if not TRAINING_METRICS_CSV.exists():
        return None
    try:
        with TRAINING_METRICS_CSV.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        return float(last.get("f1_mean") or 0.0)
    except Exception as e:
        log(f"[WARN] last_f1_from_csv: {e}")
        return None

def pair_trades_by_symbol(trades: list[dict]) -> list[dict]:
    """
    Формируем пары BUY->SELL (chrono). Если SELL нет — пара не считается закрытой (уйдёт в 'open').
    """
    buys = []
    closed = []
    for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
        side = (t.get("action") or "").upper()
        if side == "BUY":
            buys.append(t)
        elif side == "SELL":
            # сводим к ближайшему предыдущему BUY
            if buys:
                b = buys.pop(0)
                closed.append({"buy": b, "sell": t})
            else:
                # SELL без BUY — игнор/лог
                pass
    # Оставшиеся buys — открытые
    return closed, buys

def compute_realized_stats(trade_log: dict, since: datetime):
    closed_pairs = []
    open_buys = []
    # Сгруппировано по символу
    for symbol, entries in trade_log.items():
        if not isinstance(entries, list):
            continue
        # фильтруем по окну (берём по buy‑timestamp)
        entries2 = []
        for e in entries:
            ts = parse_iso(e.get("timestamp") or "")
            if ts and ts >= since:
                entries2.append(e)
        if not entries2:
            continue
        c, o = pair_trades_by_symbol(entries2)
        for p in c:
            p["symbol"] = symbol
        for b in o:
            b["symbol"] = symbol
        closed_pairs.extend(c)
        open_buys.extend(o)

    realized = []
    for p in closed_pairs:
        b = p["buy"]; s = p["sell"]; sym = p["symbol"]
        qty = float(b.get("qty") or 0)
        buy_price = float(b.get("price") or 0)
        sell_price = float(s.get("price") or 0)
        pnl = (sell_price - buy_price) * qty
        realized.append({
            "symbol": sym,
            "qty": qty,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "pnl": pnl,
            "buy_time": b.get("timestamp"),
            "sell_time": s.get("timestamp")
        })

    # Нереализованный pnl по открытым
    unrealized = []
    for b in open_buys:
        sym = b["symbol"]
        qty = float(b.get("qty") or 0)
        buy_price = float(b.get("price") or 0)
        mkt = latest_price(sym)
        if qty > 0 and buy_price > 0 and mkt > 0:
            pnl = (mkt - buy_price) * qty
            unrealized.append({
                "symbol": sym,
                "qty": qty,
                "buy_price": buy_price,
                "mkt_price": mkt,
                "unrealized_pnl": pnl,
                "buy_time": b.get("timestamp")
            })

    # сводные метрики по закрытым
    wins = sum(1 for r in realized if r["pnl"] > 0)
    losses = sum(1 for r in realized if r["pnl"] <= 0)
    total_closed = len(realized)
    win_rate = (wins / total_closed) if total_closed else 0.0
    sum_pnl = sum(r["pnl"] for r in realized)
    avg_pnl = (sum_pnl / total_closed) if total_closed else 0.0

    sum_unreal = sum(u["unrealized_pnl"] for u in unrealized)

    return {
        "window_days": DAYS_LOOKBACK,
        "since": since.isoformat(),
        "total_closed": total_closed,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "sum_pnl": round(sum_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "unrealized_sum_pnl": round(sum_unreal, 2),
        "realized": realized,
        "unrealized": unrealized
    }

def main():
    log("📈 MODEL QUALITY CHECK START")
    since = datetime.now(timezone.utc) - timedelta(days=DAYS_LOOKBACK)

    trade_log = read_trade_log()
    open_pos = read_open_positions()
    stats = compute_realized_stats(trade_log, since)

    last_f1 = last_f1_from_csv()
    status = "OK"
    reasons = []

    if last_f1 is not None and last_f1 < MIN_F1:
        status = "WARN"; reasons.append(f"F1 {last_f1:.3f} < {MIN_F1:.3f}")
    if stats["total_closed"] > 0:
        if stats["win_rate"] < MIN_WIN_RATE:
            status = "WARN"; reasons.append(f"WinRate {stats['win_rate']:.2f} < {MIN_WIN_RATE:.2f}")
        if stats["avg_pnl"] < MIN_AVG_PNL:
            status = "WARN"; reasons.append(f"AvgPnL {stats['avg_pnl']:.2f} < {MIN_AVG_PNL:.2f}")
    else:
        reasons.append("Нет закрытых сделок в окне — оцениваем только F1/нереализованный PnL")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reasons": reasons,
        "thresholds": {
            "MIN_F1": MIN_F1,
            "MIN_WIN_RATE": MIN_WIN_RATE,
            "MIN_AVG_PNL": MIN_AVG_PNL,
            "DAYS_LOOKBACK": DAYS_LOOKBACK
        },
        "last_f1": last_f1,
        "trade_stats": stats,
        "open_positions_count": len(open_pos or {})
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"✅ report saved → {REPORT_PATH}")

    # Сообщение в Telegram
    lines = []
    lines.append("📊 *Ежедневный контроль качества модели*")
    if last_f1 is not None:
        lines.append(f"• F1 (последняя): *{last_f1:.3f}* (порог {MIN_F1:.2f})")
    lines.append(f"• Окно: *{DAYS_LOOKBACK}d*, закрыто: *{stats['total_closed']}*")
    lines.append(f"• Win‑rate: *{stats['win_rate']:.2f}* (порог {MIN_WIN_RATE:.2f})")
    lines.append(f"• ΣPnL: *{stats['sum_pnl']:.2f}* | Avg: *{stats['avg_pnl']:.2f}* (порог {MIN_AVG_PNL:.2f})")
    lines.append(f"• Нереализованный ΣPnL (открытые): *{stats['unrealized_sum_pnl']:.2f}*")
    if reasons:
        lines.append("• Причины: " + ", ".join(reasons))
    if status == "WARN":
        lines.append("⚠️ *ВНИМАНИЕ:* качество ниже порогов. Рекомендуется проверить сигналы/датасет.")

    try:
        send_telegram_message("\n".join(lines))
    except Exception as e:
        log(f"[WARN] telegram send failed: {e}")

    log("🏁 MODEL QUALITY CHECK END")

if __name__ == "__main__":
    main()