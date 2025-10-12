from core.utils.paths import TRADE_LOG_PATH
# core/trading/pnl_tracker.py

import os
import json
from core.utils.telegram import send_telegram_message


def load_log():
    if not os.path.exists(TRADE_LOG_PATH):
        print("❌ trade_log.json не найден.")
        return []
    with open(TRADE_LOG_PATH, "r") as f:
        raw = json.load(f)

    trades = []
    for symbol, entries in raw.items():
        for trade in entries:
            if "exit_price" in trade and "entry_price" in trade:
                try:
                    change = (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"] * 100
                    trades.append({
                        "symbol": trade["symbol"],
                        "change_pct": round(change, 2),
                        "reason": trade.get("reason", ""),
                        "timestamp_exit": trade.get("timestamp_exit", "")
                    })
                except:
                    continue
    return trades

def analyze(trades):
    total = len(trades)
    wins = [t for t in trades if t["change_pct"] > 0]
    losses = [t for t in trades if t["change_pct"] <= 0]
    total_pct = sum(t["change_pct"] for t in trades)

    sorted_trades = sorted(trades, key=lambda x: -abs(x["change_pct"]))[:3]

    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "total_pct": round(total_pct, 2),
        "top": sorted_trades
    }

def send_report(stats):
    if stats["total"] == 0:
        send_telegram_message("📭 Нет завершённых сделок для анализа.")
        return

    msg = (
        f"💰 Итоги торговли:\n"
        f"📌 Всего сделок: {stats['total']}\n"
        f"📈 Прибыльных: {stats['wins']} | 📉 Убыточных: {stats['losses']}\n"
        f"💹 Доход: {stats['total_pct']}%\n"
    )

    if stats["top"]:
        msg += "\n🔍 Топ-3:\n"
        for t in stats["top"]:
            emoji = "📈" if t["change_pct"] > 0 else "📉"
            msg += f"{emoji} {t['symbol']}: {t['change_pct']}%\n"

    send_telegram_message(msg)

def main():
    trades = load_log()
    stats = analyze(trades)
    send_report(stats)

if __name__ == "__main__":
    main()