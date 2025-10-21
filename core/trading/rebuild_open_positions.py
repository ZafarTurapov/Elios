from core.utils.paths import TRADE_LOG_PATH
import json
from datetime import datetime


DEFAULT_TP = 0.05
DEFAULT_SL = -0.03


def rebuild_positions():
    with open(TRADE_LOG_PATH, "r") as f:
        trade_data = json.load(f)

    positions = {}
    for symbol, trades in trade_data.items():
        if not isinstance(trades, list) or len(trades) == 0:
            continue

        # Ищем самую последнюю покупку (BUY)
        for entry in reversed(trades):
            if entry.get("action", "").upper() == "BUY":
                qty = entry.get("qty", 1)
                price = entry.get("entry_price", entry.get("price", 0.0))
                timestamp = entry.get("timestamp", datetime.utcnow().isoformat())

                if price == 0:
                    continue

                positions[symbol] = {
                    "qty": qty,
                    "entry_price": price,
                    "timestamp": timestamp,
                    "take_profit": round(price * (1 + DEFAULT_TP), 2),
                    "stop_loss": round(price * (1 + DEFAULT_SL), 2),
                }
                break  # одна позиция на тикер

    return positions


if __name__ == "__main__":
    result = rebuild_positions()
    print(json.dumps(result, indent=2))
