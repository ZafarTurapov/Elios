from core.utils.alpaca_headers import alpaca_headers

# core/trading/execution_monitor.py

import os
import json
import requests

POSITIONS_PATH = "/core/trading/open_positions.json"

# 🔐 Вшитые ключи Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = alpaca_headers()


def monitor_orders():
    url = f"{ALPACA_BASE_URL}/v2/orders?status=closed&limit=50"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        orders = response.json()

        if not os.path.exists(POSITIONS_PATH):
            print("[WARN] open_positions.json не найден.")
            return

        with open(POSITIONS_PATH, "r") as f:
            positions = json.load(f)

        updated = positions.copy()
        closed_symbols = []

        for order in orders:
            if order["status"] != "filled":
                continue
            symbol = order["symbol"]
            side = order["side"]
            if side == "sell" and symbol in updated:
                closed_symbols.append(symbol)
                del updated[symbol]

        if closed_symbols:
            print(f"✅ Закрыты и удалены позиции: {', '.join(closed_symbols)}")
            with open(POSITIONS_PATH, "w") as f:
                json.dump(updated, f, indent=2)
        else:
            print("ℹ️ Нет новых закрытых позиций.")

    except Exception as e:
        print(f"[ERROR] Ошибка проверки ордеров: {e}")


if __name__ == "__main__":
    monitor_orders()
