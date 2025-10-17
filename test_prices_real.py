import json

from core.connectors.alpaca_connector import get_last_prices_batch

with open("/root/stockbot/core/trading/open_positions.json", "r") as f:
    data = json.load(f)

symbols = list(data.keys())
print(f"🔍 Проверка цен для: {symbols}")

prices = get_last_prices_batch(symbols)

print("\n📊 Результаты:")
if prices:
    for sym, price in prices.items():
        print(f"{sym}: {price}")
else:
    print("❌ Цены не получены.")
