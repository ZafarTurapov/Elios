import os
import requests
from datetime import datetime, timedelta

# === Получаем ключи из окружения ===
ALPACA_API_KEY = os.getenv("APCA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"

HEADERS = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}

# === Получаем дату начала (3 дня назад)
start_time = (datetime.utcnow() - timedelta(days=3)).isoformat() + "Z"

params = {"symbols": "AAPL", "timeframe": "1Day", "start": start_time, "limit": 5}

print("📡 Запрос баров (AAPL за 3 дня)...")
r = requests.get(DATA_URL, headers=HEADERS, params=params)

print(f"📥 HTTP статус: {r.status_code}")
print(f"📄 Тело ответа:\n{r.text}\n")

try:
    bars = r.json()
    aapl_data = bars["bars"]["AAPL"]
    print("📊 Полученные бары:")
    for bar in aapl_data:
        print(f"🕒 {bar['t']} — close: {bar['c']}, volume: {bar['v']}")
except Exception as e:
    print("⚠️ Ошибка при обработке JSON:", e)

# === Тестовый ордер (MARKET BUY 1 шт AAPL)
print("\n📤 Отправка тестового ордера...")
order_url = f"{BASE_URL}/v2/orders"
order_data = {
    "symbol": "AAPL",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "time_in_force": "gtc",
}

r = requests.post(order_url, json=order_data, headers=HEADERS)
if r.status_code in [200, 201]:
    print("✅ Ордер успешно отправлен!")
else:
    print("❌ Ошибка при отправке ордера:", r.text)
