from core.utils.alpaca_headers import alpaca_headers
import os
import requests
from dotenv import load_dotenv

# === Загрузка переменных из .env ===
load_dotenv(dotenv_path="/root/stockbot/core/trading/.env")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

HEADERS = alpaca_headers()

params = {"symbols": "AAPL", "timeframe": "1Day", "limit": 5}

response = requests.get(BASE_URL, headers=HEADERS, params=params)

print(f"Status Code: {response.status_code}")
try:
    print(response.json())
except Exception as e:
    print(f"[ERROR] Не удалось распарсить JSON: {e}")
