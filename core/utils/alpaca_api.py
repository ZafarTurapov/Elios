import os

import requests

from core.utils.alpaca_headers import alpaca_headers

# 🔐 Вшитые ключи Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = alpaca_headers()


def fetch_alpaca_cash():
    try:
        response = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            return float(data.get("cash", 0))
        else:
            print(
                f"[ERROR] Failed to fetch cash: {response.status_code} — {response.text}"
            )
            return 0.0
    except Exception as e:
        print(f"[EXCEPTION] Error while fetching cash: {str(e)}")
        return 0.0
