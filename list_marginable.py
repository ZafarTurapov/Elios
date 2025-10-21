from core.utils.alpaca_headers import alpaca_headers

# list_marginable.py

import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="/core/trading/.env")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

HEADERS = alpaca_headers(content_json=True)


def main():
    url = f"{BASE_URL}/v2/positions"
    response = requests.get(url, headers=HEADERS)
    positions = response.json()

    marginable = []
    not_marginable = []

    for pos in positions:
        symbol = pos["symbol"]
        is_marginable = pos.get("asset_marginable", True)

        if not is_marginable:
            not_marginable.append(symbol)
        else:
            marginable.append(symbol)

    print(f"✅ Marginable: {len(marginable)}")
    print(f"🔴 Not Marginable: {len(not_marginable)}")
    print("\n🧨 Исключённые тикеры (not marginable):")
    for sym in not_marginable:
        print(f"  - {sym}")


if __name__ == "__main__":
    main()
