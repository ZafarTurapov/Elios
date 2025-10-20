import os
import sys

import requests

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


def err(msg):
    print(msg, file=sys.stderr)


def get_order(order_id: str):
    url = f"{ALPACA_BASE_URL}/v2/orders/{order_id}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        err("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY (or APCA_* fallbacks).")
        sys.exit(2)
    if len(sys.argv) < 2:
        err(f"Usage: {sys.argv[0]} <order_id>")
        sys.exit(1)
    print(get_order(sys.argv[1]))
