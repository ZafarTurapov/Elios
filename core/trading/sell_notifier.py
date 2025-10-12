from core.utils.alpaca_headers import alpaca_headers
import os
import requests

# === Alpaca creds strictly from ENV (with APCA_* fallback) ===
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

HEADERS = alpaca_headers()

def _alpaca_headers():
    # совместимость с проверками
    return dict(HEADERS)

def ping_account():
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=10)
        return r.ok
    except Exception:
        return False

if __name__ == "__main__":
    print("sell_notifier alpaca auth ok:", ping_account())
