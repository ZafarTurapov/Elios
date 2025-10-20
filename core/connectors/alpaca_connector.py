import os

from core.utils.alpaca_headers import alpaca_headers

BASE_URL = (
    os.getenv("APCA_API_BASE_URL")
    or os.getenv("ALPACA_BASE_URL")
    or "https://paper-api.alpaca.markets"
)
# core/connectors/alpaca_connector.py

import os

import requests
import yfinance as yf

# === Вшитые ключи Alpaca ===
APCA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or ""
APCA_SEC = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or ""
BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = alpaca_headers()


def submit_order(symbol, qty, side, type="market", time_in_force="gtc"):
    url = f"{BASE_URL}/v2/orders"
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": type,
        "time_in_force": time_in_force,
    }
    response = requests.post(url, json=order, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"{response.status_code} — {response.text}")
    return response.json()


def get_account_equity():
    url = f"{BASE_URL}/v2/account"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return float(response.json()["equity"])


def get_positions():
    url = f"{BASE_URL}/v2/positions"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"{response.status_code} — {response.text}")
    return response.json()


def get_positions_with_pnl():
    url = f"{BASE_URL}/v2/positions"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    positions_raw = response.json()

    result = {}
    for p in positions_raw:
        symbol = p["symbol"]
        qty = int(float(p["qty"]))
        entry = float(p["avg_entry_price"])
        current = float(p["current_price"])
        pnl_pct = float(p["unrealized_plpc"])
        pnl_usd = float(p["unrealized_pl"])
        result[symbol] = {
            "qty": qty,
            "entry_price": entry,
            "current_price": current,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
        }
    return result


def close_position(symbol):
    url = f"{BASE_URL}/v2/positions/{symbol}"
    response = requests.delete(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"{response.status_code} — {response.text}")
    return response.json()


def get_last_prices(symbols, days=15):
    prices = {}

    for symbol in symbols:
        try:
            data = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
            if data is not None and not data.empty:
                close_data = data["Close"].dropna().squeeze()
                price = (
                    close_data.iloc[-1] if hasattr(close_data, "iloc") else close_data
                )
                prices[symbol] = float(price)
            else:
                print(f"[WARN] Нет данных по {symbol}")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue

    if not prices:
        raise ValueError("❌ Нет ни одного тикера с данными — прерываем.")

    return prices


def get_account_summary():
    url = f"{BASE_URL}/v2/account"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"{response.status_code} — {response.text}")
