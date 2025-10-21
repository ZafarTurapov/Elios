from __future__ import annotations
from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta, timezone
import requests

# Торговые ключи подходят и для Data API
TRADE_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip(
    "/"
)
DATA_BASE = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()

# Пороговые параметры (ENV):
SPREAD_MAX_PCT = float(os.getenv("ELIOS_QG_SPREAD_MAX_PCT", "0.003"))  # 0.3%
MIN_DOLLAR_VOL_5 = float(
    os.getenv("ELIOS_QG_MIN_DOLLAR_VOL_5M", "300000")
)  # $300k за 5м
FEED = os.getenv("ELIOS_ALPACA_FEED", "iex")  # iex (free) / sip (paid)


def jget_trade(p, params=None):
    r = requests.get(f"{TRADE_BASE}{p}", headers=H, params=params or {}, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"TRADE GET {p} -> {r.status_code} {r.text}")
    return r.json()


def jget_data(p, params=None):
    params = dict(params or {})
    # для фри-плана обязателен feed=iex, иначе 403/пусто
    params.setdefault("feed", FEED)
    r = requests.get(f"{DATA_BASE}{p}", headers=H, params=params, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"DATA GET {p} -> {r.status_code} {r.text}")
    return r.json()


def latest_quote(sym: str):
    d = jget_data(f"/v2/stocks/{sym}/quotes/latest")
    q = d.get("quote") or {}
    bp = float(q.get("bp") or 0.0)
    ap = float(q.get("ap") or 0.0)
    return bp, ap


def dollar_volume_5m(sym: str):
    d = jget_data(
        f"/v2/stocks/{sym}/bars", {"timeframe": "1Min", "limit": 5, "adjustment": "raw"}
    )
    total = 0.0
    for b in d.get("bars") or []:
        v = float(b.get("v") or 0.0)  # volume (shares)
        c = float(b.get("c") or 0.0)  # close
        total += v * c
    return total


def main():
    # забираем открытые BUY с TRADE API
    after = (
        (datetime.now(timezone.utc) - timedelta(days=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    arr = jget_trade(
        "/v2/orders",
        {"status": "open", "direction": "asc", "after": after, "limit": 200},
    )
    buys = [o for o in arr if (o.get("side") or "").lower() == "buy"]
    if not buys:
        print("[QGate] no open BUY orders")
        return

    canceled = 0
    for o in buys:
        sym = (o.get("symbol") or "?").upper()
        oid = o.get("id")
        try:
            bid, ask = latest_quote(sym)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            dvol5 = dollar_volume_5m(sym)
        except Exception as e:
            print(f"[QGate] {sym}: data err {e} -> CANCEL")
            requests.delete(f"{TRADE_BASE}/v2/orders/{oid}", headers=H, timeout=15)
            canceled += 1
            continue

        if spread_pct > SPREAD_MAX_PCT or dvol5 < MIN_DOLLAR_VOL_5:
            requests.delete(f"{TRADE_BASE}/v2/orders/{oid}", headers=H, timeout=15)
            canceled += 1
            print(f"[QGate] CANCEL {sym} spread={spread_pct:.4%} dvol5=${dvol5:,.0f}")
        else:
            print(f"[QGate] KEEP   {sym} spread={spread_pct:.4%} dvol5=${dvol5:,.0f}")

    print(f"[QGate] scanned={len(buys)} canceled={canceled}")


if __name__ == "__main__":
    main()
