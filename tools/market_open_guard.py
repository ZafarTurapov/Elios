#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import requests

# Подтянуть .env.local (как unit)
envf = Path("/root/stockbot/.env.local")
if envf.exists():
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def clean(x):
    return (x or "").strip().strip('"').strip("'")


base = clean(os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")).rstrip(
    "/"
)
k = clean(os.getenv("ALPACA_API_KEY", ""))
s = clean(os.getenv("ALPACA_SECRET_KEY", ""))

hdr = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s, "Accept": "application/json"}

try:
    r = requests.get(f"{base}/v2/clock", headers=hdr, timeout=6)
    r.raise_for_status()
    is_open = bool(r.json().get("is_open"))
    sys.exit(0 if is_open else 1)
except Exception as e:
    print(f"[market_open_guard] WARN: {e}", file=sys.stderr)
    sys.exit(1)  # при ошибке — лучше пропустить запуск
