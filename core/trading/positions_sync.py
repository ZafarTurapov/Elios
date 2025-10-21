from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT = Path("/root/stockbot")


def _load_env(paths=("/root/stockbot/.env.local", "/root/stockbot/.env", ".env")):
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v
        except Exception:
            pass


_load_env()

# Telegram helper (мягкий импорт)
try:
    from core.utils.telegram import send_telegram_message
except Exception:

    def send_telegram_message(text: str):
        print(f"[TG] {text}")


# --- Alpaca ENV/headers (нормализация имён) ---
API_KEY = (
    os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
    or ""
).strip()

API_SECRET = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET_KEY")
    or ""
).strip()

BASE_URL = (
    (
        os.getenv("APCA_API_BASE_URL")
        or os.getenv("ALPACA_API_BASE_URL")
        or "https://paper-api.alpaca.markets"
    )
    .strip()
    .rstrip("/")
)

HEADERS = alpaca_headers()

OUT_POS = ROOT / "core/trading/open_positions.json"
OUT_SUMM = ROOT / "core/trading/account_summary.json"


def _get_json(url: str, **kw):
    r = requests.get(url, headers=HEADERS, timeout=20, **kw)
    r.raise_for_status()
    return r.json()


def main():
    if not API_KEY or not API_SECRET:
        raise SystemExit("APCA keys missing (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")

    # account
    acct = _get_json(f"{BASE_URL}/v2/account")
    now_local = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    equity = float(acct.get("equity") or 0.0)
    last_equity = float(acct.get("last_equity") or equity)

    summary = {
        "ts": now_local,
        "equity": equity,
        "cash": float(acct.get("cash") or 0.0),
        "buying_power": float(acct.get("buying_power") or 0.0),
        "portfolio_value": float(acct.get("portfolio_value") or equity),
        "pl_day": equity - last_equity,
        "currency": acct.get("currency") or "USD",
        "status": acct.get("status") or "?",
        "account_number": acct.get("account_number") or "",
        "paper": acct.get("paper"),
    }

    # positions
    pos_resp = requests.get(f"{BASE_URL}/v2/positions", headers=HEADERS, timeout=20)
    if pos_resp.status_code == 404:
        positions = []
    else:
        pos_resp.raise_for_status()
        positions = pos_resp.json()

    summary["positions_count"] = len(positions)

    # normalized positions map
    snap = {"__timestamp__": datetime.now(timezone.utc).isoformat()}
    for p in positions or []:
        try:
            sym = (p.get("symbol") or "").upper()
            qty = float(p.get("qty") or 0)
            if qty <= 0 or not sym:
                continue
            snap[sym] = {
                "qty": qty,
                "avg_price": float(p.get("avg_entry_price") or 0),
                "current_price": float(
                    p.get("current_price") or p.get("lastday_price") or 0
                ),
                "unrealized_pl": float(p.get("unrealized_pl") or 0),
                "unrealized_plpc": float(p.get("unrealized_plpc") or 0),
                "qty_available": float(p.get("qty_available") or 0),
                "side": p.get("side") or "",
                "exchange": p.get("exchange") or "",
            }
        except Exception:
            continue

    OUT_POS.parent.mkdir(parents=True, exist_ok=True)
    OUT_POS.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_SUMM.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    send_telegram_message(
        "📦 Elios — Positions Sync\n"
        f"💼 Equity: ${summary['equity']:,.2f} | Cash: ${summary['cash']:,.2f} | BP: ${summary['buying_power']:,.2f}\n"
        f"📊 P/L day: ${summary['pl_day']:,.2f} | Positions: {summary['positions_count']}"
    )
    print("✅ positions_sync: done")


if __name__ == "__main__":
    main()
