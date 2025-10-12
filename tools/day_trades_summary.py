# /root/stockbot/tools/day_trades_summary.py
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TRADE_LOG = Path("/root/stockbot/core/trading/trade_log.json")
LOCAL_TZ  = ZoneInfo("Asia/Tashkent")
UTC_TZ    = ZoneInfo("UTC")

def parse_iso(ts):
    if not ts: return None
    s = str(ts).strip().replace("Z","+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_TZ)
        return dt.astimezone(UTC_TZ)
    except Exception:
        return None

def best_ts(item: dict):
    ts = item.get("timestamp") or item.get("created_at") or item.get("updated_at") or item.get("submitted_at")
    if not ts and isinstance(item.get("alpaca_response"), dict):
        ar = item["alpaca_response"]
        ts = ar.get("created_at") or ar.get("updated_at") or ar.get("submitted_at") or ar.get("filled_at")
    return parse_iso(ts) if ts else None

def iter_trade_records(d):
    if isinstance(d, list):
        for r in d:
            if isinstance(r, dict): yield r
    elif isinstance(d, dict):
        for sym, arr in d.items():
            if isinstance(arr, list):
                for r in arr:
                    if isinstance(r, dict):
                        r.setdefault("symbol", sym)
                        yield r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD локальная (Asia/Tashkent). По умолчанию — вчера.")
    args = ap.parse_args()

    today_local = datetime.now(LOCAL_TZ).date()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else (today_local - timedelta(days=1))

    if not TRADE_LOG.exists():
        print("❌ trade_log.json не найден:", TRADE_LOG)
        return

    data = json.loads(TRADE_LOG.read_text(encoding="utf-8"))

    buys = []
    sells = []
    other = []
    gross_buy_notional = 0.0

    for it in iter_trade_records(data):
        ts = best_ts(it)
        if not ts: 
            continue
        if ts.astimezone(LOCAL_TZ).date() != target_date:
            continue
        sym = it.get("symbol", "?")
        act = (it.get("action") or it.get("module") or "").upper()
        qty = None
        # qty может быть строкой в alpaca_response
        qty = it.get("qty") or it.get("quantity") or (it.get("alpaca_response") or {}).get("qty")
        try:
            qty = float(qty) if qty is not None else None
        except Exception:
            qty = None
        price = it.get("price") or it.get("entry") or it.get("exit")
        try:
            price = float(price) if price is not None else None
        except Exception:
            price = None

        row = (ts.isoformat(), act, sym, qty, price, it.get("pnl"))
        if act == "BUY":
            buys.append(row)
            if qty is not None and price is not None:
                gross_buy_notional += float(qty) * float(price)
        elif act.startswith("SELL"):
            sells.append(row)
        else:
            other.append(row)

    buys.sort(key=lambda x: x[0])
    sells.sort(key=lambda x: x[0])
    other.sort(key=lambda x: x[0])

    print(f"📅 {target_date} (Asia/Tashkent)")
    print(f"BUY: {len(buys)}  SELL: {len(sells)}  OTHER: {len(other)}")
    print(f"🧾 Пример первых 10 BUY:")
    for r in buys[:10]:
        print("  ", r)
    if sells:
        print(f"🧾 Пример первых 10 SELL:")
        for r in sells[:10]:
            print("  ", r)
    print(f"💵 Грязный оборот покупок (оценка): ${gross_buy_notional:,.2f}")

if __name__ == "__main__":
    main()
