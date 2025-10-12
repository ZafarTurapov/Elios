from core.utils.alpaca_headers import alpaca_headers
# см. предыдущие шаги — это тот же Entry Orders Gate (окно, MAX_OPEN_POS, MAX_ORDER_NOTIONAL)
from __future__ import annotations
import os, sys, json, math
from datetime import datetime, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo
import requests

TZ = os.environ.get("ELIOS_TZ", "Asia/Tashkent"); Z = ZoneInfo(TZ)
ENTRY_FROM = os.environ.get("ELIOS_ENTRY_FROM", "18:35")
ENTRY_TILL = os.environ.get("ELIOS_ENTRY_TILL", "19:50")   # твои новые границы
MAX_OPEN_POS = int(os.environ.get("ELIOS_MAX_OPEN_POS", "2"))
MAX_NOTIONAL = float(os.environ.get("ELIOS_MAX_ORDER_NOTIONAL", "2500"))

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY  = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC  = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers()

def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=30)
    if r.status_code >= 400: raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()
def jdel(path, params=None):
    r = requests.delete(f"{BASE}{path}", headers=H, params=params or {}, timeout=30)
    if r.status_code >= 400: raise RuntimeError(f"DELETE {path} -> {r.status_code} {r.text}")
    return r.status_code

def in_entry_window(t):
    hh1, mm1 = ENTRY_FROM.split(":"); hh2, mm2 = ENTRY_TILL.split(":")
    start = dtime(int(hh1), int(mm1)); end = dtime(int(hh2), int(mm2))
    tt = dtime(t.hour, t.minute); return start <= tt <= end

def approx_notional(o):
    for k in ("notional","notional_value","order_notional"):
        v = o.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    try: qty=float(o.get("qty") or o.get("filled_qty") or 0)
    except: qty=0.0
    price=None
    for k in ("limit_price","stop_price","filled_avg_price"):
        v=o.get(k)
        try:
            if v is not None: price=float(v); break
        except: pass
    return qty*price if (qty and price) else None

def main():
    if not (KEY and SEC):
        print("[EntryGate] missing keys"); sys.exit(0)
    now = datetime.now(Z)
    pos = jget("/v2/positions"); open_cnt = len(pos) if isinstance(pos, list) else 0

    # все открытые ордера
    from datetime import timezone, timedelta
    after = (datetime.now(timezone.utc) - timedelta(days=2)).replace(microsecond=0).isoformat().replace("+00:00","Z")
    orders = jget("/v2/orders?status=open&direction=asc&after="+after+"&limit=500")
    buys = [o for o in orders if o.get("side")=="buy"]

    if not buys:
        print(f"[EntryGate] {now.isoformat()} — no buy orders; positions={open_cnt}")
        return

    for o in buys:
        oid=o.get("id"); sym=o.get("symbol","?")
        if not in_entry_window(now):
            code=jdel(f"/v2/orders/{oid}")
            print(f"[EntryGate] CANCEL {oid} {sym} — out-of-window {ENTRY_FROM}-{ENTRY_TILL} -> {code}")
            continue
        if open_cnt >= MAX_OPEN_POS:
            code=jdel(f"/v2/orders/{oid}")
            print(f"[EntryGate] CANCEL {oid} {sym} — max-open-pos {open_cnt}/{MAX_OPEN_POS} -> {code}")
            continue
        n=approx_notional(o)
        if n is not None and n > MAX_NOTIONAL:
            code=jdel(f"/v2/orders/{oid}")
            print(f"[EntryGate] CANCEL {oid} {sym} — notional {n:.2f} > {MAX_NOTIONAL:.2f} -> {code}")
            continue
    print(f"[EntryGate] {now.isoformat()} — scanned {len(buys)} buy orders; positions={open_cnt}")

if __name__ == "__main__":
    main()
