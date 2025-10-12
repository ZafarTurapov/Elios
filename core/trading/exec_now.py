from core.utils.alpaca_headers import alpaca_headers
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, uuid, sys
from pathlib import Path
from datetime import datetime, timezone
import requests

BASE = Path(__file__).resolve().parent
MERGED = BASE / "signals_merged.json"
LONGS  = BASE / "signals.json"
SHORTS = BASE / "signals_short.json"
ORDERS_LOG = BASE / "orders_now.jsonl"

ALPACA_API_BASE = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")
ALPACA_KEY  = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SEC  = os.getenv("ALPACA_API_SECRET_KEY")
DRY_RUN     = os.getenv("DRY_RUN", "1") == "1" or not (ALPACA_KEY and ALPACA_SEC)

EQUITY_USD   = float(os.getenv("ELIOS_EQUITY_USD", "10000"))
POSITION_USD = float(os.getenv("ELIOS_POSITION_USD", "0"))
ORDER_TYPE   = os.getenv("ELIOS_ORDER_TYPE", "market")  # market|limit
SLIPPAGE_BPS = float(os.getenv("ELIOS_SLIPPAGE_BPS", "5"))
TIF          = os.getenv("ELIOS_TIF", "day")

def _load_any():
    if MERGED.exists():
        data = json.loads(MERGED.read_text(encoding="utf-8"))
        out = []
        for r in data:
            side = str(r.get("side","LONG")).upper()
            out.append({
                "symbol": r.get("symbol"),
                "side": "sell_short" if side=="SHORT" else "buy",
                "entry": float(r.get("entry") or 0),
                "qty_hint": r.get("qty_hint"),
            })
        return out
    longs = json.loads(LONGS.read_text(encoding="utf-8")) if LONGS.exists() else []
    shorts = json.loads(SHORTS.read_text(encoding="utf-8")) if SHORTS.exists() else []
    out = []
    for r in shorts:
        out.append({"symbol": r.get("symbol"), "side":"sell_short", "entry": float(r.get("entry") or 0), "qty_hint": r.get("qty_hint")})
    for r in longs:
        out.append({"symbol": r.get("symbol"), "side":"buy", "entry": float(r.get("entry") or 0), "qty_hint": r.get("qty_hint")})
    return out

def _qty(entry: float, qty_hint, n:int)->int:
    if qty_hint:
        try:
            q=int(qty_hint); 
            if q>0: return q
        except: pass
    if entry<=0: return 0
    usd = POSITION_USD if POSITION_USD>0 else max(100.0, EQUITY_USD/max(1,n))
    return max(0, int(usd//entry))

def _alpaca_headers():
    return alpaca_headers(content_json=True)

def _place(symbol, side, qty, entry):
    url = f"{ALPACA_API_BASE}/v2/orders"
    payload = {
        "symbol": symbol,
        "side": "sell" if side=="sell_short" else "buy",
        "type": ORDER_TYPE,
        "time_in_force": TIF,
        "qty": str(qty),
        "client_order_id": f"execnow-{symbol}-{datetime.now(timezone.utc).date()}-{uuid.uuid4().hex[:6]}",
    }
    if ORDER_TYPE=="limit":
        bump = (SLIPPAGE_BPS/10000.0)*entry
        payload["limit_price"] = f"{(entry - bump) if side=='sell_short' else (entry + bump):.4f}"
    r = requests.post(url, headers=_alpaca_headers(), data=json.dumps(payload), timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    sigs = _load_any()
    if not sigs:
        print("[NOW] Нет сигналов (merged/long/short)."); return
    n = len(sigs)
    print(f"[NOW] Сигналов: {n}  DRY_RUN={'ON' if DRY_RUN else 'OFF (Alpaca)'}  TYPE={ORDER_TYPE}")

    results=[]
    for i,s in enumerate(sigs,1):
        sym = s["symbol"]; side=s["side"]; entry=float(s["entry"] or 0)
        if not sym or entry<=0:
            print(f"  [{i}/{n}] SKIP {sym}: bad entry.")
            continue
        qty=_qty(entry, s.get("qty_hint"), n)
        if qty<=0:
            print(f"  [{i}/{n}] SKIP {sym}: qty=0.")
            continue
        if DRY_RUN:
            info={"ts":datetime.now(timezone.utc).isoformat(),"mode":"DRY","symbol":sym,"side":side,"qty":qty,"entry_ref":entry}
            print(f"  [{i}/{n}] DRY {('SHORT' if side=='sell_short' else 'BUY '):>5} {sym:<6} x{qty} @~{entry:.2f}")
        else:
            try:
                resp=_place(sym,side,qty,entry)
                info={"ts":datetime.now(timezone.utc).isoformat(),"mode":"ALPACA","symbol":sym,"side":side,"qty":qty,"entry_ref":entry,"resp":resp}
                oid=resp.get("id","?"); print(f"  [{i}/{n}] SENT {('SHORT' if side=='sell_short' else 'BUY '):>5} {sym:<6} x{qty} (id={oid})")
            except Exception as e:
                info={"ts":datetime.now(timezone.utc).isoformat(),"mode":"ALPACA_ERR","symbol":sym,"side":side,"qty":qty,"entry_ref":entry,"error":str(e)}
                print(f"  [{i}/{n}] FAIL {('SHORT' if side=='sell_short' else 'BUY '):>5} {sym:<6} x{qty} -> {e}", file=sys.stderr)
        results.append(info); time.sleep(0.15)

    with ORDERS_LOG.open("a", encoding="utf-8") as fw:
        for r in results: fw.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[NOW] Готово. Лог: {ORDERS_LOG}")

if __name__=="__main__":
    main()
