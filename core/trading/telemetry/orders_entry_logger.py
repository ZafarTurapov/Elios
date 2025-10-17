from __future__ import annotations

from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
"""
Orders Entry Logger -> trade_log.json
Назначение:
- Раз в 2 мин тянет /v2/orders (status=all) с пагинацией
- Для BUY: создаёт/обновляет запись в trade_log (entry по filled_at)
- Для SELL: находит последнюю "open" запись по тому же символу и проставляет exit_time, статус closed
- Пишет атомарно в logs/trade_log.json (если нет – создаёт), сохраняет офсет в temp/orders_entry_state.json
Важно: это телеметрия, не меняет торговлю. Нужна для качественной диагностики и постмортема.

ENV:
  ALPACA_BASE_URL, APCA_API_KEY_ID/ALPACA_API_KEY_ID, APCA_API_SECRET_KEY/ALPACA_API_SECRET_KEY
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path("/root/stockbot")
LOG = ROOT / "logs" / "trade_log.json"
TMP = ROOT / "temp"
TMP.mkdir(parents=True, exist_ok=True)
STATE = TMP / "orders_entry_state.json"

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = alpaca_headers(content_json=True)


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json(), r.headers


def read_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json_atomic(p: Path, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def iso(dt: datetime) -> str:
    return (
        dt.replace(microsecond=0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def get_orders_since(after_iso: str):
    out = []
    token = None
    while True:
        params = {"status": "all", "direction": "asc", "after": after_iso, "limit": 200}
        if token:
            params["page_token"] = token
        data, hdr = jget("/v2/orders", params)
        if not isinstance(data, list) or not data:
            break
        out.extend(data)
        token = hdr.get("x-next-page-token") or hdr.get("X-Next-Page-Token")
        if not token:
            break
    return out


def load_log():
    data = read_json(LOG, [])
    if isinstance(data, dict) and isinstance(data.get("trades"), list):
        return data["trades"]
    if isinstance(data, list):
        return data
    return []


def save_log(trades: list):
    # Храним как список (проще). Если хочешь структуру {"trades": [...]}, поменяем тут.
    write_json_atomic(LOG, trades)


def find_open_trade_for_symbol(trades, symbol: str):
    # Берём самую свежую открытую по символу
    for rec in reversed(trades):
        if rec.get("symbol") == symbol and str(rec.get("status", "")).lower() in (
            "open",
            "opened",
            "buy_filled",
        ):
            return rec
    return None


def main():
    if not (KEY and SEC):
        print("[ERR] Alpaca keys missing")
        return
    st = read_json(STATE, {})
    last = st.get("after")  # ISO
    if not last:
        last = iso(datetime.now(timezone.utc) - timedelta(days=7))  # стартуем с 7 дней
    orders = get_orders_since(last)
    if not orders:
        print("[EntryLogger] no new orders since", last)
        return

    trades = load_log()
    newest_ts = last

    for o in orders:
        created_at = o.get("created_at")
        if created_at:
            newest_ts = max(newest_ts, created_at)
        side = (o.get("side") or "").lower()
        sym = (o.get("symbol") or "?").upper()
        oid = o.get("id")
        filled_at = o.get("filled_at")
        qty = o.get("filled_qty") or o.get("qty") or "0"
        try:
            qtyf = float(qty)
        except:
            qtyf = 0.0
        avg_price = None
        for k in ("filled_avg_price", "limit_price", "stop_price"):
            v = o.get(k)
            try:
                if v is not None:
                    avg_price = float(v)
                    break
            except:
                pass

        # BUY filled -> создаём/обновляем open запись
        if side == "buy" and filled_at and qtyf > 0:
            # если уже есть открытая по символу — не задваиваем, но отметим обновление
            open_rec = find_open_trade_for_symbol(trades, sym)
            if open_rec is None:
                trades.append(
                    {
                        "order_id": oid,
                        "symbol": sym,
                        "side": "long",
                        "qty": qtyf,
                        "entry_time": filled_at,
                        "entry_price": avg_price,
                        "status": "open",
                    }
                )
                print(f"[EntryLogger] OPEN {sym} qty={qtyf} at {filled_at}")
            else:
                # если у открытой не было entry_time — заполним
                if not open_rec.get("entry_time"):
                    open_rec["entry_time"] = filled_at
                if avg_price is not None and not open_rec.get("entry_price"):
                    open_rec["entry_price"] = avg_price

        # SELL filled -> закрываем последнюю открытую по этому символу
        if side == "sell" and filled_at and qtyf > 0:
            open_rec = find_open_trade_for_symbol(trades, sym)
            if open_rec is not None:
                open_rec["exit_time"] = filled_at
                open_rec["exit_price"] = avg_price
                open_rec["status"] = "closed"
                print(f"[EntryLogger] CLOSE {sym} at {filled_at}")
            else:
                # нет открытой — создадим одиночную закрытую запись для консистентности
                trades.append(
                    {
                        "order_id": oid,
                        "symbol": sym,
                        "side": "long",
                        "qty": qtyf,
                        "entry_time": "",  # неизвестно
                        "exit_time": filled_at,
                        "exit_price": avg_price,
                        "status": "closed",
                    }
                )
                print(f"[EntryLogger] CLOSE(no-open) {sym} at {filled_at}")

    save_log(trades)
    write_json_atomic(STATE, {"after": newest_ts})
    print(f"[EntryLogger] processed {len(orders)} orders; state after={newest_ts}")


if __name__ == "__main__":
    main()
