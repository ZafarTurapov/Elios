# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path("/root/stockbot")
OUT_DIR = ROOT / "logs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
if not (KEY and SEC):
    print("[ERR] Alpaca API keys not set", file=sys.stderr)
    sys.exit(2)
H = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC, "Accept": "application/json"}


def jget(path, params=None):
    u = f"{BASE}{path}"
    r = requests.get(u, headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def save_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")


now = datetime.now(timezone.utc)
since = now - timedelta(hours=72)

# 1) Аккаунт и кривая equity (7D, 1D и 5Min для последнего дня)
acc = jget("/v2/account")
equity = float(acc.get("equity", 0))
last_eq = float(acc.get("last_equity") or acc.get("equity_prev_day") or 0)
day_change_pct = ((equity - last_eq) / last_eq * 100) if last_eq > 0 else None

ph_7d = jget(
    "/v2/account/portfolio/history",
    {"period": "7D", "timeframe": "1D", "extended_hours": "true"},
)
eq_rows = []
for t, eq in zip(ph_7d.get("timestamp", []), ph_7d.get("equity", [])):
    dt = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    eq_rows.append({"timestamp_utc": dt, "equity": eq})
save_csv(OUT_DIR / "equity_7d_1d.csv", eq_rows, ["timestamp_utc", "equity"])

# 2) Открытые позиции — нереализованные PnL
pos = jget("/v2/positions")
pos_rows = []
for p in pos:
    pos_rows.append(
        {
            "symbol": p.get("symbol"),
            "side": p.get("side"),
            "qty": p.get("qty"),
            "avg_entry_price": p.get("avg_entry_price"),
            "current_price": p.get("current_price"),
            "unrealized_pl": p.get("unrealized_pl"),
            "unrealized_plpc": p.get("unrealized_plpc"),
            "market_value": p.get("market_value"),
            "asset_class": p.get("asset_class"),
        }
    )
save_csv(
    OUT_DIR / "positions_open_now.csv",
    pos_rows,
    [
        "symbol",
        "side",
        "qty",
        "avg_entry_price",
        "current_price",
        "unrealized_pl",
        "unrealized_plpc",
        "market_value",
        "asset_class",
    ],
)


# 3) Все ордера за 72ч (status=all) — ловим проблемы исполнения
def iso(dt):
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


orders = jget(
    "/v2/orders",
    {
        "status": "all",
        "after": iso(since),
        "until": iso(now),
        "direction": "asc",
        "limit": 500,
    },
)
ord_rows = []
rej = []
for o in orders:
    rec = {
        "id": o.get("id"),
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "type": o.get("type"),
        "time_in_force": o.get("time_in_force"),
        "status": o.get("status"),
        "created_at": o.get("created_at"),
        "filled_at": o.get("filled_at"),
        "expired_at": o.get("expired_at"),
        "canceled_at": o.get("canceled_at"),
        "failed_at": o.get("failed_at"),
        "filled_qty": o.get("filled_qty"),
        "qty": o.get("qty"),
        "limit_price": o.get("limit_price"),
        "stop_price": o.get("stop_price"),
        "trail_percent": o.get("trail_percent"),
        "trail_price": o.get("trail_price"),
        "hwm": o.get("hwm"),
        "extended_hours": o.get("extended_hours"),
        "order_class": o.get("order_class"),
        "client_order_id": o.get("client_order_id"),
        "notional": o.get("notional"),
        "reject_reason": o.get("reject_reason"),
        "subtag": o.get("subtag"),
        "source": o.get("source"),
    }
    ord_rows.append(rec)
    if rec["status"] in ("rejected", "canceled"):
        rej.append(rec)

save_csv(
    OUT_DIR / "orders_72h_all.csv",
    ord_rows,
    [
        "id",
        "symbol",
        "side",
        "type",
        "time_in_force",
        "status",
        "created_at",
        "filled_at",
        "expired_at",
        "canceled_at",
        "failed_at",
        "qty",
        "filled_qty",
        "limit_price",
        "stop_price",
        "order_class",
        "reject_reason",
        "extended_hours",
        "client_order_id",
        "source",
        "subtag",
    ],
)

# 4) Сводка по отказам
rej_by_reason = defaultdict(int)
rej_by_symbol = defaultdict(int)
for r in rej:
    rej_by_reason[(r.get("reject_reason") or r.get("status") or "NA")] += 1
    rej_by_symbol[r.get("symbol", "?")] += 1


def pairs(dd):
    return sorted(((k, v) for k, v in dd.items()), key=lambda kv: kv[1], reverse=True)


save_csv(
    OUT_DIR / "orders_72h_rejected_summary.csv",
    [{"key": k, "count": v} for k, v in pairs(rej_by_reason)],
    ["key", "count"],
)
save_csv(
    OUT_DIR / "orders_72h_rejected_by_symbol.csv",
    [{"key": k, "count": v} for k, v in pairs(rej_by_symbol)],
    ["key", "count"],
)

# 5) Печать краткого итога
print("=== Alpaca Probe 72h — краткая сводка ===")
print(
    f"Equity now: {equity:.2f}  | Last day equity: {last_eq:.2f}  | Day Δ%: {day_change_pct:.2f}%"
    if day_change_pct is not None
    else f"Equity now: {equity:.2f}  | Last day equity: {last_eq:.2f}  | Day Δ%: NA"
)

print(f"Открытых позиций: {len(pos_rows)}")
worst = sorted(
    [r for r in pos_rows if r.get("unrealized_pl") is not None],
    key=lambda r: float(r["unrealized_pl"]),
    reverse=False,
)[:5]
if worst:
    print("\nТОП-5 текущих убыточных позиций (unrealized):")
    for r in worst:
        print(
            f"  • {r['symbol']:6s}  UPL: {r['unrealized_pl']:>10}  UPL%: {r['unrealized_plpc'] or 'NA'}  Qty:{r['qty']}"
        )

print(f"\nВсего ордеров за 72ч: {len(ord_rows)}  | rejected/canceled: {len(rej)}")
top_reasons = pairs(rej_by_reason)[:5]
if top_reasons:
    print("ТОП причин отказов:")
    for k, v in top_reasons:
        print(f"  • {k}: {v}")

print("\nФайлы сохранены в logs/diagnostics/:")
print(" - equity_7d_1d.csv")
print(" - positions_open_now.csv")
print(" - orders_72h_all.csv")
print(" - orders_72h_rejected_summary.csv")
print(" - orders_72h_rejected_by_symbol.csv")
