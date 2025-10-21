# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
import csv

ROOT = Path("/root/stockbot")
OUT = ROOT / "logs" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TZ = os.getenv("ELIOS_TZ", "Asia/Tashkent")
Z = ZoneInfo(TZ)
BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = {
    "APCA-API-KEY-ID": KEY or "",
    "APCA-API-SECRET-KEY": SEC or "",
    "Accept": "application/json",
}


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def save_csv(path, rows, headers):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})


def main():
    # «вчера» в Ташкенте
    now_local = datetime.now(Z)
    y = (now_local.date() - timedelta(days=1)).isoformat()  # YYYY-MM-DD
    # 1) Интрадея equity 5м
    ph = jget(
        "/v2/account/portfolio/history",
        {"period": "2D", "timeframe": "5Min", "extended_hours": "true", "date_end": y},
    )
    stamps = ph.get("timestamp", []) or []
    equities = ph.get("equity", []) or []
    rows = []
    for t, eq in zip(stamps, equities):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        if dt.astimezone(Z).date().isoformat() == y:
            rows.append({"ts_utc": dt.isoformat(), "equity": f"{float(eq):.2f}"})
    if rows:
        save_csv(OUT / f"y_intraday_5m_{y}.csv", rows, ["ts_utc", "equity"])
    # 2) Ордера и филлы за вчера
    a = f"{y}T00:00:00Z"
    b = (
        datetime.fromisoformat(y) + timedelta(days=1)
    ).date().isoformat() + "T00:00:00Z"
    orders = jget(
        "/v2/orders",
        {"status": "all", "after": a, "until": b, "direction": "asc", "limit": 500},
    )
    fills = []
    try:
        fills = jget(
            "/v2/account/activities/FILL",
            {"after": a, "until": b, "direction": "asc", "page_size": 100},
        )
    except Exception:
        fills = []
    save_csv(
        OUT / f"y_orders_{y}.csv",
        orders,
        [
            "id",
            "symbol",
            "side",
            "type",
            "status",
            "created_at",
            "filled_at",
            "qty",
            "filled_qty",
            "limit_price",
            "stop_price",
            "order_class",
            "reject_reason",
        ],
    )
    save_csv(
        OUT / f"y_fills_{y}.csv",
        fills,
        [
            "activity_type",
            "id",
            "symbol",
            "side",
            "qty",
            "price",
            "transaction_time",
            "order_id",
        ],
    )
    # 3) Сводка
    # старт/конец дня по equity
    ph1d = jget(
        "/v2/account/portfolio/history",
        {"period": "5D", "timeframe": "1D", "extended_hours": "true"},
    )
    ds = ph1d.get("timestamp", [])
    eqs = ph1d.get("equity", [])
    day_delta = None
    for t, eq_prev in zip(ds, eqs):
        d = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(Z).date().isoformat()
        # найдём пару (предыдущий, текущий) для y
    # вычислим Δ как в портфолио/истории 1D (последовательные точки)
    day_equity = []
    for t, eq in zip(ds, eqs):
        d = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(Z).date().isoformat()
        day_equity.append((d, float(eq)))
    day_equity.sort()
    for i in range(1, len(day_equity)):
        if day_equity[i][0] == y:
            prev = day_equity[i - 1][1]
            cur = day_equity[i][1]
            day_delta = cur - prev
            break
    # наличие закрытий
    fills_cnt = len(fills) if isinstance(fills, list) else 0
    buy_f = (
        sum(float(x.get("qty") or 0) for x in fills if x.get("side") == "buy")
        if isinstance(fills, list)
        else 0.0
    )
    sell_f = (
        sum(float(x.get("qty") or 0) for x in fills if x.get("side") == "sell")
        if isinstance(fills, list)
        else 0.0
    )
    # быстрый вывод
    print("=== Postmortem —", y, "(локально, Ташкент) ===")
    print(f"Equity Δ (день): {day_delta if day_delta is not None else 'NA'} USD")
    print(f"Fills: total={fills_cnt}  buy_qty={buy_f}  sell_qty={sell_f}")
    print("Файлы:")
    if rows:
        print(f" - logs/diagnostics/y_intraday_5m_{y}.csv")
    print(f" - logs/diagnostics/y_orders_{y}.csv")
    print(f" - logs/diagnostics/y_fills_{y}.csv")
    # грубая причина
    if day_delta is not None and day_delta < 0 and fills_cnt == 0:
        print("Причина: нереализованный минус (UPL), без закрытий в течение дня.")
    elif day_delta is not None and day_delta < 0 and fills_cnt > 0:
        print(
            "Причина: реализованный минус (было закрытие сделок в минус / неблагоприятные исполнения)."
        )
    elif day_delta is not None and day_delta >= 0:
        print("День не отрицательный по equity или около нуля — см. интрадею 5м.")
