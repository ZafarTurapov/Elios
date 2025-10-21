# -*- coding: utf-8 -*-
"""
Entry Window Analysis (orders-based)
- Тянем /v2/orders (status=all) за N дней (по умолчанию 60).
- Берём только заполненные ордера (filled_at). Строим события FILL: buy/sell.
- Метрики по часам (локально Asia/Tashkent): сколько входов и доля "overnight".
- "Overnight" = в день D по символу buy_fills > sell_fills (нет полного закрытия).
- "TTFS" (time-to-first-sell) — минуты от первого BUY до первого SELL в ту же дату (если был).
Запуск:
  PYTHONPATH=/root/stockbot ./venv/bin/python -m core.trading.diagnostics.entry_window_analysis --days 60
"""
from __future__ import annotations
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import defaultdict, Counter
import requests

ROOT = Path("/root/stockbot")
OUT = ROOT / "logs" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
H = {
    "APCA-API-KEY-ID": KEY or "",
    "APCA-API-SECRET-KEY": SEC or "",
    "Accept": "application/json",
}

TZ = os.getenv("ELIOS_TZ", "Asia/Tashkent")
Z = ZoneInfo(TZ)


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json(), r.headers


def get_orders_paginated(after_iso: str):
    url = "/v2/orders"
    token = None
    out = []
    while True:
        params = {"status": "all", "direction": "asc", "after": after_iso, "limit": 200}
        if token:
            params["page_token"] = token
        data, headers = jget(url, params)
        if isinstance(data, list):
            out.extend(data)
        token = headers.get("x-next-page-token") or headers.get("X-Next-Page-Token")
        if not token or not data:
            break
    return out


def parse_ts(s: str):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def to_local_hour_bucket(dt_utc: datetime) -> str:
    dt_local = dt_utc.astimezone(Z)
    return f"{dt_local.hour:02d}:00"


def save_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--days", type=int, default=60, help="Сколько дней назад анализировать (orders)"
    )
    args = ap.parse_args()

    if not (KEY and SEC):
        print("[ERR] Alpaca keys missing", file=sys.stderr)
        sys.exit(2)

    now = datetime.now(timezone.utc)
    after = (
        (now - timedelta(days=args.days))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    orders = get_orders_paginated(after)

    fills = []
    for o in orders:
        filled_at = parse_ts(o.get("filled_at"))
        if not filled_at:
            continue
        try:
            qty = float(o.get("filled_qty") or o.get("qty") or 0)
        except:
            qty = 0.0
        if qty <= 0:
            continue
        sym = str(o.get("symbol", "?")).upper()
        side = str(o.get("side", "")).lower()
        price = 0.0
        for k in ("filled_avg_price", "limit_price", "stop_price"):
            v = o.get(k)
            try:
                if v is not None:
                    price = float(v)
                    break
            except:
                pass
        fills.append(
            {
                "date_utc": filled_at.date().isoformat(),
                "ts_utc": filled_at.isoformat(),
                "ts_local_hour_bucket": to_local_hour_bucket(filled_at),
                "symbol": sym,
                "side": side,  # buy / sell
                "qty": qty,
                "price": price,
            }
        )

    if not fills:
        print(
            f"[INFO] За {args.days} дней нет filled_at (возможно, входы > {args.days}д назад)."
        )
        sys.exit(0)

    # сырые fills
    save_csv(
        OUT / "entry_fills_raw_60d.csv",
        fills,
        [
            "date_utc",
            "ts_utc",
            "ts_local_hour_bucket",
            "symbol",
            "side",
            "qty",
            "price",
        ],
    )

    # overnight по дням/символам
    buy_qty = defaultdict(float)
    sell_qty = defaultdict(float)
    first_buy_ts = {}  # (date,sym) -> ts first buy
    first_sell_ts = {}  # (date,sym) -> ts first sell (если был)
    for r in fills:
        k = (r["date_utc"], r["symbol"])
        ts = parse_ts(r["ts_utc"])
        if r["side"] == "buy":
            buy_qty[k] += r["qty"]
            if k not in first_buy_ts or ts < first_buy_ts[k]:
                first_buy_ts[k] = ts
        elif r["side"] == "sell":
            sell_qty[k] += r["qty"]
            if k not in first_sell_ts or ts < first_sell_ts[k]:
                first_sell_ts[k] = ts

    overnight = []
    for k, bq in buy_qty.items():
        d, sym = k
        sq = sell_qty.get(k, 0.0)
        if bq > sq + 1e-6:
            # “overnight-кандидат”
            # выпишем все BUY-филлы этого дня/символа
            for e in [
                r
                for r in fills
                if r["date_utc"] == d and r["symbol"] == sym and r["side"] == "buy"
            ]:
                ttfs = ""
                if k in first_sell_ts:
                    ttfs = (
                        f"{(first_sell_ts[k]-first_buy_ts[k]).total_seconds()/60.0:.1f}"
                    )
                overnight.append(
                    {
                        "date_utc": d,
                        "symbol": sym,
                        "entry_ts_utc": e["ts_utc"],
                        "entry_hour_bucket_local": e["ts_local_hour_bucket"],
                        "buy_qty_day": f"{bq:g}",
                        "sell_qty_day": f"{sq:g}",
                        "ttfs_min_if_any": ttfs,
                    }
                )
    overnight = sorted(
        overnight, key=lambda r: (r["date_utc"], r["symbol"], r["entry_ts_utc"])
    )
    save_csv(
        OUT / "overnight_candidates_60d.csv",
        overnight,
        [
            "date_utc",
            "symbol",
            "entry_ts_utc",
            "entry_hour_bucket_local",
            "buy_qty_day",
            "sell_qty_day",
            "ttfs_min_if_any",
        ],
    )

    # агрегат по бакетам
    bucket_stats = defaultdict(lambda: {"buys": 0, "overnight_candidates": 0})
    for r in fills:
        if r["side"] == "buy":
            bucket_stats[r["ts_local_hour_bucket"]]["buys"] += 1
    for o in overnight:
        bucket_stats[o["entry_hour_bucket_local"]]["overnight_candidates"] += 1

    bucket_rows = []
    for b, stat in sorted(bucket_stats.items(), key=lambda kv: kv[0]):
        oc = stat["overnight_candidates"]
        bs = stat["buys"]
        rate = (oc / bs * 100.0) if bs > 0 else 0.0
        bucket_rows.append(
            {
                "hour_bucket_local": b,
                "buys": bs,
                "overnight_candidates": oc,
                "overnight_rate_pct": f"{rate:.1f}",
            }
        )
    save_csv(
        OUT / "entry_buckets_60d.csv",
        bucket_rows,
        ["hour_bucket_local", "buys", "overnight_candidates", "overnight_rate_pct"],
    )

    # сводка по символам
    sym_buy = Counter()
    sym_ovr = Counter()
    for r in fills:
        if r["side"] == "buy":
            sym_buy[r["symbol"]] += 1
    for o in overnight:
        sym_ovr[o["symbol"]] += 1
    sym_rows = []
    for s in sorted(set(list(sym_buy.keys()) + list(sym_ovr.keys()))):
        sym_rows.append(
            {
                "symbol": s,
                "buys": sym_buy.get(s, 0),
                "overnight_candidates": sym_ovr.get(s, 0),
            }
        )
    save_csv(
        OUT / "symbols_stats_60d.csv",
        sym_rows,
        ["symbol", "buys", "overnight_candidates"],
    )

    # консольная выжимка
    toks = sorted(
        bucket_rows,
        key=lambda r: (float(r["overnight_rate_pct"]), r["buys"]),
        reverse=True,
    )[:6]
    print("=== Entry Window Analysis (orders-based, 60d) ===")
    print("Файлы:")
    print(" - entry_fills_raw_60d.csv")
    print(" - overnight_candidates_60d.csv")
    print(" - entry_buckets_60d.csv")
    print(" - symbols_stats_60d.csv")
    print("\nТОП бакетов (локальное время) по риску overnight:")
    for r in toks:
        print(
            f"  • {r['hour_bucket_local']}  buys={r['buys']}  overnight={r['overnight_candidates']}  rate={r['overnight_rate_pct']}%"
        )
