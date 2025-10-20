# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def jget(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def load_trades():
    for p in [
        ROOT / "data/trades/trade_log.json",
        ROOT / "core/trading/trade_log.json",
        ROOT / "logs/trade_log.json",
    ]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return (
                    data["trades"]
                    if isinstance(data, dict) and isinstance(data.get("trades"), list)
                    else (data if isinstance(data, list) else [])
                )
            except Exception:
                pass
    return []


def save_csv(name, rows, headers):
    p = OUT / name
    with p.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    return p


def daily_equity_30d():
    ph = jget(
        "/v2/account/portfolio/history",
        {"period": "30D", "timeframe": "1D", "extended_hours": "true"},
    )
    stamps = ph.get("timestamp", []) or []
    eqs = ph.get("equity", []) or []
    rows = []
    for t, eq in zip(stamps, eqs):
        dt = datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()
        rows.append({"date": dt, "equity": float(eq)})
    rows = sorted(rows, key=lambda x: x["date"])
    out = []
    for i, r in enumerate(rows):
        if i == 0:
            out.append(
                {
                    "date": r["date"],
                    "equity": f'{r["equity"]:.2f}',
                    "delta_usd": "",
                    "delta_pct": "",
                }
            )
            continue
        prev = rows[i - 1]["equity"]
        d = r["equity"] - prev
        pct = (d / prev * 100) if prev > 0 else None
        out.append(
            {
                "date": r["date"],
                "equity": f'{r["equity"]:.2f}',
                "delta_usd": f"{d:.2f}",
                "delta_pct": f"{(pct if pct is not None else 0):.2f}",
            }
        )
    save_csv("rca_equity_30d.csv", out, ["date", "equity", "delta_usd", "delta_pct"])
    losers = [x for x in out[1:] if float(x["delta_usd"]) < 0]
    worst = sorted(losers, key=lambda z: float(z["delta_usd"]))[:5]
    return [x["date"] for x in losers], worst


def intraday_5m_for_dates(dates):
    # Alpaca portfolio/history не принимает список дат; возьмём период 3D вокруг каждой
    picked = set()
    for ds in dates:
        try:
            d0 = datetime.fromisoformat(ds).date()
        except Exception:
            continue
        # 2 дня охвата
        ph = jget(
            "/v2/account/portfolio/history",
            {
                "period": "3D",
                "timeframe": "5Min",
                "extended_hours": "true",
                "date_end": d0.isoformat(),
            },
        )
        stamps = ph.get("timestamp", []) or []
        eqs = ph.get("equity", []) or []
        rows = []
        for t, eq in zip(stamps, eqs):
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            if dt.date() == d0:
                rows.append({"timestamp_utc": dt.isoformat(), "equity": eq})
        if rows:
            save_csv(f"rca_intraday_5m_{ds}.csv", rows, ["timestamp_utc", "equity"])
            picked.add(ds)
    return sorted(picked)


def parse_trade_dt(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def realized_by_day_from_trades():
    t = load_trades()
    out = defaultdict(float)
    for r in t:
        st = str(r.get("status", "")).lower()
        if st not in (
            "closed",
            "sold",
            "closed_full",
            "closed_partial",
            "exit",
            "flatten",
        ):
            continue
        dt = parse_trade_dt(
            r.get("exit_time") or r.get("closed_at") or r.get("sell_time")
        )
        if not dt:
            continue
        pnl = None
        for k in ("pnl", "profit", "pnl_usd", "profit_usd"):
            if k in r:
                try:
                    pnl = float(r[k])
                    break
                except Exception:
                    pass
        if pnl is not None:
            out[dt.date().isoformat()] += pnl
    return out


def orders_and_fills_for_days(days):
    # orders (all) + activities (FILL) на каждую «плохую» дату
    res = {}
    for ds in days:
        d = datetime.fromisoformat(ds)
        a = (d).strftime("%Y-%m-%dT00:00:00Z")
        b = (d + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        orders = jget(
            "/v2/orders",
            {"status": "all", "after": a, "until": b, "direction": "asc", "limit": 500},
        )
        save_csv(
            f"rca_orders_{ds}.csv",
            orders,
            [
                "id",
                "symbol",
                "side",
                "type",
                "status",
                "created_at",
                "filled_at",
                "canceled_at",
                "qty",
                "filled_qty",
                "limit_price",
                "stop_price",
                "order_class",
                "reject_reason",
            ],
        )
        # FILL активности
        try:
            fills = jget(
                "/v2/account/activities/FILL",
                {"after": a, "until": b, "direction": "asc", "page_size": 250},
            )
        except Exception:
            fills = []
        save_csv(
            f"rca_fills_{ds}.csv",
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
        res[ds] = {"orders": len(orders), "fills": len(fills)}
    return res


def journal_systemd_for_days(days):
    # вытащим журналы sell/eod за даты
    units = [
        "elios-sell-once.service",
        "sell_engine",
        "elios-eod-guard.service",
        "eod-flatten.service",
        "elios-entry-gate.service",
        "elios-intraday-clamp.service",
    ]
    out = {}
    for ds in days:
        d = datetime.fromisoformat(ds)
        since = d.strftime("%Y-%m-%d 00:00:00")
        until = (d + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
        snippets = []
        for u in units:
            try:
                r = subprocess.run(
                    [
                        "journalctl",
                        "-u",
                        u,
                        "--since",
                        since,
                        "--until",
                        until,
                        "--no-pager",
                        "-n",
                        "200",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                txt = r.stdout.strip()
                if txt:
                    # оставим только первые/последние строки и ключевые слова
                    key = []
                    for line in txt.splitlines():
                        if any(
                            k in line
                            for k in [
                                "Close",
                                "SELL",
                                "EOD",
                                "flatten",
                                "ERROR",
                                "SL",
                                "TP",
                                "unrealized",
                                "UPL",
                                "Close ALL",
                                "positions=",
                            ]
                        ):
                            key.append(line)
                    if not key:
                        key = txt.splitlines()[:5] + ["..."] + txt.splitlines()[-5:]
                    (OUT / f"rca_journal_{u}_{ds}.log").write_text(
                        "\n".join(key), encoding="utf-8"
                    )
                    snippets.append((u, len(key)))
            except Exception:
                pass
        out[ds] = snippets
    return out


def main():
    bad_days, worst = daily_equity_30d()
    print("=== Equity 30D: убыточные дни ===")
    for (
        d,
        du,
    ) in ((x["date"], x["delta_usd"]) for x in worst):
        print(f"  • {d}: Δ$={du}")
    # Intraday 5m для убыточных
    picked = intraday_5m_for_dates(bad_days[-6:])  # максимум 6 последних убыточных дат
    # Realized vs Equity
    realized = realized_by_day_from_trades()
    print("\n=== Сравнение (Equity Δ vs Realized по trade_log) ===")
    for d in bad_days[-10:]:
        eq_row = None
        # найдём дельту из файла
        # (быстро читаем CSV)
        pass
    # Заказ ордеров/филлов/журналов по убыточным датам
    meta = orders_and_fills_for_days(bad_days[-6:])
    jmeta = journal_systemd_for_days(bad_days[-6:])
    print("\nФайлы созданы в logs/diagnostics/:")
    print("  - rca_equity_30d.csv (дневные equity и дельты)")
    for d in picked:
        print(f"  - rca_intraday_5m_{d}.csv (интрадея 5м)")
    for d, v in meta.items():
        print(f"  - rca_orders_{d}.csv (orders: {v['orders']})")
        print(f"  - rca_fills_{d}.csv (fills : {v['fills']})")
    for d, items in jmeta.items():
        for u, _ in items:
            print(f"  - rca_journal_{u}_{d}.log")


if __name__ == "__main__":
    main()
