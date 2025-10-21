# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
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
    r = requests.get(f"{BASE}{path}", headers=H, params=params or {}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} -> {r.status_code} {r.text}")
    return r.json()


def load_trades():
    cands = [
        ROOT / "data/trades/trade_log.json",
        ROOT / "core/trading/trade_log.json",
        ROOT / "logs/trade_log.json",
    ]
    for p in cands:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("trades"), list):
                    return data["trades"]
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []


def dkey(dt: datetime):
    return dt.astimezone(timezone.utc).date().isoformat()


def parse_dt(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    # 1) Alpaca equity history — 14D (берём 10-12 последних дат с equity)
    ph = jget(
        "/v2/account/portfolio/history",
        {"period": "14D", "timeframe": "1D", "extended_hours": "true"},
    )
    stamps = ph.get("timestamp", []) or []
    equities = ph.get("equity", []) or []
    if not stamps or not equities:
        print("[ERR] Нет данных portfolio/history", file=sys.stderr)
        sys.exit(2)

    rows = []
    for t, eq in zip(stamps, equities):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        rows.append({"date": dkey(dt), "equity": float(eq)})

    # 2) Подсчёт дневных дельт (Δ$ и Δ%)
    rows = sorted(rows, key=lambda r: r["date"])
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        prev_eq = prev["equity"]
        cur_eq = cur["equity"]
        delta = cur_eq - prev_eq
        pct = (delta / prev_eq * 100.0) if prev_eq > 0 else None
        cur["delta_usd"] = delta
        cur["delta_pct"] = pct

    # 3) Realized PnL по дням из trade_log (по времени закрытия)
    trades = load_trades()
    realized_by_day = defaultdict(float)
    counts_by_day = defaultdict(lambda: {"W": 0, "L": 0})
    for t in trades:
        st = str(t.get("status", "")).lower()
        if st not in (
            "closed",
            "sold",
            "closed_full",
            "closed_partial",
            "exit",
            "flatten",
        ):
            continue
        dt = parse_dt(t.get("exit_time") or t.get("closed_at") or t.get("sell_time"))
        if not dt:
            continue
        day = dkey(dt)
        pnl = None
        for k in ("pnl", "profit", "pnl_usd", "profit_usd"):
            if k in t:
                try:
                    pnl = float(t[k])
                    break
                except Exception:
                    pass
        if pnl is None:
            # если только в процентах — пропустим, чтобы не искажать $
            continue
        realized_by_day[day] += pnl
        if pnl >= 0:
            counts_by_day[day]["W"] += 1
        else:
            counts_by_day[day]["L"] += 1

    # 4) Сводная таблица по датам
    out = []
    for r in rows:
        d = r["date"]
        out.append(
            {
                "date": d,
                "equity": f'{r["equity"]:.2f}',
                "delta_usd": f'{r.get("delta_usd",0.0):.2f}',
                "delta_pct": f'{(r.get("delta_pct") if r.get("delta_pct") is not None else 0.0):.2f}',
                "realized_usd": f"{realized_by_day.get(d,0.0):.2f}",
                "wins": counts_by_day.get(d, {}).get("W", 0),
                "losses": counts_by_day.get(d, {}).get("L", 0),
                "unrealized_component_usd~est": f'{(float(r.get("delta_usd",0.0)) - float(realized_by_day.get(d,0.0))):.2f}',
            }
        )

    # 5) Запись CSV и печать ТОП убыточных дней
    csvp = OUT / "daily_pnl_10d.csv"
    with csvp.open("w", encoding="utf-8") as f:
        f.write(
            "date,equity,delta_usd,delta_pct,realized_usd,wins,losses,unrealized_component_usd~est\n"
        )
        for r in out:
            f.write(
                ",".join(
                    str(r[k])
                    for k in [
                        "date",
                        "equity",
                        "delta_usd",
                        "delta_pct",
                        "realized_usd",
                        "wins",
                        "losses",
                        "unrealized_component_usd~est",
                    ]
                )
                + "\n"
            )

    # консольная сводка
    losers = []
    for r in out[1:]:  # пропускаем первую строку (нет дельты)
        try:
            losers.append((r["date"], float(r["delta_usd"]), float(r["realized_usd"])))
        except Exception:
            pass
    losers = sorted([x for x in losers if x[1] < 0], key=lambda x: x[1])[:5]

    print("=== Daily PnL 10D — сводка ===")
    print("Файл: logs/diagnostics/daily_pnl_10d.csv")
    if losers:
        print("ТОП убыточных дней (Equity Δ$):")
        for d, du, ru in losers:
            print(
                f"  • {d}: Δ$={du:.2f}  | Realized={ru:.2f}  | Unrealized≈{(du-ru):.2f}"
            )
    else:
        print("За 10 дней убыточных дней не найдено или дельты близки к нулю.")


if __name__ == "__main__":
    main()
