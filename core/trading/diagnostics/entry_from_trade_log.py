# -*- coding: utf-8 -*-
"""
Entry Analysis from trade_log (+fallback orders.created_at)
Что делаем:
- Читаем trade_log.json (возможные пути), берём ВСЕ записи с entry_time (включая ещё не закрытые)
- Группируем входы по локальным часовым бакетам (Asia/Tashkent)
- Считаем, сколько из этих входов было закрыто в тот же день (по exit_time), а сколько — нет
- Даём сводку по символам: сколько входов и доля, не закрытая в день входа
- Если trade_log пуст/скуп — дополняем входами из /v2/orders по created_at (status=all), чтобы понять «когда ставили BUY»

Выходные файлы (logs/diagnostics):
- tl_entries_raw_90d.csv                  — сырые входы из trade_log (90д)
- tl_entry_buckets_90d.csv               — бакеты по часу (локально)
- tl_symbols_stats_90d.csv               — сводка по символам
- orders_created_bucket_90d.csv          — бакеты по created_at (если доступно из Alpaca)
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import defaultdict, Counter
import requests

ROOT = Path("/root/stockbot")
OUT = ROOT / "logs" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TZ = os.getenv("ELIOS_TZ", "Asia/Tashkent")
Z = ZoneInfo(TZ)


def parse_dt(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def to_local_bucket(dt_utc):
    dt_loc = dt_utc.astimezone(Z)
    return f"{dt_loc.hour:02d}:00"


def load_trade_log():
    cands = [
        ROOT / "data/trades/trade_log.json",
        ROOT / "core/trading/trade_log.json",
        ROOT / "logs/trade_log.json",
    ]
    for p in cands:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and isinstance(data.get("trades"), list):
                    return data["trades"]
            except Exception:
                pass
    return []


def save_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")


def analyze_trade_log(days: int = 90):
    trades = load_trade_log()
    if not trades:
        print("[INFO] trade_log.json не найден или пуст.")
        return [], [], []

    since = datetime.now(timezone.utc) - timedelta(days=days)

    entries = []
    for t in trades:
        sym = str(t.get("symbol") or t.get("ticker") or "?").upper()
        et = parse_dt(t.get("entry_time") or t.get("buy_time"))
        if not et or et < since:
            continue
        xt = parse_dt(t.get("exit_time") or t.get("closed_at") or t.get("sell_time"))
        status = str(t.get("status", "")).lower()
        entries.append(
            {
                "symbol": sym,
                "entry_ts_utc": et.isoformat(),
                "entry_day_utc": et.date().isoformat(),
                "entry_hour_bucket_local": to_local_bucket(et),
                "exit_ts_utc": xt.isoformat() if xt else "",
                "exit_day_utc": xt.date().isoformat() if xt else "",
                "closed_same_day": int(bool(xt and xt.date() == et.date())),
                "status": status,
            }
        )

    if not entries:
        print("[INFO] В trade_log нет записей с entry_time за последние 90 дней.")
        return [], [], []

    # сырые
    save_csv(
        OUT / "tl_entries_raw_90d.csv",
        entries,
        [
            "symbol",
            "entry_ts_utc",
            "entry_day_utc",
            "entry_hour_bucket_local",
            "exit_ts_utc",
            "exit_day_utc",
            "closed_same_day",
            "status",
        ],
    )

    # бакеты
    bucket = defaultdict(lambda: {"entries": 0, "closed_same_day": 0})
    for r in entries:
        b = r["entry_hour_bucket_local"]
        bucket[b]["entries"] += 1
        bucket[b]["closed_same_day"] += r["closed_same_day"]
    bucket_rows = []
    for b, st in sorted(bucket.items(), key=lambda kv: kv[0]):
        e = st["entries"]
        c = st["closed_same_day"]
        same_rate = (c / e * 100.0) if e > 0 else 0.0
        bucket_rows.append(
            {
                "hour_bucket_local": b,
                "entries": e,
                "closed_same_day": c,
                "same_day_close_rate_pct": f"{same_rate:.1f}",
            }
        )
    save_csv(
        OUT / "tl_entry_buckets_90d.csv",
        bucket_rows,
        ["hour_bucket_local", "entries", "closed_same_day", "same_day_close_rate_pct"],
    )

    # символы
    sym_stat = defaultdict(lambda: {"entries": 0, "closed_same_day": 0})
    for r in entries:
        s = r["symbol"]
        sym_stat[s]["entries"] += 1
        sym_stat[s]["closed_same_day"] += r["closed_same_day"]
    sym_rows = []
    for s, st in sorted(sym_stat.items(), key=lambda kv: (-kv[1]["entries"], kv[0])):
        e = st["entries"]
        c = st["closed_same_day"]
        rate = (c / e * 100.0) if e > 0 else 0.0
        sym_rows.append(
            {
                "symbol": s,
                "entries": e,
                "closed_same_day": c,
                "same_day_close_rate_pct": f"{rate:.1f}",
            }
        )
    save_csv(
        OUT / "tl_symbols_stats_90d.csv",
        sym_rows,
        ["symbol", "entries", "closed_same_day", "same_day_close_rate_pct"],
    )
    return entries, bucket_rows, sym_rows


# --- Fallback: по orders.created_at (если нужно понять «когда ставили BUY», даже если не исполнилось)
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
        # не падаем — просто фолбэк игнорируем
        return [], {}
    return r.json(), r.headers


def orders_created_buckets(days: int = 90):
    now = datetime.now(timezone.utc)
    after = (
        (now - timedelta(days=days))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    out = []
    token = None
    while True:
        params = {"status": "all", "direction": "asc", "after": after, "limit": 200}
        if token:
            params["page_token"] = token
        data, headers = jget("/v2/orders", params)
        if not isinstance(data, list) or not data:
            break
        out.extend(data)
        token = headers.get("x-next-page-token") or headers.get("X-Next-Page-Token")
        if not token:
            break

    rows = []
    for o in out:
        cat = o.get("created_at")
        if not cat:
            continue
        dt = parse_dt(cat)
        if not dt:
            continue
        sym = str(o.get("symbol", "?")).upper()
        side = str(o.get("side", "")).lower()
        rows.append(
            {
                "date_utc": dt.date().isoformat(),
                "created_ts_utc": dt.isoformat(),
                "created_hour_bucket_local": to_local_bucket(dt),
                "symbol": sym,
                "side": side,
            }
        )
    if not rows:
        return []

    # бакеты по created_at для BUY
    b = Counter()
    for r in rows:
        if r["side"] == "buy":
            b[r["created_hour_bucket_local"]] += 1
    br = [
        {"created_hour_bucket_local": k, "buy_orders_created": v}
        for k, v in sorted(b.items())
    ]
    save_csv(
        OUT / "orders_created_bucket_90d.csv",
        br,
        ["created_hour_bucket_local", "buy_orders_created"],
    )
    return br


def main():
    entries, buckets, sym_rows = analyze_trade_log(days=90)
    if buckets:
        print("=== Entry Analysis (trade_log, 90d) ===")
        top = sorted(
            buckets, key=lambda r: (float(r["same_day_close_rate_pct"]), r["entries"])
        )[:6]
        print("ТОП «токсичных» бакетов (низкий same_day_close_rate):")
        for r in top:
            print(
                f"  • {r['hour_bucket_local']}  entries={r['entries']}  same-day-close={r['same_day_close_rate_pct']}%"
            )

    br = orders_created_buckets(days=90)
    if br:
        print(
            "\n(Доп.) Buckets по created_at (BUY orders): см. orders_created_bucket_90d.csv"
        )


if __name__ == "__main__":
    main()
