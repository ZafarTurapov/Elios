# /root/stockbot/tools/export_trade_day.py
# -*- coding: utf-8 -*-
"""
Единый экспорт торгового дня:
- Берёт события из trade_log.json (BUY/SELL) — ПРИОРИТЕТНО (JSON/NDJSON)
- Дополнительно парсит текстовые trade-логи (строки "TRADE BUY/SELL …")
- Добирает SELECTED/GPT/rejected из сопутствующих источников
- Также берёт SELL из core/trading/pnl_tracker.json (факт закрытия позиции)
- Пишет /logs/trade_day_YYYY-MM-DD.log и .csv
"""

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# === Базовые пути проекта ===
ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"

P_SIGNAL_LOG = LOGS / "signal_log.json"
P_REJECTED_CSV = LOGS / "rejected.csv"

P_SIGNALS = ROOT / "core" / "trading" / "signals.json"
P_GPT_DECISIONS = ROOT / "core" / "trading" / "gpt_decisions.json"
P_REJECTED_JSON = ROOT / "core" / "trading" / "rejected.json"
P_PNL_TRACKER = (
    ROOT / "core" / "trading" / "pnl_tracker.json"
)  # list[{symbol, qty, entry, exit, ...}]


# === Попытка использовать единый путь TRADE_LOG_PATH из проекта ===
def resolve_trade_log_path() -> Path:
    try:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
        from core.utils.paths import (
            TRADE_LOG_PATH as PROJECT_TRADE_LOG_PATH,
        )  # type: ignore

        return Path(PROJECT_TRADE_LOG_PATH)
    except Exception:
        return ROOT / "trade_log.json"  # fallback


TRADE_LOG_PATH = resolve_trade_log_path()

# === Кандидаты путей trade_log (проверяем все) ===
CANDIDATE_TRADE_LOGS = list(
    dict.fromkeys(
        [  # dedup, сохраняем порядок
            TRADE_LOG_PATH,
            ROOT / "trade_log.json",
            ROOT / "core" / "trading" / "trade_log.json",
            ROOT / "logs" / "trade_log.json",
            ROOT / "legacy_src" / "core" / "trading" / "trade_log.json",
        ]
    )
)

# === Время/даты ===
TZ_TASHKENT = ZoneInfo("Asia/Tashkent")
TZ_UTC = ZoneInfo("UTC")


def parse_iso_any(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_UTC)
        return dt.astimezone(TZ_UTC)
    except Exception:
        return None


def mk_local_dt(date_str: str, hh=12, mm=0, ss=0) -> Optional[datetime]:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        dt_loc = datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=TZ_TASHKENT)
        return dt_loc.astimezone(TZ_UTC)
    except Exception:
        return None


def ymd(dt: datetime, tz=TZ_TASHKENT) -> str:
    return dt.astimezone(tz).strftime("%Y-%m-%d")


# === IO ===
def read_json(path: Path) -> Any:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] cannot read {path}: {e}")
    return None


def read_text(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] cannot read text {path}: {e}")
    return None


def read_rejected_csv(path: Path) -> Dict[str, str]:
    res: Dict[str, str] = {}
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                r = csv.reader(f)
                _ = next(r, None)  # header or first row
                for row in r:
                    if not row:
                        continue
                    symbol = (row[0] or "").strip().upper()
                    reason = (row[1] if len(row) > 1 else "").strip()
                    if symbol:
                        res[symbol] = reason or res.get(symbol, "")
    except Exception as e:
        print(f"[WARN] cannot read CSV {path}: {e}")
    return res


# === Выход ===
def ensure_logs_dir():
    LOGS.mkdir(parents=True, exist_ok=True)


def out_paths(day_str: str):
    return (LOGS / f"trade_day_{day_str}.log", LOGS / f"trade_day_{day_str}.csv")


# === Структура выходной строки ===
@dataclass
class Row:
    ts_utc: str
    ts_local: str
    module: str
    side: str
    symbol: str
    qty: Optional[float]
    entry: Optional[float]
    exit: Optional[float]
    pnl: Optional[float]
    reason: str
    extra: str


def mk_row(
    ts_utc: datetime,
    module: str,
    side: str,
    symbol: str,
    qty: Optional[float],
    entry: Optional[float],
    exitp: Optional[float],
    pnl: Optional[float],
    reason: str = "",
    extra: str = "",
) -> Row:
    return Row(
        ts_utc=ts_utc.astimezone(TZ_UTC).isoformat(),
        ts_local=ts_utc.astimezone(TZ_TASHKENT).strftime("%Y-%m-%d %H:%M:%S"),
        module=module,
        side=side,
        symbol=symbol,
        qty=qty if qty is not None else None,
        entry=round(entry, 4) if isinstance(entry, (int, float)) else None,
        exit=round(exitp, 4) if isinstance(exitp, (int, float)) else None,
        pnl=round(pnl, 4) if isinstance(pnl, (int, float)) else None,
        reason=reason or "",
        extra=extra or "",
    )


# === Парсинг TRADE из текстовых строк ===
TRADE_RE = re.compile(
    r"""
    \bTRADE\                          # префикс
    \s+(?P<side>BUY|SELL)\            # сторона
    \s+(?P<symbol>[A-Z][A-Z0-9\.\-]*) # тикер
    \s+qty=(?P<qty>\d+)
    \s+entry=(?P<entry>[0-9]+(?:\.[0-9]+)?|None)
    \s+exit=(?P<exit>[0-9]+(?:\.[0-9]+)?|None)
    """,
    re.VERBOSE,
)

TS_IN_LINE_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")


def extract_ts_from_line(line: str, day_str: str) -> datetime:
    m = TS_IN_LINE_RE.search(line or "")
    if m:
        s = m.group("ts").replace("T", " ")
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=TZ_TASHKENT)
            return dt.astimezone(TZ_UTC)
        except Exception:
            pass
    # если таймстемпа в строке нет — ставим безопасный дефолт дня
    return mk_local_dt(day_str, 20, 0, 0)


def parse_trade_text(ts_utc: datetime, text: str) -> Optional[Tuple[Row, str]]:
    m = TRADE_RE.search(text or "")
    if not m:
        return None
    side = m.group("side").upper()
    symbol = m.group("symbol").upper()
    qty = int(m.group("qty"))
    entry = None if m.group("entry") == "None" else float(m.group("entry"))
    exitp = None if m.group("exit") == "None" else float(m.group("exit"))
    pnl = None
    if (entry is not None) and (exitp is not None) and side == "SELL":
        pnl = (exitp - entry) * qty
    return (
        mk_row(
            ts_utc,
            module="trade",
            side=side,
            symbol=symbol,
            qty=qty,
            entry=entry,
            exitp=exitp,
            pnl=pnl,
            reason="",
            extra="text",
        ),
        "text",
    )


# === Утилиты нормализации сделок (JSON/NDJSON) ===
def _norm_list_from_maybe_dict(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("trades", "items", "history", "log", "orders", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    return None


def _first_num(*vals):
    for v in vals:
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _first_str_upper(*vals):
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s.upper()
    return ""


# === Загрузка сделок из любого trade_log.json (JSON/NDJSON) ===
def load_trades_any_json() -> List[dict]:
    records: List[dict] = []
    tried = []
    for p in CANDIDATE_TRADE_LOGS:
        try:
            if not p:
                continue
            path = Path(p)
            if not path.exists():
                continue
            tried.append(str(path))
            raw = read_text(path)
            if not raw:
                continue
            obj = None
            # JSON / dict с листом
            try:
                obj = json.loads(raw)
                lst = _norm_list_from_maybe_dict(obj)
                if lst is not None:
                    records.extend([x for x in lst if isinstance(x, dict)])
                    continue
                if isinstance(obj, list):
                    records.extend([x for x in obj if isinstance(x, dict)])
                    continue
            except Exception:
                pass
            # NDJSON
            tmp = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    if isinstance(o, dict):
                        tmp.append(o)
                except Exception:
                    continue
            if tmp:
                records.extend(tmp)
        except Exception as e:
            print(f"[WARN] cannot read {p}: {e}")
    print(
        f"[INFO] trade logs (JSON/NDJSON) loaded={len(records)} from {len(tried)} files"
    )
    return records


def load_trade_text_rows_for_day(day_str: str) -> List[Row]:
    rows: List[Row] = []
    seen_keys = set()
    cnt_checked = 0
    for p in CANDIDATE_TRADE_LOGS:
        path = Path(p)
        if not path.exists():
            continue
        cnt_checked += 1
        text = read_text(path)
        if not text:
            continue
        for line in text.splitlines():
            if "TRADE " not in line:
                continue
            ts = extract_ts_from_line(line, day_str)
            if not ts or ymd(ts, TZ_TASHKENT) != day_str:
                continue
            res = parse_trade_text(ts, line)
            if not res:
                continue
            row, _ = res
            # dedup по ключу
            key = (
                row.ts_utc,
                row.module,
                row.side,
                row.symbol,
                row.qty,
                row.entry,
                row.exit,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # пометим источник
            row.extra = f"trade_log:{path.name}"
            rows.append(row)
    print(f"[INFO] trade logs (TEXT) checked={cnt_checked} | parsed_rows={len(rows)}")
    return rows


# === Фильтр по дню (локальному) ===
def same_local_day(dt_utc: datetime, target_day: str) -> bool:
    return ymd(dt_utc, TZ_TASHKENT) == target_day


# === Основной сбор данных ===
def collect_rows_for_day(day_str: str) -> List[Row]:
    rows: List[Row] = []
    target_day = day_str

    # 1a) TRADE LOG — JSON/NDJSON
    trades = load_trades_any_json()
    for t in trades:
        try:
            symbol = _first_str_upper(t.get("symbol"), t.get("ticker"), t.get("sym"))
            if not symbol:
                continue

            side = _first_str_upper(t.get("side"), t.get("action"), t.get("type"))
            qty = _first_num(t.get("qty"), t.get("quantity"), t.get("filled_qty"))

            entry = _first_num(
                t.get("entry"),
                t.get("entry_price"),
                t.get("buy_price"),
                (t.get("price") if side == "BUY" else None),
            )
            exitp = _first_num(
                t.get("exit"),
                t.get("exit_price"),
                t.get("sell_price"),
                (t.get("price") if side == "SELL" else None),
            )
            pnl = _first_num(t.get("pnl"))
            reason = str(t.get("reason", "") or t.get("note", "")).strip()

            ts = (
                parse_iso_any(t.get("timestamp"))
                or parse_iso_any(t.get("filled_at"))
                or parse_iso_any(t.get("created_at"))
                or parse_iso_any(t.get("updated_at"))
                or parse_iso_any(t.get("submitted_at"))
            )
            if not ts:
                if _first_str_upper(t.get("exit_date")):
                    ts = mk_local_dt(str(t.get("exit_date")), 23, 59, 0)
                elif _first_str_upper(t.get("entry_date")):
                    ts = mk_local_dt(str(t.get("entry_date")), 18, 30, 0)
                else:
                    ts = mk_local_dt(target_day, 20, 0, 0)

            if not ts or not same_local_day(ts, target_day):
                continue

            if not side:
                if exitp is not None or isinstance(pnl, (int, float)):
                    side = "SELL"
                else:
                    side = "BUY"

            if pnl is None and (
                isinstance(entry, (int, float))
                and isinstance(exitp, (int, float))
                and side == "SELL"
            ):
                try:
                    q = float(qty) if qty is not None else 0.0
                    pnl = (float(exitp) - float(entry)) * q
                except Exception:
                    pnl = None

            rows.append(
                mk_row(
                    ts_utc=ts,
                    module="trade",
                    side=side,
                    symbol=symbol,
                    qty=float(qty) if qty is not None else None,
                    entry=float(entry) if entry is not None else None,
                    exitp=float(exitp) if exitp is not None else None,
                    pnl=float(pnl) if isinstance(pnl, (int, float)) else None,
                    reason=reason,
                    extra="json",
                )
            )
        except Exception as e:
            print(f"[WARN] bad trade record: {t} | {e}")

    # 1b) TRADE LOG — TEXT
    rows.extend(load_trade_text_rows_for_day(target_day))

    # 1c) PnL tracker — считаем как SELL (факт закрытия позиции)
    pnlrecs = read_json(P_PNL_TRACKER) or []
    if isinstance(pnlrecs, list):
        for r in pnlrecs:
            try:
                symbol = _first_str_upper(r.get("symbol"))
                if not symbol:
                    continue
                qty = _first_num(r.get("qty"), r.get("quantity"))
                entry = _first_num(r.get("entry"), r.get("entry_price"))
                exitp = _first_num(r.get("exit"), r.get("exit_price"))
                # время — из timestamp, иначе берём конец дня локального (логично для факта выхода)
                ts = parse_iso_any(r.get("timestamp")) or mk_local_dt(
                    target_day, 23, 59, 0
                )
                if not ts or not same_local_day(ts, target_day):
                    continue
                # PnL: если нет — досчитаем
                pnl = _first_num(r.get("pnl"))
                if pnl is None and all(
                    isinstance(x, (int, float)) for x in (entry, exitp, qty)
                ):
                    try:
                        pnl = (float(exitp) - float(entry)) * float(qty)
                    except Exception:
                        pnl = None

                rows.append(
                    mk_row(
                        ts_utc=ts,
                        module="trade",
                        side="SELL",
                        symbol=symbol,
                        qty=float(qty) if qty is not None else None,
                        entry=float(entry) if entry is not None else None,
                        exitp=float(exitp) if exitp is not None else None,
                        pnl=float(pnl) if isinstance(pnl, (int, float)) else None,
                        reason="PNL_TRACKER",
                        extra="pnl_tracker.json",
                    )
                )
            except Exception as e:
                print(f"[WARN] bad pnl record: {r} | {e}")

    # 2) SELECTED
    signals = read_json(P_SIGNALS) or {}
    if isinstance(signals, dict):
        siglog = read_json(P_SIGNAL_LOG) or []
        when_by_symbol: Dict[str, datetime] = {}
        if isinstance(siglog, list):
            for ev in siglog:
                try:
                    if (ev or {}).get("event") in ("SELECTED", "SIGNAL", "ACCEPTED"):
                        sym = str(ev.get("symbol", "")).upper()
                        tss = parse_iso_any(ev.get("timestamp") or ev.get("ts"))
                        if sym and tss and same_local_day(tss, target_day):
                            when_by_symbol[sym] = tss
                except Exception:
                    pass

        for sym in signals.keys():
            try:
                symbol = str(sym).upper()
                ts = when_by_symbol.get(symbol) or mk_local_dt(target_day, 18, 30, 0)
                if not ts or not same_local_day(ts, target_day):
                    continue
                rows.append(
                    mk_row(
                        ts_utc=ts,
                        module="selected",
                        side="N/A",
                        symbol=symbol,
                        qty=None,
                        entry=None,
                        exitp=None,
                        pnl=None,
                        reason="SELECTED",
                        extra="",
                    )
                )
            except Exception:
                pass

    # 3) GPT DECISIONS
    gpt = read_json(P_GPT_DECISIONS) or {}
    if isinstance(gpt, dict):
        siglog = read_json(P_SIGNAL_LOG) or []
        when_by_symbol_gpt: Dict[str, datetime] = {}
        if isinstance(siglog, list):
            for ev in siglog:
                try:
                    if (ev or {}).get("event") in (
                        "GPT",
                        "GPT_DECISION",
                        "ACCEPTED",
                        "REJECTED",
                    ):
                        sym = str(ev.get("symbol", "")).upper()
                        tss = parse_iso_any(ev.get("timestamp") or ev.get("ts"))
                        if sym and tss and same_local_day(tss, target_day):
                            when_by_symbol_gpt[sym] = tss
                except Exception:
                    pass

        for sym, decision in gpt.items():
            try:
                symbol = str(sym).upper()
                ts = when_by_symbol_gpt.get(symbol) or mk_local_dt(
                    target_day, 18, 31, 0
                )
                if not ts or not same_local_day(ts, target_day):
                    continue
                rows.append(
                    mk_row(
                        ts_utc=ts,
                        module="gpt",
                        side="N/A",
                        symbol=symbol,
                        qty=None,
                        entry=None,
                        exitp=None,
                        pnl=None,
                        reason=str(decision),
                        extra="",
                    )
                )
            except Exception:
                pass

    # 4) REJECTED
    rej_json = read_json(P_REJECTED_JSON) or {}
    rej_csv = read_rejected_csv(P_REJECTED_CSV)
    rejected: Dict[str, str] = {}
    if isinstance(rej_json, dict):
        for k, v in rej_json.items():
            rejected[str(k).upper()] = str(v or "")
    for k, v in rej_csv.items():
        rejected[str(k).upper()] = v or rejected.get(str(k).upper(), "")

    if rejected:
        siglog = read_json(P_SIGNAL_LOG) or []
        ts_by_reject: Dict[str, datetime] = {}
        if isinstance(siglog, list):
            for ev in siglog:
                try:
                    if (ev or {}).get("event") in ("REJECTED", "FILTER_REJECT", "DENY"):
                        sym = str(ev.get("symbol", "")).upper()
                        tss = parse_iso_any(ev.get("timestamp") or ev.get("ts"))
                        if sym and tss and same_local_day(tss, target_day):
                            ts_by_reject[sym] = tss
                except Exception:
                    pass

        for symbol, reason in rejected.items():
            ts = ts_by_reject.get(symbol) or mk_local_dt(target_day, 18, 35, 0)
            if not ts or not same_local_day(ts, target_day):
                continue
            rows.append(
                mk_row(
                    ts_utc=ts,
                    module="rejected",
                    side="N/A",
                    symbol=symbol,
                    qty=None,
                    entry=None,
                    exitp=None,
                    pnl=None,
                    reason=str(reason or ""),
                    extra="",
                )
            )

    # Сортировка + дедуп
    rows.sort(key=lambda r: (r.ts_utc, r.module, r.symbol, r.side))
    uniq: List[Row] = []
    seen = set()
    for r in rows:
        key = (r.ts_utc, r.module, r.side, r.symbol, r.qty, r.entry, r.exit, r.reason)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


# === Запись результатов ===
def write_csv(path: Path, rows: List[Row]):
    header = [
        "ts_utc",
        "ts_local",
        "module",
        "side",
        "symbol",
        "qty",
        "entry",
        "exit",
        "pnl",
        "reason",
        "extra",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            d = asdict(r)
            w.writerow([d[h] for h in header])


def write_log(path: Path, rows: List[Row]):
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"📒 Trade Day Export — {rows[0].ts_local.split(' ')[0] if rows else ''}\n"
        )
        for r in rows:
            line = (
                f"[{r.ts_local}] {r.module.upper():8s} "
                f"{(r.side or 'N/A'):4s} {r.symbol:6s} "
                f"qty={r.qty if r.qty is not None else '-'} "
                f"entry={r.entry if r.entry is not None else '-'} "
                f"exit={r.exit if r.exit is not None else '-'} "
                f"pnl={r.pnl if r.pnl is not None else '-'} "
                f"reason={r.reason or '-'} "
                f"src={r.extra or '-'}"
            )
            f.write(line + "\n")


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Export trade day log + csv")
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD (локальная Asia/Tashkent). По умолчанию — вчера.",
        default=None,
    )
    args = parser.parse_args()

    now_loc = datetime.now(TZ_TASHKENT)
    day_str = args.date or (now_loc - timedelta(days=1)).strftime("%Y-%m-%d")

    ensure_logs_dir()
    rows = collect_rows_for_day(day_str)
    p_log, p_csv = out_paths(day_str)

    write_log(p_log, rows)
    write_csv(p_csv, rows)

    print(f"✅ Экспорт завершён: {p_log}")
    print(f"✅ Экспорт завершён: {p_csv}")
    buys = sum(1 for r in rows if r.module == "trade" and r.side == "BUY")
    sells = sum(1 for r in rows if r.module == "trade" and r.side == "SELL")
    selected = sum(1 for r in rows if r.module == "selected")
    gpt = sum(1 for r in rows if r.module == "gpt")
    rejected = sum(1 for r in rows if r.module == "rejected")
    print(
        f"Σ BUY: {buys} | Σ SELL: {sells} | SELECTED: {selected} | GPT: {gpt} | REJECTED: {rejected} | Всего событий: {len(rows)}"
    )


if __name__ == "__main__":
    main()
