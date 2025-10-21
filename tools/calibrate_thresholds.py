import os
import json
import datetime as dt
from pathlib import Path
import yfinance as yf

ROOT = Path("/root/stockbot")
ADAPTIVE = ROOT / "core/trading/adaptive_config.json"
HIST_JL = ROOT / "logs/signals_history.jsonl"

# Параметры (можно регулировать env-переменными)
TARGET_PREC = float(os.getenv("ELIOS_CAL_TARGET_PREC", "0.62"))  # целевой precision
WINDOW_TRADES = int(
    os.getenv("ELIOS_CAL_WINDOW_TRADES", "80")
)  # сколько последних сделок считаем
MIN_TRADES = int(os.getenv("ELIOS_CAL_MIN_TRADES", "30"))  # минимум для калибровки
HIT_PCT = float(os.getenv("ELIOS_HIT_PCT", "1.50"))  # порог "хита" в %
DELTA_UP = float(os.getenv("ELIOS_CAL_DELTA_UP", "5.0"))  # шаг поднятия порога
DELTA_DOWN = float(os.getenv("ELIOS_CAL_DELTA_DOWN", "3.0"))  # шаг опускания порога
EPS = float(
    os.getenv("ELIOS_CAL_BAND", "0.03")
)  # мёртвая зона вокруг целевого precision


def next_bday(d: dt.date) -> dt.date:
    x = d + dt.timedelta(days=1)
    while x.weekday() >= 5:
        x += dt.timedelta(days=1)
    return x


def _flatten_yf(df):
    # приводим MultiIndex к плоским именам
    try:
        import pandas as pd

        if isinstance(df.columns, pd.MultiIndex):
            if {"Open", "High", "Low", "Close", "Volume"}.issubset(
                set(df.columns.get_level_values(-1))
            ):
                df = df.droplevel(0, axis=1)
            else:
                df.columns = [
                    c[-1] if isinstance(c, tuple) and c else c for c in df.columns
                ]
    except Exception:
        pass
    return df


def day_ohlc(sym: str, day: dt.date):
    # пытаемся взять ровно следующий день
    df = yf.download(
        sym,
        start=str(day),
        end=str(day + dt.timedelta(days=2)),
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if df is None or df.empty:
        return None, None
    df = _flatten_yf(df)
    need = {"Open", "High"}
    if not need.issubset(set(df.columns)):
        return None, None
    # найдём строку ровно для 'day'
    df = df.copy()
    df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) else df.index
    row = df.loc[df.index.date == day]
    if row.empty:
        # иногда day попадает во второй бар при сдвигах — возьмём первую строку
        row = df.iloc[[0]]
    o = float(row["Open"].iloc[0])
    h = float(row["High"].iloc[0])
    if o <= 0 or h <= 0:
        return None, None
    return o, h


def parse_date_from_ts(ts: str) -> dt.date:
    try:
        # isoformat; обрезаем микросекунды/таймзону
        return (
            dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            .astimezone(dt.timezone.utc)
            .date()
        )
    except Exception:
        return dt.datetime.utcnow().date()


def load_history_items():
    if not HIST_JL.exists():
        return []
    items = []
    with open(HIST_JL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
                if isinstance(it, dict) and it.get("symbol") and it.get("timestamp"):
                    items.append(it)
            except Exception:
                continue
    # последние по времени
    items.sort(key=lambda x: x.get("timestamp", ""))
    return items[-WINDOW_TRADES:]


def label_hits(items):
    labeled = []
    for it in items:
        sym = str(it["symbol"]).upper()
        d0 = parse_date_from_ts(it["timestamp"])
        nd = next_bday(d0)
        o, h = day_ohlc(sym, nd)
        if not o or not h:
            continue
        gain = (h / o - 1.0) * 100.0
        hit = gain >= HIT_PCT
        labeled.append(
            {
                "symbol": sym,
                "date": str(d0),
                "next_day": str(nd),
                "open_next": o,
                "high_next": h,
                "gain_pct": round(gain, 3),
                "hit": hit,
                "score": it.get("score", 0.0),
            }
        )
    return labeled


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_adaptive():
    if ADAPTIVE.exists():
        try:
            with open(ADAPTIVE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "thresholds" in cfg:
                    return cfg
        except Exception:
            pass
    # базовый скелет, если файла нет
    return {
        "thresholds": {"MODEL_SCORE_MIN": 55.0},
        "last_count": None,
        "last_update": None,
    }


def save_adaptive(cfg):
    ADAPTIVE.parent.mkdir(parents=True, exist_ok=True)
    with open(ADAPTIVE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    items = load_history_items()
    if len(items) < MIN_TRADES:
        print(f"[cal] недостаточно сделок для калибровки: {len(items)} < {MIN_TRADES}")
        return

    lab = label_hits(items)
    if not lab:
        print("[cal] нет размеченных исходов (yfinance не дал дневной бар)")
        return

    total = len(lab)
    hits = sum(1 for x in lab if x["hit"])
    prec = hits / total if total else 0.0

    cfg = load_adaptive()
    th = cfg.get("thresholds", {})
    ms = float(th.get("MODEL_SCORE_MIN", 55.0))

    # границы (синхронизируем с BOUNDS из движка)
    lo, hi = 35.0, 85.0
    new_ms = ms
    msg_decision = "hold"

    if prec < (TARGET_PREC - EPS):
        new_ms = clamp(ms + DELTA_UP, lo, hi)
        msg_decision = f"raise +{DELTA_UP}"
    elif prec > (TARGET_PREC + EPS):
        new_ms = clamp(ms - DELTA_DOWN, lo, hi)
        msg_decision = f"lower -{DELTA_DOWN}"

    th["MODEL_SCORE_MIN"] = new_ms
    cfg["thresholds"] = th
    cfg["last_update"] = dt.datetime.now().isoformat()
    save_adaptive(cfg)

    print(
        json.dumps(
            {
                "window_trades": total,
                "hits": hits,
                "precision": round(prec, 4),
                "target": TARGET_PREC,
                "old_MODEL_SCORE_MIN": ms,
                "new_MODEL_SCORE_MIN": new_ms,
                "decision": msg_decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
