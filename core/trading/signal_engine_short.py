from core.utils.alpaca_headers import alpaca_headers

## -*- coding: utf-8 -*-
"""
Elios v0.3.6 — signal_engine_short.py (Dux-style short)
- One-pass strict SHORT filters:
  * Bearish day/candle gate (red day OR red body)
  * Bearish EMA trend gate: Price<EMA20<EMA50 (uses live price)
  * GapDown + Volume ratio
  * Risk guards: ATR%, Volatility, VolumeTrend
  * Scores: alpha_short, model_short
- Near-EMA: "almost trend" → SECOND_PASS (with epsilon, weekend bonus)
- Second pass: local softening + floors + GPT confirm + Telegram
- Rescue pass: momentum/risk guards + GPT confirm + Telegram
- Precision mode TOP-K
- Auto-adaptation with LOCKs for MAX_ATR_PCT / MAX_VOLATILITY
- Rejections CSV export
- yfinance auto_adjust=False
- Safe logging (no broken f-strings)
"""

import sys
import os


# === OHLC normalizer ===
def _extract_ohlc(df, symbol=None):
    try:
        import pandas as pd

        if df is None:
            return None
        if hasattr(df, "empty") and df.empty:
            return df
        cols = getattr(df, "columns", None)

        # MultiIndex от yfinance: (TICKER, Field)
        if hasattr(cols, "levels") and getattr(cols, "nlevels", 1) == 2:
            want = ["Open", "High", "Low", "Close", "Volume"]
            if symbol and all((symbol, c) in cols for c in want):
                sub = df[(symbol, want)].copy()
                sub.columns = want
                return sub
            # single-ticker MultiIndex (без символа в колонках)
            if "Close" in cols.get_level_values(1):
                # соберём по уровню 1
                parts = {}
                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    key = [(k0, k1) for (k0, k1) in cols if k1 == c]
                    if key:
                        parts[c] = df.loc[:, key[0]]
                if parts:
                    out = pd.concat(parts.values(), axis=1)
                    out.columns = list(parts.keys())
                    return out

        # Обычный DataFrame: убедимся в наличии столбцов
        # допускаем 'Adj Close' -> 'Close' при необходимости
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        need = {"Open", "High", "Low", "Close"}
        if need.issubset(set(df.columns)):
            return df
    except Exception:
        pass
    return df


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from openai import OpenAI
import shutil
import requests
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

# Optional project modules (fail-safe imports)
try:
    from core.trading.anomaly_detector import detect_anomalies  # type: ignore
except Exception:

    def detect_anomalies(symbol: str):
        return (False, "n/a")


try:
    from core.trading.alpha_utils import calculate_alpha_score  # type: ignore
except Exception:

    def calculate_alpha_score(day_change, volume_ratio, rsi, ema_dev):
        # long-style alpha (not used here, but kept for compat)
        score = (
            0.4 * max(0.0, day_change) / 5.0
            + 0.3 * min(volume_ratio, 2.0) / 2.0
            + 0.2 * max(0.0, rsi - 50.0) / 50.0
            + 0.1 * max(0.0, ema_dev) / 5.0
        )
        return float(max(0.0, min(1.0, score)))


# Short-specific alpha (fallback)
def calculate_alpha_short(day_drop_abs, volume_ratio, rsi, ema_dev_neg_abs):
    # Emphasize magnitude of drop, volume expansion, low RSI, negative EMA deviation
    score = (
        0.45 * min(day_drop_abs, 10.0) / 10.0
        + 0.30 * min(volume_ratio, 2.0) / 2.0
        + 0.15 * max(0.0, (50.0 - rsi)) / 50.0
        + 0.10 * min(ema_dev_neg_abs, 10.0) / 10.0
    )
    return float(max(0.0, min(1.0, score)))


try:
    from core.utils.telegram import send_telegram_message  # type: ignore
except Exception:

    def send_telegram_message(msg: str):
        try:
            text = (str(msg) if msg is not None else "").replace("\n", " | ")
        except Exception:
            text = "[unprintable message]"
        print("[TELEGRAM MOCK] " + text[:300])


# === Telegram helpers ===
def _tg_safe(msg: str):
    try:
        send_telegram_message(msg)
    except Exception as e:
        try:
            print("[TELEGRAM WARN]", str(e)[:200])
        except Exception:
            pass


def _tg_signal(
    pass_label: str,
    sym: str,
    price: float,
    body: float,
    day: float,
    gapd: float,
    vr: float,
):
    try:
        msg = "{}: SHORT {} @ {:.2f}\nbody={:.2f}% day={:.2f}% gapDown={:.2f}% vr={:.2f}×".format(
            pass_label, sym, price, body, day, gapd, vr
        )
        _tg_safe(msg)
    except Exception:
        pass


# === Keys/URLs, GPT client ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
)
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    client = None


def gpt_confirm_short(
    symbol: str,
    price: float,
    day_change: float,
    rsi: float,
    ema_dev: float,
    volume_ratio: float,
    alpha_score: float,
    model_score: float,
    atr_pct: float,
    volatility: float,
    volume_trend: float,
    bearish_body: float,
    gap_down: float,
):
    if client is None:
        return True, "ДА (GPT: client=None)"
    prompt = (
        "Ты торговый ассистент Искры. Оцени сделку SHORT.\n"
        "Тикер: {sym}\n"
        "Цена: {pr:.2f}\n"
        "∆%: {dc:.2f}% | RSI: {rsi:.2f}\n"
        "EMA dev: {ed:.2f}% | VolRatio: {vr:.2f}×\n"
        "AlphaShort: {al:.2f} | ModelShort: {ms:.2f}\n"
        "ATR%: {atr:.2f}% | Vol: {vol:.2f}% | VTrend: {vt:.2f}×\n"
        "Свеча: body={body:.2f}% | GapDown={gap:.2f}%\n"
        "Открывать SHORT? Ответь 'ДА' или 'НЕТ' и кратко почему."
    ).format(
        sym=symbol,
        pr=price,
        dc=day_change,
        rsi=rsi,
        ed=ema_dev,
        vr=volume_ratio,
        al=alpha_score,
        ms=model_score,
        atr=atr_pct,
        vol=volatility,
        vt=volume_trend,
        body=bearish_body,
        gap=gap_down,
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        reply = (chat.choices[0].message.content or "").strip()
        ok = "ДА" in reply.upper()
        return ok, reply
    except Exception as e:
        return True, "ДА (GPT недоступен: {})".format(e)


# Data history
TIMEFRAME = "30d"
INTERVAL = "1d"

# === Paths (short) ===
CANDIDATES_PATH = "core/trading/candidates.json"
SIGNALS_PATH = "core/trading/signals_short.json"
REJECTED_PATH = "core/trading/rejected_short.json"
GPT_DECISIONS_PATH = "core/trading/gpt_decisions_short.json"
SIGNALS_BACKUP_PATH = "core/trading/signals_short_backup.json"
SIGNAL_LOG_PATH = "logs/signal_log_short.json"
ADAPTIVE_CFG_PATH = "core/trading/adaptive_config_short.json"
OPEN_POSITIONS_PATH = "core/trading/open_positions.json"
REJECTED_CSV_PATH = "logs/rejected_short.csv"

# === Base thresholds (short) ===
BASE = {
    "MAX_ATR_PCT": 6.0,
    "MAX_VOLATILITY": 4.0,
    "MIN_VOLUME_TREND": 0.10,
    "MIN_RISK_SCORE": 10.0,
    "MIN_BEARISH_BODY": 0.50,  # require body <= -0.50%
    "MIN_DAY_DROP": 0.20,  # require day_change <= -0.20%
    "MIN_GAP_DOWN": 0.10,  # require gap_down <= -0.10%
    "MIN_VOLUME_RATIO": 1.05,
    "MODEL_SCORE_MIN": 60.0,
}

# === Hard bounds ===
BOUNDS = {
    "MAX_ATR_PCT": (3.0, 8.0),
    "MAX_VOLATILITY": (2.0, 8.0),
    "MIN_VOLUME_TREND": (0.05, 1.5),
    "MIN_RISK_SCORE": (5.0, 40.0),
    "MIN_BEARISH_BODY": (0.20, 2.0),
    "MIN_DAY_DROP": (0.05, 2.0),
    "MIN_GAP_DOWN": (0.05, 2.0),
    "MIN_VOLUME_RATIO": (0.95, 3.0),
    "MODEL_SCORE_MIN": (40.0, 85.0),
}

# === Adaptation steps ===
STEPS = {
    "MAX_ATR_PCT": 0.5,
    "MAX_VOLATILITY": 0.3,
    "MIN_VOLUME_TREND": 0.05,
    "MIN_RISK_SCORE": 2.0,
    "MIN_BEARISH_BODY": 0.05,
    "MIN_DAY_DROP": 0.05,
    "MIN_GAP_DOWN": 0.05,
    "MIN_VOLUME_RATIO": 0.05,
    "MODEL_SCORE_MIN": 5.0,
}

# === Target number of signals ===
TARGET_MIN = 5
TARGET_MAX = 12
MEAN_REVERT = 0.33

# === SECOND_PASS local deltas & floors (short) ===
SP_BODY_DELTA = 0.05
SP_DAY_DELTA = 0.05
SP_GAP_DELTA = 0.05
SP_VR_DELTA = 0.05
SP_VR_FLOOR = 0.95
# for gap down, floor is still <=0; allow slight negatives
SP_GAPDN_FLOOR = -0.00

# === RESCUE guards (short) ===
RESCUE_MIN_VOLUME_RATIO = 0.88
RESCUE_MAX_GAP_DOWN = -0.00  # must be ≤ this (i.e., non-positive)
RESCUE_MIN_BODY = (
    0.10  # magnitude threshold body ≤ -max(RESCUE_MIN_BODY, TH['MIN_BEARISH_BODY']*0.5)
)
RESCUE_MIN_DAY = 0.20  # magnitude threshold day_change ≤ -...
RESCUE_MAX_RSI = 40.0
RESCUE_MAX_EMA_DEV = -1.00
RESCUE_MODEL_FACTOR = 0.95
RESCUE_VTREND_FACTOR = 0.90
RESCUE_MAX_ATR_FACTOR = 1.00
RESCUE_MAX_VOL_FACTOR = 1.00

# === Precision mode ===
PRECISION_MODE = True
PRECISION_TOP_K = 5
PRECISION_EARNINGS_BLACKOUT_DAYS = 3

# === Locks ===
LOCK_ENABLE = True
LOCK_THRESHOLDS = {
    "MAX_ATR_PCT": 5.0,
    "MAX_VOLATILITY": 3.5,
}

# === Near-EMA config (short) ===
TREND_NEAR_EPS = 0.005
WEEKEND_NEAR_BONUS = 0.003


def is_weekend():
    try:
        return datetime.utcnow().weekday() >= 5
    except Exception:
        return False


def _trend_near_ok_short(price_now, ema20, ema50, rsi, ema_dev, eps):
    try:
        return ((price_now < ema20 * (1 + eps)) and (ema20 < ema50 * (1 + eps))) or (
            rsi <= 40.0 and ema_dev < 0.0
        )
    except Exception:
        return False


def safe_squeeze(series):
    if hasattr(series, "ndim") and series.ndim > 1:
        return series.squeeze()
    return series


def _alpaca_headers():
    return alpaca_headers()


def load_active_symbols():
    """Load open positions from local file or Alpaca API as fallback."""
    try:
        p = Path(OPEN_POSITIONS_PATH)
        data = {}
        if p.exists():
            txt = p.read_text() or "{}"
            try:
                data = json.loads(txt)
            except Exception:
                data = {}

        symbols = set()
        if isinstance(data, dict):
            if isinstance(data.get("positions"), list):
                for it in data["positions"]:
                    sym = it.get("symbol") or it.get("ticker")
                    qty = it.get("qty") or it.get("quantity") or 0
                    if sym and float(qty or 0) != 0.0:
                        symbols.add(sym)
            else:
                for k, v in data.items():
                    if isinstance(v, dict) and (
                        v.get("qty")
                        or v.get("quantity")
                        or v.get("position_size")
                        or v.get("entry_price")
                    ):
                        symbols.add(k)
        return symbols
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] load_active_symbols: {e}")
    return set()


def get_price_from_alpaca(symbol):
    try:
        rq = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rq.status_code == 200:
            q = rq.json().get("quote", {}) or {}
            ap = q.get("ap") or 0
            bp = q.get("bp") or 0
            if ap > 0 or bp > 0:
                return float(ap if ap > 0 else bp)
        rt = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rt.status_code == 200:
            p = (rt.json().get("trade", {}) or {}).get("p") or 0
            if p > 0:
                return float(p)
        rb = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars/latest",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rb.status_code == 200:
            c = (rb.json().get("bar", {}) or {}).get("c") or 0
            if c > 0:
                return float(c)
    except Exception as e:
        print(f"[WARN] price {symbol}: {e}")
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[WARN] yf price {symbol}: {e}")
    return 0.0


# === Helpers for "today" metrics from Alpaca daily bars ===
def _utc_today_date():
    try:
        return datetime.now(timezone.utc).date()
    except Exception:
        return datetime.utcnow().date()


def _parse_bar_date(ts: str):
    try:
        if not ts:
            return None
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _get_daily_bars(symbol: str, limit: int = 3):
    try:
        url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
        params = {"timeframe": "1Day", "limit": str(limit)}
        r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=10)
        if r.status_code == 200:
            return (r.json() or {}).get("bars", []) or []
        else:
            print(f"[WARN] daily bars {symbol}: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[WARN] daily bars {symbol} err: {e}")
    return []


def get_today_open_prev_close(symbol: str):
    bars = _get_daily_bars(symbol, limit=3)
    today = _utc_today_date()
    today_open = None
    prev_close = None
    if bars:
        try:
            bars = sorted(bars, key=lambda b: b.get("t", ""))
        except Exception:
            pass
        last = bars[-1]
        tdate = _parse_bar_date(last.get("t"))
        if tdate == today:
            try:
                today_open = float(last.get("o") or 0.0)
            except Exception:
                today_open = None
            if len(bars) >= 2:
                try:
                    prev_close = float(bars[-2].get("c") or 0.0)
                except Exception:
                    prev_close = None
        else:
            try:
                prev_close = float(last.get("c") or 0.0)
            except Exception:
                prev_close = None
    if (today_open is None) or (prev_close is None or prev_close == 0.0):
        try:
            df = yf.Ticker(symbol).history(
                period="3d", interval="1d", auto_adjust=False
            )
            if not df.empty and df.shape[0] >= 2:
                if today_open is None and df.shape[0] >= 1:
                    try:
                        today_open = float(df["Open"].iloc[-1])
                    except Exception:
                        pass
                if (prev_close is None or prev_close == 0.0) and df.shape[0] >= 2:
                    try:
                        prev_close = float(df["Close"].iloc[-2])
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] yf fallback today_open/prev_close {symbol}: {e}")
    return today_open, prev_close


def _ensure_threshold_keys(th: dict) -> dict:
    for k, v in BASE.items():
        if k not in th:
            th[k] = v
    for k, v in list(th.items()):
        if k in BOUNDS:
            lo, hi = BOUNDS[k]
            try:
                th[k] = max(lo, min(hi, float(v)))
            except Exception:
                th[k] = BASE[k]
    return th


def load_adaptive():
    cfg = {"thresholds": dict(BASE), "last_count": None, "last_update": None}
    if os.path.exists(ADAPTIVE_CFG_PATH):
        try:
            with open(ADAPTIVE_CFG_PATH, "r") as f:
                raw = json.load(f)
                if isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
                    cfg["thresholds"] = _ensure_threshold_keys(raw["thresholds"])
                    cfg["last_count"] = raw.get("last_count")
                    cfg["last_update"] = raw.get("last_update")
        except Exception as e:
            print(f"[WARN] bad adaptive_config: {e}")
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_CFG_PATH), exist_ok=True)
        with open(ADAPTIVE_CFG_PATH, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] save adaptive_config (ensure): {e}")
    return cfg


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def adjust_thresholds(cfg, count):
    th = _ensure_threshold_keys(cfg["thresholds"].copy())

    MAX_KEYS = ["MAX_ATR_PCT", "MAX_VOLATILITY"]
    MIN_KEYS = [
        "MIN_VOLUME_TREND",
        "MIN_RISK_SCORE",
        "MIN_BEARISH_BODY",
        "MIN_DAY_DROP",
        "MIN_GAP_DOWN",
        "MIN_VOLUME_RATIO",
        "MODEL_SCORE_MIN",
    ]

    direction = 0
    if count < TARGET_MIN:
        direction = -1
    elif count > TARGET_MAX:
        direction = +1

    def step_of(k):
        return STEPS[k]

    def bounds_of(k):
        return BOUNDS[k]

    def base_of(k):
        return BASE[k]

    if direction == -1:
        for k in MAX_KEYS:
            lo, hi = bounds_of(k)
            th[k] = clamp(th[k] + step_of(k), lo, hi)
        for k in MIN_KEYS:
            lo, hi = bounds_of(k)
            th[k] = clamp(th[k] - step_of(k), lo, hi)
    elif direction == +1:
        for k in MAX_KEYS:
            lo, hi = bounds_of(k)
            th[k] = clamp(th[k] - step_of(k), lo, hi)
        for k in MIN_KEYS:
            lo, hi = bounds_of(k)
            th[k] = clamp(th[k] + step_of(k), lo, hi)
    else:
        for k in MAX_KEYS + MIN_KEYS:
            lo, hi = bounds_of(k)
            th[k] = clamp(th[k] + (base_of(k) - th[k]) * MEAN_REVERT, lo, hi)

    if LOCK_ENABLE:
        for k, v in LOCK_THRESHOLDS.items():
            if k in BOUNDS:
                lo, hi = BOUNDS[k]
                th[k] = clamp(v, lo, hi)

    cfg["thresholds"] = th
    cfg["last_count"] = count
    cfg["last_update"] = datetime.now().isoformat()
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_CFG_PATH), exist_ok=True)
        with open(ADAPTIVE_CFG_PATH, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] save adaptive_config: {e}")
    return direction, th


def second_pass(retry_pool, TH, active_symbols):
    added, reasons_no = [], []
    eps = TREND_NEAR_EPS + (WEEKEND_NEAR_BONUS if is_weekend() else 0.0)
    for item in retry_pool:
        sym = item.get("symbol")
        if not sym or sym in active_symbols:
            continue
        reason = item.get("reject_reason", "")
        body = float(item.get("bearish_body", 0.0))
        day = float(item.get("day_change", item.get("percent_change", 0.0)))
        gapd = float(item.get("gap_down", 0.0))
        volr = float(item.get("volume_ratio", 0.0))

        pass_gap_vol = (gapd <= -max(TH["MIN_GAP_DOWN"] - SP_GAP_DELTA, 0.0)) and (
            volr >= max(SP_VR_FLOOR, TH["MIN_VOLUME_RATIO"] - SP_VR_DELTA)
        )
        pass_body_or_day = (body <= -(TH["MIN_BEARISH_BODY"] - SP_BODY_DELTA)) or (
            day <= -(TH["MIN_DAY_DROP"] - SP_DAY_DELTA)
        )

        ok = False
        tag = reason
        if reason == "gap_vol":
            ok = pass_gap_vol and pass_body_or_day
        elif reason in ("body_or_day", "body"):
            ok = pass_body_or_day and pass_gap_vol
        elif reason in ("trend", "trend_near"):
            ema20v = float(item.get("ema20", 0.0))
            ema50v = float(item.get("ema50", 0.0))
            price_now = float(item.get("alpaca_price", 0.0))
            rsi = float(item.get("rsi", 0.0))
            ema_dev = float(item.get("ema_dev", 0.0))
            near_ok = _trend_near_ok_short(price_now, ema20v, ema50v, rsi, ema_dev, eps)
            ok = near_ok and pass_gap_vol and pass_body_or_day
        else:
            ok = False

        if ok:
            added.append(item)
        else:
            reasons_no.append(
                "[SECOND_PASS no] {} reason={} body={:.2f} day={:.2f} gapd={:.2f} vr={:.2f}".format(
                    sym, tag, body, day, gapd, volr
                )
            )
    return added, reasons_no, len(added)


def rescue_pass(
    retry_pool,
    TH,
    signals,
    active_symbols,
    rejected,
    reasons_count,
    limit: int = 9999,
    gpt_decisions=None,
):
    rescued = 0
    if limit <= 0:
        print("[RESCUE] skipped: no slots")
        return 0
    for item in retry_pool:
        sym = item.get("symbol")
        if not sym or sym in signals or sym in active_symbols:
            continue
        reason = item.get("reject_reason", "")
        if reason not in ("gap_vol", "body_or_day"):
            continue

        volr = float(item.get("volume_ratio", 0.0))
        gapd = float(item.get("gap_down", 0.0))
        body = float(item.get("bearish_body", 0.0))
        day = float(item.get("day_change", item.get("percent_change", 0.0)))
        rsi = float(item.get("rsi", 0.0))
        ema = float(item.get("ema_dev", 0.0))
        atrp = float(item.get("atr_pct", 0.0))
        vola = float(item.get("volatility", 0.0))
        vtr = float(item.get("volume_trend", 0.0))
        msc = float(item.get("model_score", 0.0))
        alp = float(item.get("alpaca_price", 0.0))
        alp_s = float(item.get("alpha_score", 0.0))

        if volr < RESCUE_MIN_VOLUME_RATIO or gapd > RESCUE_MAX_GAP_DOWN:
            print(
                f"[RESCUE no] {sym} vr={volr:.2f} gapd={gapd:.2f} (min_vr={RESCUE_MIN_VOLUME_RATIO})"
            )
            continue
        if not (
            (body <= -max(RESCUE_MIN_BODY, TH["MIN_BEARISH_BODY"] * 0.5))
            or (day <= -max(RESCUE_MIN_DAY, TH["MIN_DAY_DROP"] * 0.5))
        ):
            print(f"[RESCUE no] {sym} body/day weak (body={body:.2f} day={day:.2f})")
            continue
        if not (rsi <= RESCUE_MAX_RSI and ema <= RESCUE_MAX_EMA_DEV):
            print(
                f"[RESCUE no] {sym} momentum guards fail (rsi={rsi:.1f} ema_dev={ema:.2f})"
            )
            continue
        if not (
            atrp <= TH["MAX_ATR_PCT"] * RESCUE_MAX_ATR_FACTOR
            and vola <= TH["MAX_VOLATILITY"] * RESCUE_MAX_VOL_FACTOR
        ):
            print(f"[RESCUE no] {sym} risk too high (atr%={atrp:.2f} vol={vola:.2f})")
            continue
        if not (
            vtr >= TH["MIN_VOLUME_TREND"] * RESCUE_VTREND_FACTOR
            and msc >= TH["MODEL_SCORE_MIN"] * RESCUE_MODEL_FACTOR
        ):
            print(
                f"[RESCUE no] {sym} vtrend/model fail (vtrend={vtr:.2f} mscore={msc:.2f})"
            )
            continue

        ok_r, rep_r = gpt_confirm_short(
            sym, alp, day, rsi, ema, volr, alp_s, msc, atrp, vola, vtr, body, gapd
        )
        if gpt_decisions is not None:
            gpt_decisions[sym] = rep_r
        if not ok_r:
            print(f"[RESCUE GPT no] {sym} {rep_r}")
            try:
                _tg_safe("Rescue GPT отказ по {}: {}".format(sym, rep_r))
            except Exception:
                pass
            rejected[sym] = "GPT отклонил (RESCUE): {}".format(rep_r)
            reasons_count["GPT отклонил"] += 1
            continue

        signals[sym] = {
            "price": round(alp, 2),
            "action": "SELL_SHORT",
            "confidence": round(alp_s, 2),
            "score": round(msc, 2),
            "atr": round(float(item["atr_value"]), 2),
            "atr_pct": round(atrp, 2),
            "volatility": round(vola, 2),
            "volume_trend": round(vtr, 2),
            "bearish_body": round(body, 2),
            "gap_down": round(gapd, 2),
            "day_change": round(day, 2),
            "rescue_pass": True,
        }
        print(
            f"[RESCUE yes] {sym} reason={reason} body={body:.2f} day={day:.2f} gapd={gapd:.2f} vr={volr:.2f}"
        )
        try:
            _tg_signal(
                "Rescue-pass (RESCUE)",
                sym,
                signals[sym]["price"],
                body,
                day,
                gapd,
                volr,
            )
        except Exception:
            pass
        rescued += 1
        if rescued >= limit:
            break

    print(f"[RESCUE] added: {rescued}")
    return rescued


# === Precision scoring helpers (short) ===
def _clip(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _norm(x, lo, hi):
    x = _clip(x, lo, hi)
    return 0.0 if hi == lo else (x - lo) / (hi - lo)


def _get_52w_proximity_low(sym):
    try:
        df = yf.download(
            sym, period="250d", interval="1d", progress=False, auto_adjust=False
        )
        if df is None or df.empty:
            return 0.0
        c = float(df["Close"].iloc[-1])
        mn = float(df["Close"].min())
        if c <= 0:
            return 0.0
        # Closer to 52w low → higher value
        return _clip((c - mn) / max(1e-9, c), 0.0, 1.0)
    except Exception:
        return 0.0


def _earnings_blackout(sym):
    try:
        t = yf.Ticker(sym)
        cal = getattr(t, "calendar", None)
        if cal is None or cal.empty:
            return False
        if "Earnings Date" in cal.index:
            dt = cal.loc["Earnings Date"][0]
        else:
            dt = cal.iloc[0, 0]
        if not hasattr(dt, "to_pydatetime"):
            return False
        edate = dt.to_pydatetime().date()
        today = datetime.utcnow().date()
        delta = abs((edate - today).days)
        return delta <= PRECISION_EARNINGS_BLACKOUT_DAYS
    except Exception:
        return False


def compute_precision_score_short(sym, d, TH):
    ms = _clip(d.get("score", 0.0), 0.0, 100.0) / 100.0
    alpha = _clip(d.get("confidence", 0.0), -5.0, 5.0) / 3.0
    vr = _norm(d.get("volume_ratio", 0.0), 0.8, 2.0)
    rsi_inv = _norm(max(0.0, 70.0 - d.get("rsi", 50.0)), 0.0, 20.0)  # lower RSI better
    emad_neg = _norm(max(0.0, -d.get("ema_dev", 0.0)), 0.0, 5.0)
    daypos_low = _clip(1.0 - _clip(d.get("day_range_pos", 0.0), 0.0, 1.0), 0.0, 1.0)
    prox52low = _clip(_get_52w_proximity_low(sym), 0.0, 1.0)

    atrp = _clip(d.get("atr_pct", 0.0), 0.0, TH["MAX_ATR_PCT"]) / max(
        1e-9, TH["MAX_ATR_PCT"]
    )
    vola = _clip(d.get("volatility", 0.0), 0.0, TH["MAX_VOLATILITY"]) / max(
        1e-9, TH["MAX_VOLATILITY"]
    )
    earn_pen = 1.0 if _earnings_blackout(sym) else 0.0

    base = (
        0.22 * ms
        + 0.18 * alpha
        + 0.16 * vr
        + 0.14 * rsi_inv
        + 0.12 * emad_neg
        + 0.10 * daypos_low
        + 0.08 * prox52low
    )
    penalty = 0.12 * atrp + 0.10 * vola + 0.25 * earn_pen
    score = _clip(base - penalty, 0.0, 1.0)
    return score


def precision_finalize(signals: dict, rejected: dict, TH: dict):
    ranked = []
    for sym, d in signals.items():
        q = compute_precision_score_short(sym, d, TH)
        ranked.append((sym, q))
    ranked.sort(key=lambda x: x[1], reverse=True)
    keep = set([s for s, _ in ranked[:PRECISION_TOP_K]])
    dropped = [(s, q) for s, q in ranked[PRECISION_TOP_K:]]
    if dropped:
        for s, q in dropped:
            rejected[s] = f"precision_cut (q={q:.2f})"
        try:
            with open("logs/precision_debug_short.json", "w") as f:
                json.dump(
                    {"ranked": ranked, "kept": list(keep)},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass
    return {k: v for k, v in signals.items() if k in keep}


def main():
    with open(CANDIDATES_PATH, "r") as f:
        tickers = json.load(f)

    adaptive_cfg = load_adaptive()
    TH = adaptive_cfg["thresholds"]

    active_symbols = load_active_symbols()
    print(f"Check {len(tickers)} tickers (SHORT)...")
    print(f"Thresholds: {json.dumps(TH, ensure_ascii=False)}")

    signals = {}
    rejected = {}
    gpt_decisions = {}
    reasons_count = defaultdict(int)
    retry_pool = []

    for symbol in tickers:
        if symbol in active_symbols:
            print(f"[SKIP] {symbol} — already in position")
            continue

        print(f"\nProcess {symbol} (SHORT)...")
        try:
            data = yf.download(
                symbol,
                period=TIMEFRAME,
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
            )
            if data is None or getattr(data, "empty", True) or data.shape[0] < 2:
                rejected[symbol] = "Недостаточно данных"
                reasons_count["Недостаточно данных"] += 1
                continue

            close = safe_squeeze(data["Close"]).dropna()
            volume = safe_squeeze(data["Volume"]).dropna()
            high = safe_squeeze(data["High"]).dropna()
            low = safe_squeeze(data["Low"]).dropna()
            open_price = safe_squeeze(data["Open"]).dropna()

            if len(close) < 15 or len(volume) < 5:
                rejected[symbol] = "Недостаточно чистых данных"
                reasons_count["Недостаточно чистых данных"] += 1
                continue

            alpaca_price = get_price_from_alpaca(symbol)
            if alpaca_price is None or alpaca_price == 0:
                rejected[symbol] = "Цена Alpaca отсутствует или равна 0"
                reasons_count["Цена Alpaca отсутствует или равна 0"] += 1
                continue

            # Indicators (daily context)
            rsi = float(RSIIndicator(close=close).rsi().iloc[-1])
            ema10 = float(EMAIndicator(close=close, window=10).ema_indicator().iloc[-1])
            ema20 = float(EMAIndicator(close=close, window=20).ema_indicator().iloc[-1])
            ema50 = float(EMAIndicator(close=close, window=50).ema_indicator().iloc[-1])

            # Risk context
            atr_value = float(
                AverageTrueRange(high=high, low=low, close=close, window=14)
                .average_true_range()
                .iloc[-1]
            )
            atr_pct = float((atr_value / max(1e-12, close.iloc[-1])) * 100.0)
            volatility = float(close.pct_change().std() * 100.0)
            volume_ema = float(EMAIndicator(volume, window=10).ema_indicator().iloc[-1])
            volume_trend = float(volume.iloc[-1] / max(1e-9, volume_ema))

            # Today metrics (live-aware)
            last_close = float(close.iloc[-1])
            _tod_open, _prev_close = get_today_open_prev_close(symbol)
            prev_close = (
                float(_prev_close)
                if (_prev_close is not None and _prev_close != 0)
                else float(close.iloc[-2])
            )
            today_open = (
                float(_tod_open)
                if (_tod_open is not None and _tod_open != 0)
                else float(open_price.iloc[-1])
            )
            live_price = float(alpaca_price)

            ema_dev = float(((live_price - ema10) / max(1e-12, ema10)) * 100.0)
            day_change = ((live_price - prev_close) / max(1e-12, prev_close)) * 100.0
            bearish_body = ((live_price - today_open) / max(1e-12, today_open)) * 100.0
            gap_down = (
                (today_open - prev_close) / max(1e-12, prev_close)
            ) * 100.0  # negative if gap down
            day_range_pos = 0.0
            rng = float(high.iloc[-1] - low.iloc[-1])
            if rng != 0.0:
                day_range_pos = float(
                    (last_close - float(low.iloc[-1])) / max(1e-9, rng)
                )

            volume_ratio = float(volume.iloc[-1] / (float(volume[:-1].mean()) + 1e-6))

            # Short alpha & model
            alpha_s = float(
                calculate_alpha_short(
                    abs(min(day_change, 0.0)), volume_ratio, rsi, abs(min(ema_dev, 0.0))
                )
            )
            risk_score = float(alpha_s * 100.0)
            model_score = (
                alpha_s * 0.5
                + min(volume_ratio, 2.0) * 0.3
                + (1.0 - min(max(rsi / 100.0, 0.0), 1.0)) * 0.2
            ) * 100.0

            base_info = {
                "symbol": symbol,
                "alpaca_price": round(float(alpaca_price), 2),
                "percent_change": float(day_change),
                "day_change": float(day_change),
                "rsi": float(rsi),
                "ema_dev": float(ema_dev),
                "volume_ratio": float(volume_ratio),
                "alpha_score": float(alpha_s),
                "model_score": float(model_score),
                "atr_value": float(atr_value),
                "atr_pct": float(atr_pct),
                "volatility": float(volatility),
                "volume_trend": float(volume_trend),
                "bearish_body": float(bearish_body),
                "gap_down": float(gap_down),
                "day_range_pos": float(day_range_pos),
                "ema20": float(ema20),
                "ema50": float(ema50),
            }

            # === Gates diagnostic ===
            pass_body_or_day = (bearish_body <= -TH["MIN_BEARISH_BODY"]) or (
                day_change <= -TH["MIN_DAY_DROP"]
            )
            pass_trend = (live_price < ema20) and (ema20 < ema50)
            pass_gap_vol = (gap_down <= -TH["MIN_GAP_DOWN"]) and (
                volume_ratio >= TH["MIN_VOLUME_RATIO"]
            )
            pass_risk = (
                (atr_pct <= TH["MAX_ATR_PCT"])
                and (volatility <= TH["MAX_VOLATILITY"])
                and (volume_trend >= TH["MIN_VOLUME_TREND"])
            )
            pass_scores = (risk_score >= TH["MIN_RISK_SCORE"]) and (
                model_score >= TH["MODEL_SCORE_MIN"]
            )
            print(
                "[GATES S] {sym} body/day={bod} (body={body:.2f}≤-{mbb:.2f} | day={day:.2f}≤-{mdc:.2f}) | "
                "trend={tr} | gap_vol={gv} (gapd={gap:.2f}≤-{mg:.2f} | vr={vr:.2f}≥{mvr:.2f}) | "
                "risk={rk} | scores={sc}".format(
                    sym=symbol,
                    bod=("OK" if pass_body_or_day else "NO"),
                    body=bearish_body,
                    mbb=TH["MIN_BEARISH_BODY"],
                    day=day_change,
                    mdc=TH["MIN_DAY_DROP"],
                    tr=("OK" if pass_trend else "NO"),
                    gv=("OK" if pass_gap_vol else "NO"),
                    gap=gap_down,
                    mg=TH["MIN_GAP_DOWN"],
                    vr=volume_ratio,
                    mvr=TH["MIN_VOLUME_RATIO"],
                    rk=("OK" if pass_risk else "NO"),
                    sc=("OK" if pass_scores else "NO"),
                )
            )

            # 1) Candle/Day gate
            if not pass_body_or_day:
                rejected[symbol] = "Слабая красная свеча/день"
                reasons_count["Слабая красная свеча/день"] += 1
                near = (
                    bearish_body
                    <= -max(
                        BOUNDS["MIN_BEARISH_BODY"][0],
                        TH["MIN_BEARISH_BODY"] - STEPS["MIN_BEARISH_BODY"],
                    )
                ) or (
                    day_change
                    <= -max(
                        BOUNDS["MIN_DAY_DROP"][0],
                        TH["MIN_DAY_DROP"] - STEPS["MIN_DAY_DROP"],
                    )
                )
                if near:
                    retry_pool.append({**base_info, "reject_reason": "body_or_day"})
                continue

            # 2) EMA trend gate (with Near-EMA)
            if not pass_trend:
                eps = TREND_NEAR_EPS + (WEEKEND_NEAR_BONUS if is_weekend() else 0.0)
                if _trend_near_ok_short(live_price, ema20, ema50, rsi, ema_dev, eps):
                    rejected[symbol] = "Тренд-фильтр EMA (почти вниз) — во SECOND_PASS"
                    reasons_count["Тренд-фильтр EMA (почти)"] += 1
                    retry_pool.append({**base_info, "reject_reason": "trend"})
                    continue
                rejected[symbol] = "Тренд-фильтр EMA (Price>=EMA20 или EMA20>=EMA50)"
                reasons_count["Тренд-фильтр EMA"] += 1
                continue

            # 3) GAP and volume
            if not pass_gap_vol:
                rejected[symbol] = (
                    "GapDown/объём недостаточны (gapd={:.2f}%, vr={:.2f}×)".format(
                        gap_down, volume_ratio
                    )
                )
                reasons_count["GapDown/объём недостаточны"] += 1
                if (gap_down <= -(TH["MIN_GAP_DOWN"] - STEPS["MIN_GAP_DOWN"])) and (
                    volume_ratio >= TH["MIN_VOLUME_RATIO"] - STEPS["MIN_VOLUME_RATIO"]
                ):
                    retry_pool.append({**base_info, "reject_reason": "gap_vol"})
                continue

            # 4) Risk & volatility
            if atr_pct > TH["MAX_ATR_PCT"]:
                rejected[symbol] = "ATR% слишком высокий ({:.2f}%)".format(atr_pct)
                reasons_count["ATR высокий"] += 1
                continue
            if volatility > TH["MAX_VOLATILITY"]:
                rejected[symbol] = "Волатильность слишком высокая ({:.2f}%)".format(
                    volatility
                )
                reasons_count["Волатильность высокая"] += 1
                continue
            if volume_trend < TH["MIN_VOLUME_TREND"]:
                rejected[symbol] = "Тренд объёма отрицательный ({:.2f}×)".format(
                    volume_trend
                )
                reasons_count["Тренд объёма отрицательный"] += 1
                continue

            # 5) Scores
            if risk_score < TH["MIN_RISK_SCORE"]:
                rejected[symbol] = "Низкий риск-скор SHORT ({:.2f})".format(risk_score)
                reasons_count["Низкий риск-скор SHORT"] += 1
                if risk_score >= TH["MIN_RISK_SCORE"] - 2.0:
                    retry_pool.append({**base_info, "reject_reason": "risk"})
                continue
            if model_score < TH["MODEL_SCORE_MIN"]:
                rejected[symbol] = "Слабый модельный скор SHORT ({:.2f})".format(
                    model_score
                )
                reasons_count["Слабый модельный скор SHORT"] += 1
                if model_score >= TH["MODEL_SCORE_MIN"] - 5.0:
                    retry_pool.append({**base_info, "reject_reason": "model"})
                continue

            # GPT-confirm (base)
            gpt_reply = "ДА (без GPT: ключ не задан)"
            ok_gpt, gpt_reply = gpt_confirm_short(
                symbol,
                float(alpaca_price),
                float(day_change),
                float(rsi),
                float(ema_dev),
                float(volume_ratio),
                float(alpha_s),
                float(model_score),
                float(atr_pct),
                float(volatility),
                float(volume_trend),
                float(bearish_body),
                float(gap_down),
            )
            gpt_decisions[symbol] = gpt_reply

            summary_msg = (
                "Новый сигнал (SHORT)\n"
                "{} @ {:.2f}\n"
                "∆%={:.2f}% | RSI={:.2f} | EMA dev={:.2f}%\n"
                "ATR%={:.2f} | Vol={:.2f}% | VTrend={:.2f}×\n"
                "body={:.2f}% | gapDown={:.2f}%\n"
                "GPT: {}".format(
                    symbol,
                    float(alpaca_price),
                    float(day_change),
                    float(rsi),
                    float(ema_dev),
                    float(atr_pct),
                    float(volatility),
                    float(volume_trend),
                    float(bearish_body),
                    float(gap_down),
                    gpt_reply,
                )
            )

            if ok_gpt:
                signals[symbol] = {
                    "price": round(float(alpaca_price), 2),
                    "action": "SELL_SHORT",
                    "confidence": round(alpha_s, 2),
                    "score": round(model_score, 2),
                    "atr": round(atr_value, 2),
                    "atr_pct": round(atr_pct, 2),
                    "volatility": round(volatility, 2),
                    "volume_trend": round(volume_trend, 2),
                    "bearish_body": round(bearish_body, 2),
                    "gap_down": round(gap_down, 2),
                    "day_change": round(day_change, 2),
                    "day_range_pos": round(day_range_pos, 3),
                }
                try:
                    _tg_safe(summary_msg)
                except Exception:
                    pass
            else:
                rejected[symbol] = "GPT отклонил: {}".format(gpt_reply)
                reasons_count["GPT отклонил"] += 1
                try:
                    _tg_safe(summary_msg)
                except Exception:
                    pass

        except Exception as e:
            rejected[symbol] = "Исключение/Ошибка: {}".format(str(e))
            reasons_count["Исключение/Ошибка"] += 1
            print(f"ERROR {symbol}: {e}")

    # === SECOND PASS ===
    added_second = 0
    if retry_pool:
        print("\nRetry pool (почти-прошедшие, SHORT):")
        for item in retry_pool[:100]:
            print(
                "  - {} reason: {}".format(
                    item.get("symbol"), item.get("reject_reason")
                )
            )

        sp_add, sp_no_logs, _ = second_pass(retry_pool, TH, active_symbols)
        for line in sp_no_logs:
            print(line)

        slots = max(0, TARGET_MAX - len(signals))
        sp_add_sorted = sorted(
            sp_add,
            key=lambda it: (
                it.get("model_score", 0.0),
                -abs(it.get("day_change", it.get("percent_change", 0.0))),
            ),
            reverse=True,
        )
        for item in sp_add_sorted[:slots]:
            sym = item["symbol"]
            if sym in signals or sym in active_symbols:
                continue
            body_val = float(item.get("bearish_body", 0.0))
            day_val = float(item.get("day_change", item.get("percent_change", 0.0)))
            gapd_val = float(item.get("gap_down", 0.0))
            volr_val = float(item.get("volume_ratio", 0.0))
            rsi_val = float(item.get("rsi", 0.0))
            ema_val = float(item.get("ema_dev", 0.0))
            atrp_val = float(item.get("atr_pct", 0.0))
            vola_val = float(item.get("volatility", 0.0))
            vtr_val = float(item.get("volume_trend", 0.0))
            alp_pr = float(item.get("alpaca_price", 0.0))
            alpha_s = float(item.get("alpha_score", 0.0))
            model_s = float(item.get("model_score", 0.0))

            print(
                "[SECOND_PASS candidate S] {} reason={} body={:.2f} day={:.2f} gapd={:.2f} vr={:.2f}".format(
                    sym,
                    item.get("reject_reason"),
                    body_val,
                    day_val,
                    gapd_val,
                    volr_val,
                )
            )

            ok_sp, rep_sp = gpt_confirm_short(
                sym,
                alp_pr,
                day_val,
                rsi_val,
                ema_val,
                volr_val,
                alpha_s,
                model_s,
                atrp_val,
                vola_val,
                vtr_val,
                body_val,
                gapd_val,
            )
            gpt_decisions[sym] = rep_sp
            if not ok_sp:
                rejected[sym] = "GPT отклонил (SECOND_PASS): {}".format(rep_sp)
                reasons_count["GPT отклонил"] += 1
                try:
                    _tg_safe("SECOND_PASS GPT отказ по {}: {}".format(sym, rep_sp))
                except Exception:
                    pass
                continue

            signals[sym] = {
                "price": round(alp_pr, 2),
                "action": "SELL_SHORT",
                "confidence": round(alpha_s, 2),
                "score": round(model_s, 2),
                "atr": round(float(item["atr_value"]), 2),
                "atr_pct": round(atrp_val, 2),
                "volatility": round(vola_val, 2),
                "volume_trend": round(vtr_val, 2),
                "bearish_body": round(body_val, 2),
                "gap_down": round(gapd_val, 2),
                "day_change": round(day_val, 2),
                "second_pass": True,
            }
            added_second += 1
            try:
                _tg_signal(
                    "Второй проход (SECOND_PASS, SHORT)",
                    sym,
                    signals[sym]["price"],
                    body_val,
                    day_val,
                    gapd_val,
                    volr_val,
                )
            except Exception:
                pass
        print(f"[SECOND_PASS] added: {added_second}")

    # === RESCUE PASS ===
    remaining_slots = max(0, TARGET_MAX - len(signals))
    rescue_added = rescue_pass(
        retry_pool,
        TH,
        signals,
        active_symbols,
        rejected,
        reasons_count,
        limit=remaining_slots,
        gpt_decisions=gpt_decisions,
    )

    # === PRECISION FINALIZE ===
    prec_before = len(signals)
    prec_cuts = 0
    if PRECISION_MODE and len(signals) > PRECISION_TOP_K:
        signals = precision_finalize(signals, rejected, TH)
        prec_cuts = max(0, prec_before - len(signals))

    # === Save results ===
    os.makedirs("logs", exist_ok=True)

    with open(SIGNALS_PATH, "w") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)
    try:
        shutil.copy(SIGNALS_PATH, SIGNALS_BACKUP_PATH)
    except Exception as e:
        print(f"[WARN] backup signals: {e}")

    with open(REJECTED_PATH, "w") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    try:
        with open(GPT_DECISIONS_PATH, "w") as f:
            json.dump(gpt_decisions, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        with open(SIGNAL_LOG_PATH, "w") as f:
            json.dump(
                [
                    {
                        "symbol": s,
                        "timestamp": datetime.now().isoformat(),
                        "score": d.get("score", 0),
                        "confidence": d.get("confidence", 0),
                    }
                    for s, d in signals.items()
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    # === Rejection summary ===
    print("\nSummary (rejections, SHORT):")
    for k in sorted(reasons_count, key=reasons_count.get, reverse=True):
        print(f"  • {k}: {reasons_count[k]}")

    print(f"\nSignals (SHORT): {len(signals)} -> {SIGNALS_PATH}")
    print(f"Rejected (SHORT): {len(rejected)} -> {REJECTED_PATH}")
    print(f"GPT decisions (SHORT): {GPT_DECISIONS_PATH}")

    # === Telegram summary ===
    try:
        top_reasons = sorted(reasons_count.items(), key=lambda kv: kv[1], reverse=True)[
            :5
        ]
        reasons_lines = (
            "\n".join("  • {}: {}".format(k, v) for k, v in top_reasons)
            if top_reasons
            else "  • n/a"
        )
        msg_lines = [
            "Итоговый отчёт (signals SHORT)",
            "Сигналы: {} (SECOND_PASS={}, RESCUE={})".format(
                len(signals), added_second, rescue_added
            ),
            "Precision cuts: {}".format(prec_cuts),
            "Отклонено: {}".format(len(rejected)),
            "Топ причин отказов:",
            reasons_lines,
            "Tickers: {}".format(", ".join(sorted(signals.keys())) or "—"),
        ]
        _tg_safe("\n".join(msg_lines))
    except Exception:
        pass

    # === Auto-adapt for next run ===
    direction, new_th = adjust_thresholds(adaptive_cfg, len(signals))
    if direction == -1:
        mode = "ослабил"
    elif direction == +1:
        mode = "ужесточил"
    else:
        mode = "возврат к базовым"

    change_msg = (
        "Авто-адаптация фильтров (SHORT, {})\n"
        "Сигналов: {} (цель {}-{})\n"
        "Новые пороги: {}".format(
            mode,
            len(signals),
            TARGET_MIN,
            TARGET_MAX,
            json.dumps(new_th, ensure_ascii=False),
        )
    )
    try:
        _tg_safe(change_msg)
    except Exception as e:
        print(f"[WARN] Telegram adapt msg: {e}")


if __name__ == "__main__":
    main()
