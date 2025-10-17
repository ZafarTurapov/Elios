# -*- coding: utf-8 -*-
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from openai import OpenAI
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# === Ensure /root/stockbot on sys.path ===
if "/root/stockbot" not in sys.path:
    sys.path.insert(0, "/root/stockbot")

# --- Market calendar (soft import)
try:
    from core.utils.market_calendar import is_market_open_today
except ImportError:

    def is_market_open_today():
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            r = requests.get(
                f"{ALPACA_BASE_URL}/v2/calendar",
                params={"start": today, "end": today},
                headers=_alpaca_headers(),
                timeout=10,
            )
            if r.status_code == 200:
                days = r.json() or []
                return bool(days)
        except Exception:
            return datetime.now(timezone.utc).weekday() < 5


from core.trading.alpha_utils import calculate_alpha_score

# --- Imports for trading utilities ---
from core.trading.anomaly_detector import detect_anomalies
from core.utils.telegram import send_telegram_message

# === Alpaca Configuration ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
ALPACA_NEWS_BASE = os.getenv("ALPACA_NEWS_BASE", "https://data.alpaca.markets/v1beta1")


def _alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}


# === OpenAI Configuration ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_SIGNAL_MODEL = os.getenv("ELIOS_SIGNAL_GPT_MODEL", "gpt-4o-mini")

# === IO Paths ===
CANDIDATES_PATH = "core/trading/candidates.json"
OPEN_POSITIONS_LOCAL = "core/trading/open_positions.json"
SIGNALS_PATH = "core/trading/signals.json"
REJECTED_PATH = "core/trading/rejected.json"
GPT_DECISIONS_PATH = "core/trading/gpt_decisions.json"
SIGNALS_BACKUP_PATH = "core/trading/signals_backup.json"
SIGNAL_LOG_PATH = "logs/signal_log.json"
ADAPTIVE_CFG_PATH = "core/trading/adaptive_config.json"
REJECTED_CSV_PATH = "logs/rejected.csv"
MACRO_LOG_PATH = "logs/macro.json"
DECISIONS_DIR = Path("logs/decisions")
SNAPSHOT_ROOT = Path("logs/snapshots")

# === Modes / Flags ===
HEARTBEAT_N = int(os.getenv("ELIOS_HEARTBEAT_N", "25"))
HEARTBEAT_TG_EVERY = int(os.getenv("ELIOS_HEARTBEAT_TG_EVERY", "4"))
TICKER_BUDGET_S = float(os.getenv("ELIOS_TICKER_BUDGET_S", "6"))
DEBUG_MODE = os.getenv("ELIOS_DEBUG", "0") == "1"
RESCUE_ENABLED = True
FALLBACK_ENABLED = True
HYBRID_ENABLED = True
NEWS_GUARD_ENABLED = os.getenv("ELIOS_NEWS_GUARD", "1") == "1"
WICK_GUARD_ENABLED = os.getenv("ELIOS_WICK_GUARD", "1") == "1"
MACRO_GUARD_ENABLED = os.getenv("ELIOS_MACRO_GUARD", "1") == "1"
WICK_MODE = os.getenv("ELIOS_WICK_MODE", "penalty").strip().lower()
WICK_PENALTY_BASE = float(os.getenv("ELIOS_WICK_PENALTY_BASE", "8.0"))
WICK_PENALTY_MAX = float(os.getenv("ELIOS_WICK_PENALTY_MAX", "12.0"))
LIVE_VWAP_CHECK = os.getenv("ELIOS_LIVE_VWAP_CHECK", "1") == "1"
VWAP_RECLAIM_TOL = float(os.getenv("ELIOS_VWAP_RECLAIM_TOL", "0.001"))
ORH_TOUCH_TOL = float(os.getenv("ELIOS_ORH_TOUCH_TOL", "0.001"))
UPPER_WICK_MULT = float(os.getenv("ELIOS_UPPER_WICK_MULT", "1.5"))
WICK_STRONG_BODY_MAX = float(os.getenv("ELIOS_WICK_STRONG_BODY_MAX", "3.5"))
NEWS_LOOKBACK_HOURS = int(os.getenv("ELIOS_NEWS_LOOKBACK_HOURS", "36"))
NEWS_NEG_THRESH = int(os.getenv("ELIOS_NEWS_NEG_THRESHOLD", "2"))
BYPASS_FILTERS = os.getenv("ELIOS_NO_FILTERS", "0") == "1"
HYBRID_MODEL_MARGIN = 5.0
HYBRID_MAX_OVERRIDES = 3
HYBRID_STRICT_NOISE = True
QUALITY_GATE_ENABLED = os.getenv("ELIOS_QUALITY_GATE", "1") == "1"
QUALITY_MIN_GAP_PCT = float(os.getenv("ELIOS_QUALITY_MIN_GAP", "1.0"))
QUALITY_MIN_BODY_PCT = float(os.getenv("ELIOS_QUALITY_MIN_BODY", "1.0"))
QUALITY_MIN_RANK = float(os.getenv("ELIOS_QUALITY_MIN_RANK", "-0.02"))
QUALITY_MODEL_BONUS = float(os.getenv("ELIOS_QUALITY_MODEL_BONUS", "3.0"))

# === Thresholds and Bounds ===
BASE = {
    "MAX_ATR_PCT": 8.0,
    "MAX_VOLATILITY": 5.0,
    "MIN_VOLUME_TREND": 0.10,
    "MIN_RISK_SCORE": 10.0,
    "MIN_BULLISH_BODY": 1.0,
    "MIN_GAP_UP": 1.0,
    "MIN_VOLUME_RATIO": 0.95,
    "MODEL_SCORE_MIN": 55.0,
}
BOUNDS = {
    "MAX_ATR_PCT": (0.5, 15.0),
    "MAX_VOLATILITY": (0.5, 15.0),
    "MIN_VOLUME_TREND": (0.00, 2.0),
    "MIN_RISK_SCORE": (0.0, 100.0),
    "MIN_BULLISH_BODY": (0.0, 5.0),
    "MIN_GAP_UP": (-2.0, 5.0),
    "MIN_VOLUME_RATIO": (0.0, 5.0),
    "MODEL_SCORE_MIN": (0.0, 100.0),
}
STEPS = {
    "MAX_ATR_PCT": 0.5,
    "MAX_VOLATILITY": 0.3,
    "MIN_VOLUME_TREND": 0.05,
    "MIN_RISK_SCORE": 2.0,
    "MIN_BULLISH_BODY": 1.0,
    "MIN_GAP_UP": 1.0,
    "MIN_VOLUME_RATIO": 0.05,
    "MODEL_SCORE_MIN": 5.0,
}
TARGET_MIN = 3
TARGET_MAX = 4
MEAN_REVERT = 0.33
E_TARGET_MIN = TARGET_MIN
E_TARGET_MAX = TARGET_MAX
E_HYBRID_MAX_OVERRIDES = HYBRID_MAX_OVERRIDES
DISABLE_LIMITS = False
FORCE_RUN = ("--force" in sys.argv) or (os.getenv("ELIOS_FORCE_OPEN", "0") == "1")

# === Relative Strength (RS) ===
RS_CACHE_PATH = "logs/cache/rs.json"
RS_ENABLED = os.getenv("ELIOS_RS_ENABLED", "1") == "1"
RS_LOOKBACK_D = int(os.getenv("ELIOS_RS_LOOKBACK_D", "63"))
RS_MIN_PCTL_BASE = int(os.getenv("ELIOS_RS_MIN_PCTL", "65"))
RS_BETA_ADJ = os.getenv("ELIOS_RS_BETA_ADJ", "1") == "1"
RS_SLOPE_REQ = os.getenv("ELIOS_RS_SLOPE_REQ", "0") == "1"
RS_WEIGHT = float(os.getenv("ELIOS_RS_WEIGHT", "0.10"))
RS_TTL_MIN = int(os.getenv("ELIOS_RS_TTL_MIN", "720"))

# === Squeeze Features Import (Safe) ===
try:
    from core.trading.squeeze_features import get_squeeze_features
except ImportError:

    def get_squeeze_features(symbol: str, **kwargs):
        return None


def safe_squeeze(series):
    try:
        if hasattr(series, "ndim") and series.ndim > 1:
            return series.squeeze()
    except Exception:
        pass
    return series


# --- HTTP Session with Retries ---
def _init_http_session():
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        sess = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        requests.get = sess.get
        requests.post = sess.post
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] HTTP session init: {e}")


_init_http_session()


# --- Entry Window Helper (Asia/Tashkent) ---
def _within_entry_window():
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Tashkent")
        now = datetime.now(tz)
        ENTRY_FROM = os.getenv("ENTRY_FROM", "09:00")  # Default entry window start
        ENTRY_TILL = os.getenv("ENTRY_TILL", "17:00")  # Default entry window end

        def _hm(s):
            hh, mm = map(int, str(s).split(":")[:2])
            return hh, mm

        fh, fm = _hm(ENTRY_FROM)
        th, tm = _hm(ENTRY_TILL)
        start = now.replace(hour=fh, minute=fm, second=0, microsecond=0)
        end = now.replace(hour=th, minute=tm, second=0, microsecond=0)
        return start <= now <= end
    except Exception:
        return True  # Fail-open if parsing fails


# --- Intra Data Quality Gate using Alpaca Data API ---
def _data_quality_ok(symbol: str):
    QG_ENABLED = globals().get(
        "QG_ENABLED", os.getenv("ELIOS_QUALITY_GATE_IN_SIGNALS", "1") == "1"
    )
    QG_FEED = globals().get("QG_FEED", os.getenv("ELIOS_ALPACA_FEED", "iex"))
    QG_SPREAD_MAX = globals().get(
        "QG_SPREAD_MAX", float(os.getenv("ELIOS_QG_SPREAD_MAX_PCT", "0.003"))
    )
    QG_DVOL_5M = globals().get(
        "QG_DVOL_5M", float(os.getenv("ELIOS_QG_MIN_DOLLAR_VOL_5M", "300000"))
    )
    if not QG_ENABLED:
        return True, "qg:disabled"
    try:
        rq = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest",
            params={"feed": QG_FEED},
            headers=_alpaca_headers(),
            timeout=7,
        )
        if rq.status_code != 200:
            return True, f"qg:quotes:{rq.status_code}"
        q = rq.json().get("quote", {}) or {}
        bp = float(q.get("bp") or 0)
        ap = float(q.get("ap") or 0)
        if bp > 0 and ap > 0:
            spr = (ap - bp) / (((ap + bp) / 2.0) or 1e-9)
            if spr > QG_SPREAD_MAX:
                return False, f"qg:spread>{QG_SPREAD_MAX*100:.2f}%"
        r1m = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars",
            params={
                "timeframe": "1Min",
                "limit": 5,
                "adjustment": "raw",
                "feed": QG_FEED,
            },
            headers=_alpaca_headers(),
            timeout=7,
        )
        if r1m.status_code != 200:
            return True, f"qg:bars1m:{r1m.status_code}"
        bars = (r1m.json() or {}).get("bars") or []
        dvol = sum(float(b.get("c") or 0) * float(b.get("v") or 0) for b in bars)
        if dvol < QG_DVOL_5M:
            return False, f"qg:$vol5m<{QG_DVOL_5M:,.0f}"
        return (
            True,
            f"qg:ok spr<= {QG_SPREAD_MAX*100:.2f}% & $vol5m>= {QG_DVOL_5M:,.0f}",
        )
    except Exception as e:
        return True, f"qg:error:{e}"


# --- Alpaca Helpers ---
def is_tradable(symbol: str) -> bool:
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/assets/{symbol}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return True
        a = r.json() or {}
        return bool(a.get("tradable", False)) and (
            a.get("status", "active") == "active"
        )
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] assets {symbol}: {e}")
        return True


def load_active_symbols():
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/positions", headers=_alpaca_headers(), timeout=10
        )
        if r.status_code == 200:
            return set((p.get("symbol") or "").upper() for p in r.json())
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] positions: {e}")
    return set()


def load_open_order_symbols():
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/orders",
            params={"status": "open", "limit": 200},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code == 200:
            return set((o.get("symbol") or "").upper() for o in r.json())
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] orders: {e}")
    return set()


def load_local_open_positions():
    try:
        if not os.path.exists(OPEN_POSITIONS_LOCAL):
            return set()
        with open(OPEN_POSITIONS_LOCAL, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return set()
        ts = data.get("__timestamp__")
        if ts:
            try:
                age_h = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                ).total_seconds() / 3600.0
                if age_h > 48:
                    return set()
            except Exception:
                pass
        syms = []
        for k, v in data.items():
            if str(k).startswith("__"):
                continue
            qty = 0.0
            if isinstance(v, dict):
                try:
                    qty = float(v.get("qty") or v.get("quantity") or 0)
                except Exception:
                    qty = 0.0
            elif isinstance(v, (int, float)):
                qty = float(v)
            if qty > 0:
                syms.append(str(k).upper())
        if len(syms) > 1000:
            return set()
        return set(syms)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] local open_positions: {e}")
        return set()


def _intraday_metrics(symbol: str):
    """Return dict {'vwap':..., 'orh':..., 'last':..., 'n':int} or None."""
    try:
        op, cl = _today_open_close_utc()
        start = (
            (datetime.now(timezone.utc) - timedelta(hours=7))
            .isoformat()
            .replace("+00:00", "Z")
            if op is None
            else op.isoformat().replace("+00:00", "Z")
        )
        params = {
            "timeframe": "1Min",
            "start": start,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "iex",
        }
        r = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars",
            params=params,
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            if DEBUG_MODE:
                print(f"[VWAP WARN] {symbol} bars1m status={r.status_code} {r.text}")
            return None
        bars = (r.json() or {}).get("bars") or []
        if not bars:
            return None
        v_sum = 0.0
        pv_sum = 0.0
        orh = 0.0
        last = None
        for b in bars:
            h = float(b.get("h") or 0.0)
            l = float(b.get("l") or 0.0)
            c = float(b.get("c") or 0.0)
            v = float(b.get("v") or 0.0)
            tp = (h + l + c) / 3.0
            v_sum += v
            pv_sum += tp * v
            if h > orh:
                orh = h
            last = c
        vwap = (pv_sum / v_sum) if v_sum > 0 else None
        if vwap is None or last is None:
            return None
        return {
            "vwap": float(vwap),
            "orh": float(orh),
            "last": float(last),
            "n": len(bars),
        }
    except Exception as e:
        if DEBUG_MODE:
            print(f"[VWAP ERROR] {symbol}: {e}")
        return None


# --- Placeholder Functions (To be implemented based on requirements) ---
def _fetch_history(symbol: str, days: int):
    """Placeholder: Fetch historical data for a symbol."""
    try:
        data = yf.download(
            symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=False
        )
        return data, "yfinance"
    except Exception as e:
        if DEBUG_MODE:
            print(f"[FETCH ERROR] {symbol}: {e}")
        return None, None


def get_price_from_alpaca(symbol: str):
    """Placeholder: Get current price from Alpaca."""
    try:
        r = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest",
            params={"feed": "iex"},
            headers=_alpaca_headers(),
            timeout=7,
        )
        if r.status_code == 200:
            return float(r.json().get("quote", {}).get("ap", 0))
        return None
    except Exception:
        return None


def get_today_open_from_alpaca(symbol: str):
    """Placeholder: Get today's open and previous close from Alpaca."""
    try:
        r = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars",
            params={
                "timeframe": "1Day",
                "limit": 2,
                "adjustment": "raw",
                "feed": "iex",
            },
            headers=_alpaca_headers(),
            timeout=7,
        )
        if r.status_code == 200:
            bars = r.json().get("bars", [])
            if len(bars) >= 2:
                return float(bars[-1].get("o", 0)), float(bars[-2].get("c", 0))
        return None, None
    except Exception:
        return None, None


def _today_open_close_utc():
    """Placeholder: Get today's market open and close times in UTC."""
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/calendar",
            params={"start": today, "end": today},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code == 200 and r.json():
            day = r.json()[0]
            open_time = datetime.fromisoformat(
                day["date"] + "T" + day["open"] + ":00-05:00"
            )
            close_time = datetime.fromisoformat(
                day["date"] + "T" + day["close"] + ":00-05:00"
            )
            return open_time, close_time
        return None, None
    except Exception:
        return None, None


def _is_market_open_now():
    """Placeholder: Check if market is currently open."""
    try:
        op, cl = _today_open_close_utc()
        if op is None or cl is None:
            return False
        now = datetime.now(timezone.utc)
        return op <= now <= cl
    except Exception:
        return False


def _minutes_since_open():
    """Placeholder: Calculate minutes since market open."""
    try:
        op, _ = _today_open_close_utc()
        if op is None:
            return None
        return (datetime.now(timezone.utc) - op).total_seconds() / 60.0
    except Exception:
        return None


def _ensure_dirs(snapshot_stamp: str):
    """Placeholder: Ensure snapshot directory exists and return Path."""
    snapshot_dir = SNAPSHOT_ROOT / snapshot_stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _log_decision(
    symbol: str,
    macro_regime: str,
    TH: dict,
    base_info: dict,
    accepted: bool,
    reason: str,
    gpt_reply: str,
    ohlc_source: str,
    live_used: bool,
):
    """Placeholder: Log trading decision."""
    try:
        DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "macro_regime": macro_regime,
            "thresholds": TH,
            "base_info": base_info,
            "accepted": accepted,
            "reason": reason,
            "gpt_reply": gpt_reply,
            "ohlc_source": ohlc_source,
            "live_used": live_used,
        }
        with open(
            DECISIONS_DIR / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] log_decision {symbol}: {e}")


# --- Utility Functions ---
def load_adaptive():
    if os.path.exists(ADAPTIVE_CFG_PATH):
        try:
            with open(ADAPTIVE_CFG_PATH, "r") as f:
                cfg = json.load(f)
                if "thresholds" in cfg:
                    return cfg
        except Exception as e:
            if DEBUG_MODE:
                print(f"[WARN] bad adaptive_config: {e}")
    return {"thresholds": dict(BASE), "last_count": None, "last_update": None}


def clamp(x, lo, hi):
    return x if DISABLE_LIMITS else max(lo, min(hi, x))


def _pct(a, b):
    try:
        return 100.0 * (float(a) - float(b)) / (float(b) if float(b) != 0 else 1e-9)
    except Exception:
        return 0.0


def _safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def get_macro_score():
    if not MACRO_GUARD_ENABLED:
        return 0.0, {"enabled": False}
    details = {"enabled": True}
    try:
        spy = yf.download(
            "SPY", period="3mo", interval="1d", progress=False, auto_adjust=False
        )
        vix = yf.download(
            "^VIX", period="3mo", interval="1d", progress=False, auto_adjust=False
        )
        tnx = yf.download(
            "^TNX", period="3mo", interval="1d", progress=False, auto_adjust=False
        )
        uup = yf.download(
            "UUP", period="3mo", interval="1d", progress=False, auto_adjust=False
        )
        hyg = yf.download(
            "HYG", period="3mo", interval="1d", progress=False, auto_adjust=False
        )
        if spy.empty or vix.empty:
            return 0.0, {"enabled": True, "error": "empty SPY/VIX"}
        spy_close = safe_squeeze(spy["Close"]).dropna()
        ema20 = EMAIndicator(spy_close, window=20).ema_indicator()
        ema50 = EMAIndicator(spy_close, window=50).ema_indicator()
        s_above20 = float(spy_close.iloc[-1] > ema20.iloc[-1])
        s_above50 = float(spy_close.iloc[-1] > ema50.iloc[-1])
        slope20 = float(ema20.iloc[-1] - ema20.iloc[max(0, len(ema20) - 6)])
        dayret = float(_pct(spy_close.iloc[-1], spy_close.iloc[-2]))
        spy_score = _clamp_unit(
            0.25 * (1 if s_above50 else -1)
            + 0.20 * (1 if s_above20 else -1)
            + 0.15 * (1 if slope20 > 0 else -1)
            + 0.10 * (1 if dayret >= 0 else -1)
        )
        vix_close = float(safe_squeeze(vix["Close"]).dropna().iloc[-1])
        vix_score = _clamp_unit((18.0 - vix_close) / 10.0)
        tnx_score = 0.0
        if not tnx.empty:
            tclose = safe_squeeze(tnx["Close"]).dropna()
            tema50 = EMAIndicator(tclose, window=50).ema_indicator()
            tnx_score += -0.10 if tclose.iloc[-1] > tema50.iloc[-1] else 0.05
            tchg = _pct(tclose.iloc[-1], tclose.iloc[-2])
            tnx_score += -0.10 if tchg > 1.0 else (0.05 if tchg < -1.0 else 0.0)
            tnx_score = _clamp_unit(tnx_score)
        uup_score = 0.0
        if not uup.empty:
            uclose = safe_squeeze(uup["Close"]).dropna()
            uema20 = EMAIndicator(uclose, window=20).ema_indicator()
            uup_score += -0.10 if uclose.iloc[-1] > uema20.iloc[-1] else 0.05
            uchg5 = _pct(uclose.iloc[-1], uclose.iloc[max(0, len(uclose) - 6)])
            uup_score += -0.05 if uchg5 > 0 else 0.02
            uup_score = _clamp_unit(uup_score)
        hyg_score = 0.0
        if not hyg.empty:
            hclose = safe_squeeze(hyg["Close"]).dropna()
            hema50 = EMAIndicator(hclose, window=50).ema_indicator()
            hyg_score += 0.15 if hclose.iloc[-1] > hema50.iloc[-1] else -0.15
            hchg5 = _pct(hclose.iloc[-1], hclose.iloc[max(0, len(hclose) - 6)])
            hyg_score += 0.05 if hchg5 > 0 else -0.05
            hyg_score = _clamp_unit(hyg_score)
        macro = _clamp_unit(
            0.35 * spy_score
            + 0.35 * vix_score
            + 0.15 * hyg_score
            + 0.10 * tnx_score
            + 0.05 * uup_score
        )
        details.update(
            {
                "spy": {"score": round(spy_score, 3), "dayret": round(dayret, 2)},
                "vix": {"score": round(vix_score, 3), "last": round(vix_close, 2)},
                "tnx": {"score": round(tnx_score, 3)},
                "uup": {"score": round(uup_score, 3)},
                "hyg": {"score": round(hyg_score, 3)},
                "macro": round(macro, 3),
            }
        )
        return macro, details
    except Exception as e:
        if DEBUG_MODE:
            print(f"[MACRO ERROR] {e}")
        return 0.0, {"enabled": True, "error": str(e)}


def apply_macro_regime(TH, macro_score):
    global E_TARGET_MIN, E_TARGET_MAX, E_HYBRID_MAX_OVERRIDES
    E_TARGET_MIN, E_TARGET_MAX = TARGET_MIN, TARGET_MAX
    E_HYBRID_MAX_OVERRIDES = HYBRID_MAX_OVERRIDES
    regime = "neutral"
    block = False
    th = dict(TH)

    def add(name, delta):
        if name not in th:
            return
        lo, hi = BOUNDS[name]
        th[name] = clamp(th[name] + delta, lo, hi)

    if macro_score <= -0.6:
        regime = "panic"
        block = True
        E_HYBRID_MAX_OVERRIDES = 0
        E_TARGET_MIN, E_TARGET_MAX = 0, 0
        add("MODEL_SCORE_MIN", +8.0)
        add("MIN_GAP_UP", +0.30)
        add("MIN_BULLISH_BODY", +0.30)
        add("MAX_ATR_PCT", -1.0)
    elif macro_score <= -0.2:
        regime = "risk_off"
        E_HYBRID_MAX_OVERRIDES = 0
        E_TARGET_MIN, E_TARGET_MAX = 2, 3
        add("MODEL_SCORE_MIN", +3.0)
        add("MIN_GAP_UP", +0.10)
        add("MIN_BULLISH_BODY", +0.10)
        add("MAX_ATR_PCT", -0.5)
    elif macro_score >= 0.2:
        regime = "risk_on"
        E_HYBRID_MAX_OVERRIDES = min(3, HYBRID_MAX_OVERRIDES)
        E_TARGET_MIN, E_TARGET_MAX = 3, 4
        add("MODEL_SCORE_MIN", -2.0)
        add("MIN_GAP_UP", -0.05)
        add("MIN_BULLISH_BODY", -0.05)
        add("MAX_ATR_PCT", +0.5)
    else:
        regime = "neutral"
        E_HYBRID_MAX_OVERRIDES = min(1, HYBRID_MAX_OVERRIDES)
        E_TARGET_MIN, E_TARGET_MAX = TARGET_MIN, TARGET_MAX

    return th, regime, block


def adjust_thresholds(cfg, count):
    th = cfg["thresholds"].copy()
    direction = -1 if count < E_TARGET_MIN else (+1 if count > E_TARGET_MAX else 0)
    mid = (E_TARGET_MIN + E_TARGET_MAX) / 2.0
    k = clamp(1.0 + 0.5 * abs(count - mid), 1.0, 3.0)

    def move(name, sign_for_tight):
        cur = th[name]
        base = BASE[name]
        lo, hi = BOUNDS[name]
        step = STEPS[name] * (k if direction != 0 else 1.0)
        if direction == -1:
            th[name] = clamp(cur - sign_for_tight * step, lo, hi)
        elif direction == +1:
            th[name] = clamp(cur + sign_for_tight * step, lo, hi)
        else:
            th[name] = clamp(cur + (base - cur) * MEAN_REVERT, lo, hi)

    move("MAX_ATR_PCT", -1)
    move("MAX_VOLATILITY", -1)
    move("MIN_VOLUME_TREND", +1)
    move("MIN_RISK_SCORE", +1)
    move("MIN_BULLISH_BODY", +1)
    move("MIN_GAP_UP", +1)
    move("MIN_VOLUME_RATIO", +1)
    move("MODEL_SCORE_MIN", +1)

    cfg["thresholds"] = th
    cfg["last_count"] = count
    cfg["last_update"] = datetime.now().isoformat()
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_CFG_PATH), exist_ok=True)
        with open(ADAPTIVE_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] save adaptive_config: {e}")
    return direction, th


def _smart_model_score(alpha_score, volume_ratio, rsi, atr_pct, volatility):
    sa = float(max(0.0, min(1.0, alpha_score)))
    sv = float(max(0.0, min(1.0, volume_ratio / 2.0)))
    sr = float(max(0.0, min(1.0, rsi / 100.0)))
    satr = float(max(0.0, 1.0 - min(1.0, atr_pct / 8.0)))
    svol = float(max(0.0, 1.0 - min(1.0, volatility / 8.0)))
    return float(
        (0.35 * sa + 0.25 * sv + 0.15 * sr + 0.15 * satr + 0.10 * svol) * 100.0
    )


def _write_rejected_csv(rejected_dict):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(REJECTED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "reason"])
            for sym, reason in rejected_dict.items():
                w.writerow([sym, reason])
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] write rejected.csv: {e}")


def is_gpt_yes(text: str) -> bool:
    if not text:
        return False
    t = text.strip().upper()
    if re.match(r"^\s*(НЕТ|NO)\b", t):
        return False
    return bool(re.match(r"^\s*(ДА|YES)\b", t))


def _has_negative_news(symbol: str, hours: int = NEWS_LOOKBACK_HOURS) -> bool:
    if not NEWS_GUARD_ENABLED:
        return False
    try:
        start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        params = {"symbols": symbol, "limit": 50, "start": start}
        r = requests.get(
            f"{ALPACA_NEWS_BASE}/news",
            params=params,
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            if DEBUG_MODE:
                print(f"[NEWS WARN] {symbol}: {r.status_code} {r.text}")
            return False
        payload = r.json()
        items = payload.get("news") if isinstance(payload, dict) else payload
        neg_words = (
            "downgrade",
            "miss",
            "guidance cut",
            "cuts guidance",
            "lowered guidance",
            "profit warning",
            "fraud",
            "probe",
            "investigation",
            "charges",
            "lawsuit",
            "recall",
            "bankruptcy",
            "chapter 11",
            "sec",
            "subpoena",
            "cease and desist",
            "halt",
            "delist",
            "going concern",
            "liquidity issue",
            "warning",
            "restatement",
            "short seller",
            "whistleblower",
            "data breach",
            "hack",
        )
        neg = sum(
            1
            for it in (items or [])
            if any(
                k in f"{it.get('headline', '')} {it.get('summary', '')}".lower()
                for k in neg_words
            )
        )
        return neg >= NEWS_NEG_THRESH
    except Exception as e:
        if DEBUG_MODE:
            print(f"[NEWS ERROR] {symbol}: {e}")
        return False


def _long_upper_wick(open_p, high, low, close_p) -> bool:
    try:
        body = abs(close_p - open_p)
        up_wick = float(high - max(open_p, close_p))
        return (up_wick > body * UPPER_WICK_MULT) and (
            100.0 * (close_p - open_p) / max(1e-9, open_p) < WICK_STRONG_BODY_MAX
        )
    except Exception:
        return False


def _log_macro_and_notify(details, regime, block):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(MACRO_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "regime": regime, **details},
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] macro log: {e}")
    try:
        msg = f"🌐 Macro: {regime.upper()} | score={details.get('macro', '?')}"
        vix = details.get("vix", {}).get("last")
        if vix is not None:
            msg += f" | VIX={vix}"
        send_telegram_message(msg + (" — новые лонги ОТКЛЮЧЕНЫ" if block else ""))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] macro telegram: {e}")


def _load_rs_cache():
    try:
        if not os.path.exists(RS_CACHE_PATH):
            return None
        with open(RS_CACHE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        ts = float(obj.get("_ts", 0))
        if (time.time() - ts) / 60.0 > RS_TTL_MIN:
            return None
        if obj.get("_lookback") != RS_LOOKBACK_D:
            return None
        data = obj.get("data")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_rs_cache(data):
    try:
        os.makedirs(Path(RS_CACHE_PATH).parent, exist_ok=True)
        with open(RS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"_ts": time.time(), "_lookback": RS_LOOKBACK_D, "data": data},
                f,
                ensure_ascii=False,
            )
    except Exception as e:
        if DEBUG_MODE:
            print(f"[RS WARN] cache save: {e}")


def _compute_rs_table(tickers, lookback=63, beta_adj=True):
    import pandas as pd

    unq = sorted(set(t for t in tickers if isinstance(t, str) and t))
    if "SPY" not in unq:
        unq.append("SPY")
    df = yf.download(
        tickers=unq,
        period=f"{max(lookback+10, lookback)}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    closes = {}
    if hasattr(df, "columns") and isinstance(df.columns, pd.MultiIndex):
        for t in unq:
            key = (t, "Close")
            if key in df.columns:
                s = df[key].dropna()
                if not s.empty:
                    closes[t] = s
    elif "Close" in df.columns and len(unq) == 1:
        closes[unq[0]] = df["Close"].dropna()
    spy = closes.get("SPY")
    if spy is None or len(spy) < max(20, lookback // 2):
        return {}
    out = {}
    for t in [x for x in unq if x != "SPY"]:
        st = closes.get(t)
        if st is None:
            continue
        j = pd.concat([st, spy], axis=1, join="inner").tail(lookback + 1)
        if j.shape[0] < max(20, lookback // 2):
            continue
        st2, sp2 = j.iloc[:, 0], j.iloc[:, 1]
        ret_t = float(st2.iloc[-1] / st2.iloc[0] - 1.0)
        ret_sp = float(sp2.iloc[-1] / sp2.iloc[0] - 1.0)
        beta = 1.0
        if beta_adj:
            rt = st2.pct_change().dropna()
            rm = sp2.pct_change().dropna()
            k = pd.concat([rt, rm], axis=1, join="inner").tail(lookback)
            if k.shape[0] >= max(20, lookback // 3):
                x = k.iloc[:, 1].values
                y = k.iloc[:, 0].values
                xm = x - x.mean()
                ym = y - y.mean()
                denom = float((xm * xm).sum())
                if denom > 0:
                    beta = float((xm * ym).sum() / denom)
        rs_raw = float(ret_t - beta * ret_sp)
        slope = None
        try:
            ratio = (st2 / sp2).dropna().tail(lookback)
            if len(ratio) >= max(20, lookback // 3):
                x = np.arange(len(ratio))
                y = np.log(ratio.values + 1e-12)
                slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            slope = None
        out[t] = {"raw": rs_raw, "slope": slope}
    vals = np.array([v["raw"] for v in out.values() if np.isfinite(v["raw"])])
    if vals.size > 0:
        for k, v in out.items():
            r = v["raw"]
            try:
                pctl = float((vals < r).mean() * 100.0)
            except Exception:
                pctl = None
            v["pctl"] = pctl
    return out


def _prepare_rs(tickers, macro_regime):
    RS_MIN = RS_MIN_PCTL_BASE
    if RS_ENABLED:
        if macro_regime == "risk_off":
            RS_MIN = min(100, RS_MIN + 10)
        elif macro_regime == "panic":
            RS_MIN = min(100, RS_MIN + 15)
        elif macro_regime == "risk_on":
            RS_MIN = max(0, RS_MIN - 5)
    rs_table = {}
    if RS_ENABLED:
        rs_table = _load_rs_cache() or {}
        if not rs_table:
            try:
                rs_table = _compute_rs_table(
                    tickers, lookback=RS_LOOKBACK_D, beta_adj=RS_BETA_ADJ
                )
                _save_rs_cache(rs_table)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[RS WARN] compute failed: {e}")
                rs_table = {}
    return rs_table, RS_MIN


# === Main Function ===
def main():
    try:
        # Initial market and time checks
        if not is_market_open_today() and not FORCE_RUN:
            print("⛔ Рынок закрыт сегодня — отбор сигналов пропущен.")
            return
        if datetime.now(timezone.utc).weekday() >= 5 and not FORCE_RUN:
            print("⛔ Выходной (weekend) — отбор сигналов пропущен.")
            return
        if not _within_entry_window() and not FORCE_RUN:
            print("⛔ Вне окна входа (.env) — отбор сигналов пропущен.")
            return

        # Load tickers
        with open(CANDIDATES_PATH, "r") as f:
            raw_tickers = json.load(f)
        tickers = [str(t).upper() for t in raw_tickers]

        # Load adaptive thresholds and macro score
        adaptive_cfg = load_adaptive()
        TH = adaptive_cfg["thresholds"]
        macro, macro_details = get_macro_score()
        TH, macro_regime, macro_block = apply_macro_regime(TH, macro)
        _log_macro_and_notify(
            macro_details if macro_details else {"macro": round(macro, 3)},
            macro_regime,
            macro_block,
        )

        if macro_block and not BYPASS_FILTERS:
            print("🛑 Macro PANIC: новые лонги отключены этим прогоном.")
            for pth, obj in [
                (SIGNALS_PATH, {}),
                (REJECTED_PATH, {"__macro__": "panic block"}),
                (GPT_DECISIONS_PATH, {}),
            ]:
                with open(pth, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
            _write_rejected_csv({"__macro__": "panic block"})
            return

        # Prepare Relative Strength table
        rs_table, RS_MIN_PCTL = _prepare_rs(tickers, macro_regime)

        # Load existing positions and orders
        active_pos = load_active_symbols()
        open_orders = load_open_order_symbols()
        local_open = load_local_open_positions()
        already_held = active_pos | open_orders | local_open

        print(f"📅 Проверяем {len(tickers)} тикеров...")
        print(f"⚙️ Пороги (TH): {json.dumps(TH, ensure_ascii=False)}")
        print(f"🎛️ Цели: {E_TARGET_MIN}-{E_TARGET_MAX} сигналов")
        print(f"🚧 BYPASS_FILTERS={'ON' if BYPASS_FILTERS else 'OFF'}")
        if already_held:
            print(
                f"🔒 Исключены: {len(already_held)} — {', '.join(sorted(list(already_held))[:20])}{'…' if len(already_held) > 20 else ''}"
            )

        signals, rejected, gpt_decisions = {}, {}, {}
        reasons_count = defaultdict(int)
        retry_pool, hybrid_accepts = [], []
        hybrid_used = 0
        count_for_adapt = None
        snapshot_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        snapshot_dir = _ensure_dirs(snapshot_stamp)
        early_window_min = 7.0
        mins_since_open = _minutes_since_open()
        in_early_window = (
            mins_since_open is not None and mins_since_open < early_window_min
        )
        idx = 0
        heartbeat_cnt = 0

        for symbol in tickers:
            try:
                idx += 1
                _t0 = time.perf_counter()
                if (idx % max(1, HEARTBEAT_N)) == 0:
                    heartbeat_cnt += 1
                    print(f"⏱️ Heartbeat: {idx}/{len(tickers)} обработано...")

                # Early rejection checks
                if not is_tradable(symbol):
                    rejected[symbol] = (
                        "Не торгуется в Alpaca (tradable=false/status!=active)"
                    )
                    reasons_count["Не торгуется"] += 1
                    continue
                if symbol in already_held and not BYPASS_FILTERS:
                    rejected[symbol] = "Уже в портфеле/ордерах"
                    reasons_count["Уже в портфеле/ордерах"] += 1
                    continue
                if _has_negative_news(symbol):
                    rejected[symbol] = "Негативный новостной фон (последние ~36ч)"
                    reasons_count["Негативный новостной фон"] += 1
                    continue

                print(f"\n🔎 Обрабатываем {symbol}...")
                data, ohlc_source = _fetch_history(symbol, days=45)
                if data is None or getattr(data, "empty", True):
                    data = yf.download(
                        symbol,
                        period="30d",
                        interval="1d",
                        progress=False,
                        auto_adjust=False,
                    )
                    ohlc_source = ohlc_source or "yfinance"
                if data is None or getattr(data, "empty", True) or data.shape[0] < 2:
                    rejected[symbol] = "Недостаточно данных"
                    reasons_count["Недостаточно данных"] += 1
                    continue

                # Snapshot daily OHLC
                try:
                    (snapshot_dir / f"{symbol}.csv").write_text(
                        data.to_csv(index=True), encoding="utf-8"
                    )
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[WARN] snapshot {symbol}: {e}")

                # Calculate technical indicators
                close = safe_squeeze(data["Close"]).dropna()
                volume = safe_squeeze(data["Volume"]).dropna()
                high = safe_squeeze(data["High"]).dropna()
                low = safe_squeeze(data["Low"]).dropna()
                open_s = safe_squeeze(data["Open"]).dropna()

                if len(close) < 15 or len(volume) < 5:
                    rejected[symbol] = "Недостаточно чистых данных"
                    reasons_count["Недостаточно чистых данных"] += 1
                    continue

                alpaca_price = get_price_from_alpaca(symbol)
                ok_qg, qg_msg = _data_quality_ok(symbol)
                if not ok_qg and not BYPASS_FILTERS:
                    rejected[symbol] = f"QualityGate: {qg_msg}"
                    reasons_count["Pre-trade качество (спред/$vol)"] += 1
                    _log_decision(
                        symbol,
                        macro_regime,
                        TH,
                        {"alpaca_price": alpaca_price},
                        False,
                        rejected[symbol],
                        qg_msg,
                        ohlc_source or "unknown",
                        False,
                    )
                    continue
                if alpaca_price is None or alpaca_price == 0:
                    rejected[symbol] = "Цена Alpaca отсутствует или равна 0"
                    reasons_count["Цена Alpaca отсутствует или равна 0"] += 1
                    continue

                rsi = float(RSIIndicator(close=close).rsi().iloc[-1])
                ema = float(
                    EMAIndicator(close=close, window=10).ema_indicator().iloc[-1]
                )
                percent_change = float(
                    ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
                )
                volume_ratio = float(volume.iloc[-1] / (volume[:-1].mean() + 1e-6))
                ema_deviation = float(((close.iloc[-1] - ema) / ema) * 100)
                alpha_score = float(
                    calculate_alpha_score(
                        percent_change, volume_ratio, rsi, ema_deviation
                    )
                )
                atr_value = float(
                    AverageTrueRange(high=high, low=low, close=close, window=14)
                    .average_true_range()
                    .iloc[-1]
                )
                atr_pct = (
                    float((atr_value / close.iloc[-1]) * 100)
                    if close.iloc[-1] != 0
                    else 0.0
                )
                volatility = float(close.pct_change().std() * 100)
                volume_ema = float(
                    EMAIndicator(volume, window=10).ema_indicator().iloc[-1]
                )
                volume_trend = float(
                    (volume.iloc[-1] / volume_ema) if volume_ema != 0 else 1.0
                )
                risk_score = float(alpha_score * 100)
                prev_close = float(close.iloc[-2])
                today_open_hist = float(open_s.iloc[-1])
                today_close_hist = float(close.iloc[-1])
                bullish_body = (
                    ((today_close_hist - today_open_hist) / today_open_hist) * 100
                    if today_open_hist != 0
                    else 0.0
                )
                gap_up = (
                    ((today_open_hist - prev_close) / prev_close) * 100
                    if prev_close != 0
                    else 0.0
                )

                # Wick handling
                wick_flag = _long_upper_wick(
                    today_open_hist,
                    float(high.iloc[-1]),
                    float(low.iloc[-1]),
                    today_close_hist,
                )
                wick_penalized = False
                vwap_tail = ""
                if WICK_GUARD_ENABLED and wick_flag and not BYPASS_FILTERS:
                    allow_reclaim = False
                    if LIVE_VWAP_CHECK:
                        live = _intraday_metrics(symbol)
                        if live:
                            last = live["last"]
                            vwap = live["vwap"]
                            orh = live["orh"]
                            vwap_tail = f"\nLive VWAP={vwap:.2f}, ORH={orh:.2f}, last={last:.2f}"
                            if (last >= vwap * (1.0 - VWAP_RECLAIM_TOL)) and (
                                last >= orh * (1.0 - ORH_TOUCH_TOL)
                            ):
                                allow_reclaim = True
                    if WICK_MODE == "strict" and not allow_reclaim:
                        rejected[symbol] = "Длинная верхняя тень (strict)"
                        reasons_count["Длинная верхняя тень"] += 1
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            {
                                "alpaca_price": alpaca_price,
                                "bullish_body": bullish_body,
                                "gap_up": gap_up,
                            },
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            False,
                        )
                        continue
                    if not allow_reclaim:
                        penalty = WICK_PENALTY_BASE * (
                            1.0
                            + min(
                                1.0,
                                max(
                                    0.0, atr_pct / max(1e-6, TH.get("MAX_ATR_PCT", 8.0))
                                ),
                            )
                        )
                        penalty = float(min(WICK_PENALTY_MAX, penalty))
                        wick_penalized = True

                model_score = _smart_model_score(
                    alpha_score, volume_ratio, rsi, atr_pct, volatility
                )
                if wick_penalized:
                    model_score = max(0.0, model_score - penalty)

                # Relative Strength check
                rs_pctl = None
                rs_slope = None
                if RS_ENABLED and isinstance(rs_table, dict):
                    _rs = rs_table.get(symbol)
                    if _rs:
                        rs_pctl = (
                            float(_rs.get("pctl", None))
                            if _rs.get("pctl") is not None
                            else None
                        )
                        rs_slope = (
                            float(_rs.get("slope", None))
                            if _rs.get("slope") is not None
                            else None
                        )
                        if rs_pctl is not None and not BYPASS_FILTERS:
                            if rs_pctl < RS_MIN_PCTL:
                                rejected[symbol] = (
                                    f"RS ниже порога ({rs_pctl:.1f}% < {RS_MIN_PCTL:.0f}%)"
                                )
                                reasons_count["RS ниже порога"] += 1
                                continue
                            if RS_SLOPE_REQ and rs_slope is not None and rs_slope <= 0:
                                rejected[symbol] = "RS тренд отрицательный"
                                reasons_count["RS тренд отрицательный"] += 1
                                continue
                        if rs_pctl is not None:
                            model_score = float(
                                min(100.0, model_score + RS_WEIGHT * float(rs_pctl))
                            )

                # Short-Squeeze features
                day_vol = float(volume.iloc[-1]) if volume.iloc[-1] is not None else 0.0
                squeeze = get_squeeze_features(
                    symbol,
                    day_volume=day_vol,
                    use_nasdaq=True,
                    use_yahoo=True,
                    allow_stale_cache=False,
                )
                sq_score = float(squeeze.get("squeeze_score", 0.0)) if squeeze else 0.0
                sq_si = squeeze.get("si_pct", 0.0) if squeeze else 0.0
                sq_dtc = squeeze.get("dtc", 0.0) if squeeze else 0.0
                sq_float = squeeze.get("float_m", 0.0) if squeeze else 0.0
                sq_fee = squeeze.get("fee_pct", 0.0) if squeeze else 0.0
                sq_util = squeeze.get("util_pct", 0.0) if squeeze else 0.0
                sq_ssr = bool(squeeze.get("ssr_flag", False)) if squeeze else False
                sq_long_risk = (
                    bool(squeeze.get("is_squeeze_long_risk", False))
                    if squeeze
                    else False
                )
                sq_short_op = (
                    bool(squeeze.get("is_squeeze_short_opportunity", False))
                    if squeeze
                    else False
                )

                SQUEEZE_WEIGHT = float(os.getenv("ELIOS_SQUEEZE_WEIGHT", "0.12"))
                model_score = float(
                    min(
                        100.0,
                        max(
                            0.0,
                            model_score
                            + min(max(SQUEEZE_WEIGHT, 0.0), 0.25) * sq_score,
                        ),
                    )
                )

                if (
                    sq_long_risk
                    and not os.getenv("ELIOS_ALLOW_SQUEEZE_LONGS", "0") == "1"
                    and not BYPASS_FILTERS
                ):
                    rejected[symbol] = "Squeeze long risk (высокие SI/DTC/Fee)"
                    reasons_count["Squeeze long risk"] += 1
                    continue

                base_info = {
                    "symbol": symbol,
                    "alpaca_price": round(alpaca_price, 2),
                    "percent_change": percent_change,
                    "rsi": rsi,
                    "ema_dev": ema_deviation,
                    "volume_ratio": volume_ratio,
                    "alpha_score": alpha_score,
                    "model_score": model_score,
                    "atr_value": atr_value,
                    "atr_pct": atr_pct,
                    "volatility": volatility,
                    "volume_trend": volume_trend,
                    "bullish_body": bullish_body,
                    "gap_up": gap_up,
                }

                # Live validation
                summary_live_tail = ""
                live_used = False
                if (time.perf_counter() - _t0) > TICKER_BUDGET_S:
                    rejected[symbol] = f"watchdog>={TICKER_BUDGET_S:.1f}s@live"
                    reasons_count["Watchdog (live)"] += 1
                    _log_decision(
                        symbol,
                        macro_regime,
                        TH,
                        base_info,
                        False,
                        rejected[symbol],
                        "",
                        ohlc_source or "unknown",
                        False,
                    )
                    continue
                if _is_market_open_now():
                    open_today_live, prev_close_alp = get_today_open_from_alpaca(symbol)
                    if open_today_live and open_today_live > 0:
                        prev_for_gap = (
                            prev_close_alp
                            if prev_close_alp and prev_close_alp > 0
                            else prev_close
                        )
                        bullish_body_live = (
                            ((alpaca_price - open_today_live) / open_today_live) * 100
                            if open_today_live != 0
                            else 0.0
                        )
                        gap_up_live = (
                            ((open_today_live - prev_for_gap) / prev_for_gap) * 100
                            if prev_for_gap != 0
                            else 0.0
                        )
                        TOL = 0.05
                        if not BYPASS_FILTERS and not in_early_window:
                            if bullish_body_live <= 0:
                                rejected[symbol] = (
                                    f"Live красная свеча (body={bullish_body_live:.2f}%)"
                                )
                                reasons_count["Live красная свеча"] += 1
                                _log_decision(
                                    symbol,
                                    macro_regime,
                                    TH,
                                    base_info,
                                    False,
                                    rejected[symbol],
                                    "",
                                    ohlc_source or "unknown",
                                    True,
                                )
                                continue
                            if bullish_body_live < (TH["MIN_BULLISH_BODY"] - TOL):
                                rejected[symbol] = (
                                    f"Live свеча слабая (body={bullish_body_live:.2f}%)"
                                )
                                reasons_count["Live свеча слабая"] += 1
                                _log_decision(
                                    symbol,
                                    macro_regime,
                                    TH,
                                    base_info,
                                    False,
                                    rejected[symbol],
                                    "",
                                    ohlc_source or "unknown",
                                    True,
                                )
                                continue
                            if gap_up_live < (TH["MIN_GAP_UP"] - TOL):
                                rejected[symbol] = (
                                    f"Live gap недостаточен (gap={gap_up_live:.2f}%)"
                                )
                                reasons_count["Live gap недостаточен"] += 1
                                _log_decision(
                                    symbol,
                                    macro_regime,
                                    TH,
                                    base_info,
                                    False,
                                    rejected[symbol],
                                    "",
                                    ohlc_source or "unknown",
                                    True,
                                )
                                continue
                        summary_live_tail = f"\nLive: body={bullish_body_live:.2f}% | gap={gap_up_live:.2f}%"
                        live_used = True

                # Standard filters
                if not BYPASS_FILTERS:
                    if bullish_body <= 0:
                        rejected[symbol] = f"Красная свеча (body={bullish_body:.2f}%)"
                        reasons_count["Красная свеча"] += 1
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if bullish_body < TH["MIN_BULLISH_BODY"]:
                        rejected[symbol] = (
                            f"Слабая зелёная свеча (body={bullish_body:.2f}%)"
                        )
                        reasons_count["Слабая зелёная свеча"] += 1
                        if (
                            bullish_body
                            >= TH["MIN_BULLISH_BODY"] - STEPS["MIN_BULLISH_BODY"]
                        ):
                            retry_pool.append({**base_info, "reject_reason": "body"})
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if (gap_up < TH["MIN_GAP_UP"]) or (
                        volume_ratio < TH["MIN_VOLUME_RATIO"]
                    ):
                        rejected[symbol] = (
                            f"Нет достаточного Gap/объёма (gap={gap_up:.2f}%, vol_ratio={volume_ratio:.2f}×)"
                        )
                        reasons_count["Gap/объём недостаточны"] += 1
                        if (gap_up >= TH["MIN_GAP_UP"] - STEPS["MIN_GAP_UP"]) and (
                            volume_ratio
                            >= TH["MIN_VOLUME_RATIO"] - STEPS["MIN_VOLUME_RATIO"]
                        ):
                            retry_pool.append({**base_info, "reject_reason": "gap_vol"})
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if atr_pct > TH["MAX_ATR_PCT"]:
                        rejected[symbol] = f"ATR% слишком высокий ({atr_pct:.2f}%)"
                        reasons_count["ATR высокий"] += 1
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if volatility > TH["MAX_VOLATILITY"]:
                        rejected[symbol] = (
                            f"Волатильность слишком высокая ({volatility:.2f}%)"
                        )
                        reasons_count["Волатильность высокая"] += 1
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if volume_trend < TH["MIN_VOLUME_TREND"]:
                        rejected[symbol] = (
                            f"Тренд объёма отрицательный ({volume_trend:.2f}×)"
                        )
                        reasons_count["Тренд объёма отрицательный"] += 1
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if risk_score < TH["MIN_RISK_SCORE"]:
                        rejected[symbol] = f"Низкий риск-скор ({risk_score:.2f})"
                        reasons_count["Низкий риск-скор"] += 1
                        if risk_score >= TH["MIN_RISK_SCORE"] - STEPS["MIN_RISK_SCORE"]:
                            retry_pool.append({**base_info, "reject_reason": "risk"})
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue
                    if model_score < TH["MODEL_SCORE_MIN"]:
                        rejected[symbol] = f"Слабый модельный скор ({model_score:.2f})"
                        reasons_count["Слабый модельный скор"] += 1
                        if (
                            model_score
                            >= TH["MODEL_SCORE_MIN"] - STEPS["MODEL_SCORE_MIN"]
                        ):
                            retry_pool.append({**base_info, "reject_reason": "model"})
                        _log_decision(
                            symbol,
                            macro_regime,
                            TH,
                            base_info,
                            False,
                            rejected[symbol],
                            "",
                            ohlc_source or "unknown",
                            live_used,
                        )
                        continue

                # GPT vote
                is_anomaly, anomaly_reason = detect_anomalies(symbol)
                prompt = (
                    f"Ты торговый ассистент для Искры. Данные по {symbol}:\n"
                    f"Цена: {alpaca_price:.2f}\n∆%: {percent_change:.2f}% | RSI: {rsi:.2f}\n"
                    f"EMA dev: {ema_deviation:.2f}% | VolRatio: {volume_ratio:.2f}×\n"
                    f"Alpha: {alpha_score:.2f} | ModelScore: {model_score:.2f}\n"
                    f"ATR%: {atr_pct:.2f}% | Vol: {volatility:.2f}% | VTrend: {volume_trend:.2f}×\n"
                    f"Свеча: body={bullish_body:.2f}% | GapUp={gap_up:.2f}%\n"
                    f"Short Squeeze: SI={sq_si:.0f}% | DTC={sq_dtc:.1f} | Fee={sq_fee:.0f}% | Score={sq_score:.0f}\n"
                    f"Аномалия объёма: {'ОБНАРУЖЕНА' if is_anomaly else anomaly_reason}\n"
                    f"Войти в сделку? Ответь 'ДА' или 'НЕТ' и кратко почему."
                )
                gpt_reply = "НЕТ (fallback)"
                try:
                    chat = client.chat.completions.create(
                        model=GPT_SIGNAL_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=150,
                    )
                    gpt_reply = chat.choices[0].message.content.strip()
                except Exception as e:
                    gpt_reply = f"НЕТ (ошибка GPT: {e})"
                gpt_decisions[symbol] = gpt_reply

                rs_tail = ""
                if RS_ENABLED and symbol in rs_table:
                    _rsi = rs_table[symbol]
                    rs_tail = f"\nRS={_rsi.get('pctl', 0):.0f}pctl"
                    if rs_slope is not None:
                        rs_tail += "↑" if rs_slope > 0 else "↓"

                squeeze_tail = (
                    f"\nSqueeze: SI {sq_si:.0f}% | DTC {sq_dtc:.1f} | Float {sq_float:.1f}M | "
                    f"Fee {sq_fee:.0f}% | Util {sq_util:.2f}% | Score {sq_score:.0f}"
                )
                if sq_ssr:
                    squeeze_tail += " | SSR"
                if sq_short_op:
                    squeeze_tail += " | short_op"
                if sq_long_risk:
                    squeeze_tail += " | long_risk"

                summary_lines = [
                    "📊 Новый сигнал (BUY)",
                    f"📌 ${symbol} @ {alpaca_price:.2f}",
                    f"∆%={percent_change:.2f}% | RSI={rsi:.2f} | EMA dev={ema_deviation:.2f}%",
                    f"ATR%={atr_pct:.2f} | Vol={volatility:.2f}%",
                    f'🤖 GPT: "{gpt_reply}"{summary_live_tail}{rs_tail}{vwap_tail}{squeeze_tail}',
                ]
                if wick_penalized:
                    summary_lines.append(
                        f"⚠️ wick-penalty: -{penalty:.1f} к model_score"
                    )
                summary_msg = "\n".join(summary_lines)

                accepted = False
                accept_reason = "GPT_OK"
                if BYPASS_FILTERS:
                    accepted = True
                    accept_reason = "NO_FILTERS"
                elif is_gpt_yes(gpt_reply):
                    accepted = True
                    accept_reason = "GPT_OK"
                elif HYBRID_ENABLED and hybrid_used < E_HYBRID_MAX_OVERRIDES:
                    clean_noise = (
                        (
                            atr_pct <= TH["MAX_ATR_PCT"]
                            and volatility <= TH["MAX_VOLATILITY"]
                        )
                        if HYBRID_STRICT_NOISE
                        else True
                    )
                    no_neg_trend = volume_trend >= TH["MIN_VOLUME_TREND"]
                    strong_model_possible = model_score >= (
                        TH["MODEL_SCORE_MIN"] + HYBRID_MODEL_MARGIN
                    )
                    if (
                        strong_model_possible
                        and clean_noise
                        and no_neg_trend
                        and (bullish_body >= TH["MIN_BULLISH_BODY"] + 0.20)
                        and (gap_up >= TH["MIN_GAP_UP"] + 0.10)
                    ):
                        accepted = True
                        accept_reason = "HYBRID_ACCEPT"
                        hybrid_used += 1
                        hybrid_accepts.append(symbol)

                if accepted:
                    signals[symbol] = {
                        "price": round(alpaca_price, 2),
                        "action": "BUY",
                        "confidence": round(alpha_score, 2),
                        "score": round(model_score, 2),
                        "atr": round(atr_value, 2),
                        "atr_pct": round(atr_pct, 2),
                        "volatility": round(volatility, 2),
                        "volume_trend": round(volume_trend, 2),
                        "bullish_body": round(bullish_body, 2),
                        "gap_up": round(gap_up, 2),
                        "reason": accept_reason,
                        "squeeze": {
                            "score": round(sq_score, 2),
                            "si_pct": round(sq_si, 2),
                            "dtc": round(sq_dtc, 3),
                            "float_m": round(sq_float, 2),
                            "fee_pct": round(sq_fee, 2),
                            "util_pct": round(sq_util, 1),
                            "ssr_flag": bool(sq_ssr),
                            "long_risk": bool(sq_long_risk),
                            "short_opportunity": bool(sq_short_op),
                        },
                    }
                    if accept_reason == "HYBRID_ACCEPT":
                        summary_msg += (
                            "\n✅ Принят по гибридной логике (сильный квант, GPT=НЕТ)."
                        )
                    if accept_reason == "NO_FILTERS":
                        summary_msg += "\n🟢 Принят: режим без фильтров."
                    send_telegram_message(summary_msg)
                    _log_decision(
                        symbol,
                        macro_regime,
                        TH,
                        base_info,
                        True,
                        accept_reason,
                        gpt_reply,
                        ohlc_source or "unknown",
                        live_used,
                    )
                else:
                    rejected[symbol] = f"GPT отклонил: {gpt_reply}"
                    reasons_count["GPT отклонил"] += 1
                    send_telegram_message(summary_msg)
                    _log_decision(
                        symbol,
                        macro_regime,
                        TH,
                        base_info,
                        False,
                        rejected[symbol],
                        gpt_reply,
                        ohlc_source or "unknown",
                        live_used,
                    )

            except Exception as e:
                rejected[symbol] = f"Ошибка: {str(e)}"
                reasons_count["Исключение/Ошибка"] += 1
                print(f"❗ Ошибка {symbol}: {e}")

        # Rescue-pass
        rescued = []
        if not BYPASS_FILTERS and RESCUE_ENABLED and len(signals) < E_TARGET_MIN:
            need = E_TARGET_MIN - len(signals)
            soft_TH = dict(TH)
            soft_TH["MIN_BULLISH_BODY"] = clamp(
                soft_TH["MIN_BULLISH_BODY"] - STEPS["MIN_BULLISH_BODY"],
                *BOUNDS["MIN_BULLISH_BODY"],
            )
            soft_TH["MIN_GAP_UP"] = clamp(
                soft_TH["MIN_GAP_UP"] - STEPS["MIN_GAP_UP"], *BOUNDS["MIN_GAP_UP"]
            )
            soft_TH["MIN_VOLUME_RATIO"] = clamp(
                soft_TH["MIN_VOLUME_RATIO"] - STEPS["MIN_VOLUME_RATIO"],
                *BOUNDS["MIN_VOLUME_RATIO"],
            )
            soft_TH["MODEL_SCORE_MIN"] = clamp(
                soft_TH["MODEL_SCORE_MIN"] - STEPS["MODEL_SCORE_MIN"],
                *BOUNDS["MODEL_SCORE_MIN"],
            )
            soft_TH["MIN_RISK_SCORE"] = clamp(
                soft_TH["MIN_RISK_SCORE"] - STEPS["MIN_RISK_SCORE"],
                *BOUNDS["MIN_RISK_SCORE"],
            )

            retry_sorted = sorted(
                retry_pool, key=lambda x: x.get("model_score", 0), reverse=True
            )
            for info in retry_sorted:
                if need <= 0:
                    break
                sym = info["symbol"]
                if sym in signals:
                    continue
                conds = [
                    info["bullish_body"] >= soft_TH["MIN_BULLISH_BODY"],
                    info["gap_up"] >= soft_TH["MIN_GAP_UP"],
                    info["volume_ratio"] >= soft_TH["MIN_VOLUME_RATIO"],
                    info["atr_pct"] <= soft_TH["MAX_ATR_PCT"],
                    info["volatility"] <= soft_TH["MAX_VOLATILITY"],
                    info["volume_trend"] >= soft_TH["MIN_VOLUME_TREND"],
                    (info["alpha_score"] * 100) >= soft_TH["MIN_RISK_SCORE"],
                    info["model_score"] >= soft_TH["MODEL_SCORE_MIN"],
                ]
                if not all(conds):
                    continue
                reply = "НЕТ (fallback)"
                try:
                    chat = client.chat.completions.create(
                        model=GPT_SIGNAL_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Rescue-проверка {sym}. Данные: body={info['bullish_body']:.2f}%, gap={info['gap_up']:.2f}%, vol_ratio={info['volume_ratio']:.2f}×, modelScore={info['model_score']:.1f}, ATR%={info['atr_pct']:.2f}, Vol={info['volatility']:.2f}%.",
                            }
                        ],
                        temperature=0.0,
                        max_tokens=40,
                    )
                    reply = chat.choices[0].message.content.strip()
                except Exception as e:
                    if info["model_score"] >= soft_TH["MODEL_SCORE_MIN"] + 10:
                        reply = "ДА (fallback: model_score высокий)"
                    else:
                        reply = f"НЕТ (ошибка GPT: {e})"
                accepted = False
                reason_lbl = "RESCUE"
                if is_gpt_yes(reply):
                    accepted = True
                elif HYBRID_ENABLED and (hybrid_used < E_HYBRID_MAX_OVERRIDES):
                    strong_model = info["model_score"] >= (
                        soft_TH["MODEL_SCORE_MIN"] + HYBRID_MODEL_MARGIN
                    )
                    clean_noise = (info["atr_pct"] <= soft_TH["MAX_ATR_PCT"]) and (
                        info["volatility"] <= soft_TH["MAX_VOLATILITY"]
                    )
                    no_neg_trend = info["volume_trend"] >= soft_TH["MIN_VOLUME_TREND"]
                    if (
                        strong_model
                        and clean_noise
                        and no_neg_trend
                        and (
                            (info["bullish_body"] >= soft_TH["MIN_BULLISH_BODY"] + 0.10)
                            and (info["gap_up"] >= soft_TH["MIN_GAP_UP"] + 0.05)
                        )
                    ):
                        accepted = True
                        reason_lbl = "HYBRID_ACCEPT"
                        hybrid_used += 1
                        hybrid_accepts.append(sym)
                if accepted:
                    signals[sym] = {
                        "price": round(info["alpaca_price"], 2),
                        "action": "BUY",
                        "confidence": round(info["alpha_score"], 2),
                        "score": round(info["model_score"], 2),
                        "atr": round(info["atr_value"], 2),
                        "atr_pct": round(info["atr_pct"], 2),
                        "volatility": round(info["volatility"], 2),
                        "volume_trend": round(info["volume_trend"], 2),
                        "bullish_body": round(info["bullish_body"], 2),
                        "gap_up": round(info["gap_up"], 2),
                        "reason": reason_lbl,
                    }
                    rescued.append(sym)
                    need -= 1
            if rescued:
                try:
                    send_telegram_message(
                        "🛟 Rescue-pass добрал сигналы: "
                        + ", ".join(f"${s}" for s in rescued)
                    )
                except Exception:
                    if DEBUG_MODE:
                        print("[WARN] telegram send failed")

        # Fallback-pass
        fallbacked = []
        if (
            not BYPASS_FILTERS
            and FALLBACK_ENABLED
            and len(signals) < E_TARGET_MAX
            and retry_pool
        ):
            need_fb = E_TARGET_MAX - len(signals)
            ultra_TH = dict(TH)
            ultra_TH["MIN_BULLISH_BODY"] = clamp(
                ultra_TH["MIN_BULLISH_BODY"] - 2 * STEPS["MIN_BULLISH_BODY"],
                *BOUNDS["MIN_BULLISH_BODY"],
            )
            ultra_TH["MIN_GAP_UP"] = clamp(
                ultra_TH["MIN_GAP_UP"] - 2 * STEPS["MIN_GAP_UP"], *BOUNDS["MIN_GAP_UP"]
            )
            ultra_TH["MIN_VOLUME_RATIO"] = clamp(
                ultra_TH["MIN_VOLUME_RATIO"] - 2 * STEPS["MIN_VOLUME_RATIO"],
                *BOUNDS["MIN_VOLUME_RATIO"],
            )
            ultra_model_req = clamp(
                TH["MODEL_SCORE_MIN"] + 5.0, *BOUNDS["MODEL_SCORE_MIN"]
            )
            retry_sorted = sorted(
                retry_pool,
                key=lambda x: (x.get("model_score", 0), x.get("volume_ratio", 0)),
                reverse=True,
            )
            for info in retry_sorted:
                if need_fb <= 0:
                    break
                sym = info["symbol"]
                if sym in signals:
                    continue
                conds = [
                    info["bullish_body"] >= ultra_TH["MIN_BULLISH_BODY"],
                    info["gap_up"] >= ultra_TH["MIN_GAP_UP"],
                    info["volume_ratio"] >= ultra_TH["MIN_VOLUME_RATIO"],
                    info["atr_pct"] <= TH["MAX_ATR_PCT"],
                    info["volatility"] <= TH["MAX_VOLATILITY"],
                    info["volume_trend"] >= TH["MIN_VOLUME_TREND"],
                    info["model_score"] >= ultra_model_req,
                ]
                if not all(conds):
                    continue
                reply = "ДА (light)"
                try:
                    chat = client.chat.completions.create(
                        model=GPT_SIGNAL_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": "Fallback-проверка. Дай ДА/НЕТ. Если явных красных флагов нет — отвечай ДА.",
                            }
                        ],
                        temperature=0.0,
                        max_tokens=10,
                    )
                    reply = chat.choices[0].message.content.strip()
                except Exception:
                    reply = "ДА (fallback no-GPT)"
                accepted = False
                reason_lbl = "FALLBACK"
                if is_gpt_yes(reply):
                    accepted = True
                elif HYBRID_ENABLED and (hybrid_used < E_HYBRID_MAX_OVERRIDES):
                    accepted = True
                    reason_lbl = "HYBRID_ACCEPT"
                    hybrid_used += 1
                    hybrid_accepts.append(sym)
                if accepted:
                    signals[sym] = {
                        "price": round(info["alpaca_price"], 2),
                        "action": "BUY",
                        "confidence": round(info["alpha_score"], 2),
                        "score": round(info["model_score"], 2),
                        "atr": round(info["atr_value"], 2),
                        "atr_pct": round(info["atr_pct"], 2),
                        "volatility": round(info["volatility"], 2),
                        "volume_trend": round(info["volume_trend"], 2),
                        "bullish_body": round(info["bullish_body"], 2),
                        "gap_up": round(info["gap_up"], 2),
                        "reason": reason_lbl,
                    }
                    fallbacked.append(sym)
                    need_fb -= 1
            if fallbacked:
                try:
                    send_telegram_message(
                        "🧩 Fallback-pass добрал до цели: "
                        + ", ".join(f"${s}" for s in fallbacked)
                    )
                except Exception:
                    if DEBUG_MODE:
                        print("[WARN] telegram send failed")

        # Tiebreaker + Quality Gate
        if QUALITY_GATE_ENABLED and len(signals) > E_TARGET_MAX:

            def _cap(v, lo=None, hi=None):
                x = _safe(v, 0.0)
                if lo is not None:
                    x = max(lo, x)
                if hi is not None:
                    x = min(hi, x)
                return x

            def _rank(sig: dict) -> float:
                model = _cap(sig.get("score", 0.0) / 100.0, 0.0, 1.0)
                conf = _cap(sig.get("confidence", 0.0), 0.0, 1.0)
                gap = _cap(sig.get("gap_up", 0.0), 0.0, 0.20)
                body = _cap(sig.get("bullish_body", 0.0), -0.20, 0.20)
                atrp = _cap(sig.get("atr_pct", 0.0), 0.0, 10.0)
                vola = _cap(sig.get("volatility", 0.0), 0.0, 8.0)
                vtrend = _cap(sig.get("volume_trend", 1.0), 0.0, 2.0)
                return float(
                    0.24 * model
                    + 0.22 * gap
                    + 0.18 * body
                    + 0.12 * conf
                    + 0.08 * vtrend
                    - 0.10 * atrp
                    - 0.10 * vola
                )

            original_items = list(signals.items())
            ranked = sorted(original_items, key=lambda kv: _rank(kv[1]), reverse=True)
            kept = ranked[:E_TARGET_MAX]
            dropped = ranked[E_TARGET_MAX:]

            if len(kept) < min(E_TARGET_MIN, len(original_items)):
                kept = original_items[
                    : max(E_TARGET_MIN, min(len(original_items), E_TARGET_MAX))
                ]

            def _passes_quality(sym, info):
                rank_val = _rank(info)
                need_gap = max(TH.get("MIN_GAP_UP", 0.0), QUALITY_MIN_GAP_PCT)
                need_body = max(TH.get("MIN_BULLISH_BODY", 0.0), QUALITY_MIN_BODY_PCT)
                need_model = TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS
                if info.get("score", 0.0) < need_model:
                    return False, f"model<{need_model:.1f}", rank_val
                if info.get("gap_up", 0.0) < need_gap:
                    return False, f"gap<{need_gap:.2f}%", rank_val
                if info.get("bullish_body", 0.0) < need_body:
                    return False, f"body<{need_body:.2f}%", rank_val
                if rank_val <= QUALITY_MIN_RANK:
                    return False, f"rank≤{QUALITY_MIN_RANK:.2f}", rank_val
                return True, "OK", rank_val

            final_kept, failed = [], []
            if QUALITY_GATE_ENABLED:
                for sym, info in kept:
                    ok, reason, r = _passes_quality(sym, info)
                    if ok:
                        final_kept.append((sym, info, r))
                    else:
                        failed.append((sym, info, reason, r))
                if len(final_kept) < E_TARGET_MIN:
                    pool = [
                        (sym, info, _rank(info))
                        for sym, info in kept
                        if sym not in [s for s, *_ in final_kept]
                    ]
                    pool = sorted(pool, key=lambda t: t[2], reverse=True)
                    for sym, info, r in pool:
                        if len(final_kept) >= E_TARGET_MIN:
                            break
                        final_kept.append((sym, info, r))
                final_kept = [(sym, info, _rank(info)) for sym, info in kept]

            signals = {sym: info for sym, info, _ in final_kept}
            try:
                lines = [
                    f"✂️ Тай-брейкер: оставил TOP {E_TARGET_MAX} из {len(original_items)} сигналов"
                ]
                for i, (sym, info, r) in enumerate(
                    sorted(final_kept, key=lambda t: t[2], reverse=True), 1
                ):
                    lines.append(
                        f"  {i}. {sym}  score={r:.3f} (ms={info.get('score',0):.1f}, gap={info.get('gap_up',0):.2f}%, body={info.get('bullish_body',0):.2f}%, atr%={info.get('atr_pct',0):.2f}, vol%={info.get('volatility',0):.2f})"
                    )
                if dropped:
                    lines.append(
                        "  └─ Отсечены ранжированием: "
                        + ", ".join(sym for sym, _ in dropped)
                    )
                if QUALITY_GATE_ENABLED:
                    if failed:
                        lines.append(
                            "🧰 Quality gate: отклонены в TOP из-за порогов → "
                            + ", ".join(
                                f"{sym}({reason})" for sym, _, reason, _ in failed
                            )
                        )
                    lines.append(
                        f"🧰 Quality gate итог: {len(final_kept)}/{len(kept)} в финале (мин. rank>{QUALITY_MIN_RANK:.2f}, gap≥{QUALITY_MIN_GAP_PCT:.2f}%, body≥{QUALITY_MIN_BODY_PCT:.2f}%, model≥TH+{QUALITY_MODEL_BONUS:.1f})"
                    )
                send_telegram_message("\n".join(lines))
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[WARN] tiebreak/quality log: {e}")

        if isinstance(signals, dict) and len(signals) <= E_TARGET_MAX:
            try:
                send_telegram_message(
                    f"✂️ Тай-брейкер: пропущен (сигналов {len(signals)} ≤ цель {E_TARGET_MAX})"
                )
            except Exception:
                if DEBUG_MODE:
                    print("[WARN] telegram send failed")

        with open(SIGNALS_PATH, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)
        try:
            shutil.copy(SIGNALS_PATH, SIGNALS_BACKUP_PATH)
        except Exception as e:
            if DEBUG_MODE:
                print(f"[WARN] backup signals: {e}")

        with open(REJECTED_PATH, "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)
        with open(GPT_DECISIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(gpt_decisions, f, indent=2, ensure_ascii=False)

        os.makedirs("logs", exist_ok=True)
        with open(SIGNAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "symbol": s,
                        "timestamp": datetime.now().isoformat(),
                        "score": d.get("score", 0),
                        "confidence": d.get("confidence", 0),
                        "reason": d.get("reason", "GPT_OK"),
                    }
                    for s, d in signals.items()
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

        _write_rejected_csv(rejected)

        print("\n📊 Сводка причин отказов:", flush=True)
        for k in sorted(reasons_count, key=reasons_count.get, reverse=True):
            print(f"  • {k}: {reasons_count[k]}")
        print(f"\n📦 Сигналы: {len(signals)} → {SIGNALS_PATH}")
        print(f"🚫 Отклонено: {len(rejected)} → {REJECTED_PATH}")
        print(f"🧠 GPT ответы: {GPT_DECISIONS_PATH}")
        print(
            f"🟡 Hybrid-accepts: {len([k for k in signals.values() if k.get('reason')=='HYBRID_ACCEPT'])} → {', '.join([s for s,d in signals.items() if d.get('reason')=='HYBRID_ACCEPT']) or '—'}"
        )

        if not BYPASS_FILTERS:
            adaptive_cfg = load_adaptive()
            adaptive_count = (
                count_for_adapt if count_for_adapt is not None else len(signals)
            )
            direction, new_th = adjust_thresholds(adaptive_cfg, adaptive_count)
            mode = (
                "ослабил"
                if direction == -1
                else ("ужесточил" if direction == +1 else "возврат к базовым")
            )
            change_msg = (
                f"🛠 Авто-адаптация фильтров ({mode})\n"
                f"Сигналов (до TOP-{E_TARGET_MAX}/доборов): {adaptive_count} | итог: {len(signals)} (цель {E_TARGET_MIN}-{E_TARGET_MAX})\n"
                f"Новые пороги: {json.dumps(new_th, ensure_ascii=False)}"
            )
            try:
                send_telegram_message(change_msg)
            except Exception:
                if DEBUG_MODE:
                    print("[WARN] telegram send failed")
        else:
            try:
                send_telegram_message(
                    f"🟢 Режим без фильтров активен. Принято сигналов: {len(signals)}."
                )
            except Exception:
                if DEBUG_MODE:
                    print("[WARN] telegram send failed")

    except Exception as e:
        print(f"❗ Критическая ошибка в main(): {e}")
        send_telegram_message(f"❗ Ошибка в signal_engine: {e}")


if __name__ == "__main__":
    main()
