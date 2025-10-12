from core.utils.alpaca_headers import alpaca_headers
# -*- coding: utf-8 -*-
import sys


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


if "/root/stockbot" not in sys.path:
    sys.path.insert(0, "/root/stockbot")
import os, json, shutil, re, csv, hashlib, time
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict


# --- .env autoload (simple) ---
def _load_env_from_files(paths=("/root/stockbot/.env", ".env")):
    import os

    for fp in paths:
        try:
            p = Path(fp)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ or not os.environ.get(k)):
                    os.environ[k] = v
        except Exception:
            pass


from pathlib import Path
from core.trading.signals.providers import get_default_providers
from core.trading.signals.config import SignalsCfg
from core.trading.signals.filters import (
    rs_filter,
    vwap_orh_reclaim,
    tradable_guard,
)
from core.trading.signals.features import (
    intraday_metrics as f_intraday_metrics,
    long_upper_wick as f_long_upper_wick,
    get_today_open_close_utc as f_today_open_close_utc,
    smart_model_score,
    compute_daily_features,
)
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from openai import OpenAI

# === Runtime flags ===
OFFLINE_MODE = (
    os.getenv("ELIOS_OFFLINE", "0") == "1"
)  # жёсткий офлайн: без сети, только локальные CSV
NO_GPT_MODE = os.getenv("ELIOS_NO_GPT", "0") == "1"  # не вызывать GPT вовсе
FAST_MACRO = os.getenv("ELIOS_MACRO_FAST", "0") == "1"  # макро: нейтраль/быстрый путь


# === Safe yf.download with timeout (thread wrapper) ===
def yf_download_safe(
    *,
    tickers,
    period="3mo",
    interval="1d",
    auto_adjust=False,
    progress=False,
    group_by="ticker",
    threads=True,
    timeout_sec=15,
):
    """
    Возвращает DataFrame или None по таймауту/ошибке.
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeout

        def _call():
            try:
                return yf.download(
                    tickers=tickers,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=progress,
                    group_by=group_by,
                    threads=threads,
                )
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            return fut.result(timeout=timeout_sec)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[YF SAFE ERROR] tickers={tickers}: {e}")
        return None


# --- Hard timeout runner (process-based) ---
def run_with_timeout(
    fn, *, seconds=30, default=None, args=(), kwargs=None, name="task"
):
    import threading, queue

    q = queue.Queue(maxsize=1)

    def _worker():
        try:
            res = fn(*args, **(kwargs or {}))
            q.put(("ok", res))
        except Exception as e:
            q.put(("err", f"{type(e).__name__}: {e}"))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return (False, default, "timeout")
    if q.empty():
        return (False, default, "no-result")
    status, payload = q.get()
    return (
        status == "ok",
        (payload if status == "ok" else default),
        (None if status == "ok" else payload),
    )


# --- Market calendar (soft import)
try:
    from core.utils.market_calendar import is_market_open_today
except Exception:
    # Фоллбэк по будням, попытка спросить Alpaca при вызове
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
            pass
        return datetime.now(timezone.utc).weekday() < 5


from core.trading.anomaly_detector import detect_anomalies
from core.trading.alpha_utils import calculate_alpha_score
from core.utils.telegram import send_telegram_message

# ensure .env
_load_env_from_files()
# === Alpaca ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
ALPACA_NEWS_BASE = os.getenv("ALPACA_NEWS_BASE", "https://data.alpaca.markets/v1beta1")


def _alpaca_headers():
    return alpaca_headers()


# === OpenAI ===
client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY",""),
    )
)
GPT_SIGNAL_MODEL = os.getenv("ELIOS_SIGNAL_GPT_MODEL", "gpt-4o-mini")
# === IO paths ===
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
# --- Providers (Signals v2) ---
CFG_SIGNALS = SignalsCfg.from_env()
PROVIDERS = get_default_providers(CFG_SIGNALS)
# === Modes / Flags ===
DEBUG_MODE = os.getenv("ELIOS_DEBUG", "0") == "1"
RESCUE_ENABLED = True
FALLBACK_ENABLED = True
HYBRID_ENABLED = True
NEWS_GUARD_ENABLED = os.getenv("ELIOS_NEWS_GUARD", "1") == "1"
WICK_GUARD_ENABLED = os.getenv("ELIOS_WICK_GUARD", "1") == "1"
MACRO_GUARD_ENABLED = os.getenv("ELIOS_MACRO_GUARD", "1") == "1"
# --- Wick guard controls ---
WICK_MODE = os.getenv("ELIOS_WICK_MODE", "penalty").strip().lower()  # penalty|strict
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
# --- Per-ticker hard time budget (sec)
TICKER_BUDGET_SEC = int(os.getenv("ELIOS_TICKER_BUDGET_SEC", "30"))
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
QUALITY_GATE_ENABLED = os.getenv("ELIOS_QUALITY_GATE", "1") == "1"
QUALITY_MIN_GAP_PCT = float(os.getenv("ELIOS_QUALITY_MIN_GAP", "1.0"))
QUALITY_MIN_BODY_PCT = float(os.getenv("ELIOS_QUALITY_MIN_BODY", "1.0"))
QUALITY_MIN_RANK = float(os.getenv("ELIOS_QUALITY_MIN_RANK", "-0.02"))
QUALITY_MODEL_BONUS = float(os.getenv("ELIOS_QUALITY_MODEL_BONUS", "3.0"))


# --- helper: строгая проверка качества для гибрид-приёма ---
def _passes_hybrid_quality(
    TH: dict, bullish_body: float, gap_up: float, model_score: float
) -> bool:
    need_gap = max(float(TH.get("MIN_GAP_UP", 0.0)), float(QUALITY_MIN_GAP_PCT))
    need_body = max(float(TH.get("MIN_BULLISH_BODY", 0.0)), float(QUALITY_MIN_BODY_PCT))
    need_model = float(TH.get("MODEL_SCORE_MIN", 0.0)) + float(QUALITY_MODEL_BONUS)
    return (
        (model_score >= (need_model + HYBRID_MODEL_MARGIN))
        and (bullish_body >= need_body + 0.10)
        and (gap_up >= need_gap + 0.05)
    )


PRE_QUALITY_GUARD_ENABLED = os.getenv("ELIOS_PRE_QUALITY_GUARD", "1") == "1"
# === Relative Strength (RS) ===
RS_CACHE_PATH = "logs/cache/rs.json"
RS_ENABLED = os.getenv("ELIOS_RS_ENABLED", "1") == "1"
RS_LOOKBACK_D = int(os.getenv("ELIOS_RS_LOOKBACK_D", "63"))
RS_MIN_PCTL_BASE = int(os.getenv("ELIOS_RS_MIN_PCTL", "65"))
RS_BETA_ADJ = os.getenv("ELIOS_RS_BETA_ADJ", "1") == "1"
RS_SLOPE_REQ = os.getenv("ELIOS_RS_SLOPE_REQ", "0") == "1"
RS_WEIGHT = float(os.getenv("ELIOS_RS_WEIGHT", "0.10"))
RS_TTL_MIN = int(os.getenv("ELIOS_RS_TTL_MIN", "720"))
# === NEW: Squeeze features import (safe) ===
try:
    from core.trading.squeeze_features import get_squeeze_features
except Exception:

    def get_squeeze_features(symbol: str, **kwargs):
        return None


def safe_squeeze(series):
    try:
        if hasattr(series, "ndim") and series.ndim > 1:
            return series.squeeze()
    except Exception:
        pass
    return series


# --- Reliable Wilder RSI (no-ta) ---
def _wilder_rsi(series, period: int = 14) -> float:
    import pandas as pd

    try:
        s = pd.Series(series).astype(float).dropna()
        if s.shape[0] < max(20, period + 1):
            return 50.0
        delta = s.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0, 1e-9))
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1])
        import numpy as np

        if not np.isfinite(val):
            return 50.0
        return max(0.0, min(100.0, val))
    except Exception:
        return 50.0


# --- HTTP Session w/ retries
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


# --- Alpaca helpers ---
def is_tradable(symbol: str) -> bool:
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/assets/{symbol}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return True  # не блокируем, если API недоступно
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
        if isinstance(data, dict):
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
                if isinstance(v, dict):
                    sym = (k or "").upper()
                    qty = (
                        v.get("qty")
                        or v.get("quantity")
                        or v.get("position_size")
                        or v.get("entry_price")
                    )
                    if qty:
                        syms.append(sym)
            return set(syms)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] load_local_open_positions: {e}")
    return set()


def _minutes_since_open():
    try:
        rc = requests.get(
            f"{ALPACA_BASE_URL}/v2/clock", headers=_alpaca_headers(), timeout=10
        )
        rc.raise_for_status()
        ts_str = (rc.json() or {}).get("timestamp")
        if not ts_str:
            return None
        now_utc = (
            datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts_str.endswith("Z")
            else datetime.fromisoformat(ts_str).astimezone(timezone.utc)
        )
        today_iso = now_utc.date().isoformat()
        rcal = requests.get(
            f"{ALPACA_BASE_URL}/v2/calendar",
            params={"start": today_iso, "end": today_iso},
            headers=_alpaca_headers(),
            timeout=10,
        )
        rcal.raise_for_status()
        days = rcal.json() or []
        if not days:
            return None
        op_raw = days[0].get("open")
        open_utc = _parse_calendar_dt(op_raw, now_utc)
        if not open_utc:
            return None
        return (now_utc - open_utc).total_seconds() / 60.0
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] minutes_since_open: {e}")
        return None


def _parse_calendar_dt(val, now_utc):
    """Парсит Alpaca calendar open/close: ISO / 'HH:MM' -> UTC datetime"""
    try:
        if not val:
            return None
        s = str(val)
        if "T" in s or s.endswith("Z") or "+" in s or "-" in s[10:]:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except Exception:
                pass
        if re.match(r"^\d{2}:\d{2}", s):
            from zoneinfo import ZoneInfo

            h, m = map(int, s.split(":")[:2])
            dt_ny = datetime(
                now_utc.year,
                now_utc.month,
                now_utc.day,
                h,
                m,
                tzinfo=ZoneInfo("America/New_York"),
            )
            return dt_ny.astimezone(timezone.utc)
    except Exception:
        return None
    return None


def _today_open_close_utc():
    """(open_utc, close_utc) из Alpaca через feature-слой"""
    try:
        return f_today_open_close_utc(PROVIDERS)
    except Exception:
        return None, None


def _ensure_dirs(snapshot_stamp: str | None = None) -> Path:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    if snapshot_stamp:
        d = SNAPSHOT_ROOT / snapshot_stamp / "ohlc"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return SNAPSHOT_ROOT


def _log_decision(
    symbol, regime, TH, features, accepted, reason, gpt_reply, ohlc_source, live_used
):
    try:
        DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
        date_key = datetime.now().date().isoformat()
        digest_src = json.dumps(
            {
                k: (
                    round(features.get(k, 0), 6)
                    if isinstance(features.get(k), (int, float))
                    else features.get(k)
                )
                for k in sorted(features.keys())
            },
            ensure_ascii=False,
        ).encode("utf-8")
        features_digest = hashlib.sha256(digest_src).hexdigest()[:16]
        record = {
            "ts": datetime.now().isoformat(),
            "symbol": symbol,
            "regime": regime,
            "accepted": bool(accepted),
            "reason": reason,
            "gpt_reply": gpt_reply,
            "ohlc_source": ohlc_source,
            "live_used": bool(live_used),
            "TH_snapshot": TH,
            "features_digest": features_digest,
        }
        with open(DECISIONS_DIR / f"{date_key}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] decision log: {e}")


def _fetch_history(symbol: str, days: int = 30):
    if OFFLINE_MODE:
        # Только локальная история, без Yahoo/Alpaca
        try:
            df, src = PROVIDERS.local.history_daily(symbol)
            if df is not None:
                import pandas as pd

                ok = True
                try:
                    n = int(getattr(df, "shape", [0])[0])
                    ok = ok and (n >= max(30, int(days)))
                    core = df[["Open", "High", "Low", "Close"]].astype(float)
                    flat = float(core.std(numeric_only=True).sum()) == 0.0
                    ok = ok and (not flat)
                except Exception:
                    ok = False
                if ok:
                    if DEBUG_MODE:
                        print(f"[HIST-OFFLINE] {symbol}: use LOCAL rows={len(df)}")
                    return df, src
        except Exception as e:
            if DEBUG_MODE:
                print(f"[WARN] Local history OFFLINE {symbol}: {e}")
        return None, None
    """
    Daily OHLC history: LocalCSV -> Yahoo -> Alpaca
    Returns (DataFrame, source) or (None, None)
    """
    # 0) Local CSV — офлайн история
    try:
        df, src = PROVIDERS.local.history_daily(symbol)
        if df is not None:
            import pandas as pd, numpy as np

            ok = True
            try:
                n = int(getattr(df, "shape", [0])[0])
                ok = ok and (n >= max(30, int(days)))
                core = df[["Open", "High", "Low", "Close"]].astype(float)
                flat = float(core.std(numeric_only=True).sum()) == 0.0
                ok = ok and (not flat)
            except Exception:
                ok = False
            if ok:
                if DEBUG_MODE:
                    print(f"[HIST] {symbol}: use LOCAL rows={len(df)}")
                return df, src
            else:
                if DEBUG_MODE:
                    rows = getattr(df, "shape", [0])[0]
                    try:
                        flat_flag = (
                            "yes"
                            if (
                                "core" in locals()
                                and float(core.std(numeric_only=True).sum()) == 0.0
                            )
                            else "?"
                        )
                    except Exception:
                        flat_flag = "?"
                    print(
                        f"[HIST] {symbol}: skip LOCAL (rows={rows}, flat={flat_flag})"
                    )
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] Local history {symbol}: {e}")
    # 1) Yahoo first
    try:
        df, src = PROVIDERS.yahoo.history_daily(symbol, period_days=max(5, int(days)))
        if df is not None:
            try:
                if DEBUG_MODE:
                    print(f"[HIST] {symbol}: use YAHOO rows={len(df)}")
            except Exception:
                pass
            return df, src
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] Yahoo history {symbol}: {e}")
    # 2) Alpaca fallback
    try:
        from datetime import datetime, timezone, timedelta

        start = (
            datetime.now(timezone.utc) - timedelta(days=max(40, int(days * 2)))
        ).date().isoformat() + "T00:00:00Z"
        end = datetime.now(timezone.utc).date().isoformat() + "T00:00:00Z"
        bars = PROVIDERS.alpaca.bars(
            symbol, timeframe="1Day", start=start, end=end, limit=10000
        )
        if not bars:
            return None, None
        Open = [float(b.get("o", 0) or 0) for b in bars]
        High = [float(b.get("h", 0) or 0) for b in bars]
        Low = [float(b.get("l", 0) or 0) for b in bars]
        Close = [float(b.get("c", 0) or 0) for b in bars]
        Vol = [float(b.get("v", 0) or 0) for b in bars]
        import pandas as pd

        idx = pd.to_datetime([b.get("t") for b in bars], utc=True)
        df = pd.DataFrame(
            {"Open": Open, "High": High, "Low": Low, "Close": Close, "Volume": Vol},
            index=idx,
        )
        df = df[df[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)]
        if df is None or df.empty or df.shape[0] < 2:
            return None, None
        try:
            if DEBUG_MODE:
                print(f"[HIST] {symbol}: use ALPACA rows={len(df)}")
        except Exception:
            pass
        return df, "alpaca_iex"
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] Alpaca daily history {symbol}: {e}")
    return None, None


def get_price_from_alpaca(symbol):
    try:
        rq = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest",
            params={"feed": "iex"},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rq.status_code == 200:
            q = rq.json().get("quote", {}) or {}
            ap = q.get("ap") or 0
            bp = q.get("bp") or 0
            if ap > 0 or bp > 0:
                return ap if ap > 0 else bp
        rt = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest",
            params={"feed": "iex"},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rt.status_code == 200:
            p = (rt.json().get("trade", {}) or {}).get("p") or 0
            if p > 0:
                return p
        rb = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars/latest",
            params={"feed": "iex"},
            headers=_alpaca_headers(),
            timeout=10,
        )
        if rb.status_code == 200:
            c = (rb.json().get("bar", {}) or {}).get("c") or 0
            if c > 0:
                return c
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] price {symbol}: {e}")
    # yfinance fallback
    try:
        hist = yf_download_safe(
            tickers=symbol,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=10,
        )
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] yf price {symbol}: {e}")
    # LocalCSV fallback — последний Close
    try:
        prov = get_default_providers(SignalsCfg.from_env())
        if hasattr(prov, "local"):
            df, src = prov.local.history_daily(symbol)  # type: ignore
            if df is not None and getattr(df, "empty", False) is False:
                return float(df["Close"].iloc[-1])
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] local price {symbol}: {e}")
    return 0.0


def _is_market_open_now():
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/clock", headers=_alpaca_headers(), timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            return bool(data.get("is_open", False))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] clock: {e}")
    return False


# --- Intraday VWAP / ORH (1m) ---
def _intraday_metrics(symbol: str):
    """VWAP/ORH/last через feature-слой"""
    try:
        return f_intraday_metrics(PROVIDERS, symbol)
    except Exception:
        return None


# --- Adaptive thresholds ---
def get_today_open_from_alpaca(symbol: str):
    """
    Возвращает (today_open_live, prev_close) из Alpaca (feed=iex) или (None, None) при ошибке.
    today_open_live берём как open первой 1-мин свечи сегодняшнего дня, prev_close — close из последней дневной свечи до сегодня.
    """
    try:
        # Часы/дата по /v2/clock
        rc = requests.get(
            f"{ALPACA_BASE_URL}/v2/clock", headers=_alpaca_headers(), timeout=10
        )
        if rc.status_code != 200:
            return None, None
        now_iso = (rc.json() or {}).get("timestamp")
        if not now_iso:
            return None, None
        now_utc = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
        start = now_utc.date().isoformat() + "T00:00:00Z"
        end = now_utc.date().isoformat() + "T23:59:59Z"
        # 1) Минутные бары за сегодня — берём открытие первой свечи
        r1 = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars",
            params={
                "timeframe": "1Min",
                "start": start,
                "end": end,
                "limit": 5000,
                "feed": "iex",
            },
            headers=_alpaca_headers(),
            timeout=10,
        )
        open_today = None
        if r1.status_code == 200:
            arr = (r1.json() or {}).get("bars") or []
            if arr:
                open_today = float(arr[0].get("o") or 0.0) or None
        # 2) Дневные бары — берём предыдущий close
        from datetime import timedelta

        start_d = (now_utc - timedelta(days=15)).date().isoformat() + "T00:00:00Z"
        r2 = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars",
            params={
                "timeframe": "1Day",
                "start": start_d,
                "end": start,
                "limit": 30,
                "feed": "iex",
            },
            headers=_alpaca_headers(),
            timeout=10,
        )
        prev_close = None
        if r2.status_code == 200:
            days = (r2.json() or {}).get("bars") or []
            if days:
                prev_close = float(days[-1].get("c") or 0.0) or None
        return open_today, prev_close
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] get_today_open_from_alpaca {symbol}: {e}")
        return None, None


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


# --- Macro Guard ---
def _clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def _pct(a, b):
    try:
        return 100.0 * (a - b) / (b if b != 0 else 1e-9)
    except Exception:
        return 0.0


def get_macro_score():
    # Fast/Offline path
    if OFFLINE_MODE or FAST_MACRO:
        return 0.0, {"enabled": MACRO_GUARD_ENABLED, "fast": True, "macro": 0.0}
    if not MACRO_GUARD_ENABLED:
        return 0.0, {"enabled": False}
    details = {"enabled": True}
    try:
        spy = yf_download_safe(
            tickers="SPY",
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=15,
        )
        vix = yf_download_safe(
            tickers="^VIX",
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=15,
        )
        tnx = yf_download_safe(
            tickers="^TNX",
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=15,
        )
        uup = yf_download_safe(
            tickers="UUP",
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=15,
        )
        hyg = yf_download_safe(
            tickers="HYG",
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout_sec=15,
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
        with open(ADAPTIVE_CFG_PATH, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] save adaptive_config: {e}")
    return direction, th


def _reasons_top3(reasons_count: dict):
    try:
        items = sorted(reasons_count.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in items[:3]]
    except Exception:
        return []


def adjust_thresholds_v2(cfg: dict, count: int, reasons_count: dict, macro_regime: str):
    """
    Смарт-адаптация: смягчаем/ужесточаем только узкие места.
    - Гистерезис: меняем, только если ушли за TARGET_MIN/MAX.
    - Cooldown: не чаще, чем раз в 20 минут.
    - Приоритеты по причинам отказов.
    """
    from datetime import datetime, timezone
    import time, math, json, os

    # базовые
    th = dict(cfg.get("thresholds") or {})
    if not th:
        th = dict(BASE)
    # cooldown 20 минут
    last_ts = cfg.get("last_update")
    if last_ts:
        try:
            last = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
            dtm = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
            if dtm < 20:
                # слишком рано — только мягкий возврат к базовым
                for k in th:
                    base = BASE.get(k, th[k])
                    th[k] = clamp(
                        th[k] + (base - th[k]) * MEAN_REVERT,
                        *BOUNDS.get(k, (th[k], th[k])),
                    )
                cfg["thresholds"] = th
                return 0, th
        except Exception:
            pass
    # Определяем направление (как раньше)
    direction = -1 if count < E_TARGET_MIN else (+1 if count > E_TARGET_MAX else 0)
    # Если в цели — чуть возвращаем к базовым и выходим
    if direction == 0:
        for k in th:
            base = BASE.get(k, th[k])
            th[k] = clamp(
                th[k] + (base - th[k]) * MEAN_REVERT, *BOUNDS.get(k, (th[k], th[k]))
            )
        cfg["thresholds"] = th
        cfg["last_count"] = count
        cfg["last_update"] = datetime.now().isoformat()
        try:
            os.makedirs(os.path.dirname(ADAPTIVE_CFG_PATH), exist_ok=True)
            with open(ADAPTIVE_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return 0, th
    # Усилим/ослабим адресно на основе топ-причин
    top = _reasons_top3(reasons_count or {})
    # Базовый множитель шага: чем дальше от цели, тем сильнее шаг
    mid = (E_TARGET_MIN + E_TARGET_MAX) / 2.0
    k = clamp(1.0 + 0.5 * abs(count - mid), 1.0, 3.0)
    # === RS_MIN_PCTL adaptive knob ===
    try:
        rs_min = int(cfg.get("rs_min_pctl", RS_MIN_PCTL_BASE))
    except Exception:
        rs_min = RS_MIN_PCTL_BASE
    # если недобор — немного ослабляем RS-порог; если избыток — ужесточаем
    if direction == -1:
        if (reasons_count or {}).get("RS ниже порога", 0) > 0:
            rs_min = max(35, rs_min - 5)
    elif direction == +1:
        rs_min = min(90, rs_min + 5)
    # микро-коррекция от макро-режима (паника не даёт слишком снизить)
    if macro_regime == "panic":
        rs_min = max(rs_min, RS_MIN_PCTL_BASE)
    cfg["rs_min_pctl"] = int(rs_min)

    def move(name, sign_for_tight, mult=1.0):
        lo, hi = BOUNDS[name]
        step = STEPS[name] * k * mult
        cur = th.get(name, BASE[name])
        if direction == -1:  # недобор -> смягчаем
            th[name] = clamp(cur - sign_for_tight * step, lo, hi)
        elif direction == +1:  # избыток -> ужесточаем
            th[name] = clamp(cur + sign_for_tight * step, lo, hi)

    # Карта приоритетов: какие причины какими порогами лечить
    # Недобор сигналов: смягчаем соответствующие пороги
    # Избыток сигналов: ужесточаем те же
    priorities = {
        "RS ниже порога": [
            ("MODEL_SCORE_MIN", +1, 1.2)
        ],  # повысим/понизим требование к модели как прокси RS (RS_MIN_PCTL — через ENV)
        "Красная свеча": [("MIN_BULLISH_BODY", +1, 1.0)],
        "Слабая зелёная свеча": [("MIN_BULLISH_BODY", +1, 1.2)],
        "Gap/объём недостаточны": [
            ("MIN_GAP_UP", +1, 1.0),
            ("MIN_VOLUME_RATIO", +1, 1.0),
        ],
        "ATR высокий": [("MAX_ATR_PCT", -1, 1.0)],
        "Волатильность высокая": [("MAX_VOLATILITY", -1, 1.0)],
        "Тренд объёма отрицательный": [("MIN_VOLUME_TREND", +1, 1.0)],
        "Низкий риск-скор": [("MIN_RISK_SCORE", +1, 1.0)],
        "Слабый модельный скор": [("MODEL_SCORE_MIN", +1, 1.3)],
        "Pre-quality gate": [
            ("MIN_GAP_UP", +1, 1.0),
            ("MIN_BULLISH_BODY", +1, 1.0),
            ("MODEL_SCORE_MIN", +1, 1.0),
        ],
        "Live красная свеча": [("MIN_BULLISH_BODY", +1, 1.0)],
        "Live свеча слабая": [("MIN_BULLISH_BODY", +1, 1.0)],
        "Live gap недостаточен": [("MIN_GAP_UP", +1, 1.0)],
        "Негативный новостной фон": [],  # не трогаем пороги, это внешний фактор
        "Squeeze long risk": [],  # управляется флагом ELIOS_ALLOW_SQUEEZE_LONGS
    }
    # Применим приоритетные сдвиги
    applied = False
    for key in top:
        for name, sign_for_tight, mult in priorities.get(key, []):
            if name in th:
                move(name, sign_for_tight, mult)
                applied = True
    # Если топ-причины не распознаны — fallback на равномерную схему из v1
    if not applied:
        # переносим логику из adjust_thresholds()
        def _fallback_move(name, sign_for_tight):
            lo, hi = BOUNDS[name]
            step = STEPS[name] * (k if direction != 0 else 1.0)
            cur = th[name]
            base = BASE[name]
            if direction == -1:
                th[name] = clamp(cur - sign_for_tight * step, lo, hi)
            elif direction == +1:
                th[name] = clamp(cur + sign_for_tight * step, lo, hi)
            else:
                th[name] = clamp(cur + (base - cur) * MEAN_REVERT, lo, hi)

        _fallback_move("MAX_ATR_PCT", -1)
        _fallback_move("MAX_VOLATILITY", -1)
        _fallback_move("MIN_VOLUME_TREND", +1)
        _fallback_move("MIN_RISK_SCORE", +1)
        _fallback_move("MIN_BULLISH_BODY", +1)
        _fallback_move("MIN_GAP_UP", +1)
        _fallback_move("MIN_VOLUME_RATIO", +1)
        _fallback_move("MODEL_SCORE_MIN", +1)
    # Небольшая коррекция по macro-режиму
    if macro_regime == "panic":
        # не разжимаем слишком сильно в панике
        th["MODEL_SCORE_MIN"] = clamp(
            th["MODEL_SCORE_MIN"] + 2.0, *BOUNDS["MODEL_SCORE_MIN"]
        )
        th["MIN_BULLISH_BODY"] = clamp(
            th["MIN_BULLISH_BODY"] + 0.10, *BOUNDS["MIN_BULLISH_BODY"]
        )
        th["MIN_GAP_UP"] = clamp(th["MIN_GAP_UP"] + 0.10, *BOUNDS["MIN_GAP_UP"])
    cfg["thresholds"] = th
    cfg["last_count"] = count
    cfg["last_update"] = datetime.now().isoformat()
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_CFG_PATH), exist_ok=True)
        with open(ADAPTIVE_CFG_PATH, "w", encoding="utf-8") as f:
            import json

            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return direction, th


# --- scoring helpers ---
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
        neg = 0
        for it in items or []:
            text = f"{it.get('headline','')} {it.get('summary','')}".lower()
            if any(k in text for k in neg_words):
                neg += 1
        return neg >= NEWS_NEG_THRESH
    except Exception as e:
        if DEBUG_MODE:
            print(f"[NEWS ERROR] {symbol}: {e}")
        return False


def _long_upper_wick(open_p, high, low, close_p) -> bool:
    try:
        return f_long_upper_wick(
            float(open_p),
            float(high),
            float(low),
            float(close_p),
            mult=UPPER_WICK_MULT,
            strong_body_max=WICK_STRONG_BODY_MAX,
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
        msg = f"🌐 Macro: {regime.upper()} | score={details.get('macro','?')}"
        vix = details.get("vix", {}).get("last")
        if vix is not None:
            msg += f" | VIX={vix}"
        send_telegram_message(msg + (" — новые лонги ОТКЛЮЧЕНЫ" if block else ""))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[WARN] macro telegram: {e}")


# --- RS cache helpers ---
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

    unq = sorted(set([t for t in tickers if isinstance(t, str) and t]))
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
    if hasattr(df, "columns"):
        if isinstance(df.columns, pd.MultiIndex):
            for t in unq:
                key = (t, "Close")
                if key in df.columns:
                    s = df[key].dropna()
                    if not s.empty:
                        closes[t] = s
        else:
            if "Close" in df.columns and len(unq) == 1:
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
    return out  # dict: {symbol: {"raw":..., "slope":..., "pctl":...}}


def _prepare_rs(tickers, macro_regime, rs_min_override=None):
    RS_MIN = int(rs_min_override) if (rs_min_override is not None) else RS_MIN_PCTL_BASE
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


# === main ===
def main():
    if not is_market_open_today() and not FORCE_RUN:
        print("⛔ Рынок закрыт сегодня — отбор сигналов пропущен.")
        return
    if datetime.now(timezone.utc).weekday() >= 5 and not FORCE_RUN:
        print("⛔ Выходной (weekend) — отбор сигналов пропущен.")
        return
    with open(CANDIDATES_PATH, "r") as f:
        raw_tickers = json.load(f)
    tickers = [str(t).upper() for t in raw_tickers]
    adaptive_cfg = load_adaptive()
    TH = adaptive_cfg["thresholds"]
    print("[BOOT] get_macro_score…", flush=True)
    macro, macro_details = get_macro_score()
    print("[BOOT] macro done", flush=True)
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
    rs_table, RS_MIN_PCTL = _prepare_rs(
        tickers, macro_regime, adaptive_cfg.get("rs_min_pctl")
    )
    active_pos = load_active_symbols()
    open_orders = load_open_order_symbols()
    local_open = load_local_open_positions()
    already_held = set([s for s in (active_pos | open_orders | local_open) if s])
    print(f"📅 Проверяем {len(tickers)} тикеров...")
    print(f"⚙️ Пороги (TH): {json.dumps(TH, ensure_ascii=False)}")
    # print(f"💪 RS_MIN_PCTL={RS_MIN_PCTL} (adaptive)")
    print(f"🎛️ Цели: {E_TARGET_MIN}-{E_TARGET_MAX} сигналов")
    print(f"🚧 BYPASS_FILTERS={'ON' if BYPASS_FILTERS else 'OFF'}")
    if already_held:
        print(
            f"🔒 Исключены: {len(already_held)} — {', '.join(sorted(list(already_held))[:20])}{'…' if len(already_held)>20 else ''}"
        )
    signals, rejected, gpt_decisions = {}, {}, {}
    features_map = {}
    reasons_count = defaultdict(int)
    retry_pool, hybrid_accepts = [], []
    hybrid_used = 0
    count_for_adapt = None
    snapshot_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    snapshot_dir = _ensure_dirs(snapshot_stamp)
    early_window_min = 7.0
    mins_since_open = _minutes_since_open()
    in_early_window = (mins_since_open is not None) and (
        mins_since_open < early_window_min
    )
    for symbol in tickers:
        t0 = time.monotonic()

        def _budget_exceeded():
            return (time.monotonic() - t0) > TICKER_BUDGET_SEC

        # Ранний отсев неторгуемых через Alpaca (с таймаутом)
        try:
            ok, trad, _ = run_with_timeout(
                is_tradable, seconds=5, args=(symbol,), name="tradable"
            )
            if ok and not trad:
                rejected[symbol] = (
                    "Не торгуется в Alpaca (tradable=false/status!=active)"
                )
                reasons_count["Не торгуется"] += 1
                continue
        except Exception:
            pass
        if (symbol in already_held) and not BYPASS_FILTERS:
            rejected[symbol] = "Уже в портфеле/ордерах"
            reasons_count["Уже в портфеле/ордерах"] += 1
            continue
        # Негативные новости — с жёстким таймаутом
        ok, neg, _ = run_with_timeout(
            _has_negative_news, seconds=5, args=(symbol,), name="news"
        )
        if ok and neg:
            rejected[symbol] = "Негативный новостной фон (последние ~36ч)"
            reasons_count["Негативный новостной фон"] += 1
            continue
        print(f"\n🔎 Обрабатываем {symbol}...", flush=True)
        try:
            # defaults to avoid UnboundLocal in dry/no-filters paths
            rsi = 50.0
            percent_change = 0.0
            volume_ratio = 1.0
            ema_deviation = 0.0
            alpha_score = 0.0
            atr_value = 0.0
            atr_pct = 0.0
            volatility = 0.0
            volume_ema = 0.0
            volume_trend = 1.0
            risk_score = 0.0
            prev_close = 0.0
            today_open_hist = 0.0
            today_close_hist = 0.0
            bullish_body = 0.0
            gap_up = 0.0
            data, ohlc_source = _fetch_history(symbol, days=45)
            if (data is None) or getattr(data, "empty", True):
                # пытаемся добрать историю через yfinance в рамках бюджета на тикер
                if _budget_exceeded():
                    rejected[symbol] = (
                        f"Превышен лимит времени на тикер ({TICKER_BUDGET_SEC}s) — history"
                    )
                    reasons_count["Лимит времени на тикер"] += 1
                    continue
                data = yf_download_safe(
                    tickers=symbol,
                    period="30d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                    timeout_sec=20,
                )
                ohlc_source = ohlc_source or "yfinance"
            if (data is None) or (getattr(data, "empty", True)) or (data.shape[0] < 2):
                rejected[symbol] = "Недостаточно данных"
                reasons_count["Недостаточно данных"] += 1
                continue
            # snapshot daily OHLC
            try:
                (snapshot_dir / f"{symbol}.csv").write_text(
                    data.to_csv(index=True), encoding="utf-8"
                )
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[WARN] snapshot {symbol}: {e}")
            close = safe_squeeze(data["Close"]).dropna()
            volume = safe_squeeze(data["Volume"]).dropna()
            high = safe_squeeze(data["High"]).dropna()
            low = safe_squeeze(data["Low"]).dropna()
            open_s = safe_squeeze(data["Open"]).dropna()
            # compute reliable RSI now (Wilder)
            rsi_fixed = _wilder_rsi(close, 14)
            if len(close) < 5 or len(volume) < 5:
                rejected[symbol] = "Недостаточно чистых данных"
                reasons_count["Недостаточно чистых данных"] += 1
                continue
            alpaca_price = get_price_from_alpaca(symbol)
            if alpaca_price is None or alpaca_price == 0:
                rejected[symbol] = "Цена Alpaca отсутствует или равна 0"
                reasons_count["Цена Alpaca отсутствует или равна 0"] += 1
                continue
            # === DAILY FEATURES ONCE ===
            feats = compute_daily_features(data)
            # FORCE: подменяем RSI на Wilder, чтобы исключить '50.00' по умолчанию
            try:
                rsi = float(_wilder_rsi(close, 14))
                feats["rsi"] = rsi
            except Exception:
                rsi = float(feats.get("rsi", 50.0))
            if DEBUG_MODE:
                print(f"[RSI] {symbol}: Wilder={rsi:.2f}  feats_rsi={feats.get('rsi')}")
            # force RSI from Wilder implementation to avoid 50.00 bug
            try:
                import numpy as np

                if np.isfinite(rsi_fixed):
                    feats["rsi"] = float(rsi_fixed)
            except Exception:
                pass
            features_map[symbol] = feats
            # unpack feats (с дефолтами)
            rsi = float(feats.get("rsi", rsi))
            percent_change = float(feats.get("percent_change", percent_change))
            volume_ratio = float(feats.get("volume_ratio", volume_ratio))
            ema_deviation = float(feats.get("ema_deviation", ema_deviation))
            alpha_score = float(feats.get("alpha_score", alpha_score))
            atr_value = float(feats.get("atr_value", atr_value))
            atr_pct = float(feats.get("atr_pct", atr_pct))
            volatility = float(feats.get("volatility", volatility))
            volume_ema = float(feats.get("volume_ema", volume_ema))
            volume_trend = float(feats.get("volume_trend", volume_trend))
            risk_score = float(feats.get("risk_score", risk_score))
            prev_close = float(feats.get("prev_close", prev_close))
            today_open_hist = float(feats.get("today_open_hist", today_open_hist))
            today_close_hist = float(feats.get("today_close_hist", today_close_hist))
            bullish_body = float(feats.get("bullish_body", bullish_body))
            gap_up = float(feats.get("gap_up", gap_up))
            # --- NEW: Wick handling (penalty + VWAP/ORH live check) ---
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
                    ok, live, _ = run_with_timeout(
                        _intraday_metrics, seconds=10, args=(symbol,), name="intraday"
                    )
                    if ok and live:
                        last = live["last"]
                        vwap = live["vwap"]
                        orh = live["orh"]
                        vwap_tail = (
                            f"\nLive VWAP={vwap:.2f}, ORH={orh:.2f}, last={last:.2f}"
                        )
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
                            max(0.0, atr_pct / max(1e-6, TH.get("MAX_ATR_PCT", 8.0))),
                        )
                    )
                    penalty = float(min(WICK_PENALTY_MAX, penalty))
                    wick_penalized = True
            model_score = _smart_model_score(
                alpha_score, volume_ratio, rsi, atr_pct, volatility
            )
            if wick_penalized:
                model_score = max(0.0, model_score - penalty)
            # --- RS bonus / filter ---
            rs_pctl = None
            rs_slope = None
            if RS_ENABLED and isinstance(rs_table, dict):
                _rs = rs_table.get(symbol)
                if _rs:
                    try:
                        rs_pctl = float(_rs.get("pctl"))
                    except Exception:
                        rs_pctl = None
                    try:
                        rs_slope = float(_rs.get("slope"))
                    except Exception:
                        rs_slope = None
                    if (rs_pctl is not None) and not BYPASS_FILTERS:
                        if rs_pctl < RS_MIN_PCTL:
                            rejected[symbol] = (
                                f"RS ниже порога ({rs_pctl:.1f}% < {RS_MIN_PCTL:.0f}%)"
                            )
                            reasons_count["RS ниже порога"] += 1
                            continue
                        if RS_SLOPE_REQ and (rs_slope is not None) and (rs_slope <= 0):
                            rejected[symbol] = "RS тренд отрицательный"
                            reasons_count["RS тренд отрицательный"] += 1
                            continue
                    if rs_pctl is not None:
                        try:
                            model_score = float(
                                min(100.0, model_score + RS_WEIGHT * float(rs_pctl))
                            )
                        except Exception:
                            pass
            # --- Short-Squeeze features ---
            try:
                day_vol = float(volume.iloc[-1])
            except Exception:
                day_vol = 0.0
            squeeze = None
            try:
                squeeze = get_squeeze_features(
                    symbol,
                    day_volume=day_vol,
                    use_nasdaq=True,
                    use_yahoo=True,
                    allow_stale_cache=False,
                )
            except Exception:
                squeeze = None
            if _budget_exceeded():
                rejected[symbol] = (
                    f"Превышен лимит времени на тикер ({TICKER_BUDGET_SEC}s) — squeeze"
                )
                reasons_count["Лимит времени на тикер"] += 1
                continue
            sq_score = float(squeeze.get("squeeze_score", 0.0)) if squeeze else 0.0
            sq_si = squeeze.get("si_pct", 0.0) if squeeze else 0.0
            sq_dtc = squeeze.get("dtc", 0.0) if squeeze else 0.0
            sq_float = squeeze.get("float_m", 0.0) if squeeze else 0.0
            sq_fee = squeeze.get("fee_pct", 0.0) if squeeze else 0.0
            sq_util = squeeze.get("util_pct", 0.0) if squeeze else 0.0
            sq_ssr = bool(squeeze.get("ssr_flag", False)) if squeeze else False
            sq_long_risk = (
                bool(squeeze.get("is_squeeze_long_risk", False)) if squeeze else False
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
                        model_score + min(max(SQUEEZE_WEIGHT, 0.0), 0.25) * sq_score,
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
            # --- enforce feats before base_info
            try:
                _ms = _smart_model_score(
                    alpha_score, volume_ratio, rsi, atr_pct, volatility
                )
                if wick_penalized:
                    model_score = max(0.0, _ms - float(penalty))
                else:
                    model_score = float(_ms)
            except Exception as _e:
                if DEBUG_MODE:
                    print(f"[FEATS ENFORCE WARN] {symbol}: {_e}")
            base_info = {
                "symbol": symbol,
                "alpaca_price": float(alpaca_price),
                "percent_change": float(feats.get("percent_change", percent_change)),
                "rsi": float(feats.get("rsi", 50.0)),
                "ema_dev": float(feats.get("ema_deviation", 0.0)),
                "volume_ratio": float(feats.get("volume_ratio", 1.0)),
                "alpha_score": float(feats.get("alpha_score", alpha_score)),
                "model_score": float(model_score),
                "atr_value": float(feats.get("atr_value", 0.0)),
                "atr_pct": float(feats.get("atr_pct", 0.0)),
                "volatility": float(feats.get("volatility", 0.0)),
                "volume_trend": float(feats.get("volume_trend", 1.0)),
                "bullish_body": float(feats.get("bullish_body", 0.0)),
                "gap_up": float(feats.get("gap_up", 0.0)),
            }
            # --- Live validation window
            summary_live_tail = ""
            live_used = False
            if _is_market_open_now():
                open_today_live, prev_close_alp = get_today_open_from_alpaca(symbol)
                if open_today_live and open_today_live > 0:
                    prev_for_gap = (
                        prev_close_alp
                        if (prev_close_alp and prev_close_alp > 0)
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
                    if risk_score >= TH["MIN_RISK_SCORE"] - 2.0:
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
                    if model_score >= TH["MODEL_SCORE_MIN"] - 5.0:
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
            # Pre-quality gate (до GPT): отсекаем серые и слабые сетапы заранее
            if PRE_QUALITY_GUARD_ENABLED and not BYPASS_FILTERS:
                need_gap = max(TH.get("MIN_GAP_UP", 0.0), QUALITY_MIN_GAP_PCT)
                need_body = max(TH.get("MIN_BULLISH_BODY", 0.0), QUALITY_MIN_BODY_PCT)
                need_model = TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS
                if (
                    (gap_up < need_gap)
                    or (bullish_body < need_body)
                    or (model_score < need_model)
                ):
                    rejected[symbol] = (
                        f"Pre-quality gate: gap<{need_gap:.2f}%|body<{need_body:.2f}%|model<{need_model:.1f}"
                    )
                    reasons_count["Pre-quality gate"] += 1
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
            if _budget_exceeded():
                rejected[symbol] = (
                    f"Превышен лимит времени на тикер ({TICKER_BUDGET_SEC}s) — before GPT"
                )
                reasons_count["Лимит времени на тикер"] += 1
                continue
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
            gpt_reply = "НЕТ (no-gpt)" if NO_GPT_MODE else "НЕТ (fallback)"
            if not NO_GPT_MODE:
                try:

                    def _gpt_call(prompt_):
                        chat = client.chat.completions.create(
                            model=GPT_SIGNAL_MODEL,
                            messages=[{"role": "user", "content": prompt_}],
                            temperature=0.0,
                            max_tokens=150,
                        )
                        return (chat.choices[0].message.content or "").strip()

                    ok, out, err = run_with_timeout(
                        _gpt_call, seconds=15, args=(prompt,), name="GPT"
                    )
                    if ok and out:
                        gpt_reply = out
                    else:
                        gpt_reply = f"НЕТ (gpt-{err or 'error'})"
                except Exception as _e:
                    gpt_reply = f"НЕТ (gpt-exc: {_e})"
            gpt_decisions[symbol] = gpt_reply
            rs_tail = ""
            if RS_ENABLED and (symbol in rs_table):
                _rsi = rs_table[symbol]
                rs_tail = f"\nRS={_rsi.get('pctl',0):.0f}pctl"
                if rs_slope is not None:
                    rs_tail += "↑" if (rs_slope or 0.0) > 0 else "↓"
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
            # обновим summary показатели из feats (на случай их корректировки)
            try:
                percent_change = float(feats.get("percent_change", percent_change))
                # Жёстко закрепляем Wilder RSI
                rsi = float(feats.get("rsi", 50.0))
                rsi = float(feats.get("rsi", rsi))
                ema_deviation = float(feats.get("ema_deviation", ema_deviation))
                atr_pct = float(feats.get("atr_pct", atr_pct))
                volatility = float(feats.get("volatility", volatility))
                bullish_body = float(feats.get("bullish_body", bullish_body))
                gap_up = float(feats.get("gap_up", gap_up))
            except Exception:
                pass
            summary_lines = [
                f"📊 Новый сигнал (BUY)",
                f"📌 ${symbol} @ {alpaca_price:.2f}",
                f"∆%={percent_change:.2f}% | RSI={rsi:.2f} | EMA dev={ema_deviation:.2f}%",
                f"ATR%={atr_pct:.2f} | Vol={volatility:.2f}%",
                f"🤖 GPT: {gpt_reply}{summary_live_tail}{rs_tail}{vwap_tail}{squeeze_tail}",
            ]
            if wick_penalized:
                summary_lines.append(f"⚠️ wick-penalty: -{penalty:.1f} к model_score")
            summary_msg = "\n".join(summary_lines)
            accepted = False
            accept_reason = "GPT_OK"
            if BYPASS_FILTERS:
                accepted = True
                accept_reason = "NO_FILTERS"
            else:
                strong_model_possible = model_score >= (
                    TH["MODEL_SCORE_MIN"] + HYBRID_MODEL_MARGIN
                )
                if is_gpt_yes(gpt_reply):
                    accepted = True
                    accept_reason = "GPT_OK"
                else:
                    if HYBRID_ENABLED and (hybrid_used < E_HYBRID_MAX_OVERRIDES):
                        clean_noise = (
                            (atr_pct <= TH["MAX_ATR_PCT"])
                            and (volatility <= TH["MAX_VOLATILITY"])
                            if HYBRID_STRICT_NOISE
                            else True
                        )
                        no_neg_trend = volume_trend >= TH["MIN_VOLUME_TREND"]
                        need_gap = max(TH.get("MIN_GAP_UP", 0.0), QUALITY_MIN_GAP_PCT)
                        need_body = max(
                            TH.get("MIN_BULLISH_BODY", 0.0), QUALITY_MIN_BODY_PCT
                        )
                        need_model = (
                            TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS
                        )
                        strong_model_possible = model_score >= (
                            need_model + HYBRID_MODEL_MARGIN
                        )
                        if (
                            strong_model_possible
                            and clean_noise
                            and no_neg_trend
                            and (bullish_body >= need_body + 0.10)
                            and (gap_up >= need_gap + 0.05)
                        ):
                            accepted = True
                            accept_reason = "HYBRID_ACCEPT"
                            hybrid_used += 1
                            hybrid_accepts.append(symbol)
            if accepted:
                signals[symbol] = {
                    "price": round(float(alpaca_price), 2),
                    "action": "BUY",
                    "confidence": round(
                        float(feats.get("alpha_score", alpha_score)), 2
                    ),
                    "score": round(float(model_score), 2),
                    "atr": round(float(feats.get("atr_value", 0.0)), 2),
                    "atr_pct": round(float(feats.get("atr_pct", 0.0)), 2),
                    "volatility": round(float(feats.get("volatility", 0.0)), 2),
                    "volume_trend": round(float(feats.get("volume_trend", 1.0)), 2),
                    "bullish_body": round(float(feats.get("bullish_body", 0.0)), 2),
                    "gap_up": round(float(feats.get("gap_up", 0.0)), 2),
                    "rsi": round(float(rsi), 2),
                    "reason": accept_reason,
                    "squeeze": {
                        "score": round(float(sq_score), 2),
                        "si_pct": round(float(sq_si), 2),
                        "dtc": round(float(sq_dtc), 3),
                        "float_m": round(float(sq_float), 2),
                        "fee_pct": round(float(sq_fee), 2),
                        "util_pct": round(float(sq_util), 1),
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
    # pre-rescue count
    count_for_adapt = len(signals)
    # === Rescue-pass ===
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
            need_gap = max(soft_TH.get("MIN_GAP_UP", 0.0), QUALITY_MIN_GAP_PCT)
            need_body = max(soft_TH.get("MIN_BULLISH_BODY", 0.0), QUALITY_MIN_BODY_PCT)
            need_model = soft_TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS
            conds = [
                info["bullish_body"] >= need_body,
                info["gap_up"] >= need_gap,
                info["volume_ratio"] >= soft_TH["MIN_VOLUME_RATIO"],
                info["atr_pct"] <= soft_TH["MAX_ATR_PCT"],
                info["volatility"] <= soft_TH["MAX_VOLATILITY"],
                info["volume_trend"] >= soft_TH["MIN_VOLUME_TREND"],
                (info["alpha_score"] * 100) >= soft_TH["MIN_RISK_SCORE"],
                info["model_score"] >= need_model,
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
                    "rsi": round(float(rsi), 2),
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
                pass
    # === Fallback-pass ===
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
        ultra_model_req = clamp(TH["MODEL_SCORE_MIN"] + 5.0, *BOUNDS["MODEL_SCORE_MIN"])
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
            need_gap = max(ultra_TH.get("MIN_GAP_UP", 0.0), QUALITY_MIN_GAP_PCT)
            need_body = max(ultra_TH.get("MIN_BULLISH_BODY", 0.0), QUALITY_MIN_BODY_PCT)
            ultra_model_req = max(
                ultra_model_req, TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS
            )
            conds = [
                info["bullish_body"] >= need_body,
                info["gap_up"] >= need_gap,
                info["volume_ratio"] >= ultra_TH["MIN_VOLUME_RATIO"],
                info["atr_pct"] <= TH["MAX_ATR_PCT"],
                info["volatility"] <= TH["MAX_VOLATILITY"],
                info["volume_trend"] >= TH["MIN_VOLUME_TREND"],
                info["model_score"]
                >= max(
                    ultra_model_req,
                    TH.get("MODEL_SCORE_MIN", 0.0) + QUALITY_MODEL_BONUS,
                ),
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
                    "rsi": round(float(rsi), 2),
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
                pass
    # === Tiebreaker + Quality Gate ===
    if isinstance(signals, dict) and len(signals) > E_TARGET_MAX:

        def _safe(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return float(default)

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
        else:
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
                        "🧰 Quality gate: отклонены в TOP из-за порогов -> "
                        + ", ".join(f"{sym}({reason})" for sym, _, reason, _ in failed)
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
            pass
    # === POST-FEATS SYNC -> signals: жёстко подменяем ключевые поля значениями feats ===
    try:
        for _sym, _info in list(signals.items()):
            _f = features_map.get(_sym)
            if not isinstance(_f, dict):
                continue

            def _flt(k, d=0.0):
                try:
                    return float(_f.get(k, d))
                except Exception:
                    return float(d)

            _info["atr"] = round(_flt("atr_value", _info.get("atr", 0.0)), 2)
            _info["atr_pct"] = round(_flt("atr_pct", _info.get("atr_pct", 0.0)), 2)
            _info["volatility"] = round(
                _flt("volatility", _info.get("volatility", 0.0)), 2
            )
            _info["volume_trend"] = round(
                _flt("volume_trend", _info.get("volume_trend", 1.0)), 2
            )
            _info["bullish_body"] = round(
                _flt("bullish_body", _info.get("bullish_body", 0.0)), 2
            )
            _info["gap_up"] = round(_flt("gap_up", _info.get("gap_up", 0.0)), 2)
    except Exception as _e:
        if DEBUG_MODE:
            print(f"[POST-FEATS WARN] {_e}")
    # --- Save artifacts ---
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
    print("\n📊 Сводка причин отказов:")
    for k in sorted(reasons_count, key=reasons_count.get, reverse=True):
        print(f"  • {k}: {reasons_count[k]}")
    print(f"\n📦 Сигналы: {len(signals)} -> {SIGNALS_PATH}")
    print(f"🚫 Отклонено: {len(rejected)} -> {REJECTED_PATH}")
    print(f"🧠 GPT ответы: {GPT_DECISIONS_PATH}")
    print(
        f"🟡 Hybrid-accepts: {len([k for k in signals.values() if k.get('reason')=='HYBRID_ACCEPT'])} -> {', '.join([s for s,d in signals.items() if d.get('reason')=='HYBRID_ACCEPT']) or '—'}"
    )
    # --- Adaptive update / no-filters notice
    if not BYPASS_FILTERS:
        adaptive_cfg = load_adaptive()
        adaptive_count = (
            count_for_adapt if count_for_adapt is not None else len(signals)
        )
        direction, new_th = adjust_thresholds_v2(
            adaptive_cfg, adaptive_count, reasons_count, macro_regime
        )
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
            pass
    else:
        try:
            send_telegram_message(
                f"🟢 Режим без фильтров активен. Принято сигналов: {len(signals)}."
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
