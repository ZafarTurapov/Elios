from core.utils.alpaca_headers import alpaca_headers


def _read_signals_any_format(path="core/trading/signals.json"):
    """
    Возвращает list[dict]: [{symbol, price, action, weight, meta?}, ...]
    Поддерживает:
      • {"signals":[...]}
      • {"AAPL": {...}, "NVDA": {...}}
    """
    import json
    import os

    if not os.path.exists(path):
        return []
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("signals"), list):
        # уже нормализовано
        out = []
        for it in obj["signals"]:
            if not isinstance(it, dict):
                continue
            sym = (it.get("symbol") or it.get("ticker") or "").upper()
            if not sym:
                continue
            it = {**it}
            it.setdefault("action", "BUY")
            it.setdefault("weight", 1.0)
            out.append({"symbol": sym, **it})
        return out
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            d = v or {}
            if not isinstance(d, dict):
                d = {}
            sym = str(k).upper()
            item = {
                "symbol": sym,
                "price": float(d.get("price") or 0.0),
                "action": d.get("action", "BUY"),
                "weight": float(d.get("weight") or 1.0),
                "meta": d.get("meta"),
            }
            # прокинем полезные поля, если есть
            for fld in (
                "confidence",
                "score",
                "atr",
                "atr_pct",
                "volatility",
                "volume_trend",
                "bullish_body",
                "gap_up",
                "reason",
                "squeeze",
            ):
                if fld in d:
                    item[fld] = d[fld]
            out.append(item)
        return out
    return []


# core/trading/trade_executor.py — гибридная аллокация с реальным BP/плечом, gross-капом,
# BUY только в окно 18:30–21:00 (Asia/Tashkent). Выход делает Sell Engine.
# Новое:
# • Проверка открытия рынка США (Alpaca /v2/clock)
# • Slippage-guard по live-цене vs цене из сигнала (по умолч. 1.25%)
# • Динамическая подгонка qty под фактическую цену/бюджет
# • DRY_RUN режим и троттлинг запросов к Alpaca
# • Расширенный аудит: почему не купили (slippage/budget/caps/округление)
# • ВКЛЮЧЕНО: поддержка short-squeeze-флагов из signals, даунсайз при long_risk
# • ВКЛЮЧЕНО: OCO (bracket) с ужесточенными TP/SL при long_risk + ATR-режим

import os
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from pathlib import Path

# === ensure /root/stockbot on sys.path ===
import sys

if "/root/stockbot" not in sys.path:
    sys.path.insert(0, "/root/stockbot")

# === Alpaca Keys ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")

HEADERS = alpaca_headers()

# --- ENV-флаги/режимы ---
STRICT_ASSETS = os.getenv("ELIOS_STRICT_ASSETS_CHECK", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
SKIP_ASSETS = (
    not STRICT_ASSETS
    if os.getenv("ELIOS_SKIP_ASSETS_CHECK") is None
    else (
        os.getenv("ELIOS_SKIP_ASSETS_CHECK", "").strip().lower() in ("1", "true", "yes")
    )
)
FAILOPEN_IF_EMPTY = os.getenv("ELIOS_FAILOPEN_IF_EMPTY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

ROUNDING_RESCUE_ENABLED = os.getenv("ELIOS_ROUNDING_RESCUE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
CHECK_US_MARKET_OPEN = os.getenv("ELIOS_CHECK_US_MARKET_OPEN", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
DRY_RUN = os.getenv("ELIOS_DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
ORDER_THROTTLE_MS = int(os.getenv("ELIOS_ORDER_THROTTLE_MS", "300"))
ORDER_TIF = os.getenv("ELIOS_ORDER_TIME_IN_FORCE", "day")
ORDER_TYPE = os.getenv(
    "ELIOS_ORDER_TYPE", "market"
)  # market|limit (limit не рекомендуется в этом модуле)

# --- BUY-окно (локальное время Asia/Tashkent) ---
TZ_LOCAL = ZoneInfo("Asia/Tashkent")
BUY_WIN_START = os.getenv("ELIOS_BUY_WINDOW_START", "18:30")
BUY_WIN_END = os.getenv("ELIOS_BUY_WINDOW_END", "21:00")

# --- Глобальные лимиты плеча/риска ---
GROSS_CAP_PRIME = float(os.getenv("ELIOS_GROSS_CAP_PRIME", "1.5"))  # в окне 18:30–21:00
GROSS_CAP_OFF = float(os.getenv("ELIOS_GROSS_CAP_OFF", "1.0"))  # вне окна
DAILY_RISK_CAP_PCT = float(os.getenv("ELIOS_DAILY_RISK_CAP_PCT", "0.03"))  # 3%
BP_SAFETY_BUFFER_PCT = float(os.getenv("ELIOS_BP_SAFETY_BUFFER_PCT", "0.05"))  # 5%

# --- Slippage guard ---
SLIPPAGE_REJECT_PCT = float(
    os.getenv("ELIOS_SLIPPAGE_REJECT_PCT", "1.25")
)  # % от цены сигнала
SLIPPAGE_WARN_PCT = float(
    os.getenv("ELIOS_SLIPPAGE_WARN_PCT", "0.60")
)  # предупреждение в телеге
SLIPPAGE_DIRECTIONAL = os.getenv("ELIOS_SLIPPAGE_DIRECTIONAL", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
SLIPPAGE_WARN_FAVORABLE = os.getenv(
    "ELIOS_SLIPPAGE_WARN_FAVORABLE", "1"
).strip().lower() in ("1", "true", "yes")

# --- Squeeze-aware sizing / OCO ---
SQUEEZE_SIZE_MULT = float(
    os.getenv("ELIOS_SQUEEZE_LONG_RISK_SIZE_MULT", "0.5")
)  # 50% размера при long_risk
OCO_ENABLE = os.getenv("ELIOS_OCO_ENABLE", "1").strip().lower() in ("1", "true", "yes")
OCO_TIGHT_IF_LONG_RISK = os.getenv(
    "ELIOS_OCO_TIGHT_IF_LONG_RISK", "1"
).strip().lower() in ("1", "true", "yes")
OCO_USE_ATR = os.getenv("ELIOS_OCO_USE_ATR", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

# базовые % (если ATR не используется или нет данных)
TP_PCT_BASE = float(os.getenv("ELIOS_TP_PCT", "3.0"))  # +3.0% TP
SL_PCT_BASE = float(os.getenv("ELIOS_SL_PCT", "1.6"))  # -1.6% SL
# ужатые % для long_risk
TP_PCT_TIGHT = float(os.getenv("ELIOS_TP_PCT_TIGHT", "2.2"))
SL_PCT_TIGHT = float(os.getenv("ELIOS_SL_PCT_TIGHT", "1.0"))

# ATR-мультипликаторы: TP/SL как доля дневного ATR%
TP_ATR_MULT = float(os.getenv("ELIOS_TP_ATR_MULT", "0.90"))  # 0.9 * ATR%
SL_ATR_MULT = float(os.getenv("ELIOS_SL_ATR_MULT", "0.60"))  # 0.6 * ATR%
TP_ATR_MULT_TIGHT = float(os.getenv("ELIOS_TP_ATR_MULT_TIGHT", "0.70"))
SL_ATR_MULT_TIGHT = float(os.getenv("ELIOS_SL_ATR_MULT_TIGHT", "0.45"))

# --- АУДИТ АЛЛОКАЦИИ ---
ELIOS_ALLOCATION_AUDIT = os.getenv("ELIOS_ALLOCATION_AUDIT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
LAST_GPT_WEIGHTS = {}

# === OpenAI (для GPT-аллокации) ===

import os

USE_GPT = os.getenv("ELIOS_USE_GPT", "0") == "1"
OpenAI = None
if USE_GPT:
    try:
        from openai import OpenAI  # guarded lazy import
    except Exception:
        OpenAI = None

client = None
if USE_GPT and OpenAI is not None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    except Exception as _e:
        client = None
GPT_MODEL = os.getenv("ELIOS_GPT_MODEL", "gpt-4o-mini")

# === Paths ===
SIGNALS_PATH = Path("core/trading/signals.json")
TRADE_LOG_PATH = Path("core/trading/trade_log.json")
OPEN_POSITIONS_PATH = Path("core/trading/open_positions.json")
ACCOUNT_PATH = Path("core/trading/account_summary.json")
POSITIONS_PATH = Path("core/trading/open_positions.json")
DEBUG_LOG_PATH = Path("logs/executor_debug.json")

from core.utils.telegram import send_telegram_message, escape_markdown

# === Allocation settings ===
ALLOCATION_MODE = os.getenv(
    "ELIOS_ALLOCATION_MODE", "hybrid_gpt"
)  # proportional|capped_proportional|gpt|hybrid_gpt
RESERVE_CASH_PCT = float(os.getenv("ELIOS_RESERVE_CASH_PCT", "0.10"))

# До 100% целевого бюджета на тикер (далее ограничим gross-капом и BP):
MAX_ALLOC_PCT_PER_TICKER = float(os.getenv("ELIOS_PER_TICKER_CAP_PCT_MAX", "1.0"))

MAX_POSITIONS_NEW = int(os.getenv("ELIOS_MAX_POSITIONS_NEW", "6"))
MIN_QTY_PER_TRADE = int(os.getenv("ELIOS_MIN_QTY_PER_TRADE", "1"))
MIN_USD_PER_TRADE = float(os.getenv("ELIOS_MIN_USD_PER_TRADE", "0.0"))
W_SCORE = float(os.getenv("ELIOS_W_SCORE", "0.60"))
W_CONF = float(os.getenv("ELIOS_W_CONF", "0.40"))


def escape_num(n):
    return (
        str(n)
        .replace(".", "\\.")
        .replace("-", "\\-")
        .replace("+", "\\+")
        .replace("$", "\\$")
    )


# ---- Flexible signals loader ----
def _normalize_signal_obj(obj: dict):
    sym = (obj.get("symbol") or obj.get("ticker") or obj.get("sym") or "").upper()
    side = (obj.get("side") or obj.get("action") or "buy").lower()
    qty = obj.get("qty") or obj.get("quantity") or obj.get("size") or 0
    score = (
        obj.get("score")
        or obj.get("model_score")
        or obj.get("prob")
        or obj.get("confidence")
    )
    price = (
        obj.get("price") or obj.get("entry") or obj.get("trigger") or obj.get("last")
    )
    out = dict(obj)
    out.update(
        {"symbol": sym, "side": side, "qty": qty, "score": score, "price": price}
    )
    return sym, out


def _load_signals_flexible(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return {}
    if not raw:
        return {}
    raw = raw.strip("` \n")
    try:
        data = json.loads(raw)
    except Exception:
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        data = items

    signals = {}
    if isinstance(data, dict):
        if isinstance(data.get("signals"), list):
            items = data["signals"]
        else:
            for k, v in data.items():
                if not isinstance(v, dict):
                    v = {}
                sym, out = _normalize_signal_obj({**v, "symbol": k})
                if sym:
                    signals[sym] = out
            return signals
    elif isinstance(data, list):
        items = data
    else:
        return {}
    for obj in items:
        if not isinstance(obj, dict):
            continue
        sym, out = _normalize_signal_obj(obj)
        if sym:
            signals[sym] = out
    return signals


# ---------- Alpaca helpers ----------
def _unique_coid(symbol: str, qty: int) -> str:
    return f"elios-{symbol}-{int(time.time()*1000)}-q{qty}"


def submit_order_simple(
    symbol, qty, side="buy", type=ORDER_TYPE, time_in_force=ORDER_TIF
):
    if DRY_RUN:
        return {
            "id": "dry-run",
            "status": "accepted",
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "order_class": "simple",
        }
    url = f"{ALPACA_BASE_URL}/v2/orders"
    order_data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
        "client_order_id": _unique_coid(symbol, qty),
    }
    try:
        r = requests.post(url, json=order_data, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise Exception(
            f"Ошибка отправки ордера: {e}\n{r.text if 'r' in locals() and r is not None else ''}"
        )


def submit_order_bracket_market(
    symbol, qty, tp_price, sl_stop, sl_limit=None, time_in_force=ORDER_TIF
):
    """
    Market parent + TP/SL дочерние (Alpaca bracket).
    """
    if DRY_RUN:
        return {
            "id": "dry-run",
            "status": "accepted",
            "symbol": symbol,
            "qty": qty,
            "side": "buy",
            "type": "market",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "tp": tp_price,
            "sl_stop": sl_stop,
            "sl_limit": sl_limit,
        }
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": "buy",
        "type": "market",
        "time_in_force": time_in_force,
        "client_order_id": _unique_coid(symbol, qty),
        "order_class": "bracket",
        "take_profit": {"limit_price": round(float(tp_price), 2)},
        "stop_loss": {
            "stop_price": round(float(sl_stop), 2),
            **({"limit_price": round(float(sl_limit), 2)} if sl_limit else {}),
        },
    }
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise Exception(
            f"Ошибка bracket-ордера: {e}\n{r.text if 'r' in locals() and r is not None else ''}"
        )


def fetch_account_live():
    """Живые поля аккаунта: cash, equity, last_equity, (daytrading_)buying_power, long/short MV."""
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=10)
        r.raise_for_status()
        a = r.json() or {}

        def f(x, d=0.0):
            try:
                return float(x)
            except:
                return d

        out = {
            "cash": f(a.get("cash")),
            "equity": f(a.get("equity")),
            "last_equity": f(a.get("last_equity")),
            "buying_power": f(a.get("buying_power")),
            "dtbp": f(a.get("daytrading_buying_power")),
            "portfolio_value": f(a.get("portfolio_value")),
            "long_market_value": f(a.get("long_market_value")),
            "short_market_value": f(a.get("short_market_value")),
        }
        return out
    except Exception as e:
        print(f"[ERROR] /v2/account failed: {e}")
        return {
            "cash": 0.0,
            "equity": 0.0,
            "last_equity": 0.0,
            "buying_power": 0.0,
            "dtbp": 0.0,
            "portfolio_value": 0.0,
            "long_market_value": 0.0,
            "short_market_value": 0.0,
        }


def fetch_live_positions_symbols():
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return set((p.get("symbol") or "").upper() for p in r.json())
        print(f"[WARN] positions status={r.status_code} | {r.text}")
    except Exception as e:
        print(f"[WARN] fetch_live_positions_symbols: {e}")
    return set()


def fetch_positions_total_value():
    """Возвращает суммарную абсолютную рыночную стоимость позиций (|long|+|short|)."""
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"[WARN] positions total status={r.status_code} | {r.text}")
            return 0.0
        total = 0.0
        for p in r.json():
            try:
                total += abs(float(p.get("market_value") or 0.0))
            except:
                pass
        return total
    except Exception as e:
        print(f"[WARN] fetch_positions_total_value: {e}")
        return 0.0


def is_tradable_safe(symbol: str):
    """
    Безопасная проверка:
    - Если SKIP_ASSETS=True → разрешаем сразу.
    - HTTP 403/сетевые ошибки/не-200 → fail-open (разрешаем).
    - Явный отказ только при tradable=false или status!=active.
    """
    if SKIP_ASSETS:
        return True, "skip_env"
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/assets/{symbol}", headers=HEADERS, timeout=10
        )
        if r.status_code == 403:
            return True, "403_fallback"
        if r.status_code != 200:
            return True, f"assets {r.status_code} (fail_open)"
        a = r.json() or {}
        if not a.get("tradable", False):
            return False, "not tradable"
        if str(a.get("status", "")).lower() != "active":
            return False, f"status {a.get('status')}"
        return True, "ok"
    except Exception as e:
        return True, f"assets error fail_open: {e}"


def us_market_is_open() -> bool:
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/clock", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return bool((r.json() or {}).get("is_open", False))
    except Exception as e:
        print(f"[WARN] /v2/clock: {e}")
    return True  # fail-open


def get_last_price(symbol: str) -> float:
    """Последняя цена (quotes→trades→bars) через Alpaca data (IEX feed)."""
    try:
        rq = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/quotes/latest",
            params={"feed": "iex"},
            headers=HEADERS,
            timeout=10,
        )
        if rq.status_code == 200:
            q = (rq.json() or {}).get("quote") or {}
            ap = float(q.get("ap") or 0)
            bp = float(q.get("bp") or 0)
            if ap > 0 or bp > 0:
                return ap if ap > 0 else bp
        rt = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest",
            params={"feed": "iex"},
            headers=HEADERS,
            timeout=10,
        )
        if rt.status_code == 200:
            p = float(((rt.json() or {}).get("trade") or {}).get("p") or 0)
            if p > 0:
                return p
        rb = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars/latest",
            params={"feed": "iex"},
            headers=HEADERS,
            timeout=10,
        )
        if rb.status_code == 200:
            c = float(((rb.json() or {}).get("bar") or {}).get("c") or 0)
            if c > 0:
                return c
    except Exception as e:
        print(f"[WARN] last price {symbol}: {e}")
    return 0.0


# ---------- Local data ----------
def load_account_data():
    try:
        with ACCOUNT_PATH.open() as f:
            account = json.load(f)
    except Exception:
        account = {}
    try:
        with POSITIONS_PATH.open() as f:
            positions = json.load(f)
    except Exception:
        positions = {}
    return account, positions


# --- Coercion helpers ---
def _coerce_trade_log(obj):
    out = {}
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list):
                    out[k] = v
                elif v is None:
                    out[k] = []
                else:
                    out[k] = [v]
            return out
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    sym = item.get("symbol") or "UNKNOWN"
                    out.setdefault(sym, []).append(item)
            return out
    except Exception:
        pass
    return {}


def _coerce_positions(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        out = {}
        for p in obj:
            if isinstance(p, dict):
                sym = (p.get("symbol") or "").upper()
                if sym:
                    out[sym] = p
        return out
    return {}


# 🔧 Авто-синхронизация локальных «открытых позиций»
def reconcile_local_positions(open_positions: dict) -> dict:
    keep = set()
    keep |= fetch_live_positions_symbols()
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/orders",
            headers=HEADERS,
            params={"status": "open", "limit": "200", "nested": "true"},
            timeout=10,
        )
        if r.status_code == 200:
            keep |= set((o.get("symbol") or "").upper() for o in r.json())
        else:
            print(f"[WARN] reconcile orders status={r.status_code} | {r.text}")
    except Exception as e:
        print(f"[WARN] reconcile orders: {e}")

    pruned = {s: v for s, v in (open_positions or {}).items() if s.upper() in keep}
    removed = [s for s in (open_positions or {}).keys() if s.upper() not in keep]
    if removed:
        try:
            msg = "🧹 Очищены локальные хвосты позиций:\n" + "\n".join(
                f"• {escape_markdown(s)}" for s in removed
            )
            send_telegram_message(msg)
        except Exception:
            pass
    return pruned


# ---------- Weights & allocation ----------
def _priority_weight(info: dict) -> float:
    def f(x):
        try:
            return float(x)
        except:
            return 0.0

    score = f(info.get("score"))
    conf = f(info.get("confidence"))
    conf_scaled = conf * 100.0 if conf <= 1.0 else conf
    return W_SCORE * score + W_CONF * conf_scaled


def _normalized_weights_from_scores(signals: dict):
    arr, total = [], 0.0
    for sym, info in signals.items():
        price = float(info.get("price", 0) or 0)
        if price <= 0:
            continue
        w = _priority_weight(info)
        if w > 0:
            arr.append((sym, price, w))
            total += w
    if total <= 0:
        return []
    return [(sym, price, w / total) for sym, price, w in arr]


def _apply_caps_and_build_qty(weighted, target_notional_total, per_ticker_cap):
    remaining = target_notional_total
    position_sizes = {}
    for sym, price, w in sorted(weighted, key=lambda x: x[2], reverse=True):
        if remaining <= 0:
            break
        alloc = min(target_notional_total * w, per_ticker_cap)
        alloc = min(alloc, remaining)
        if MIN_USD_PER_TRADE > 0 and alloc < MIN_USD_PER_TRADE:
            continue
        qty = int(alloc // price)
        if qty < MIN_QTY_PER_TRADE:
            if remaining >= price and (
                MIN_USD_PER_TRADE == 0 or price >= MIN_USD_PER_TRADE
            ):
                qty = 1
            else:
                continue
        cost = qty * price
        if cost <= 0 or cost > remaining:
            continue
        position_sizes[sym] = qty
        remaining -= cost
    return position_sizes


def _weights_to_qty(weights_dict, prices_dict, target_notional_total, per_ticker_cap):
    s = sum(max(0.0, float(v)) for v in weights_dict.values()) or 1.0
    weighted = []
    for sym, w in weights_dict.items():
        try:
            w = max(0.0, float(w)) / s
        except:
            continue
        price = float(prices_dict.get(sym, 0) or 0)
        if price <= 0:
            continue
        weighted.append((sym, price, w))
    return _apply_caps_and_build_qty(weighted, target_notional_total, per_ticker_cap)


def _gpt_propose_weights(filtered_signals: dict, cash_after_reserve: float) -> dict:
    global LAST_GPT_WEIGHTS
    try:
        items = []
        for sym, info in filtered_signals.items():
            items.append(
                {
                    "symbol": sym,
                    "price": float(info.get("price", 0) or 0),
                    "score": float(info.get("score", 0) or 0),
                    "confidence": float(info.get("confidence", 0) or 0),
                    "atr_pct": float(info.get("atr_pct", 0) or 0),
                    "volatility": float(info.get("volatility", 0) or 0),
                    "volume_trend": float(info.get("volume_trend", 0) or 0),
                    "bullish_body": float(info.get("bullish_body", 0) or 0),
                    "gap_up": float(info.get("gap_up", 0) or 0),
                }
            )
        system = (
            "Ты портфельный аллокатор. Верни ЧИСТЫЙ JSON без текста, "
            'формата {"weights": {"SYM": 0.0..1.0, ...}}. Сумма весов ≤ 1.0. '
            "Предпочитай высокий score/confidence, штрафуй высокий atr_pct/volatility."
        )
        user = json.dumps(
            {
                "cash_after_reserve": cash_after_reserve,
                "max_per_ticker_pct": MAX_ALLOC_PCT_PER_TICKER,
                "max_positions": MAX_POSITIONS_NEW,
                "signals": items,
            },
            ensure_ascii=False,
        )
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
        except Exception:
            data = json.loads(raw.strip().strip("`").strip())
        weights_raw = data.get("weights", {}) or {}
        LAST_GPT_WEIGHTS = weights_raw

        if ELIOS_ALLOCATION_AUDIT:
            try:
                wf = {
                    k: (round(float(v), 4) if isinstance(v, (int, float, str)) else v)
                    for k, v in weights_raw.items()
                }
                considered = ", ".join(sorted(filtered_signals.keys()))
                send_telegram_message(
                    "🧮 *GPT веса аллокации*:\n"
                    f"`{json.dumps(wf, ensure_ascii=False)}`\n"
                    f"К рассмотрению: {escape_markdown(considered)}"
                )
            except Exception:
                pass

        valid = {}
        for sym, w in weights_raw.items():
            if sym in filtered_signals:
                try:
                    w = float(w)
                    if w > 0:
                        valid[sym] = max(0.0, min(1.0, w))
                except Exception:
                    pass
        return valid
    except Exception as e:
        print(f"[WARN] GPT allocation failed: {e}")
        LAST_GPT_WEIGHTS = {}
        return {}


# === Вспомогательные функции по времени и бюджетам ===
def _parse_hhmm(s: str):
    h, m = s.split(":")
    return int(h), int(m)


def within_buy_window(now_local: datetime) -> bool:
    sh, sm = _parse_hhmm(BUY_WIN_START)
    eh, em = _parse_hhmm(BUY_WIN_END)
    start = now_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now_local.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now_local < end


def determine_gross_cap(now_local: datetime) -> float:
    return GROSS_CAP_PRIME if within_buy_window(now_local) else GROSS_CAP_OFF


def calc_available_budget_from_bp_and_gross(account: dict, gross_cap: float) -> float:
    """
    Итоговый бюджет на новые покупки = min(доступный BP с буфером, свободная "комната" по gross-каппу).
    gross = total_mv / equity ; room = max(0, gross_cap - gross) * equity
    """
    equity = account.get("equity", 0.0) or 0.0
    last_equity = account.get("last_equity", 0.0) or 0.0
    bp = account.get("dtbp", 0.0) or account.get("buying_power", 0.0) or 0.0
    bp_eff = max(0.0, bp * (1.0 - BP_SAFETY_BUFFER_PCT))

    total_mv = fetch_positions_total_value()
    gross = (total_mv / equity) if (equity and equity > 0) else 0.0
    room = max(0.0, gross_cap - gross) * (equity if equity > 0 else 0.0)

    if equity <= 0.0 and bp_eff <= 0.0:
        return 0.0

    if (last_equity and last_equity > 0.0) and (
        equity < (1.0 - DAILY_RISK_CAP_PCT) * last_equity
    ):
        send_telegram_message(
            f"🛑 *Дневной риск-кап сработал*\n"
            f"Equity: {equity:.2f} < {(1.0 - DAILY_RISK_CAP_PCT) * last_equity:.2f} "
            f"(порог {DAILY_RISK_CAP_PCT*100:.1f}%)\nНовые покупки заблокированы до завтра."
        )
        return 0.0

    return max(0.0, min(bp_eff, room))


# ---------- Slippage helpers ----------
def _slippage_info(sig_price: float, live_price: float):
    """
    Возвращает:
      sp_abs   — абсолютный % слиппиджа
      sp_signed— signed % (положит. = хуже для buy, отрицат. = лучше)
      favorable — bool, True если лучше для buy (live < signal)
    """
    if not sig_price or sig_price <= 0 or not live_price or live_price <= 0:
        return 0.0, 0.0, False
    sp_signed = (live_price - sig_price) / sig_price * 100.0
    sp_abs = abs(sp_signed)
    favorable = sp_signed < 0.0
    return sp_abs, sp_signed, favorable


def _slippage_ok(sig_price: float, live_price: float) -> (bool, float, float, bool):
    sp_abs, sp_signed, favorable = _slippage_info(sig_price, live_price)
    if SLIPPAGE_DIRECTIONAL and favorable:
        # цена лучше сигналной — не блокируем
        return True, sp_abs, sp_signed, favorable
    # иначе блокируем только если ухудшение превысило порог
    return (sp_abs <= SLIPPAGE_REJECT_PCT), sp_abs, sp_signed, favorable
    sp = _slippage_pct(sig_price, live_price)
    return (sp <= SLIPPAGE_REJECT_PCT), sp


# ---------- SQUEEZE helpers ----------
def _squeeze_flags(info: dict):
    sq = info.get("squeeze") or {}
    # безопасные значения
    score = float(sq.get("score") or 0.0)
    long_risk = bool(sq.get("long_risk") or False)
    short_opp = bool(sq.get("short_opportunity") or False)
    return score, long_risk, short_opp


def _atr_pct(info: dict) -> float:
    try:
        return float(info.get("atr_pct") or 0.0)
    except Exception:
        return 0.0


def _compute_tp_sl_prices(info: dict, live_price: float):
    """
    Возвращает (tp_price, sl_stop, sl_limit, meta_str)
    meta_str — для телеги: "%-режим" или "ATR%-режим"
    """
    score, long_risk, _ = _squeeze_flags(info)
    use_tight = OCO_TIGHT_IF_LONG_RISK and long_risk

    if OCO_USE_ATR and _atr_pct(info) > 0:
        atrp = _atr_pct(info)
        tp_pct = (TP_ATR_MULT_TIGHT if use_tight else TP_ATR_MULT) * atrp
        sl_pct = (SL_ATR_MULT_TIGHT if use_tight else SL_ATR_MULT) * atrp
        tp_price = live_price * (1.0 + tp_pct / 100.0)
        sl_stop = live_price * (1.0 - sl_pct / 100.0)
        sl_limit = sl_stop * 0.999  # чуть ниже
        meta = f"ATR% mode: TP {tp_pct:.2f}%, SL {sl_pct:.2f}% (ATR={atrp:.2f}%)"
    else:
        tp_pct = TP_PCT_TIGHT if use_tight else TP_PCT_BASE
        sl_pct = SL_PCT_TIGHT if use_tight else SL_PCT_BASE
        tp_price = live_price * (1.0 + tp_pct / 100.0)
        sl_stop = live_price * (1.0 - sl_pct / 100.0)
        sl_limit = sl_stop * 0.999
        meta = f"% mode: TP {tp_pct:.2f}%, SL {sl_pct:.2f}%"
    return round(tp_price, 2), round(sl_stop, 2), round(sl_limit, 2), meta


# ---------- Allocation wrapper ----------
def _priority_weight(info: dict) -> float:
    # (дублируется выше — оставлено намеренно для читабельности блока)
    def f(x):
        try:
            return float(x)
        except:
            return 0.0

    score = f(info.get("score"))
    conf = f(info.get("confidence"))
    conf_scaled = conf * 100.0 if conf <= 1.0 else conf
    return W_SCORE * score + W_CONF * conf_scaled


def calculate_position_sizes(
    signals: dict, available_notional: float, mode: str
) -> dict:
    if not signals:
        return {}
    target_notional_total = max(0.0, available_notional * (1.0 - RESERVE_CASH_PCT))
    if target_notional_total <= 0.0:
        print("[WARN] Нет доступного бюджета после резерва.")
        return {}

    # ТОП по приоритету
    scored = []
    for sym, info in signals.items():
        price = float(info.get("price", 0) or 0)
        if price <= 0:
            continue
        w = _priority_weight(info)
        # ↓ корректировка веса под squeeze long risk: просто уменьшаем итоговый alloc через post-фактор при сборке qty
        scored.append((sym, price, w))
    if not scored:
        return {}
    scored.sort(key=lambda x: x[2], reverse=True)
    top_syms = [s for s, _, _ in scored[: max(1, min(MAX_POSITIONS_NEW, len(scored)))]]
    top_signals = {s: signals[s] for s in top_syms}

    per_ticker_cap = target_notional_total * MAX_ALLOC_PCT_PER_TICKER
    prices = {s: float(top_signals[s].get("price", 0) or 0) for s in top_syms}

    def _apply_with_squeeze_cap(weighted):
        """
        Применяем пер-тикер кап и дополнительный *уменьшающий множитель* для long_risk.
        """
        remaining = target_notional_total
        position_sizes = {}
        for sym, price, w in sorted(weighted, key=lambda x: x[2], reverse=True):
            if remaining <= 0:
                break
            base_alloc = min(target_notional_total * w, per_ticker_cap)
            info = top_signals.get(sym, {})
            _, long_risk, _ = _squeeze_flags(info)
            alloc = base_alloc * (
                SQUEEZE_SIZE_MULT if long_risk and SQUEEZE_SIZE_MULT > 0 else 1.0
            )
            alloc = min(alloc, remaining)
            if MIN_USD_PER_TRADE > 0 and alloc < MIN_USD_PER_TRADE:
                continue
            qty = int(alloc // price)
            if qty < MIN_QTY_PER_TRADE:
                if remaining >= price and (
                    MIN_USD_PER_TRADE == 0 or price >= MIN_USD_PER_TRADE
                ):
                    qty = 1
                else:
                    continue
            cost = qty * price
            if cost <= 0 or cost > remaining:
                continue
            position_sizes[sym] = qty
            remaining -= cost
        return position_sizes

    def fallback_capped():
        weighted = _normalized_weights_from_scores(top_signals)
        return _apply_with_squeeze_cap(weighted)

    if mode in ("proportional", "capped_proportional"):
        position_sizes = fallback_capped()
    elif mode == "gpt":
        weights = _gpt_propose_weights(top_signals, target_notional_total)
        # превращаем в triples
        if weights:
            s = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
            weighted = []
            for sym, w in weights.items():
                try:
                    w = max(0.0, float(w)) / s
                except:
                    continue
                price = prices.get(sym, 0.0)
                if price <= 0:
                    continue
                weighted.append((sym, price, w))
            position_sizes = _apply_with_squeeze_cap(weighted)
        else:
            position_sizes = fallback_capped()
    else:  # hybrid_gpt
        weights = _gpt_propose_weights(top_signals, target_notional_total)
        if weights:
            s = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
            weighted = []
            for sym, w in weights.items():
                try:
                    w = max(0.0, float(w)) / s
                except:
                    continue
                price = prices.get(sym, 0.0)
                if price <= 0:
                    continue
                weighted.append((sym, price, w))
            position_sizes = _apply_with_squeeze_cap(weighted)
        else:
            position_sizes = fallback_capped()

    # --- RESCUE: базовая аллокация дала 0 лотов
    if ROUNDING_RESCUE_ENABLED and not position_sizes:
        items = []
        for sym, info in signals.items():
            price = float(info.get("price", 0) or 0)
            if price <= 0:
                continue
            w = _priority_weight(info)
            _, long_risk, _ = _squeeze_flags(info)
            # при спасении не занижаем агрессивно — и так 1 лот
            items.append((sym, price, w, long_risk))
        # сортировка: дешевле → выше w → без long_risk
        items.sort(key=lambda t: (t[1], -t[2], t[3]))
        remaining = target_notional_total
        rescue = {}
        for sym, price, w, _lr in items:
            if price > remaining or price > per_ticker_cap:
                continue
            rescue[sym] = 1
            break
        if rescue:
            try:
                send_telegram_message(
                    "🛟 *Rounding/BP rescue*: базовая аллокация дала 0 лотов. Взял 1 лот самого дешёвого тикера."
                )
            except Exception:
                pass
            return rescue

    return position_sizes


# ---------- Main ----------
def main():
    # 0) BUY-окно + (опц.) рынок США открыт
    now_local = datetime.now(TZ_LOCAL)
    if not within_buy_window(now_local):
        send_telegram_message(
            f"🛌 *BUY-окно закрыто* — работаем только 18:30–21:00 Asia/Tashkent.\n"
            f"Сейчас: {now_local.strftime('%H:%M:%S')}"
        )
        print("BUY window closed. Exiting.")
        return
    if CHECK_US_MARKET_OPEN and not us_market_is_open():
        send_telegram_message(
            "⛔️ Рынок США закрыт по Alpaca /v2/clock — пропускаю покупки."
        )
        return

    send_telegram_message(
        "⚙️ Запущен модуль: TRADE EXECUTOR (BP+GrossCap, BUY окно 18:30–21:00, slippage-guard, squeeze-aware sizing, OCO для long_risk, выход 'sell' по умолчанию)"
    )

    # 1) Сигналы
    signals = _load_signals_flexible(SIGNALS_PATH)
    if not signals:
        print("📭 Сигналы пусты.")
        return

    # 2) Живые/локальные позиции → антидубль
    live_positions_set = fetch_live_positions_symbols()
    _, local_positions = load_account_data()
    local_positions = reconcile_local_positions(local_positions or {})
    try:
        with OPEN_POSITIONS_PATH.open("w") as f:
            json.dump(local_positions, f, indent=2)
    except Exception:
        pass
    local_positions_set = set(local_positions.keys())

    signals_no_dupes = {
        s: info
        for s, info in signals.items()
        if s not in live_positions_set and s not in local_positions_set
    }
    if not signals_no_dupes:
        msg = [
            "📭 Сигналы отфильтрованы как дубликаты.",
            f"Живые позиции Alpaca: {', '.join(sorted(live_positions_set)) or '—'}",
            f"Локальные позиции: {', '.join(sorted(local_positions_set)) or '—'}",
        ]
        send_telegram_message("\n".join(msg))
        return

    # 3) Торгуемость
    tradable_signals = {}
    skipped_nontradable = []
    for s, info in signals_no_dupes.items():
        ok, why = is_tradable_safe(s)
        if ok:
            tradable_signals[s] = info
        else:
            skipped_nontradable.append((s, why))

    if not tradable_signals:
        if FAILOPEN_IF_EMPTY:
            send_telegram_message(
                "⚠️ Alpaca assets вернул 0 торгуемых — включён fail-open-if-empty. Продолжаю с исходным набором."
            )
            tradable_signals = dict(signals_no_dupes)
        else:
            send_telegram_message(
                "📭 Нет торгуемых сигналов после фильтра Alpaca (tradable=false/status!=active)."
            )
            return

    if skipped_nontradable:
        try:
            msg = "⛔️ Пропущены неторгуемые тикеры:\n" + "\n".join(
                [
                    f"• {escape_markdown(sym)} — {escape_markdown(why)}"
                    for sym, why in skipped_nontradable
                ]
            )
            send_telegram_message(msg)
        except Exception as e:
            print(f"[WARN] telegram nontradable: {e}")

    # 4) Аккаунт, gross-каппа, бюджет
    account = fetch_account_live()
    gross_cap = determine_gross_cap(now_local)
    available_budget = calc_available_budget_from_bp_and_gross(account, gross_cap)

    # 5) Аллокация
    position_sizes = calculate_position_sizes(
        tradable_signals, available_budget, ALLOCATION_MODE
    )

    # Сообщение о распределении (по сигнал-цене; позже будет live-пересчёт)
    try:
        equity = account.get("equity", 0.0) or 0.0
        last_equity = account.get("last_equity", 0.0) or 0.0
        bp_show = account.get("dtbp", 0.0) or account.get("buying_power", 0.0) or 0.0
        dist_msg = [
            f"📊 *Распределение капитала* (режим: `{ALLOCATION_MODE}`)",
            f"Equity: ${equity:.2f} | LastEquity: ${last_equity:.2f}",
            f"BP(raw): ${bp_show:.2f} | GrossCap: ×{gross_cap:.2f}",
            f"Бюджет после ограничений: ${available_budget:.2f}",
            f"{'🔎 DRY_RUN: заказы НЕ отправляются.' if DRY_RUN else ''}",
        ]
        if not position_sizes:
            dist_msg.append(
                "_Подходящих позиций для покупки нет (после ограничений BP/Gross/Reserve)._"
            )
        for symbol, qty in position_sizes.items():
            info = tradable_signals[symbol]
            price = float(info.get("price", 0))
            score = float(info.get("score", 0))
            conf = float(info.get("confidence", 0))
            w = _priority_weight(info)
            sq_score, sq_lr, sq_so = _squeeze_flags(info)
            tail = ""
            if sq_lr:
                tail = f"  | squeeze: long_risk (x{SQUEEZE_SIZE_MULT:.2f})"
            elif sq_score > 0:
                tail = f"  | squeeze_score={sq_score:.0f}"
            dist_msg.append(
                f"• {escape_markdown(symbol)} x{qty} @ ${escape_num(price)}  (score={score:.1f}, conf={conf:.2f}, w={w:.1f}){tail}"
            )
        send_telegram_message("\n".join(dist_msg))
    except Exception as e:
        print(f"[WARN] Ошибка при формировании сообщения о распределении: {e}")

    # === AUDИТ: почему не купили некоторые тикеры (по аллокации) ===
    if ELIOS_ALLOCATION_AUDIT:
        try:
            bought_syms = set(position_sizes.keys())
            candidates = set(tradable_signals.keys())
            missed = sorted(candidates - bought_syms)
            if missed:
                tct = available_budget * (1.0 - RESERVE_CASH_PCT)
                per_cap = tct * MAX_ALLOC_PCT_PER_TICKER
                lines = []
                for s in missed:
                    w = None
                    try:
                        w = (
                            float(LAST_GPT_WEIGHTS.get(s))
                            if s in LAST_GPT_WEIGHTS
                            else None
                        )
                    except Exception:
                        w = None
                    price = float(tradable_signals.get(s, {}).get("price", 0) or 0)
                    if w is None or w <= 0:
                        lines.append(
                            f"• {escape_markdown(s)} — GPT вес {w if w is not None else '—'}"
                        )
                    else:
                        alloc_guess = min(max(0.0, tct * w), per_cap)
                        if price <= 0:
                            lines.append(f"• {escape_markdown(s)} — price<=0")
                        elif alloc_guess < price:
                            lines.append(
                                f"• {escape_markdown(s)} — округление до 0 (alloc≈${alloc_guess:.2f} < price ${price:.2f})"
                            )
                        else:
                            lines.append(
                                f"• {escape_markdown(s)} — недостаток остатка/кап (alloc≈${alloc_guess:.2f})"
                            )
                send_telegram_message(
                    "🔎 *Почему не купили (аллокация):*\n" + "\n".join(lines)
                )
        except Exception as e:
            print(f"[AUDIT] explain-missed error: {e}")

    # Логи/состояние
    log = {}
    open_positions = {}

    if TRADE_LOG_PATH.exists():
        try:
            with TRADE_LOG_PATH.open() as f:
                log = json.load(f)
        except Exception:
            log = {}
    if OPEN_POSITIONS_PATH.exists():
        try:
            with OPEN_POSITIONS_PATH.open() as f:
                open_positions = json.load(f)
        except Exception:
            open_positions = {}

    log = _coerce_trade_log(log)
    open_positions = _coerce_positions(open_positions)

    executed_any = False
    remaining_budget = available_budget
    slippage_skips = []  # [(sym, sp, sig_p, live_p)]
    budget_skips = []  # [(sym, need, have)]
    cap_skips = []  # тикеры, упёрлись в per-ticker cap/округление

    # 6) Отправка ордеров (SIMPLE/BRACKET). Перед отправкой — slippage-guard + динамика qty
    for symbol, qty_plan in position_sizes.items():
        info = tradable_signals.get(symbol, {})
        sig_price = float(info.get("price", 0))
        live_price = get_last_price(symbol) or sig_price
        if live_price <= 0:
            live_price = sig_price

        ok_slip, sp_abs, sp_signed, favorable = _slippage_ok(sig_price, live_price)
        if sp_abs >= SLIPPAGE_WARN_PCT and (not SLIPPAGE_DIRECTIONAL or not favorable):
            try:
                send_telegram_message(
                    f"⚠️ Slippage по {escape_markdown(symbol)} = {sp_signed:.2f}% "
                    f"(signal ${sig_price:.2f} → live ${live_price:.2f})"
                )
            except Exception:
                pass
        if not ok_slip:
            slippage_skips.append((symbol, sp_abs, sig_price, live_price))
            continue

        # Пересчёт qty под live-цену и текущий остаток бюджета
        qty = qty_plan
        total_cost = live_price * qty
        if total_cost > remaining_budget:
            qty_adj = int(remaining_budget // live_price)
            if qty_adj >= MIN_QTY_PER_TRADE:
                qty = qty_adj
                total_cost = live_price * qty
            else:
                budget_skips.append((symbol, total_cost, remaining_budget))
                continue

        if MIN_USD_PER_TRADE > 0 and total_cost < MIN_USD_PER_TRADE:
            cap_skips.append(symbol)
            continue

        ok, why = is_tradable_safe(symbol)
        if not ok:
            print(f"[SKIP] {symbol} — не торгуется ({why})")
            continue

        # squeeze-флаги
        sq_score, sq_lr, sq_so = _squeeze_flags(info)

        # Решение: BRACKET или SIMPLE
        use_bracket = OCO_ENABLE and sq_lr  # включаем OCO, когда риск squeeze для лонга
        order_response = None
        order_class_str = "simple"
        exit_mode = "sell_engine"
        bracket_info_tail = ""

        if use_bracket:
            # вычисляем TP/SL
            tp_price, sl_stop, sl_limit, meta = _compute_tp_sl_prices(info, live_price)
            try:
                order_response = submit_order_bracket_market(
                    symbol, qty, tp_price, sl_stop, sl_limit, time_in_force=ORDER_TIF
                )
                order_class_str = "bracket"
                exit_mode = "oco"
                bracket_info_tail = (
                    f"\n🎯 TP: ${tp_price:.2f} | 🛑 SL: ${sl_stop:.2f}  ({meta})"
                )
            except Exception as e:
                # фоллбэк в SIMPLE
                try:
                    send_telegram_message(
                        f"⚠️ BRACKET не удался по {escape_markdown(symbol)} — перехожу в SIMPLE.\nПричина: {escape_markdown(str(e)[:250])}"
                    )
                except Exception:
                    pass
                order_response = None
                use_bracket = False  # упадем в простой режим

        if not use_bracket:
            print(f"🚀 Покупка [SIMPLE]: {symbol} x{qty} @ {live_price}  -> exit: Sell")
            try:
                order_response = submit_order_simple(
                    symbol, qty, side="buy", type=ORDER_TYPE, time_in_force=ORDER_TIF
                )
            except Exception as e:
                error_msg = f"❌ Не удалось купить {symbol}: {e}"
                send_telegram_message(escape_markdown(error_msg))
                print(error_msg)
                # троттлинг и далее к следующему
                if ORDER_THROTTLE_MS > 0:
                    time.sleep(ORDER_THROTTLE_MS / 1000.0)
                continue

        status = (
            order_response.get("status", "??")
            if isinstance(order_response, dict)
            else "sent"
        )
        print(f"✅ Ответ Alpaca: {status}")
        timestamp = datetime.now(timezone.utc).isoformat()

        log_entry = {
            "symbol": symbol,
            "qty": qty,
            "price": live_price,
            "action": info.get("action", "BUY"),
            "timestamp": timestamp,
            "order_class": order_class_str,
            "exit": {"mode": exit_mode},
            "alpaca_response": order_response,
            "slippage_pct": sp,
            "sig_price": sig_price,
            "squeeze": {
                "score": sq_score,
                "long_risk": sq_lr,
                "short_opportunity": sq_so,
            },
        }
        log.setdefault(symbol, []).append(log_entry)

        open_positions[symbol] = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": live_price,
            "timestamp": timestamp,
            "managed_by": ("oco" if order_class_str == "bracket" else "sell"),
            "status": "active",
            "confirmed_by_alpaca": (not DRY_RUN),
        }

        # Телега
        sq_tail = ""
        if sq_lr:
            sq_tail = f"\n⚠️ Squeeze: *LONG RISK* (size×{SQUEEZE_SIZE_MULT:.2f}, score={sq_score:.0f})"
        elif sq_score > 0:
            sq_tail = f"\nSqueeze score: {sq_score:.0f}"

        msg = (
            f"{'🧷' if order_class_str == 'bracket' else '📂'} *Покупка подтверждена* [{order_class_str.upper()}]\n"
            f"📌 {escape_markdown(symbol)} x{qty} @ \\${escape_num(live_price)} "
            f"(slip={sp_signed:.2f}%)\n🧠 Exit: {'OCO (TP/SL)' if order_class_str=='bracket' else 'Sell (TP/SL отключены)'}"
            f"{sq_tail}{bracket_info_tail}"
        )
        send_telegram_message(msg)

        executed_any = True
        remaining_budget -= live_price * qty

        # троттлинг
        if ORDER_THROTTLE_MS > 0:
            time.sleep(ORDER_THROTTLE_MS / 1000.0)

        if remaining_budget <= available_budget * 0.01:
            break

    with TRADE_LOG_PATH.open("w") as f:
        json.dump(log, f, indent=2)
    with OPEN_POSITIONS_PATH.open("w") as f:
        json.dump(open_positions, f, indent=2)
    with DEBUG_LOG_PATH.open("w") as f:
        json.dump([], f, indent=2)

    # Итоговые уведомления и аудит причин пропуска при отправке
    if slippage_skips or budget_skips or cap_skips:
        lines = []
        if slippage_skips:
            lines.append(
                "⛔️ *Пропуск из-за slippage ухудшения (>{:.2f}%):*".format(
                    SLIPPAGE_REJECT_PCT
                )
            )
            for s, sp, sigp, livep in slippage_skips:
                lines.append(
                    f"• {escape_markdown(s)} — {sp_signed:.2f}% (sig ${sigp:.2f} → live ${livep:.2f})"
                )
        if budget_skips:
            lines.append("⛔️ *Недостаточно бюджета на live-цене:*")
            for s, need, have in budget_skips:
                lines.append(
                    f"• {escape_markdown(s)} — нужно ${need:.2f}, остаток ${have:.2f}"
                )
        if cap_skips:
            lines.append(
                "⛔️ *Ниже минимума сделки/упёрлись в кап:* "
                + ", ".join(escape_markdown(x) for x in cap_skips)
            )
        try:
            send_telegram_message("\n".join(lines))
        except Exception:
            pass

    if not executed_any:
        send_telegram_message("📭 *Цикл завершён*\nНовых покупок не было.")

    print("\n✅ Все ордера обработаны. Лог:", TRADE_LOG_PATH)
    print("📌 Открытые позиции:", OPEN_POSITIONS_PATH)


if __name__ == "__main__":
    main()
