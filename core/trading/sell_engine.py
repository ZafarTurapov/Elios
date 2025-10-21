from core.utils.alpaca_headers import alpaca_headers

# -*- coding: utf-8 -*-
import sys
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI


def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


from core.utils.paths import TRADE_LOG_PATH
from core.connectors.alpaca_connector import get_positions_with_pnl, submit_order
from core.utils.telegram import send_telegram_message

# === OpenAI client bootstrap (ENV-driven, safe) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    client  # noqa: F821
except NameError:
    try:
        _tmp = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    except Exception as _e:
        _tmp = None
    client = _tmp  # global fallback


def _get_openai_client():
    """Lazily return OpenAI client or None if not configured."""
    global client
    if client is None and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            client = None
    return client


# === end OpenAI bootstrap ===


# ==========================
# Константы / пути
# ==========================
APCA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or ""
APCA_SEC = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or ""
HEADERS = alpaca_headers()


def _alpaca_headers():
    return dict(HEADERS)


ALPACA_BASE_URL = (
    os.getenv("ALPACA_BASE_URL")
    or os.getenv("APCA_API_BASE_URL")
    or "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets/v2"

DEBUG = True
DEBUG_LOG = "/root/stockbot/logs/sell_debug.json"

POSITIONS_PATH = os.path.join(os.path.dirname(__file__), "open_positions.json")
PNL_TRACKER_PATH = Path("core/trading/pnl_tracker.json")

ROOT = Path("/root/stockbot")
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CANON_TRADE_LOG_NDJSON = LOGS_DIR / "trade_log.json"

# анти-дубликат: per-symbol cooldown
COOLDOWN_DIR = LOGS_DIR / "sell.cooldown"
SELL_COOLDOWN_SEC = int(os.getenv("SELL_COOLDOWN_SEC", "90"))

# heartbeat
HEARTBEAT = LOGS_DIR / "sell.heartbeat"

FORCE_CLOSE = "--force" in sys.argv or os.environ.get("FORCE_SELL") == "1"
SELL_DRY_RUN = os.getenv("ELIOS_SELL_DRY_RUN", "0") == "1"
DEFAULT_TAKE_PROFIT = 0.05  # +5% (фракция)
DEFAULT_STOP_LOSS = -0.03  # -3% (фракция)

# ==========================
# Grace-window (удержание молодых)
# ==========================
GRACE_ENABLED = os.getenv("GRACE_ENABLED", "1") == "1"
GRACE_MINUTES = int(os.getenv("GRACE_MINUTES", "12"))
MIN_VOL_RATIO_HOLD = float(os.getenv("MIN_VOL_RATIO_HOLD", "1.10"))
MIN_BODY_PCT_HOLD = float(os.getenv("MIN_BODY_PCT_HOLD", "0.20"))
RSI_HOLD_FLOOR = float(os.getenv("RSI_HOLD_FLOOR", "50"))
HOLD_RULES_REQUIRE = int(os.getenv("HOLD_RULES_REQUIRE", "2"))
GRACE_TIMEFRAME = os.getenv("GRACE_TIMEFRAME", "5Min")
GRACE_BARS_LIMIT = int(os.getenv("GRACE_BARS_LIMIT", "60"))

TRADING_DIR = Path(__file__).resolve().parent
META_PATH = TRADING_DIR / "position_meta.json"

# ==========================
# Микро-паттерн удержания (держать сильный импульс дольше)
# ==========================
MICRO_HOLD_ENABLED = os.getenv("MICRO_HOLD_ENABLED", "1") == "1"
MICRO_TIMEFRAME = os.getenv("MICRO_TIMEFRAME", GRACE_TIMEFRAME)
MICRO_BARS_LIMIT = int(os.getenv("MICRO_BARS_LIMIT", "120"))

# Диапазон, где имеет смысл «тянуть» (в % PnL)
MICRO_PNL_MIN_PCT = float(os.getenv("MICRO_PNL_MIN_PCT", "1.5"))
MICRO_PNL_MAX_PCT = float(os.getenv("MICRO_PNL_MAX_PCT", "12.0"))

# Силовые признаки
MICRO_RSI_FLOOR = float(os.getenv("MICRO_RSI_FLOOR", "55"))
MICRO_VOL_RATIO = float(os.getenv("MICRO_VOL_RATIO", "1.20"))
MICRO_BODY_PCT = float(os.getenv("MICRO_BODY_PCT", "0.20"))
MICRO_RULES_REQUIRE = int(os.getenv("MICRO_RULES_REQUIRE", "3"))

# Трейл-безопасность
TRAIL_LOOKBACK_BARS = int(os.getenv("TRAIL_LOOKBACK_BARS", "20"))
TRAIL_MAX_DRAWDOWN_PCT = float(
    os.getenv("TRAIL_MAX_DRAWDOWN_PCT", "1.8")
)  # 1.8% от локального макс.

# ==========================
# GPT: мягкие правила для минусовых продаж
# ==========================
GPT_NEG_MIN_PNL_PCT = float(os.getenv("GPT_NEG_MIN_PNL_PCT", "-2.0"))
GPT_NEG_MAX_PNL_PCT = float(os.getenv("GPT_NEG_MAX_PNL_PCT", "-0.4"))
GPT_NEG_MIN_AGE_MIN = float(os.getenv("GPT_NEG_MIN_AGE_MIN", "5"))
GPT_NEG_RULES_REQUIRE = int(os.getenv("GPT_NEG_RULES_REQUIRE", "3"))

# === optional deps для индикаторов ===
try:
    import pandas as pd
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    pd = None
    RSIIndicator = None
    EMAIndicator = None
    AverageTrueRange = None


# ==========================
# Утилиты
# ==========================
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_round(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return None


def _ensure_logs_dir():
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _write_heartbeat(stage: str = "tick"):
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        data = {"ts": time.time(), "iso": _iso_utc_now(), "stage": stage}
        HEARTBEAT.write_text(
            json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as e:
        print(f"[HB WARN] {e}")


def _headers():
    return alpaca_headers(content_json=True)


def _load_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data):
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"[sell_engine] WARN: save_json({path}) failed: {e}")


def _get_bars(symbol: str, timeframe: str, limit: int):
    if pd is None:
        return None
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": str(limit), "adjustment": "all"}
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        bars = j.get("bars", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
            inplace=True,
        )
        return df
    except Exception as e:
        print(f"[sell_engine] WARN: get_bars({symbol}) failed: {e}")
        return None


def _calc_age_minutes(first_seen_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(first_seen_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return 9999.0


def log_debug(symbol, data):
    if not DEBUG:
        return
    _ensure_logs_dir()
    try:
        log_entry = {"timestamp": _iso_utc_now(), "symbol": symbol, "details": data}
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")


# cooldown helpers
def _cooldown_ok(symbol: str) -> bool:
    """True — можно продавать (кулдаун истёк)."""
    try:
        COOLDOWN_DIR.mkdir(parents=True, exist_ok=True)
        f = COOLDOWN_DIR / f"{symbol.upper()}.json"
        if not f.exists():
            return True
        j = json.loads(f.read_text(encoding="utf-8") or "{}")
        ts = float(j.get("ts", 0.0))
        return (time.time() - ts) >= SELL_COOLDOWN_SEC
    except Exception:
        return True


def _mark_cooldown(
    symbol: str, qty: int, reason: str = "", order_id: str | None = None
):
    """Фиксируем попытку продажи, чтобы соседний тик не отправил второй ордер."""
    try:
        COOLDOWN_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "ts": time.time(),
            "iso": _iso_utc_now(),
            "symbol": symbol,
            "qty": int(qty),
            "reason": reason or "",
            "order_id": order_id,
        }
        (COOLDOWN_DIR / f"{symbol.upper()}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[CD WARN] {e}")


# ==========================
# Grace логика
# ==========================
def should_hold_by_grace(symbol: str, age_min: float) -> bool:
    if not GRACE_ENABLED:
        return False
    if age_min >= GRACE_MINUTES:
        return False
    if pd is None or RSIIndicator is None or EMAIndicator is None:
        return False

    df = _get_bars(symbol, GRACE_TIMEFRAME, GRACE_BARS_LIMIT)
    if df is None or len(df) < 25:
        return False

    try:
        ema20 = EMAIndicator(close=df["close"], window=20).ema_indicator()
        rsi14 = RSIIndicator(close=df["close"], window=14).rsi()
    except Exception:
        return False

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_vol = float(last["volume"] or 0)

    body_pct = ((last_close - last_open) / max(last_open, 1e-9)) * 100.0
    vol_ma = float(df["volume"].tail(20).mean() or 0.0)
    vol_ratio = (last_vol / vol_ma) if vol_ma > 0 else 1.0
    ema_ok = last_close >= float(ema20.iloc[-1] or 0.0)
    rsi_last = float(rsi14.iloc[-1] or 0.0)
    rsi_prev = float(rsi14.iloc[-3] or rsi_last)

    strength = 0
    if ema_ok:
        strength += 1
    if rsi_last >= RSI_HOLD_FLOOR and (rsi_last - rsi_prev) >= -1.0:
        strength += 1
    if vol_ratio >= MIN_VOL_RATIO_HOLD:
        strength += 1
    if body_pct >= MIN_BODY_PCT_HOLD:
        strength += 1

    weakness = False
    if rsi_last < 40:
        weakness = True
    try:
        if last_close < float(df["close"].iloc[-3] or last_close) * 0.985:
            weakness = True
    except Exception:
        pass

    if weakness:
        return False

    return strength >= HOLD_RULES_REQUIRE


# ==========================
# Микро-паттерн удержания
# ==========================
def should_hold_by_micro(symbol: str, pnl_pct: float) -> bool:
    """
    True — держать сильный импульс дольше (отменяем только мягкие продажи).
    """
    if not MICRO_HOLD_ENABLED:
        return False
    if pd is None or RSIIndicator is None or EMAIndicator is None:
        return False
    if not (MICRO_PNL_MIN_PCT <= pnl_pct <= MICRO_PNL_MAX_PCT):
        return False

    df = _get_bars(symbol, MICRO_TIMEFRAME, MICRO_BARS_LIMIT)
    if df is None or len(df) < 40:
        return False

    try:
        ema20 = EMAIndicator(close=df["close"], window=20).ema_indicator()
        rsi14 = RSIIndicator(close=df["close"], window=14).rsi()
        atr14 = (
            AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=14
            ).average_true_range()
            if AverageTrueRange
            else None
        )
    except Exception:
        return False

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_vol = float(last["volume"] or 0)

    ema_ok = last_close >= float(ema20.iloc[-1] or 0.0)
    rsi_last = float(rsi14.iloc[-1] or 0.0)
    rsi_prev = float(rsi14.iloc[-4] or rsi_last)
    rsi_up = (rsi_last >= MICRO_RSI_FLOOR) and (rsi_last >= rsi_prev - 0.5)

    vol_ma20 = float(df["volume"].tail(20).mean() or 0.0)
    vol_ratio = (last_vol / vol_ma20) if vol_ma20 > 0 else 1.0

    body_pct = ((last_close - last_open) / max(last_open, 1e-9)) * 100.0
    new_high = last_close >= float(df["close"].rolling(15).max().iloc[-1] or last_close)

    strength = 0
    if ema_ok:
        strength += 1
    if rsi_up:
        strength += 1
    if vol_ratio >= MICRO_VOL_RATIO:
        strength += 1
    if body_pct >= MICRO_BODY_PCT:
        strength += 1
    if new_high:
        strength += 1

    # защита: локальная просадка от пика за N баров
    try:
        peak_close = float(df["close"].tail(TRAIL_LOOKBACK_BARS).max())
        drawdown_pct = (last_close / peak_close - 1.0) * 100.0
        if drawdown_pct <= -TRAIL_MAX_DRAWDOWN_PCT:
            return False
    except Exception:
        pass

    return strength >= MICRO_RULES_REQUIRE


# ==========================
# Мягкий гард минусовых GPT-продаж
# ==========================
def gpt_negative_guard(symbol: str, pnl_pct: float, age_min: float) -> bool:
    """
    Разрешаем минусовой GPT-sell только при явной слабости.
    """
    if age_min < GPT_NEG_MIN_AGE_MIN:
        return False
    if not (GPT_NEG_MIN_PNL_PCT <= pnl_pct <= GPT_NEG_MAX_PNL_PCT):
        return False
    if pd is None or RSIIndicator is None or EMAIndicator is None:
        return False

    df = _get_bars(symbol, GRACE_TIMEFRAME, 60)
    if df is None or len(df) < 25:
        return False

    try:
        ema20 = EMAIndicator(close=df["close"], window=20).ema_indicator()
        rsi14 = RSIIndicator(close=df["close"], window=14).rsi()
    except Exception:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    last_close = float(last["close"])
    prev_close = float(prev["close"])
    last_high = float(last["high"])
    prev_high = float(prev["high"])
    last_low = float(last["low"])
    prev_low = float(prev["low"])
    last_vol = float(last["volume"] or 0.0)
    vol_ma20 = float(df["volume"].tail(20).mean() or 0.0)

    cond1 = last_close < float(ema20.iloc[-1] or last_close + 1)
    rsi_last = float(rsi14.iloc[-1] or 50.0)
    rsi_prev = float(rsi14.iloc[-2] or rsi_last)
    cond2 = (rsi_last < 48.0) and (rsi_last <= rsi_prev + 0.1)
    cond3 = last_close <= prev_low * 0.998
    cond4 = (last_high < prev_high) and (last_low < prev_low)
    cond5 = (vol_ma20 > 0 and last_vol / vol_ma20 >= 1.10) and (last_close < prev_close)

    score = sum([cond1, cond2, cond3, cond4, cond5])
    return score >= GPT_NEG_RULES_REQUIRE


# ==========================
# Логи/репортинг
# ==========================
def append_to_pnl_tracker(symbol, qty, entry_price, exit_price):
    try:
        pnl_data = []
        if PNL_TRACKER_PATH.exists():
            with PNL_TRACKER_PATH.open() as f:
                try:
                    pnl_data = json.load(f)
                    if not isinstance(pnl_data, list):
                        pnl_data = []
                except Exception:
                    pnl_data = []
        pnl_data.append(
            {
                "timestamp": _iso_utc_now(),
                "symbol": symbol,
                "qty": qty,
                "entry": _safe_round(entry_price, 2),
                "exit": _safe_round(exit_price, 2),
                "invested": _safe_round(entry_price * qty, 2),
                "revenue": _safe_round(exit_price * qty, 2),
            }
        )
        with PNL_TRACKER_PATH.open("w") as f:
            json.dump(pnl_data, f, indent=2)
        print(f"📈 Записано в pnl_tracker: {symbol}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать в pnl_tracker: {e}")


def append_to_trade_log(symbol, qty, entry_price, exit_price, reason):
    try:
        trade_log = []
        if TRADE_LOG_PATH.exists():
            with TRADE_LOG_PATH.open() as f:
                try:
                    trade_log = json.load(f)
                    if not isinstance(trade_log, list):
                        trade_log = []
                except Exception:
                    trade_log = []
        trade_log.append(
            {
                "timestamp": _iso_utc_now(),
                "side": "SELL",
                "action": "SELL",
                "symbol": symbol,
                "qty": qty,
                "entry": _safe_round(entry_price, 2),
                "exit": _safe_round(exit_price, 2),
                "pnl": _safe_round((exit_price - entry_price) * qty, 2),
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "exit_date": datetime.now().strftime("%Y-%m-%d"),
                "reason": reason or "",
            }
        )
        with TRADE_LOG_PATH.open("w") as f:
            json.dump(trade_log, f, indent=2)
        print(f"📝 Записано в trade_log (list JSON): {symbol}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать в trade_log: {e}")


def append_trade_event_ndjson(
    side,
    symbol,
    qty,
    entry_price,
    exit_price,
    pnl,
    reason,
    order_id=None,
    source="sell_engine",
):
    try:
        rec = {
            "timestamp": _iso_utc_now(),
            "side": str(side).upper(),
            "symbol": str(symbol).upper(),
            "qty": qty,
            "entry": _safe_round(entry_price, 4),
            "exit": _safe_round(exit_price, 4),
            "pnl": _safe_round(pnl, 4),
            "reason": reason or "",
            "order_id": order_id,
            "source": source,
            "module": "sell_engine",
        }
        with open(CANON_TRADE_LOG_NDJSON, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"📒 NDJSON trade_log +1: {rec['symbol']} {rec['side']} qty={qty}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать в NDJSON trade_log: {e}")


# ==========================
# GPT-подтверждение
# ==========================
def gpt_confirm_sell(symbol, entry_price, current_price, change_pct, pnl):
    prompt = f"""
Ты профессиональный дейтрейдер. Твоя задача — определить, стоит ли зафиксировать прибыль.

Акция: {symbol}
Цена входа: ${entry_price:.2f}
Текущая цена: ${current_price:.2f}
Изменение: {round(change_pct * 100, 2)}%
PnL: {pnl:+.2f}$

Ответь строго: ДА (если стоит продать) или НЕТ (если удерживать).
""".strip()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=50,
        )
        reply = response.choices[0].message.content.strip().upper()
        print(f"[GPT] {reply}")
        return "ДА" in reply
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return False


# ==========================
# Основной цикл
# ==========================
def main():
    _ensure_logs_dir()
    _write_heartbeat("start")

    try:
        alpaca_positions = get_positions_with_pnl()
    except Exception as e:
        print(f"[ERROR] Не удалось получить позиции с Alpaca: {e}")
        send_telegram_message(f"🚫 Ошибка получения позиций Alpaca:\n{str(e)}")
        return

    meta = _load_json(META_PATH, default={})
    meta_changed = False

    updated = {}
    sold_any = False

    for symbol, pos in alpaca_positions.items():
        if not all(k in pos for k in ("qty", "entry_price", "current_price")):
            print(f"[SKIP] Пропуск {symbol}: отсутствуют ключевые поля → {pos}")
            continue

        # Сторона и базовые поля
        raw_qty = pos["qty"]
        try:
            qty_num = float(raw_qty)
        except Exception:
            qty_num = float(int(raw_qty)) if str(raw_qty).isdigit() else 0.0

        pos_side = str(pos.get("side", "")).lower().strip()
        if pos_side not in {"long", "short"}:
            if qty_num < 0 or float(pos.get("market_value", 0) or 0) < 0:
                pos_side = "short"
            else:
                pos_side = "long"

        qty_abs = abs(int(qty_num)) if qty_num else 0
        entry_price = float(pos["entry_price"])
        current_price = float(pos["current_price"])

        # Фракционный профит
        if pos_side == "short":
            change_pct = (entry_price - current_price) / max(entry_price, 1e-9)
            pnl_preview = round((entry_price - current_price) * qty_abs, 2)
        else:
            change_pct = (current_price - entry_price) / max(entry_price, 1e-9)
            pnl_preview = round((current_price - entry_price) * qty_abs, 2)

        pnl_pct = change_pct * 100.0

        log_debug(
            symbol,
            {
                "pos_side": pos_side,
                "qty": qty_abs,
                "entry_price": entry_price,
                "current_price": current_price,
                "change_pct%": round(pnl_pct, 2),
                "pnl_preview$": pnl_preview,
            },
        )

        # Метаданные (для возраста)
        m = meta.get(symbol, {})
        if "first_seen" not in m:
            m["first_seen"] = _iso_utc_now()
            meta_changed = True
        age_min = _calc_age_minutes(m["first_seen"])
        note = None  # причина удержания для телеги

        # Базовое решение
        if FORCE_CLOSE:
            sell = True
            reason = "FORCE_CLOSE"
        elif change_pct >= DEFAULT_TAKE_PROFIT:
            sell = True
            reason = "TAKE_PROFIT_5%"
        elif change_pct <= DEFAULT_STOP_LOSS:
            sell = True
            reason = "STOP_LOSS_-3%"
        elif gpt_confirm_sell(
            symbol, entry_price, current_price, change_pct, pnl_preview
        ):
            if change_pct < 0:
                if gpt_negative_guard(symbol, pnl_pct, age_min):
                    sell = True
                    reason = "GPT_CONFIRM_NEG"
                else:
                    sell = False
                    reason = None
                    note = "NEG_GUARD"
            else:
                sell = True
                reason = "GPT_CONFIRM"
        else:
            sell = False
            reason = None

        # Grace: молодая и сильная — удерживаем (только для мягких причин TP/GPT)
        if (
            sell
            and reason in {"TAKE_PROFIT_5%", "GPT_CONFIRM"}
            and should_hold_by_grace(symbol, age_min)
        ):
            print(
                f"[sell_engine] HOLD(grace): {symbol} age={age_min:.1f}m pnl={pnl_preview:+.2f}$"
            )
            sell = False
            reason = None
            note = "GRACE_HOLD"

        # Микро: сильный импульс — удерживаем (только для TP/GPT)
        if (
            sell
            and reason in {"TAKE_PROFIT_5%", "GPT_CONFIRM"}
            and should_hold_by_micro(symbol, pnl_pct)
        ):
            print(
                f"[sell_engine] HOLD(micro):  {symbol} pnl={pnl_pct:.2f}% (структура сильная)"
            )
            sell = False
            reason = None
            note = "MICRO_HOLD"

        if sell and qty_abs > 0:
            # cooldown — не дублируем SELL по тикеру в пределах окна
            if not _cooldown_ok(symbol):
                print(f"[sell_engine] SKIP by cooldown: {symbol}")
                updated[symbol] = {
                    "qty": qty_abs,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "side": pos_side,
                }
                meta[symbol] = m
                continue

            try:
                side_out = "sell" if pos_side == "long" else "buy"

                # LIVE re-check количества у брокера, чтобы не перепродать
                try:
                    latest = get_positions_with_pnl()
                    pos_live = latest.get(symbol)
                    live_qty = (
                        int(abs(float(pos_live["qty"])))
                        if pos_live and "qty" in pos_live
                        else 0
                    )
                except Exception:
                    live_qty = qty_abs

                if live_qty <= 0:
                    print(f"[sell_engine] ABORT: no live qty for {symbol}")
                    sell = False
                    reason = None
                    updated[symbol] = {
                        "qty": 0,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "side": pos_side,
                    }
                    meta[symbol] = m
                    continue

                if live_qty < qty_abs:
                    print(f"[sell_engine] Adjust qty: {symbol} {qty_abs} -> {live_qty}")
                qty_abs = live_qty

                submit_resp = submit_order(
                    symbol=symbol,
                    qty=qty_abs,
                    side=side_out,
                    type="market",
                    time_in_force="gtc",
                )
                # отмечаем кулдаун сразу (даже если fill ещё не пришёл)
                try:
                    tmp_order_id = (
                        submit_resp.get("id")
                        if isinstance(submit_resp, dict)
                        else getattr(submit_resp, "id", None)
                    )
                except Exception:
                    tmp_order_id = None
                _mark_cooldown(symbol, qty_abs, reason or "SELL_ATTEMPT", tmp_order_id)

                # читаем fill
                try:
                    order_id = (
                        submit_resp.get("id")
                        if isinstance(submit_resp, dict)
                        else getattr(submit_resp, "id", None)
                    )
                except Exception:
                    order_id = None

                fill_info = {
                    "status": "unknown",
                    "filled_qty": 0.0,
                    "filled_avg_price": 0.0,
                }
                if order_id:
                    fill_info = wait_for_fill(order_id, timeout_s=90, poll_interval=1.5)

                status = fill_info.get("status")
                filled_qty = int(float(fill_info.get("filled_qty", 0.0) or 0))
                exec_price = float(fill_info.get("filled_avg_price", 0.0) or 0.0)

                if status == "filled" and filled_qty > 0 and exec_price > 0:
                    if pos_side == "short":
                        pnl = round((entry_price - exec_price) * filled_qty, 2)
                    else:
                        pnl = round((exec_price - entry_price) * filled_qty, 2)

                    append_to_pnl_tracker(symbol, filled_qty, entry_price, exec_price)
                    append_to_trade_log(
                        symbol, filled_qty, entry_price, exec_price, reason
                    )
                    append_trade_event_ndjson(
                        side="SELL" if pos_side == "long" else "BUY_TO_COVER",
                        symbol=symbol,
                        qty=filled_qty,
                        entry_price=entry_price,
                        exit_price=exec_price,
                        pnl=pnl,
                        reason=reason,
                        order_id=order_id,
                        source="sell_engine",
                    )
                    msg = format_position_message(
                        symbol,
                        filled_qty,
                        entry_price,
                        exec_price,
                        change_pct,
                        pnl,
                        reason,
                    )
                    send_telegram_message(msg)
                    sold_any = True

                    # частичное исполнение — оставляем остаток
                    remaining = max(qty_abs - filled_qty, 0)
                    if remaining > 0:
                        updated[symbol] = {
                            "qty": remaining,
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "side": pos_side,
                        }
                    # если остатка нет — позиция локально удаляется до следующего sync
                else:
                    msg = (
                        f"⚠️ Ордер не исполнен\n"
                        f"📌 {symbol} ({pos_side})\n"
                        f"🧾 order_id: {order_id}\n"
                        f"📋 status: {status}\n"
                        f"ℹ️ filled_qty={filled_qty}, filled_avg_price={exec_price}"
                    )
                    send_telegram_message(msg)
                    print(msg)
                    updated[symbol] = {
                        "qty": qty_abs,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "side": pos_side,
                    }

            except Exception as e:
                msg = (
                    f"⚠️ Ошибка продажи\n"
                    f"📌 {symbol} ({pos_side})\n"
                    f"📉 Предварит. PnL: {pnl_preview:+.2f}$\n"
                    f"🚫 {str(e)}"
                )
                send_telegram_message(msg)
                print(f"[ERROR] Не удалось продать {symbol}: {e}")

        else:
            # Удерживаем позицию
            msg = format_position_message(
                symbol,
                qty_abs,
                entry_price,
                current_price,
                change_pct,
                pnl_preview,
                note=note,
            )
            send_telegram_message(msg)
            updated[symbol] = {
                "qty": qty_abs,
                "entry_price": entry_price,
                "current_price": current_price,
                "side": pos_side,
            }

        meta[symbol] = m

    # сохраняем локальное зеркало открытых
    try:
        with open(POSITIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
    except Exception as e:
        print(f"[WARN] Не удалось обновить {POSITIONS_PATH}: {e}")

    # meta сохраняем
    _save_json(META_PATH, meta)

    if not sold_any:
        send_telegram_message("📊 Все позиции удерживаются. Продаж не выполнено.")
    _write_heartbeat("end")


# ==========================
# Вспомогательная: ожидание fill
# ==========================
def get_order_status(order_id: str) -> dict:
    try:
        url = f"{ALPACA_BASE_URL}/v2/orders/{order_id}"
        resp = requests.get(url, headers=_headers(), timeout=10)
        if 200 <= resp.status_code < 300:
            return resp.json()
        else:
            print(
                f"[WARN] get_order_status {order_id}: HTTP {resp.status_code} {resp.text}"
            )
            return {}
    except Exception as e:
        print(f"[WARN] get_order_status error: {e}")
        return {}


def wait_for_fill(
    order_id: str, timeout_s: int = 90, poll_interval: float = 1.5
) -> dict:
    start = time.time()
    last_status = None
    while True:
        od = get_order_status(order_id)
        status = str(od.get("status", "")).lower()
        if status and status != last_status:
            print(f"[ORDER] {order_id} status → {status}")
            last_status = status
        if status == "filled":
            fq = float(od.get("filled_qty", 0) or 0)
            fav = float(od.get("filled_avg_price", 0) or 0)
            return {
                "status": status,
                "filled_qty": fq,
                "filled_avg_price": fav,
                "raw": od,
            }
        if status in {"canceled", "expired", "rejected", "stopped", "suspended"}:
            return {
                "status": status,
                "filled_qty": 0.0,
                "filled_avg_price": 0.0,
                "raw": od,
            }
        if time.time() - start > timeout_s:
            return {
                "status": "timeout",
                "filled_qty": float(od.get("filled_qty", 0) or 0),
                "filled_avg_price": float(od.get("filled_avg_price", 0) or 0),
                "raw": od,
            }
        time.sleep(poll_interval)


# ==========================
# Формирование сообщений
# ==========================
def format_position_message(
    symbol,
    qty,
    entry_price,
    current_price,
    change_pct,
    pnl,
    reason=None,
    note: Optional[str] = None,
):
    if reason:
        return (
            f"💰 Продажа: {symbol} x{qty} @ ${current_price:.2f}\n"
            f"📈 Куплено по: ${entry_price:.2f} → Продано: ${current_price:.2f}\n"
            f"📊 Изменение: {round(change_pct * 100, 2)}% | PnL: {pnl:+.2f}$\n"
            f"🧠 Причина: {reason}"
        )
    else:
        return (
            f"📊 {symbol} удерживается\n"
            f"📈 Куплено по: ${entry_price:.2f} → Сейчас: ${current_price:.2f}\n"
            f"📊 Изменение: {round(change_pct * 100, 2)}% | PnL: {pnl:+.2f}$\n"
            f"🤖 GPT/Правила: удержание" + (f" — {note}" if note else "")
        )


if __name__ == "__main__":
    main()
