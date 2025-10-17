# -*- coding: utf-8 -*-
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# --- sys.path safety
if "/root/stockbot" not in sys.path:
    sys.path.insert(0, "/root/stockbot")

# --- Telegram helper (должен существовать в проекте)
try:
    from core.utils.telegram import send_telegram_message
except Exception:

    def send_telegram_message(text: str):
        # мягкий фоллбэк, чтобы не падать без тг
        print(f"[TG] {text}")


# --- Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")


def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY or "",
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
    }


# --- Config / paths
CANDIDATES_PATH = "core/trading/candidates.json"
SIGNALS_PATH = "core/trading/signals.json"
REJECTED_PATH = "core/trading/rejected.json"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BYPASS_FILTERS = os.getenv("ELIOS_NO_FILTERS", "0") == "1"
MAX_ATR_PCT = float(os.getenv("ELIOS_MAX_ATR_PCT", "8.0"))
MAX_VOLATILITY = float(os.getenv("ELIOS_MAX_VOLATILITY", "5.0"))
MIN_VOLUME_TREND = float(os.getenv("ELIOS_MIN_VOLUME_TREND", "0.10"))
MIN_RISK_SCORE = float(os.getenv("ELIOS_MIN_RISK_SCORE", "10.0"))  # простая заглушка
MIN_BULLISH_BODY = float(os.getenv("ELIOS_MIN_BULLISH_BODY", "1.0"))
MIN_GAP_UP = float(os.getenv("ELIOS_MIN_GAP_UP", "1.0"))
MIN_VOLUME_RATIO = float(os.getenv("ELIOS_MIN_VOLUME_RATIO", "0.95"))
MODEL_SCORE_MIN = float(os.getenv("ELIOS_MODEL_SCORE_MIN", "55.0"))

TARGET_MIN = int(os.getenv("ELIOS_TARGET_MIN", "3"))
TARGET_MAX = int(os.getenv("ELIOS_TARGET_MAX", "4"))

DEBUG_MODE = os.getenv("ELIOS_DEBUG", "0") == "1"


def safe_squeeze(series):
    try:
        if hasattr(series, "ndim") and series.ndim > 1:
            return series.squeeze()
    except Exception:
        pass
    return series


def pct(a, b):
    try:
        a = float(a)
        b = float(b)
        return 100.0 * (a - b) / (b if b != 0 else 1e-9)
    except Exception:
        return 0.0


def is_tradable(symbol: str) -> bool:
    """Не блокируем при ошибках Alpaca — fail-open."""
    try:
        r = requests.get(
            f"{ALPACA_BASE_URL}/v2/assets/{symbol}",
            headers=_alpaca_headers(),
            timeout=7,
        )
        if r.status_code != 200:
            return True
        a = r.json() or {}
        return bool(a.get("tradable", False)) and a.get("status", "active") == "active"
    except Exception:
        return True


def last_trade_price(symbol: str, fallback_close: float | None = None) -> float | None:
    """Берём last trade из Alpaca, иначе отдаём fallback_close."""
    try:
        r = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest",
            params={"feed": os.getenv("ELIOS_ALPACA_FEED", "iex")},
            headers=_alpaca_headers(),
            timeout=7,
        )
        if r.status_code == 200:
            t = (r.json() or {}).get("trade") or {}
            p = float(t.get("p") or 0)
            return (
                p if p > 0 else (fallback_close if (fallback_close or 0) > 0 else None)
            )
    except Exception:
        pass
    return fallback_close if (fallback_close or 0) > 0 else None


def fetch_history(symbol: str, period="45d"):
    df = yf.download(
        symbol, period=period, interval="1d", progress=False, auto_adjust=False
    )
    if df is None or getattr(df, "empty", True):
        return None
    return df


def smart_score(alpha_score, volume_ratio, rsi, atr_pct, volatility):
    # суррогат модельного скора 0..100
    sa = max(0.0, min(1.0, float(alpha_score)))
    sv = max(0.0, min(1.0, float(volume_ratio) / 2.0))
    sr = max(0.0, min(1.0, float(rsi) / 100.0))
    satr = max(0.0, 1.0 - min(1.0, float(atr_pct) / 8.0))
    svol = max(0.0, 1.0 - min(1.0, float(volatility) / 8.0))
    return float(
        (0.35 * sa + 0.25 * sv + 0.15 * sr + 0.15 * satr + 0.10 * svol) * 100.0
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    # простые «гейткиперы»
    wd = datetime.now(timezone.utc).weekday()
    if wd >= 5 and os.getenv("ELIOS_FORCE_OPEN", "0") != "1":
        print("⛔ Выходной — пропуск.")
        return

    # читаем кандидатов
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        tickers = [str(t).upper() for t in json.load(f)]

    print(f"📅 Проверяем {len(tickers)} тикеров...")
    print(f"🚧 BYPASS_FILTERS={'ON' if BYPASS_FILTERS else 'OFF'}")
    print(
        "⚙️ Пороги:",
        json.dumps(
            dict(
                MAX_ATR_PCT=MAX_ATR_PCT,
                MAX_VOLATILITY=MAX_VOLATILITY,
                MIN_VOLUME_TREND=MIN_VOLUME_TREND,
                MIN_RISK_SCORE=MIN_RISK_SCORE,
                MIN_BULLISH_BODY=MIN_BULLISH_BODY,
                MIN_GAP_UP=MIN_GAP_UP,
                MIN_VOLUME_RATIO=MIN_VOLUME_RATIO,
                MODEL_SCORE_MIN=MODEL_SCORE_MIN,
            ),
            ensure_ascii=False,
        ),
    )

    signals = {}
    rejected = {}
    reasons = {}

    snapshot_dir = ensure_dir(
        Path("logs/snapshots") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )

    for i, symbol in enumerate(tickers, 1):
        print(f"\n🔎 {i}/{len(tickers)} → {symbol}")
        # скорость: до yfinance — проверим торгуемость
        if not is_tradable(symbol) and not BYPASS_FILTERS:
            rejected[symbol] = "Не торгуется в Alpaca"
            reasons["Не торгуется"] = reasons.get("Не торгуется", 0) + 1
            continue

        df = fetch_history(symbol)
        if df is None or getattr(df, "empty", True) or df.shape[0] < 3:
            rejected[symbol] = "Недостаточно данных"
            reasons["Недостаточно данных"] = reasons.get("Недостаточно данных", 0) + 1
            continue

        # snapshot
        try:
            (snapshot_dir / f"{symbol}.csv").write_text(df.to_csv(), encoding="utf-8")
        except Exception:
            pass

        close = safe_squeeze(df["Close"]).dropna()
        open_s = safe_squeeze(df["Open"]).dropna()
        high = safe_squeeze(df["High"]).dropna()
        low = safe_squeeze(df["Low"]).dropna()
        vol = safe_squeeze(df["Volume"]).dropna()

        if len(close) < 3 or len(open_s) < 2:
            rejected[symbol] = "Недостаточно чистых данных"
            reasons["Недостаточно чистых данных"] = (
                reasons.get("Недостаточно чистых данных", 0) + 1
            )
            continue

        # индикаторы
        rsi_val = float(RSIIndicator(close=close).rsi().iloc[-1])
        ema10 = float(EMAIndicator(close=close, window=10).ema_indicator().iloc[-1])
        ema_dev = pct(close.iloc[-1], ema10)
        atr_val = float(
            AverageTrueRange(high=high, low=low, close=close, window=14)
            .average_true_range()
            .iloc[-1]
        )
        atr_pct = (
            float((atr_val / close.iloc[-1]) * 100.0) if close.iloc[-1] != 0 else 0.0
        )
        vola = float(close.pct_change().std() * 100.0)

        vol_ema = float(EMAIndicator(vol, window=10).ema_indicator().iloc[-1])
        vol_tr = float((vol.iloc[-1] / (vol_ema if vol_ema != 0 else 1.0)))
        vol_ratio = float(vol.iloc[-1] / (vol[:-1].mean() + 1e-6))

        pct_chg = pct(close.iloc[-1], close.iloc[-2])
        prev_close = float(close.iloc[-2])
        today_open = float(open_s.iloc[-1])
        today_close = float(close.iloc[-1])
        body_pct = pct(today_close, today_open)
        gap_up = pct(today_open, prev_close)

        # «риск-скор» (заглушка) и «модельный скор»
        alpha_score = max(
            0.0,
            min(
                1.0,
                0.4 * (pct_chg / 5.0)
                + 0.4 * (vol_ratio / 2.0)
                + 0.2 * (rsi_val / 100.0),
            ),
        )
        model_score = smart_score(alpha_score, vol_ratio, rsi_val, atr_pct, vola)

        # актуальная цена
        price = last_trade_price(symbol, fallback_close=today_close)
        if not price or price <= 0:
            rejected[symbol] = "Цена недоступна"
            reasons["Цена недоступна"] = reasons.get("Цена недоступна", 0) + 1
            continue

        # фильтры
        if not BYPASS_FILTERS:
            if body_pct <= 0:
                rejected[symbol] = f"Красная свеча (body={body_pct:.2f}%)"
                reasons["Красная свеча"] = reasons.get("Красная свеча", 0) + 1
                continue
            if body_pct < MIN_BULLISH_BODY:
                rejected[symbol] = f"Слабая зелёная свеча (body={body_pct:.2f}%)"
                reasons["Слабая зелёная свеча"] = (
                    reasons.get("Слабая зелёная свеча", 0) + 1
                )
                continue
            if gap_up < MIN_GAP_UP or vol_ratio < MIN_VOLUME_RATIO:
                rejected[symbol] = (
                    f"Gap/объём недостаточны (gap={gap_up:.2f}%, vr={vol_ratio:.2f}×)"
                )
                reasons["Gap/объём недостаточны"] = (
                    reasons.get("Gap/объём недостаточны", 0) + 1
                )
                continue
            if atr_pct > MAX_ATR_PCT:
                rejected[symbol] = f"ATR высокий ({atr_pct:.2f}%)"
                reasons["ATR высокий"] = reasons.get("ATR высокий", 0) + 1
                continue
            if vola > MAX_VOLATILITY:
                rejected[symbol] = f"Волатильность высокая ({vola:.2f}%)"
                reasons["Волатильность высокая"] = (
                    reasons.get("Волатильность высокая", 0) + 1
                )
                continue
            if vol_tr < MIN_VOLUME_TREND:
                rejected[symbol] = f"Тренд объёма отрицательный ({vol_tr:.2f}×)"
                reasons["Тренд объёма отрицательный"] = (
                    reasons.get("Тренд объёма отрицательный", 0) + 1
                )
                continue
            if (alpha_score * 100.0) < MIN_RISK_SCORE:
                rejected[symbol] = f"Низкий риск-скор ({alpha_score*100.0:.1f})"
                reasons["Низкий риск-скор"] = reasons.get("Низкий риск-скор", 0) + 1
                continue
            if model_score < MODEL_SCORE_MIN:
                rejected[symbol] = f"Слабый модельный скор ({model_score:.1f})"
                reasons["Слабый модельный скор"] = (
                    reasons.get("Слабый модельный скор", 0) + 1
                )
                continue

        # принятый сигнал
        signals[symbol] = {
            "price": round(price, 2),
            "action": "BUY",
            "confidence": round(alpha_score, 2),
            "score": round(model_score, 2),
            "atr": round(atr_val, 2),
            "atr_pct": round(atr_pct, 2),
            "volatility": round(vola, 2),
            "volume_trend": round(vol_tr, 2),
            "bullish_body": round(body_pct, 2),
            "gap_up": round(gap_up, 2),
            "reason": "NO_FILTERS" if BYPASS_FILTERS else "OK",
        }

        # Telegram уведомление по принятому
        try:
            send_telegram_message(
                "\n".join(
                    [
                        "📊 Новый сигнал (BUY)",
                        f"📌 ${symbol} @ {price:.2f}",
                        f"∆%={pct_chg:.2f}% | RSI={rsi_val:.1f} | EMA dev={ema_dev:.2f}%",
                        f"ATR%={atr_pct:.2f} | Vol%={vola:.2f} | VR={vol_ratio:.2f}× | VTrend={vol_tr:.2f}×",
                        f"Свеча: body={body_pct:.2f}% | gap={gap_up:.2f}%",
                    ]
                )
            )
        except Exception as e:
            if DEBUG_MODE:
                print(f"[WARN] telegram: {e}")

        # ограничим количество сигналов до TARGET_MAX (если фильтры включены)
        if not BYPASS_FILTERS and len(signals) >= TARGET_MAX:
            break

    # запись артефактов
    with open(SIGNALS_PATH, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)
    with open(REJECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    # сводка
    print("\n📊 Сводка причин отказов:")
    for k, v in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  • {k}: {v}")
    print(f"\n📦 Сигналы: {len(signals)} → {SIGNALS_PATH}")
    print(f"🚫 Отклонено: {len(rejected)} → {REJECTED_PATH}")

    # финальные нотификации
    try:
        send_telegram_message(
            f"✅ Итог: сигналов {len(signals)} (цель {TARGET_MIN}-{TARGET_MAX}). BYPASS={BYPASS_FILTERS}"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
# test Fri Oct 17 12:05:43 PM +05 2025
