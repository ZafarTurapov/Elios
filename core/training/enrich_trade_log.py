from core.utils.paths import TRADE_LOG_PATH

# core/training/enrich_trade_log.py

import json
import os
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from core.trading.alpha_utils import calculate_alpha_score

OUTPUT_PATH = str(TRADE_LOG_PATH)  # перезаписываем


def get_metrics(symbol):
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        if df is None or df.empty or "Close" not in df or "Volume" not in df:
            print(f"[ERROR] Нет корректных данных для {symbol}")
            return None

        # RSI
        rsi = RSIIndicator(close=df["Close"]).rsi().iloc[-1]

        # EMA Deviation
        ema = EMAIndicator(close=df["Close"], window=14).ema_indicator()
        ema_dev = ((df["Close"] - ema) / ema).iloc[-1] * 100

        # Volume Ratio
        vol_ratio = df["Volume"].iloc[-1] / df["Volume"].rolling(5).mean().iloc[-1]

        # Alpha Score
        alpha_score = calculate_alpha_score(vol_ratio, rsi, ema_dev)

        return {
            "rsi": round(rsi, 2),
            "ema_dev": round(ema_dev, 2),
            "vol_ratio": round(vol_ratio, 2),
            "alpha_score": round(alpha_score, 3),
        }

    except Exception as e:
        print(f"[ERROR] Ошибка расчёта метрик для {symbol}: {e}")
        return None


def enrich_trade(trade, symbol):
    if all(k in trade for k in ["rsi", "ema_dev", "vol_ratio", "alpha_score", "pnl"]):
        return trade  # уже обогащён

    metrics = get_metrics(symbol)
    if metrics:
        trade.update(metrics)
    else:
        trade.setdefault("rsi", 50.0)
        trade.setdefault("ema_dev", 0.0)
        trade.setdefault("vol_ratio", 1.0)
        trade.setdefault("alpha_score", 0.05)

    trade.setdefault("pnl", 0.0)
    return trade


def main():
    if not os.path.exists(TRADE_LOG_PATH):
        print("[ERROR] trade_log.json не найден.")
        return

    try:
        with open(TRADE_LOG_PATH, "r") as f:
            trades_by_symbol = json.load(f)

        modified = False
        for symbol, trades in trades_by_symbol.items():
            if not isinstance(trades, list):
                print(f"[WARN] {symbol}: не список сделок.")
                continue
            for i, trade in enumerate(trades):
                if isinstance(trade, dict):
                    enriched = enrich_trade(trade, symbol)
                    trades[i] = enriched
                    modified = True

        if modified:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(trades_by_symbol, f, indent=2)
            print("✅ trade_log.json обогащён недостающими полями.")
        else:
            print("ℹ️ Все записи уже содержат необходимые поля.")

    except Exception as e:
        print(f"[ERROR] Ошибка при обогащении trade_log.json: {e}")


if __name__ == "__main__":
    main()
