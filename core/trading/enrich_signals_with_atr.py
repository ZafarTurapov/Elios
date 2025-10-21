# core/trading/enrich_signals_with_atr.py
# Обогащение сигналов: ATR, волатильность, объёмный тренд
# Устойчиво парсит сигналы в любых форматах: str / dict / tuple / list

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf


SIGNALS_PATH = Path("/root/stockbot/core/trading/signals.json")
OUTPUT_PATH = Path("/root/stockbot/core/trading/signals_enriched.json")

# --- Настройки индикаторов ---
PERIOD_DAYS = 30  # историю берём ~ на месяц
ATR_WINDOW = 14
VOL_SMA = 5  # сглаживание объёма
VOL_TREND_W = 3  # окно тренда объёма (последние N дней)
VOLAT_WIN = 3  # окно краткосрочной волатильности (стд. отклонение %изменений)


def normalize_symbol(item: Any) -> Optional[str]:
    """
    Превращает запись сигнала в чистый тикер-строку.
    Допустимые входы: "AAPL", {"symbol":"AAPL"}, {"ticker":"AAPL"}, ("AAPL", 0.87), ["AAPL", 123], ...
    Возвращает UPPER тикер или None (если распарсить нельзя).
    """
    # Уже строка
    if isinstance(item, str):
        s = item.strip().upper()
        return s if s else None

    # Словарь с типичными ключами
    if isinstance(item, dict):
        for k in ("symbol", "ticker", "sym", "s"):
            if k in item:
                val = str(item[k]).strip().upper()
                return val if val else None
        # Иногда сигнал может лежать под ключом "data" и т.п.
        # Попробуем вытащить первую строку в словаре
        for v in item.values():
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
        return None

    # Кортеж или список: берём первый элемент, если он строка
    if isinstance(item, (list, tuple)):
        if len(item) > 0 and isinstance(item[0], str):
            s = item[0].strip().upper()
            return s if s else None
        return None

    return None


def load_symbols_from_signals(path: Path) -> List[str]:
    """
    Загружает массив сигналов из JSON и приводит к списку уникальных тикеров (str).
    Если файла нет — вернёт пустой список (скрипт завершится спокойно).
    """
    if not path.exists():
        print(f"[WARN] Файл сигналов не найден: {path}")
        return []

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {path}: {e}")
        return []

    symbols: List[str] = []
    # data может быть списком, словарём с ключом "signals" и т.д.
    candidates: Iterable[Any]
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        candidates = (
            data.get("signals") or data.get("tickers") or data.get("data") or []
        )
    else:
        candidates = []

    for item in candidates:
        sym = normalize_symbol(item)
        if sym:
            symbols.append(sym)

    # Уникализируем порядок
    symbols = list(dict.fromkeys(symbols))
    return symbols


def download_history(
    symbol: str, period_days: int = PERIOD_DAYS
) -> Optional[pd.DataFrame]:
    """
    Скачивает дневные бары через yfinance, безопасно.
    """
    try:
        df = yf.download(
            symbol,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,  # избегаем предупреждения и получаем скорректированные цены
            threads=False,
        )
        if df is None or df.empty:
            print(f"[WARN] Нет данных по {symbol}")
            return None

        # Нормализуем столбцы (иногда yfinance возвращает lower-case)
        cols_map = {c.lower(): c for c in df.columns}
        # убеждаемся, что нужные поля есть
        need = {"open", "high", "low", "close", "volume"}
        have = set(map(str.lower, df.columns))
        if not need.issubset(have):
            # Попробуем привести к стандартным именам
            rename_map = {}
            for need_c in ["Open", "High", "Low", "Close", "Volume"]:
                for c in df.columns:
                    if c.lower() == need_c.lower():
                        rename_map[c] = need_c
            if rename_map:
                df = df.rename(columns=rename_map)

        return df
    except Exception as e:
        print(f"[ERROR] download({symbol}): {e}")
        return None


def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """
    ATR по классике Уайлдера.
    Ожидаются столбцы: High, Low, Close
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    # классический Wilder smoothing можно заменить на простое SMA — оставим EMA(α=1/window*2) близко к классике
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    return atr


def volume_trend(df: pd.DataFrame, window: int = VOL_TREND_W) -> float:
    """
    Простой тренд объёма: линейная регрессия по последним N дням объёма (после SMA).
    Возвращает наклон (slope). >0 — объёмы растут.
    """
    vol = df["Volume"].rolling(VOL_SMA, min_periods=1).mean().tail(window).values
    if len(vol) < 2:
        return 0.0
    x = np.arange(len(vol))
    # slope = cov(x, y) / var(x)
    slope = float(np.cov(x, vol, bias=True)[0, 1] / (np.var(x) + 1e-9))
    return slope


def short_volatility(df: pd.DataFrame, window: int = VOLAT_WIN) -> float:
    """
    Краткосрочная волатильность: std дневных %изменений за последние N дней.
    """
    rets = df["Close"].pct_change().dropna().tail(window)
    return float(rets.std()) if not rets.empty else 0.0


def enrich_symbols(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Считает метрики на каждый тикер. Возвращает список словарей.
    """
    enriched = []
    for sym in symbols:
        df = download_history(sym, PERIOD_DAYS)
        if df is None:
            continue

        try:
            atr = compute_atr(df, ATR_WINDOW).iloc[-1]
            vol_trend = volume_trend(df, VOL_TREND_W)
            vol_short = short_volatility(df, VOLAT_WIN)

            # Дополнительно: относительный ATR (% от цены)
            last_close = float(df["Close"].iloc[-1])
            atr_pct = float(atr / last_close) if last_close else 0.0

            enriched.append(
                {
                    "symbol": sym,
                    "last_close": last_close,
                    "atr": float(atr),
                    "atr_pct": atr_pct,
                    "volume_trend": vol_trend,
                    "short_volatility": vol_short,
                }
            )
        except Exception as e:
            print(f"[ERROR] calc({sym}): {e}")
            continue
    return enriched


def main():
    symbols = load_symbols_from_signals(SIGNALS_PATH)
    if not symbols:
        print("[INFO] Список сигналов пуст — нечего обогащать.")
        return

    print(f"[INFO] Найдено сигналов: {len(symbols)}")
    data = enrich_symbols(symbols)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[OK] Сохранено: {OUTPUT_PATH} ({len(data)} записей)")


if __name__ == "__main__":
    main()
