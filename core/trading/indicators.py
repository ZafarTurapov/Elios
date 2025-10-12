# -*- coding: utf-8 -*-
"""
core/trading/indicators.py
Минималистичные технические индикаторы без сторонних библиотек.
Первым добавляем ATR (Average True Range).
Предполагаем, что вход – pandas.DataFrame с колонками: ['open','high','low','close'].
"""

from typing import Optional
import pandas as pd

def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_close = close.shift(1)
    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()

    tr = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14, use_rma: bool = True) -> Optional[float]:
    """
    Возвращает последнее значение ATR по историческим барам.
    - period: окно ATR (классика = 14)
    - use_rma: True => RMA (Wilder), False => простая SMA по TR

    Требования к df:
      - индекс по времени не обязателен, но строки должны быть в хрон. порядке
      - должны быть столбцы: 'high', 'low', 'close'
    """
    if df is None or df.empty:
        return None
    need_cols = {'high', 'low', 'close'}
    if not need_cols.issubset(set(map(str.lower, df.columns))):
        # допускаем колонки в любом регистре
        cols = {c.lower(): c for c in df.columns}
        try:
            df = df.rename(columns={cols['high']: 'high', cols['low']: 'low', cols['close']: 'close'})
        except Exception:
            return None

    tr = _true_range(df).dropna()
    if len(tr) < period:
        return None

    if use_rma:
        # Wilder's RMA:
        # RMA_t = (RMA_{t-1} * (period - 1) + TR_t) / period
        rma = tr.ewm(alpha=1/period, adjust=False).mean()
        return float(rma.iloc[-1])
    else:
        sma = tr.rolling(window=period, min_periods=period).mean()
        return float(sma.dropna().iloc[-1]) if not sma.dropna().empty else None

def atr_ratio(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Возвращает отношение ATR к последней цене закрытия (ATR / close).
    Удобно для фильтра по волатильности.
    """
    val_atr = atr(df, period=period, use_rma=True)
    try:
        last_close = float(df['close'].astype(float).iloc[-1])
    except Exception:
        last_close = None
    if val_atr is None or not last_close or last_close <= 0:
        return None
    return float(val_atr / last_close)
