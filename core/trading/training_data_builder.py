from core.utils.paths import TRADE_LOG_PATH

# /root/stockbot/core/trading/training_data_builder.py
# -*- coding: utf-8 -*-
"""
Сборщик обучающих данных — устойчивый к weird-форматам от yfinance,
с fallback-периодами, безопасной нормализацией колонок и корректными timestamp'ами.

Для безопасности: прямой запуск файла отключён по умолчанию.
Разрешить прямой запуск можно установив в окружении ALLOW_DIRECT_BUILDER=1.
( daily_training_update.py выставляет эту переменную перед запуском )
"""

import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
from core.utils.telegram import send_telegram_message
from core.trading.alpha_utils import calculate_alpha_score
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# === Абсолютные пути ===

GPT_DECISIONS_PATH = "/root/stockbot/core/trading/gpt_decisions.json"
TRAINING_DATA_PATH = "/root/stockbot/core/trading/training_data.json"
TMP_OUTPUT_PATH = TRAINING_DATA_PATH + ".new"

# === Параметры индикаторов ===
RSI_WINDOW = 14
EMA_WINDOW = 10
VOL_LOOKBACK = 20
ATR_WINDOW = 14
# Периоды для fallback при проблемах с историей (в порядке попыток)
HIST_PERIODS = ["max", "2y", "1y", "6mo", "3mo"]
HIST_INTERVAL = "1d"


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Не удалось загрузить {path}: {e}")
        return {}


def _tz_naive(ts_str):
    """Строковый ISO timestamp -> naive UTC datetime (tz=None)."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _ensure_ohlc(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            last_lvl = df.columns.get_level_values(-1)
            unique_last = pd.unique(last_lvl)
            if len(unique_last) >= 1:
                first_ticker = unique_last[0]
                df = df.xs(first_ticker, axis=1, level=-1, drop_level=True)
        except Exception:
            try:
                df.columns = [
                    c[-1] if isinstance(c, tuple) and len(c) > 1 else c
                    for c in df.columns
                ]
            except Exception:
                pass

    ren = {}
    for c in df.columns:
        lc = str(c).lower()
        if "close" in lc:
            ren[c] = "Close"
        elif "high" in lc:
            ren[c] = "High"
        elif "low" in lc:
            ren[c] = "Low"
        elif "volume" in lc or lc == "vol":
            ren[c] = "Volume"
        elif "open" in lc:
            ren[c] = "Open"
    df = df.rename(columns=ren)

    required = ["High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        candidates = {c.lower(): c for c in df.columns}
        mapped = []
        for need in required:
            key = need.lower()
            if key in candidates:
                mapped.append(candidates[key])
        if len(mapped) == 4:
            df = df[mapped].copy()
            df.columns = required
        else:
            cols = list(df.columns)
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 4:
                picked = num_cols[-4:]
                df = df[picked].copy()
                df.columns = required
            else:
                return None
    else:
        df = df[required].copy()

    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return None

    if hasattr(df.index, "tz") and df.index.tz is not None:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                pass

    return df


def manual_tr(close, high, low):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _window_upto_ref(df, ref_dt, max_lookback):
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    idx = df.index
    pos = idx.searchsorted(ref_dt, side="right")
    if pos == 0:
        end = min(len(df), max_lookback)
        return df.iloc[:end].copy() if end > 0 else None
    start = max(0, pos - max_lookback)
    win = df.iloc[start:pos].copy()
    return win if not win.empty else None


def _as_series_safe(col):
    if col is None:
        return None
    if isinstance(col, pd.Series):
        return col
    if isinstance(col, pd.DataFrame):
        for c in col.columns:
            if pd.api.types.is_numeric_dtype(col[c]):
                return col[c]
        return col.iloc[:, 0]
    try:
        s = pd.Series(col)
        return s
    except Exception:
        return None


def compute_features_flexible(df, ref_dt):
    out = {
        "rsi": None,
        "ema_dev": None,
        "vol_ratio": None,
        "atr": None,
        "alpha_score": None,
    }
    if df is None or df.empty:
        return out

    max_lookback = max(
        RSI_WINDOW + 1, EMA_WINDOW + 1, VOL_LOOKBACK + 1, ATR_WINDOW + 1, 5
    )
    win = _window_upto_ref(df, ref_dt, max_lookback)
    if win is None or win.empty:
        return out

    try:
        close_col = _as_series_safe(
            win.get("Close")
            if "Close" in win
            else win.iloc[:, -2] if win.shape[1] >= 2 else None
        )
    except Exception:
        close_col = None

    high_col = _as_series_safe(
        win.get("High")
        if "High" in win
        else (win.iloc[:, 0] if win.shape[1] >= 1 else None)
    )
    low_col = _as_series_safe(
        win.get("Low")
        if "Low" in win
        else (win.iloc[:, 1] if win.shape[1] >= 2 else None)
    )
    vol_col = _as_series_safe(
        win.get("Volume")
        if "Volume" in win
        else (win.iloc[:, -1] if win.shape[1] >= 1 else None)
    )

    try:
        close = (
            pd.to_numeric(close_col, errors="coerce")
            if close_col is not None
            else pd.Series(dtype=float)
        )
    except Exception:
        close = pd.Series(dtype=float)

    try:
        high = (
            pd.to_numeric(high_col, errors="coerce")
            if high_col is not None
            else pd.Series(dtype=float)
        )
    except Exception:
        high = pd.Series(dtype=float)

    try:
        low = (
            pd.to_numeric(low_col, errors="coerce")
            if low_col is not None
            else pd.Series(dtype=float)
        )
    except Exception:
        low = pd.Series(dtype=float)

    try:
        vol = (
            pd.to_numeric(vol_col, errors="coerce")
            if vol_col is not None
            else pd.Series(dtype=float)
        )
    except Exception:
        vol = pd.Series(dtype=float)

    if close.empty or close.dropna().empty:
        return out
    last_close = float(close.dropna().iloc[-1])

    if len(close.dropna()) >= RSI_WINDOW:
        try:
            rsi_series = RSIIndicator(close=close.dropna(), window=RSI_WINDOW).rsi()
            out["rsi"] = float(rsi_series.iloc[-1])
        except Exception:
            out["rsi"] = None

    if len(close.dropna()) >= EMA_WINDOW:
        try:
            ema_series = EMAIndicator(
                close=close.dropna(), window=EMA_WINDOW
            ).ema_indicator()
            ema_val = float(ema_series.iloc[-1])
            out["ema_dev"] = (
                float(((last_close - ema_val) / ema_val) * 100.0)
                if ema_val != 0
                else 0.0
            )
        except Exception:
            out["ema_dev"] = None
    else:
        try:
            sma_len = min(EMA_WINDOW, len(close.dropna()))
            if sma_len >= 2:
                sma = float(
                    close.dropna()
                    .rolling(window=sma_len, min_periods=2)
                    .mean()
                    .iloc[-1]
                )
                out["ema_dev"] = (
                    float(((last_close - sma) / sma) * 100.0) if sma != 0 else 0.0
                )
        except Exception:
            out["ema_dev"] = None

    if len(vol.dropna()) >= 3:
        try:
            n = min(VOL_LOOKBACK, len(vol.dropna()) - 1)
            if n > 0:
                prev_mean = float(vol.dropna().iloc[-n - 1 : -1].mean())
                if prev_mean > 0:
                    out["vol_ratio"] = float(float(vol.dropna().iloc[-1]) / prev_mean)
        except Exception:
            out["vol_ratio"] = None

    pct_chg = 0.0
    if len(close.dropna()) >= 2:
        try:
            prev = float(close.dropna().iloc[-2])
            curr = float(close.dropna().iloc[-1])
            pct_chg = float(((curr - prev) / prev) * 100.0) if prev != 0 else 0.0
        except Exception:
            pct_chg = 0.0

    if len(close.dropna()) >= 2 and not high.empty and not low.empty:
        try:
            tr = manual_tr(
                close=close.dropna(),
                high=high.dropna().reindex(close.dropna().index, method="ffill"),
                low=low.dropna().reindex(close.dropna().index, method="ffill"),
            )
            win_len = min(ATR_WINDOW, len(tr))
            atr = tr.rolling(window=win_len, min_periods=1).mean().iloc[-1]
            out["atr"] = None if pd.isna(atr) else float(round(float(atr), 3))
        except Exception:
            out["atr"] = None

    try:
        rsi_for_alpha = out["rsi"] if out["rsi"] is not None else 50.0
        ema_dev_for_alpha = out["ema_dev"] if out["ema_dev"] is not None else 0.0
        vol_ratio_for_alpha = out["vol_ratio"] if out["vol_ratio"] is not None else 1.0
        out["alpha_score"] = float(
            calculate_alpha_score(
                pct_chg, vol_ratio_for_alpha, rsi_for_alpha, ema_dev_for_alpha
            )
        )
    except Exception:
        out["alpha_score"] = None

    return out


def get_history_with_fallback(symbol):
    last_exc = None
    df_last = None
    for per in HIST_PERIODS:
        try:
            raw = yf.download(
                symbol,
                period=per,
                interval=HIST_INTERVAL,
                progress=False,
                auto_adjust=False,
            )
            df = _ensure_ohlc(raw)
            if df is None:
                last_exc = f"Ensure OHLC returned None for period={per}"
                continue
            snap = compute_features_flexible(df, df.index.max())
            if snap.get("atr") is not None:
                if per != HIST_PERIODS[0]:
                    print(f"[FALLBACK] {symbol} — ATR obtained with period={per}")
                return df
            else:
                df_last = df
                last_exc = f"ATR empty for period={per}"
        except Exception as e:
            last_exc = str(e)
            print(f"[HIST ERROR] {symbol} period={per}: {e}")
            continue
    print(f"[HIST FAIL] {symbol} — no valid history, last_reason: {last_exc}")
    return df_last


def build_training_dataset():
    trade_log = load_json(TRADE_LOG_PATH)
    gpt_decisions = load_json(GPT_DECISIONS_PATH)
    dataset = []

    if not isinstance(trade_log, dict):
        trade_log = {}

    hist_cache = {}
    counts = {
        "total_trades": 0,
        "with_hist": 0,
        "no_hist": 0,
        "appended": 0,
        "skipped_no_prices": 0,
        "skipped_nofeats": 0,
    }

    for symbol, trades in trade_log.items():
        if not isinstance(trades, list):
            continue

        if symbol not in hist_cache:
            hist_cache[symbol] = get_history_with_fallback(symbol)
        df_hist = hist_cache[symbol]

        for trade in trades:
            counts["total_trades"] += 1

            entry_price = trade.get("price") or trade.get("entry_price")
            exit_price = trade.get("exit_price")
            try:
                entry_price = float(entry_price) if entry_price is not None else None
                exit_price = float(exit_price) if exit_price is not None else None
            except Exception:
                entry_price, exit_price = None, None

            if entry_price is None or exit_price is None or entry_price == 0:
                counts["skipped_no_prices"] += 1
                continue

            ts_entry = _tz_naive(trade.get("timestamp_entry")) or _tz_naive(
                trade.get("timestamp")
            )
            ts_exit = _tz_naive(trade.get("timestamp_exit"))

            if ts_entry is not None:
                ref_dt = ts_entry
            elif ts_exit is not None:
                ref_dt = ts_exit - timedelta(days=1)
            else:
                if (
                    df_hist is not None
                    and isinstance(df_hist.index, pd.DatetimeIndex)
                    and not df_hist.empty
                ):
                    ref_dt = df_hist.index.max()
                else:
                    ref_dt = datetime.utcnow().replace(tzinfo=None)

            if df_hist is not None and not df_hist.empty:
                counts["with_hist"] += 1
                snap = compute_features_flexible(df_hist, ref_dt)
            else:
                counts["no_hist"] += 1
                snap = {
                    "rsi": None,
                    "ema_dev": None,
                    "vol_ratio": None,
                    "atr": None,
                    "alpha_score": None,
                }

            rsi = snap["rsi"] if snap.get("rsi") is not None else trade.get("rsi")
            ema_dev = (
                snap["ema_dev"]
                if snap.get("ema_dev") is not None
                else trade.get("ema_dev")
            )
            vol_ratio = (
                snap["vol_ratio"]
                if snap.get("vol_ratio") is not None
                else trade.get("vol_ratio")
            )
            atr = snap["atr"] if snap.get("atr") is not None else trade.get("atr")
            alpha_score = (
                snap["alpha_score"]
                if snap.get("alpha_score") is not None
                else (trade.get("confidence") or trade.get("alpha_score"))
            )

            if all(v is None for v in [rsi, ema_dev, vol_ratio, atr, alpha_score]):
                counts["skipped_nofeats"] += 1
                continue

            change_pct = ((exit_price - entry_price) / entry_price) * 100.0
            label = "WIN" if change_pct > 0 else "LOSS"

            gpt_reply = gpt_decisions.get(symbol, "")
            reason = trade.get("reason", "")

            try:
                atr_val = float(atr)
            except Exception:
                atr_val = 0.0

            ts_snapshot = (
                ref_dt.isoformat()
                if ref_dt is not None
                else datetime.utcnow().replace(tzinfo=None).isoformat()
            )
            ts_exit_val = None
            if ts_exit is not None:
                try:
                    ts_exit_val = (
                        ts_exit.isoformat()
                        if hasattr(ts_exit, "isoformat")
                        else str(ts_exit)
                    )
                except Exception:
                    ts_exit_val = str(ts_exit)

            data_point = {
                "symbol": symbol,
                "entry_price": round(entry_price, 3),
                "exit_price": round(exit_price, 3),
                "change_pct": round(change_pct, 2),
                "alpha_score": (
                    round(alpha_score, 3) if alpha_score is not None else None
                ),
                "rsi": round(rsi, 2) if rsi is not None else None,
                "ema_dev": round(ema_dev, 2) if ema_dev is not None else None,
                "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
                "atr": atr_val,
                "gpt_decision": gpt_reply,
                "reason": reason,
                "label": label,
                "timestamp": ts_snapshot,
                "timestamp_exit": ts_exit_val,
            }
            dataset.append(data_point)
            counts["appended"] += 1

    try:
        os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
    except Exception:
        pass

    # Записываем сначала во временный файл, затем атомарно перемещаем
    try:
        with open(TMP_OUTPUT_PATH, "w") as f:
            json.dump(dataset, f, indent=2)
        os.replace(TMP_OUTPUT_PATH, TRAINING_DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
        # попытка прямой записи в основной файл как fallback
        with open(TRAINING_DATA_PATH, "w") as f:
            json.dump(dataset, f, indent=2)

    print(f"✅ Сформировано обучающих строк: {len(dataset)} → {TRAINING_DATA_PATH}")
    print(f"ℹ️ Статистика: {counts}")

    if dataset:
        tickers = list({d["symbol"] for d in dataset if "symbol" in d})
        msg = (
            "📚 Обновление обучающих данных\n"
            f"➕ Добавлено: {len(dataset)} примеров\n"
            "📌 " + ", ".join(tickers[:30]) + ("..." if len(tickers) > 30 else "")
        )
    else:
        msg = "📚 Обновление обучающих данных — новых примеров нет."
    try:
        send_telegram_message(msg)
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")


if __name__ == "__main__":
    # Защита: запрещаем прямой запуск без переменной окружения
    if os.environ.get("ALLOW_DIRECT_BUILDER") != "1":
        print(
            "[ERROR] Direct run disabled. Use daily_training_update.py (or set ALLOW_DIRECT_BUILDER=1)."
        )
    else:
        build_training_dataset()
