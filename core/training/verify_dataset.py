# -*- coding: utf-8 -*-
"""
verify_dataset.py — быстрая диагностика обучающего датасета Искры
Проверяет:
- структуру и обязательные поля
- пропуски/NaN/inf
- распределения и допустимые диапазоны признаков
- дубликаты и покрытие по тикерам/датам
- баланс классов
- перетренировку по тикеру (слишком много примеров на один тикер)
- предварительный sanity-check модели (быстрый CV по GroupKFold на тикер)
Сохраняет подробный отчет в core/training/verify_report.json
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# ML часть — по возможности (если sklearn установлен)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_validate
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

DATA_PATH = Path("core/trading/training_data.json")
REPORT_PATH = Path("core/training/verify_report.json")

# Ожидаемые фичи (минимальный набор); остальные — как есть
EXPECTED_FEATURES = [
    "alpha_score", "rsi", "ema_dev", "vol_ratio", "atr_pct"
]
# Расширенные (если есть — проверим диапазоны)
OPTIONAL_FEATURES = [
    "bullish_body", "gap_up", "volume_trend", "volatility"
]
# Обязательные служебные поля
META_FIELDS = [
    "symbol",         # тикер
    "timestamp",      # ISO дата/время входа или бара
    "label"           # 1/0 или "WIN"/"LOSS"
]

# Диапазоны sanity для фич (soft-границы; если выходит — warning)
RANGES = {
    "rsi":            (0, 100),
    "vol_ratio":      (0.0, 20.0),
    "atr_pct":        (0.0, 50.0),
    "ema_dev":        (-50.0, 50.0),
    "bullish_body":   (-50.0, 50.0),
    "gap_up":         (-50.0, 50.0),
    "volume_trend":   (0.0, 10.0),
    "volatility":     (0.0, 50.0),
    "alpha_score":    (-1e6, 1e6),  # модельная величина — только проверка на NaN/inf
}

def is_nanlike(x):
    try:
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return False

def to_float_or_none(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def normalize_label(y):
    if isinstance(y, str):
        y = y.strip().lower()
        if y in ("win", "1", "true", "yes", "profit", "pos"):
            return 1
        if y in ("loss", "0", "false", "no", "neg"):
            return 0
    try:
        return int(y)
    except Exception:
        return None

def parse_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def main():
    report = {
        "loaded": False,
        "count": 0,
        "time_range": None,
        "unique_symbols": 0,
        "top_symbols": [],
        "class_balance": None,
        "missing_fields": {},
        "nan_counts": {},
        "out_of_range": {},
        "duplicates": 0,
        "per_symbol_stats": {},
        "notes": [],
        "sklearn_cv": None
    }

    if not DATA_PATH.exists():
        print(f"❌ Не найден {DATA_PATH}")
        report["notes"].append("training_data.json not found")
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return

    with DATA_PATH.open() as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка чтения JSON: {e}")
            report["notes"].append(f"json_load_error: {e}")
            REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            return

    if not isinstance(data, list) or not data:
        print("❌ Датасет пустой или не список.")
        report["notes"].append("empty_or_not_list")
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return

    report["loaded"] = True
    report["count"] = len(data)

    # Проверки полей
    missing_fields = Counter()
    nan_counts = Counter()
    out_of_range = Counter()
    sym_counter = Counter()
    labels_counter = Counter()
    times = []
    seen_keys = set()
    dup_count = 0

    # Пер-символьная статистика
    per_symbol = defaultdict(lambda: {"n":0, "wins":0, "losses":0})

    rows_clean_for_ml = []
    y_clean = []
    g_groups = []

    for i, row in enumerate(data):
        # Проверка обязательных полей
        for f in META_FIELDS:
            if f not in row:
                missing_fields[f] += 1

        symbol = row.get("symbol")
        ts = row.get("timestamp")
        label = normalize_label(row.get("label"))

        sym_counter[symbol] += 1
        if label is not None:
            labels_counter[label] += 1

        # Пер-символьная статистика
        per_symbol[symbol]["n"] += 1
        if label == 1:
            per_symbol[symbol]["wins"] += 1
        elif label == 0:
            per_symbol[symbol]["losses"] += 1

        # Временной ряд
        dt = parse_dt(ts) if isinstance(ts, str) else None
        if dt:
            times.append(dt)

        # Дубликаты по (symbol, timestamp)
        key = (symbol, ts)
        if key in seen_keys:
            dup_count += 1
        else:
            seen_keys.add(key)

        # Проверка NaN/inf и диапазонов фич
        for feat in EXPECTED_FEATURES + OPTIONAL_FEATURES:
            if feat in row:
                v = to_float_or_none(row.get(feat))
                if is_nanlike(v):
                    nan_counts[feat] += 1
                    continue
                lo, hi = RANGES.get(feat, (-np.inf, np.inf))
                # Только если есть установленные границы
                if lo is not None and hi is not None:
                    try:
                        if v is not None and (v < lo or v > hi):
                            out_of_range[feat] += 1
                    except Exception:
                        pass
            else:
                # отсутствие необязательных не считаем ошибкой
                if feat in EXPECTED_FEATURES:
                    missing_fields[feat] += 1

        # Соберем наблюдения для быстрого ML-саначека (только если все нужные есть и корректны)
        feats = []
        valid_row = True
        for feat in EXPECTED_FEATURES:
            v = to_float_or_none(row.get(feat))
            if v is None or is_nanlike(v):
                valid_row = False
                break
            feats.append(v)
        if valid_row and (label in (0, 1)) and isinstance(symbol, str):
            rows_clean_for_ml.append(feats)
            y_clean.append(label)
            g_groups.append(symbol)

    report["missing_fields"] = dict(missing_fields)
    report["nan_counts"] = dict(nan_counts)
    report["out_of_range"] = dict(out_of_range)
    report["duplicates"] = int(dup_count)

    # Тайм-диапазон
    if times:
        report["time_range"] = [min(times).isoformat(), max(times).isoformat()]

    # Символы и топ-10 по числу примеров
    report["unique_symbols"] = len(sym_counter)
    report["top_symbols"] = sym_counter.most_common(10)

    # Баланс классов
    total_lbl = sum(labels_counter.values())
    if total_lbl > 0:
        share_win = labels_counter.get(1, 0) / total_lbl
        report["class_balance"] = {
            "WIN": labels_counter.get(1, 0),
            "LOSS": labels_counter.get(0, 0),
            "WIN_share": round(share_win, 4)
        }

    # Пер-символьная статистика (win rate по тикеру)
    for s, st in per_symbol.items():
        n = st["n"]
        w = st["wins"]
        l = st["losses"]
        wr = (w / max(1, w + l))
        st["win_rate"] = round(wr, 4)
    # Сохраним первые 20 для отчета
    sample_sym_stats = dict(list(per_symbol.items())[:20])
    report["per_symbol_stats"] = sample_sym_stats

    # Предупреждения по перекосу на один тикер
    most_sym, most_cnt = sym_counter.most_common(1)[0]
    if most_cnt > 12:
        report["notes"].append(
            f"ticker_imbalance: {most_sym} имеет {most_cnt} примеров (>12); проверь downsampling"
        )

    # Быстрый sanity-check модели (по желанию и если есть sklearn)
    if HAVE_SKLEARN and len(rows_clean_for_ml) >= 50 and len(set(g_groups)) >= 3:
        X = np.array(rows_clean_for_ml, dtype=float)
        y = np.array(y_clean, dtype=int)
        groups = np.array(g_groups)
        try:
            cv = GroupKFold(n_splits=min(5, len(set(groups))))
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
            )
            scores = cross_validate(
                clf, X, y, cv=cv, groups=groups,
                scoring=["accuracy", "precision", "recall", "f1"],
                n_jobs=-1, error_score="raise"
            )
            report["sklearn_cv"] = {
                "folds": int(cv.get_n_splits()),
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "precision_mean": float(np.mean(scores["test_precision"])),
                "recall_mean": float(np.mean(scores["test_recall"])),
                "f1_mean": float(np.mean(scores["test_f1"])),
            }
        except Exception as e:
            report["notes"].append(f"sklearn_cv_error: {e}")
    else:
        if not HAVE_SKLEARN:
            report["notes"].append("sklearn_not_available")
        else:
            report["notes"].append("not_enough_rows_or_groups_for_cv")

    # Сохранить отчет
    try:
        os.makedirs(REPORT_PATH.parent, exist_ok=True)
        with REPORT_PATH.open("w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Не удалось сохранить отчет: {e}")

    # Короткая консольная сводка
    print("==== VERIFY DATASET REPORT (short) ====")
    print(f"rows: {report['count']}, symbols: {report['unique_symbols']}, dups: {report['duplicates']}")
    if report["time_range"]:
        print(f"time_range: {report['time_range'][0]} → {report['time_range'][1]}")
    if report["class_balance"]:
        cb = report["class_balance"]
        print(f"class_balance: WIN={cb['WIN']} LOSS={cb['LOSS']} (share_win={cb['WIN_share']})")
    if report["missing_fields"]:
        print("missing_fields:", report["missing_fields"])
    if report["nan_counts"]:
        print("nan_counts:", report["nan_counts"])
    if report["out_of_range"]:
        print("out_of_range:", report["out_of_range"])
    if report["sklearn_cv"]:
        print("cv:", report["sklearn_cv"])
    if report["notes"]:
        print("notes:", report["notes"])

if __name__ == "__main__":
    main()
