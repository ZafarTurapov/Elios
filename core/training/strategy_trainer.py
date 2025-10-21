# -*- coding: utf-8 -*-
"""
strategy_trainer.py — обучение модели Искры с контролем качества
- Читает core/trading/training_data.json
- Даунсэмплит тикеры до MAX_PER_SYMBOL (борьба с перекосом)
- Групповая CV по тикеру (GroupKFold → fallback GroupShuffleSplit)
- Сохраняет метрики в core/training/training_metrics.csv
- Сохраняет модель в core/training/trained_model.pkl, если прошла гейт по метрикам
"""

import os
import csv
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA_PATH = Path("core/trading/training_data.json")
MODEL_PATH = Path("core/training/trained_model.pkl")
METRICS_CSV = Path("core/training/training_metrics.csv")
MODEL_BACKUP_DIR = Path("core/training/model_backups")

# Ограничения/настройки
RANDOM_STATE = 42
MAX_PER_SYMBOL = 12  # даунсэмплинг на тикер
MIN_ROWS_FOR_TRAIN = 200
MIN_GROUPS_FOR_KFOLD = 5
KFOLD_SPLITS = 5  # уменьшится до 3, если групп меньше
FALLBACK_TEST_SIZE = 0.2  # если групп мало — GroupShuffleSplit
IMPROVEMENT_TOL = 0.005  # насколько новая F1 должна быть не хуже старой (или лучше)
ABS_MIN_F1_TO_SAVE = 0.62  # минимальный порог качества, чтобы вообще сохранить модель

# Базовый набор признаков (остальные добавим, если присутствуют у всех)
BASE_FEATURES = ["alpha_score", "rsi", "ema_dev", "vol_ratio", "atr_pct"]
EXTRA_FEATURES = ["bullish_body", "gap_up", "volume_trend", "volatility"]


def _normalize_label(y):
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


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"not found: {DATA_PATH}")
    data = json.loads(DATA_PATH.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("training_data.json empty or not a list")
    return data


def _available_features(rows, feat_list):
    """Вернёт подсписок фичей, которые присутствуют И валидны (не NaN) хотя бы у ~95% строк."""
    keep = []
    n = len(rows)
    for f in feat_list:
        ok = 0
        for r in rows:
            v = _to_float(r.get(f))
            if v is None or math.isnan(v) or math.isinf(v):
                continue
            ok += 1
        if ok >= 0.95 * n:
            keep.append(f)
    return keep


def _downsample_by_symbol(rows, max_per_symbol=MAX_PER_SYMBOL):
    by = defaultdict(list)
    for r in rows:
        by[r.get("symbol")].append(r)
    out = []
    rng = random.Random(RANDOM_STATE)
    for sym, lst in by.items():
        if len(lst) > max_per_symbol:
            out.extend(rng.sample(lst, max_per_symbol))
        else:
            out.extend(lst)
    rng.shuffle(out)
    return out


def _build_matrix(rows, feature_list):
    X, y, groups = [], [], []
    for r in rows:
        label = _normalize_label(r.get("label"))
        sym = r.get("symbol")
        if label not in (0, 1) or not isinstance(sym, str):
            continue
        vec = []
        ok = True
        for f in feature_list:
            v = _to_float(r.get(f))
            if v is None or math.isnan(v) or math.isinf(v):
                ok = False
                break
            vec.append(v)
        if not ok:
            continue
        X.append(vec)
        y.append(label)
        groups.append(sym)
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)


def _read_last_f1():
    if not METRICS_CSV.exists():
        return None
    try:
        with METRICS_CSV.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        return float(last.get("f1_mean") or 0.0)
    except Exception:
        return None


def _append_metrics(ts, n_rows, n_groups, feats, cv_res):
    newfile = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(
                [
                    "timestamp",
                    "rows",
                    "groups",
                    "features",
                    "folds",
                    "accuracy_mean",
                    "precision_mean",
                    "recall_mean",
                    "f1_mean",
                ]
            )
        w.writerow(
            [
                ts,
                n_rows,
                n_groups,
                "|".join(feats),
                cv_res.get("folds", ""),
                round(cv_res["accuracy_mean"], 6),
                round(cv_res["precision_mean"], 6),
                round(cv_res["recall_mean"], 6),
                round(cv_res["f1_mean"], 6),
            ]
        )


def _do_cv(X, y, groups):
    uniq_groups = len(set(groups))
    if uniq_groups >= MIN_GROUPS_FOR_KFOLD and len(X) >= MIN_ROWS_FOR_TRAIN:
        splits = min(KFOLD_SPLITS, uniq_groups)
        cv = GroupKFold(n_splits=splits)
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scores = cross_validate(
            clf,
            X,
            y,
            cv=cv,
            groups=groups,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=-1,
            error_score="raise",
        )
        return {
            "folds": int(cv.get_n_splits()),
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "precision_mean": float(np.mean(scores["test_precision"])),
            "recall_mean": float(np.mean(scores["test_recall"])),
            "f1_mean": float(np.mean(scores["test_f1"])),
        }
    # Fallback: одно случайное групповое разбиение
    gss = GroupShuffleSplit(
        n_splits=1, test_size=FALLBACK_TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups))
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    return {
        "folds": 1,
        "accuracy_mean": float(accuracy_score(y[test_idx], y_pred)),
        "precision_mean": float(precision_score(y[test_idx], y_pred, zero_division=0)),
        "recall_mean": float(recall_score(y[test_idx], y_pred, zero_division=0)),
        "f1_mean": float(f1_score(y[test_idx], y_pred, zero_division=0)),
    }


def main():
    print("🚀 Training start")
    rows = _load_data()

    # Базовые и доп. фичи (только если они «стабильны» в данных)
    feats = BASE_FEATURES.copy()
    extra_keep = _available_features(rows, EXTRA_FEATURES)
    feats.extend(extra_keep)

    # Даунсэмплинг по тикеру
    rows_ds = _downsample_by_symbol(rows, MAX_PER_SYMBOL)

    # Матрица
    X, y, groups = _build_matrix(rows_ds, feats)
    if len(X) < 100:
        raise RuntimeError(f"too few rows after cleaning: {len(X)}")

    # CV
    cv_res = _do_cv(X, y, groups)
    print(
        f"CV folds={cv_res['folds']} | F1={cv_res['f1_mean']:.4f} | Acc={cv_res['accuracy_mean']:.4f}"
    )

    # Лог метрик
    ts = datetime.now(timezone.utc).isoformat()
    _append_metrics(ts, len(X), len(set(groups)), feats, cv_res)

    # Гейт сохранения модели
    last_f1 = _read_last_f1()
    new_f1 = cv_res["f1_mean"]
    can_save = (new_f1 >= ABS_MIN_F1_TO_SAVE) and (
        last_f1 is None or (new_f1 + 1e-9) >= (last_f1 - IMPROVEMENT_TOL)
    )
    print(f"Gate: last_f1={last_f1}, new_f1={new_f1:.4f}, save={can_save}")

    if not can_save:
        print("⛔️ Model not saved (quality gate).")
        return

    # Обучаем на всём (на тех же фичах)
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Бэкап старой модели
    try:
        if MODEL_PATH.exists():
            os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
            backup_path = (
                MODEL_BACKUP_DIR
                / f"trained_model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            MODEL_PATH.replace(backup_path)
            print(f"🗄️ backup: {backup_path}")
    except Exception as e:
        print(f"[WARN] backup failed: {e}")

    # Сохраняем новую
    joblib.dump({"model": model, "features": feats}, MODEL_PATH)
    print(f"✅ saved model → {MODEL_PATH}")


if __name__ == "__main__":
    main()
