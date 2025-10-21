# -*- coding: utf-8 -*-
"""
Elios — Calibrate Model (v1)
Тренирует Platt (LogisticRegression) и IsotonicRegression на валидационном хвосте,
выбирает лучший по Brier и сохраняет калибратор рядом с моделью.
Выход: 0=OK, 1=FAIL
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Tashkent")
ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
OUT_JSON = LOGS / "calibration_report.json"
OUT_MD = LOGS / "calibration_report.md"

DS = ROOT / "core" / "data" / "train" / "dataset.parquet"
SPEC = ROOT / "core" / "models" / "feature_spec.json"
MODEL_V2 = ROOT / "core" / "models" / "xgb_spike4_v2.json"
MODEL_V1 = ROOT / "core" / "models" / "xgb_spike4_v1.json"

# Настройки (ENV-override)
N_DAYS = int(os.getenv("ELIOS_CALIB_N_DAYS", "90"))  # последние N торговых дней
USE_ISO_IF_TIE = os.getenv(
    "ELIOS_CALIB_TIE", "isotonic"
)  # isotonic|platt при равенстве Brier


def now():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def load_spec():
    feats, target = None, "target_spike4"
    if SPEC.exists():
        try:
            s = json.loads(SPEC.read_text())
            feats = s.get("features") or None
            target = s.get("target", target)
        except Exception:
            pass
    if not feats:
        feats = [
            "rsi14",
            "ema_dev_pct",
            "atr_pct",
            "volatility_pct",
            "volume_trend",
            "volume_ratio",
            "gap_up_pct",
            "bullish_body_pct",
            "mom5_pct",
            "mom20_pct",
            "alpha_score",
        ]
    return feats, target


def load_model_path():
    if MODEL_V2.exists():
        return MODEL_V2
    if MODEL_V1.exists():
        return MODEL_V1
    raise FileNotFoundError("Model not found: xgb_spike4_v2.json/v1.json")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def brier(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def main():
    try:
        if not DS.exists():
            raise FileNotFoundError(f"Dataset not found: {DS}")
        feats, target = load_spec()
        model_path = load_model_path()

        # Загружаем датасет
        df = pd.read_parquet(DS)
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df or "symbol" not in df:
            raise ValueError("dataset must contain 'date' and 'symbol'")
        if target not in df:
            raise ValueError(f"target '{target}' not found in dataset")

        # Дата в US/Eastern (дневная гранулярность)
        d = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.tz_convert("US/Eastern")
            .dt.date
        )
        df = df.assign(date=d).dropna(subset=["date"]).reset_index(drop=True)

        # Последние N торговых дат
        uniq = sorted(pd.Series(df["date"].unique()).dropna().tolist())
        if not uniq:
            raise ValueError("no dates in dataset")
        val_days = uniq[-N_DAYS:]

        feats_present = [c for c in feats if c in df.columns]
        if not feats_present:
            raise ValueError("no features from spec present in dataset")

        # Предсказания модели
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        med = {c: float(to_num(df[c]).median()) for c in feats_present}

        preds, ys = [], []
        for d_ in val_days:
            sub = df[df["date"] == d_]
            if sub.empty:
                continue
            X = sub[feats_present].copy()
            for c in feats_present:
                X[c] = to_num(X[c]).fillna(med[c])
            D = xgb.DMatrix(X.values, feature_names=feats_present)
            p = booster.predict(D)  # для binary:logistic — уже вероятности
            y = to_num(sub[target]).fillna(0.0).clip(0, 1).values
            preds.append(p)
            ys.append(y)

        if not preds:
            raise ValueError("no validation rows in last N days")

        P = np.concatenate(preds)
        Y = np.concatenate(ys)
        base_brier = brier(P, Y)

        # Калибраторы
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        # Isotonic (монотонный калибратор)
        iso = IsotonicRegression(out_of_bounds="clip")
        P_iso = iso.fit_transform(P, Y)
        brier_iso = brier(P_iso, Y)
        ll_iso = float(log_loss(Y, np.clip(P_iso, 1e-6, 1 - 1e-6)))

        # Platt (логистическая регрессия по скалярной вероятности)
        # добавим маленький регуляризатор для устойчивости
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        P_col = P.reshape(-1, 1)
        platt.fit(P_col, Y)
        P_pl = platt.predict_proba(P_col)[:, 1]
        brier_pl = brier(P_pl, Y)
        ll_pl = float(log_loss(Y, np.clip(P_pl, 1e-6, 1 - 1e-6)))

        # Выбор лучшего калибратора
        if abs(brier_iso - brier_pl) < 1e-6:
            best = USE_ISO_IF_TIE
        else:
            best = "isotonic" if brier_iso < brier_pl else "platt"

        # Сохраняем калибратор
        import joblib

        if best == "isotonic":
            calib_obj = {"type": "isotonic", "impl": iso}
        else:
            calib_obj = {"type": "platt", "impl": platt}
        calib_path = model_path.with_suffix(".calib.pkl")
        joblib.dump(calib_obj, calib_path)

        # Отчёты
        rep = {
            "ts": now(),
            "model": str(model_path),
            "days": len(val_days),
            "rows": int(len(P)),
            "base": {
                "brier": base_brier,
                "logloss": float(log_loss(Y, np.clip(P, 1e-6, 1 - 1e-6))),
            },
            "isotonic": {"brier": brier_iso, "logloss": ll_iso},
            "platt": {"brier": brier_pl, "logloss": ll_pl},
            "best": best,
            "calibrator_path": str(calib_path),
        }
        OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

        md = []
        md.append("# Calibration Report")
        md.append(f"- Model: `{model_path}`")
        md.append(f"- Val days: {len(val_days)} | Rows: {len(P)}")
        md.append(f"- Baseline Brier: {base_brier:.5f}")
        md.append(f"- Isotonic  — Brier: {brier_iso:.5f}, LogLoss: {ll_iso:.5f}")
        md.append(f"- Platt     — Brier: {brier_pl:.5f}, LogLoss: {ll_pl:.5f}")
        md.append(f"- **Selected**: {best} → saved: `{calib_path.name}`")
        OUT_MD.write_text("\n".join(md), encoding="utf-8")

        print(json.dumps(rep, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        print(f"Calibration error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
