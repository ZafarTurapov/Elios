# -*- coding: utf-8 -*-
"""
Elios — Shadow Metrics (daily p@5/10/20 & Brier) + Calibrator support
- Считает precision@K по дням (последние N дат) и Brier
- Если рядом с моделью найден калибратор *.calib.pkl — применяет его (можно отключить ELIOS_SHADOW_USE_CALIB=0)
- Сохраняет: logs/shadow_metrics.json + .md
- Exit: 0=OK, 2=WARN, 1=FAIL
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Tashkent")
ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
OUT_JSON = LOGS / "shadow_metrics.json"
OUT_MD = LOGS / "shadow_metrics.md"

DS = ROOT / "core" / "data" / "train" / "dataset.parquet"
SPEC = ROOT / "core" / "models" / "feature_spec.json"
MODEL_V2 = ROOT / "core" / "models" / "xgb_spike4_v2.json"
MODEL_V1 = ROOT / "core" / "models" / "xgb_spike4_v1.json"

# Настройки
N_DAYS = int(os.getenv("ELIOS_SHADOW_N_DAYS", "12"))
KS = [5, 10, 20]
USE_CALIB = os.getenv("ELIOS_SHADOW_USE_CALIB", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def load_spec():
    feats = None
    target = "target_spike4"
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


def load_model():
    import xgboost as xgb

    booster = xgb.Booster()
    if MODEL_V2.exists():
        booster.load_model(str(MODEL_V2))
        path = MODEL_V2
    elif MODEL_V1.exists():
        booster.load_model(str(MODEL_V1))
        path = MODEL_V1
    else:
        raise FileNotFoundError("Model not found: xgb_spike4_v2.json/v1.json")
    return booster, path


def load_calibrator(model_path: Path):
    try:
        import joblib

        calib_path = model_path.with_suffix(".calib.pkl")
        if calib_path.exists():
            return joblib.load(calib_path), str(calib_path)
    except Exception:
        pass
    return None, None


def apply_calib(p: np.ndarray, calib_obj) -> np.ndarray:
    try:
        if calib_obj is None:
            return p
        typ = calib_obj.get("type")
        impl = calib_obj.get("impl")
        if typ == "isotonic":
            return np.asarray(impl.predict(p), float)
        elif typ == "platt":
            return np.asarray(impl.predict_proba(p.reshape(-1, 1))[:, 1], float)
        else:
            return p
    except Exception:
        return p


def to_num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def main():
    try:
        if not DS.exists():
            raise FileNotFoundError(f"Dataset not found: {DS}")
        df = pd.read_parquet(DS)
        feats, target = load_spec()
        booster, model_path = load_model()
        calib, calib_path = load_calibrator(model_path)

        # Нормализация
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "date" not in df.columns or "symbol" not in df.columns:
            raise ValueError("dataset must contain 'date' and 'symbol'")
        df["date"] = (
            pd.to_datetime(df["date"], utc=True, errors="coerce")
            .dt.tz_convert("US/Eastern")
            .dt.date
        )
        if target not in df.columns:
            raise ValueError(f"target '{target}' not found in dataset")
        feats_present = [c for c in feats if c in df.columns]
        if not feats_present:
            raise ValueError("no features from spec are present in dataset")

        med = {
            c: float(pd.to_numeric(df[c], errors="coerce").median())
            for c in feats_present
        }
        dates_sorted = sorted(pd.Series(df["date"].unique()).dropna().tolist())
        if not dates_sorted:
            raise ValueError("no dates in dataset")
        last_days = dates_sorted[-N_DAYS:]

        import xgboost as xgb

        day_rows = []
        all_pred = []
        all_true = []
        for d in last_days:
            day_df = df[df["date"] == d]
            if day_df.empty:
                continue
            X = day_df[feats_present].copy()
            for c in feats_present:
                X[c] = to_num(X[c]).fillna(med[c])
            dmat = xgb.DMatrix(X.values, feature_names=feats_present)
            pred = booster.predict(dmat)
            if USE_CALIB and calib is not None:
                pred = apply_calib(pred, calib)
            y = to_num(day_df[target]).fillna(0.0).clip(0, 1).values

            order = np.argsort(-pred)
            metrics = {}
            for K in KS:
                kk = min(K, len(pred))
                metrics[f"p@{K}"] = (
                    None if kk <= 0 else float(np.nanmean(y[order[:kk]]))
                )

            all_pred.append(pred)
            all_true.append(y)
            day_rows.append({"date": str(d), "n": int(len(day_df)), **metrics})

        if not day_rows:
            raise ValueError("no rows for last days")
        pred_all = np.concatenate(all_pred)
        true_all = np.concatenate(all_true)
        brier = float(np.nanmean((pred_all - true_all) ** 2))

        def mean_safe(key):
            vals = [r[key] for r in day_rows if r.get(key) is not None]
            return None if not vals else float(np.mean(vals))

        summary = {
            "p@5": mean_safe("p@5"),
            "p@10": mean_safe("p@10"),
            "p@20": mean_safe("p@20"),
            "brier": brier,
            "days": len(day_rows),
            "n_total": int(sum(r["n"] for r in day_rows)),
            "calibrated": bool(USE_CALIB and calib is not None),
            "calibrator_path": calib_path,
        }
        report = {
            "ts": now_str(),
            "model": str(model_path),
            "features_used": feats_present,
            "target": target,
            "last_days": [r["date"] for r in day_rows],
            "by_day": day_rows,
            "summary": summary,
        }
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        lines = []
        lines.append("# Shadow Metrics")
        lines.append(f"- Model: `{model_path}`")
        lines.append(f"- Days: {summary['days']} | Rows: {summary['n_total']}")
        lines.append(f"- Calibrated: {'yes' if summary['calibrated'] else 'no'}")
        if summary["calibrated"] and summary["calibrator_path"]:
            lines.append(f"- Calibrator: `{Path(summary['calibrator_path']).name}`")
        lines.append(
            f"- p@5: {summary['p@5'] if summary['p@5'] is not None else 'n/a'}"
        )
        lines.append(
            f"- p@10: {summary['p@10'] if summary['p@10'] is not None else 'n/a'}"
        )
        lines.append(
            f"- p@20: {summary['p@20'] if summary['p@20'] is not None else 'n/a'}"
        )
        lines.append(f"- Brier: {summary['brier']:.4f}")
        lines.append("## By day")
        for r in day_rows:
            parts = [f"n={r['n']}"] + [
                f"{k}={r[k]:.3f}" for k in r if k.startswith("p@") and r[k] is not None
            ]
            lines.append(f"- {r['date']}: " + ", ".join(parts))
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        print(f"Shadow metrics error: {type(e).__name__}: {e}")
        try:
            OUT_JSON.write_text(
                json.dumps(
                    {"ts": now_str(), "status": "FAIL", "error": str(e)},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
