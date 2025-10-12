# -*- coding: utf-8 -*-
"""
Elios — Avoidance Grid for Short Risk (v1)
- Считает, как разные пороги p и daily Top-K "avoid short" снижают риск squeeze (open->high)
- Источник p: текущая XGB-модель + калибратор (если есть)
- Окно: последние N торговых дней (по умолчанию 60)
Выход:
  logs/avoidance_grid.json
  logs/avoidance_grid.md
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Tashkent")
ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
OUTJ = LOGS / "avoidance_grid.json"
OUTM = LOGS / "avoidance_grid.md"

DS   = ROOT/"core/data/train/dataset.parquet"
SPEC = ROOT/"core/models/feature_spec.json"
MODEL_V2 = ROOT/"core/models/xgb_spike4_v2.json"
MODEL_V1 = ROOT/"core/models/xgb_spike4_v1.json"

# Параметры
N_DAYS = int(os.getenv("ELIOS_AVOID_N_DAYS", "60"))
P_THRESHOLDS = [round(x,2) for x in np.linspace(0.10, 0.90, 17)]  # 0.10..0.90 шаг 0.05
TOPKS = [5, 10, 20]

def to_num(s): return pd.to_numeric(s, errors="coerce").astype(float)

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
        feats = ["rsi14","ema_dev_pct","atr_pct","volatility_pct",
                 "volume_trend","volume_ratio","gap_up_pct","bullish_body_pct",
                 "mom5_pct","mom20_pct","alpha_score"]
    return feats, target

def load_model():
    import xgboost as xgb
    booster = xgb.Booster()
    if MODEL_V2.exists():
        booster.load_model(str(MODEL_V2)); path = MODEL_V2
    elif MODEL_V1.exists():
        booster.load_model(str(MODEL_V1)); path = MODEL_V1
    else:
        raise FileNotFoundError("Model not found v1/v2")
    return booster, path

def load_calibrator(model_path: Path):
    try:
        import joblib
        p = model_path.with_suffix(".calib.pkl")
        if p.exists():
            return joblib.load(p), str(p)
    except Exception:
        pass
    return None, None

def apply_calib(p: np.ndarray, calib_obj):
    try:
        if calib_obj is None: return p
        typ = calib_obj.get("type"); impl = calib_obj.get("impl")
        if typ == "isotonic":
            return np.asarray(impl.predict(p), float)
        elif typ == "platt":
            return np.asarray(impl.predict_proba(p.reshape(-1,1))[:,1], float)
    except Exception:
        pass
    return p

def main():
    if not DS.exists():
        print(f"Dataset not found: {DS}"); sys.exit(1)

    feats, target = load_spec()
    booster, model_path = load_model()
    calib, calib_path = load_calibrator(model_path)

    df = pd.read_parquet(DS)
    df.columns = [c.lower() for c in df.columns]
    need = {"date","symbol","open_next","high_next", target}
    if not need.issubset(df.columns):
        print(f"dataset must contain: {', '.join(sorted(need))}"); sys.exit(1)

    # даты
    d = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("US/Eastern").dt.date
    df = df.assign(date=d).dropna(subset=["date"]).reset_index(drop=True)

    # признаки
    feats_present = [c for c in feats if c in df.columns]
    if not feats_present:
        print("no features present"); sys.exit(1)

    # окно последних N дней
    uniq = sorted(pd.Series(df["date"].unique()).dropna().tolist())
    last_days = uniq[-N_DAYS:] if len(uniq) >= N_DAYS else uniq

    # предсказания
    import xgboost as xgb
    med = {c: float(to_num(df[c]).median()) for c in feats_present}
    rows = []
    for d_ in last_days:
        sub = df[df["date"] == d_]
        if sub.empty: continue
        X = sub[feats_present].copy()
        for c in feats_present:
            X[c] = to_num(X[c]).fillna(med[c])
        D = xgb.DMatrix(X.values, feature_names=feats_present)
        p = booster.predict(D)
        p = apply_calib(p, calib)
        y = to_num(sub[target]).clip(0,1).values
        open_n = to_num(sub["open_next"]).values
        high_n = to_num(sub["high_next"]).values

        mae_pct = np.maximum(0.0, (high_n / open_n - 1.0) * 100.0)  # риск шорта
        for i in range(len(sub)):
            rows.append({
                "date": str(d_),
                "symbol": sub.iloc[i]["symbol"],
                "p": float(p[i]),
                "y": int(y[i]),
                "mae_pct": float(mae_pct[i]),
            })

    if not rows:
        print("no rows in last days"); sys.exit(1)

    S = pd.DataFrame(rows)

    # --- Пороговая политика: avoid if p >= thr
    thr_grid = []
    for thr in P_THRESHOLDS:
        sel = S[S["p"] >= thr]
        n_all = len(S)
        n_avoid = len(sel)
        avoid_rate = n_avoid / n_all if n_all else 0.0
        # покрытие реальных squeeze
        spikes = S[S["y"] == 1]
        tp = len(sel[sel["y"] == 1])
        spike_cov = tp / len(spikes) if len(spikes) else 0.0
        false_avoid = len(sel[sel["y"] == 0]) / n_all if n_all else 0.0
        saved_mae_mean = float(sel["mae_pct"].mean()) if n_avoid else 0.0
        saved_mae_median = float(sel["mae_pct"].median()) if n_avoid else 0.0
        thr_grid.append({
            "thr": thr, "n_all": n_all, "n_avoid": n_avoid,
            "avoid_rate": avoid_rate, "spike_coverage": spike_cov,
            "false_avoid_rate": false_avoid,
            "saved_mae_mean": saved_mae_mean, "saved_mae_median": saved_mae_median
        })

    # --- Daily Top-K политика: avoid N самых рискованных по каждому дню
    topk_grid = []
    for K in TOPKS:
        parts = []
        for d_ in S["date"].unique():
            day = S[S["date"] == d_].sort_values("p", ascending=False)
            parts.append(day.head(K))
        sel = pd.concat(parts, ignore_index=True)
        n_all = len(S)
        n_avoid = len(sel)
        avoid_rate = n_avoid / n_all if n_all else 0.0
        spikes = S[S["y"] == 1]
        tp = len(sel[sel["y"] == 1])
        spike_cov = tp / len(spikes) if len(spikes) else 0.0
        false_avoid = len(sel[sel["y"] == 0]) / n_all if n_all else 0.0
        saved_mae_mean = float(sel["mae_pct"].mean()) if n_avoid else 0.0
        saved_mae_median = float(sel["mae_pct"].median()) if n_avoid else 0.0
        topk_grid.append({
            "topk": K, "n_all": n_all, "n_avoid": n_avoid,
            "avoid_rate": avoid_rate, "spike_coverage": spike_cov,
            "false_avoid_rate": false_avoid,
            "saved_mae_mean": saved_mae_mean, "saved_mae_median": saved_mae_median
        })

    report = {
        "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"),
        "window_days": len(last_days),
        "model": str(model_path),
        "calibrator": calib_path,
        "threshold_grid": thr_grid,
        "topk_grid": topk_grid
    }
    OUTJ.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = []
    lines.append("# Avoidance Grid (short risk)")
    lines.append(f"- Window days: {len(last_days)}")
    lines.append(f"- Model: `{model_path.name}`" + (f" + `{Path(calib_path).name}`" if calib_path else ""))
    lines.append("## Threshold policy (p >= thr) — top 10 by spike_coverage")
    thr_sorted = sorted(thr_grid, key=lambda r: (r["spike_coverage"], r["saved_mae_mean"]), reverse=True)[:10]
    for r in thr_sorted:
        lines.append(f"- thr={r['thr']:.2f} | avoid={r['avoid_rate']:.1%} | spike_cov={r['spike_coverage']:.1%} | false_avoid={r['false_avoid_rate']:.1%} | saved_mae≈{r['saved_mae_mean']:.2f}% (med {r['saved_mae_median']:.2f}%)")
    lines.append("## Daily Top-K policy — summary")
    for r in topk_grid:
        lines.append(f"- K={r['topk']:>2} | avoid={r['avoid_rate']:.1%} | spike_cov={r['spike_coverage']:.1%} | false_avoid={r['false_avoid_rate']:.1%} | saved_mae≈{r['saved_mae_mean']:.2f}% (med {r['saved_mae_median']:.2f}%)")
    OUTM.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"threshold_best": thr_sorted[0] if thr_sorted else None,
                      "topk_grid": topk_grid}, indent=2, ensure_ascii=False))
    sys.exit(0)
if __name__ == "__main__":
    main()
