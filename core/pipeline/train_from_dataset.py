import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/root/stockbot")
DS = ROOT / "core/data/train/dataset.parquet"
SPEC = ROOT / "core/models/feature_spec.json"
OUTM = ROOT / "core/models/xgb_spike4_v2.json"
OUTJ = ROOT / "core/models/xgb_spike4_v2.meta.json"

assert DS.exists(), f"dataset not found: {DS}"

# features / target
if SPEC.exists():
    spec = json.loads(SPEC.read_text())
    FEATS = spec.get("features", [])
    TARGET = spec.get("target", "target_spike4")
else:
    FEATS = [
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
    ]
    TARGET = "target_spike4"

df = pd.read_parquet(DS)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=FEATS + [TARGET]).reset_index(drop=True)

# split: last 180 days => valid
cut = df["date"].max() - pd.Timedelta(days=180)
tr = df[df["date"] < cut]
va = df[df["date"] >= cut]
assert len(tr) > 0 and len(va) > 0, "empty split"

Xtr, ytr = tr[FEATS].values, tr[TARGET].astype(int).values
Xva, yva = va[FEATS].values, va[TARGET].astype(int).values

dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATS)
dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATS)

params = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc"],
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2.0,
}
bst = xgb.train(
    params,
    dtr,
    num_boost_round=2000,
    evals=[(dtr, "train"), (dva, "valid")],
    early_stopping_rounds=100,
    verbose_eval=False,
)

proba = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
auc = roc_auc_score(yva, proba)
ap = average_precision_score(yva, proba)

bst.save_model(str(OUTM))
meta = {
    "features": FEATS,
    "target": TARGET,
    "train_rows": int(len(tr)),
    "valid_rows": int(len(va)),
    "date_min": str(df["date"].min()),
    "date_max": str(df["date"].max()),
    "valid_window_days": 180,
    "best_iteration": int(bst.best_iteration),
    "auc_valid": float(auc),
    "ap_valid": float(ap),
}
OUTJ.write_text(json.dumps(meta, indent=2))
print(f"✅ trained: {OUTM}")
print(f"📊 AUC={auc:.4f}  AP={ap:.4f}  best_iter={bst.best_iteration}")
