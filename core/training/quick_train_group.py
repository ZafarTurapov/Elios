#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

p = Path("/root/stockbot/core/trading/training_data.json")
if not p.exists():
    print("training_data.json not found")
    raise SystemExit(1)
d = json.loads(p.read_text())
df = pd.DataFrame(d)
df = df.dropna(subset=["symbol"])


# features
def fe(df_):
    X = pd.DataFrame()
    X["atr_pct"] = (df_["atr"] / df_["entry_price"]).replace(
        [np.inf, -np.inf], 0
    ).fillna(0) * 100.0
    X["alpha_score"] = df_["alpha_score"].fillna(0)
    X["ema_dev"] = df_["ema_dev"].fillna(0)
    X["rsi"] = df_["rsi"].fillna(50)
    X["vol_ratio"] = df_["vol_ratio"].fillna(1.0)
    X["atr"] = df_["atr"].fillna(0)
    return X


X = fe(df)
y = (df["label"] == "WIN").astype(int)
groups = df["symbol"].astype(str)
gkf = GroupKFold(n_splits=5)
accs = []
precs = []
recs = []
f1s = []
fi_accum = np.zeros(X.shape[1])
fold = 0
for train_idx, test_idx in gkf.split(X, y, groups):
    fold += 1
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    accs.append(accuracy_score(yte, yp))
    precs.append(precision_score(yte, yp, zero_division=0))
    recs.append(recall_score(yte, yp, zero_division=0))
    f1s.append(f1_score(yte, yp, zero_division=0))
    fi_accum += clf.feature_importances_
    print(
        f"Fold {fold}: acc={accs[-1]:.4f} prec={precs[-1]:.4f} rec={recs[-1]:.4f} f1={f1s[-1]:.4f}"
    )
print(
    "GroupKFold summary:",
    "acc_mean",
    round(pd.Series(accs).mean(), 4),
    "f1_mean",
    round(pd.Series(f1s).mean(), 4),
)
# train final on full data but using class weights to mitigate freq-ticker bias
final_clf = RandomForestClassifier(
    n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
)
final_clf.fit(X, y)
MODEL_PATH = Path("/root/stockbot/core/training/trained_model.pkl")
if MODEL_PATH.exists():
    shutil.copy(MODEL_PATH, str(MODEL_PATH) + ".bak")
import joblib

joblib.dump(final_clf, MODEL_PATH)
print("Saved model to", MODEL_PATH)
print("Feature importances (final):")
fi = dict(zip(X.columns.tolist(), final_clf.feature_importances_))
for k, v in sorted(fi.items(), key=lambda x: -x[1]):
    print(f"  {k:10s}: {v:.4f}")
