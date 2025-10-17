#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

p = Path("/root/stockbot/core/trading/training_data.json")
if not p.exists():
    print("training_data.json not found:", p)
    raise SystemExit(1)
d = json.loads(p.read_text())
df = pd.DataFrame(d)
# require symbol, timestamp
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
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    accs.append(accuracy_score(yte, yp))
    precs.append(precision_score(yte, yp, zero_division=0))
    recs.append(recall_score(yte, yp, zero_division=0))
    f1s.append(f1_score(yte, yp, zero_division=0))
    fi_accum += clf.feature_importances_
    print(f"=== Fold {fold} ===")
    print("size train/test:", len(Xtr), len(Xte))
    print(
        "accuracy:",
        round(accs[-1], 4),
        "precision:",
        round(precs[-1], 4),
        "recall:",
        round(recs[-1], 4),
        "f1:",
        round(f1s[-1], 4),
    )
    print("confusion matrix:\n", confusion_matrix(yte, yp))
    print(classification_report(yte, yp, zero_division=0))
print("=== GroupKFold summary (5 folds) ===")
print(
    "Accuracy mean/std:",
    round(pd.Series(accs).mean(), 4),
    "/",
    round(pd.Series(accs).std(), 4),
)
print(
    "Precision mean/std:",
    round(pd.Series(precs).mean(), 4),
    "/",
    round(pd.Series(precs).std(), 4),
)
print(
    "Recall mean/std:",
    round(pd.Series(recs).mean(), 4),
    "/",
    round(pd.Series(recs).std(), 4),
)
print(
    "F1 mean/std:", round(pd.Series(f1s).mean(), 4), "/", round(pd.Series(f1s).std(), 4)
)
feat_names = X.columns.tolist()
fi_avg = fi_accum / 5.0
print("\nFeature importances (avg):")
for k, v in sorted(zip(feat_names, fi_avg), key=lambda x: -x[1]):
    print(f"  {k:10s}: {v:.4f}")
