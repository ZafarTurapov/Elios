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

p = Path("/root/stockbot/core/trading/training_data.json")
if not p.exists():
    print("training_data.json not found:", p)
    raise SystemExit(1)
d = json.loads(p.read_text())
df = pd.DataFrame(d)
# parse timestamp (prefer timestamp, fallback to timestamp_exit)
df["ts"] = pd.to_datetime(df.get("timestamp").astype(str), errors="coerce")
df["ts"] = df["ts"].fillna(
    pd.to_datetime(df.get("timestamp_exit").astype(str), errors="coerce")
)
df = df.dropna(subset=["ts"])
df = df.sort_values("ts").reset_index(drop=True)
n = len(df)
if n < 50:
    print("Not enough rows for time-split:", n)
    raise SystemExit(0)
cut = int(n * 0.8)
train = df.iloc[:cut].copy()
test = df.iloc[cut:].copy()


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


Xtr = fe(train)
Xte = fe(test)
ytr = (train["label"] == "WIN").astype(int)
yte = (test["label"] == "WIN").astype(int)
print("rows total, train, test:", n, len(Xtr), len(Xte))
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, yp), 4))
print(
    "Precision:",
    round(precision_score(yte, yp, zero_division=0), 4),
    "Recall:",
    round(recall_score(yte, yp, zero_division=0), 4),
    "F1:",
    round(f1_score(yte, yp, zero_division=0), 4),
)
print("\nConfusion matrix:\n", confusion_matrix(yte, yp))
print("\nClassification report:\n", classification_report(yte, yp, zero_division=0))
fi = dict(
    zip(
        ["atr_pct", "alpha_score", "ema_dev", "rsi", "vol_ratio", "atr"],
        clf.feature_importances_,
    )
)
print("\nFeature importances:")
for k, v in sorted(fi.items(), key=lambda x: -x[1]):
    print(f"  {k:10s}: {v:.4f}")
