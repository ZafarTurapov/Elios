#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import shutil
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TRAIN_PATH = Path("/root/stockbot/core/trading/training_data.json")
MODEL_PATH = Path("/root/stockbot/core/training/trained_model.pkl")
if not TRAIN_PATH.exists():
    print("training_data.json not found:", TRAIN_PATH)
    raise SystemExit(1)

d = json.loads(TRAIN_PATH.read_text())
df = pd.DataFrame(d)
print("Loaded rows:", len(df))

# === params ===
MAX_PER_SYMBOL = 12  # cap per symbol (tuneable)
RANDOM_SEED = 42
N_SPLITS = 5

# === downsample per-symbol ===
np.random.seed(RANDOM_SEED)
groups = df["symbol"].astype(str)
kept = []
for sym, g in df.groupby("symbol"):
    if len(g) <= MAX_PER_SYMBOL:
        kept.append(g)
    else:
        kept.append(g.sample(n=MAX_PER_SYMBOL, random_state=RANDOM_SEED))
df_ds = pd.concat(kept, ignore_index=True)
print("After downsample rows:", len(df_ds), f"(max_per_symbol={MAX_PER_SYMBOL})")
print("Top symbols (post-ds):")
print(df_ds["symbol"].value_counts().head(20).to_dict())


# === feature builder ===
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


X = fe(df_ds)
y = (df_ds["label"] == "WIN").astype(int)
groups = df_ds["symbol"].astype(str)

print("Feature matrix shape:", X.shape)
print("Label distribution:\n", dict(y.value_counts()))

# === GroupKFold evaluation ===
gkf = GroupKFold(n_splits=N_SPLITS)
accs = []
precs = []
recs = []
f1s = []
fi_acc = np.zeros(X.shape[1])
fold = 0
for train_idx, test_idx in gkf.split(X, y, groups):
    fold += 1
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    accs.append(accuracy_score(yte, yp))
    precs.append(precision_score(yte, yp, zero_division=0))
    recs.append(recall_score(yte, yp, zero_division=0))
    f1s.append(f1_score(yte, yp, zero_division=0))
    fi_acc += clf.feature_importances_
    print(f"\n=== Fold {fold} ===")
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

print("\n=== GroupKFold summary ===")
import pandas as pd

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
fi_avg = fi_acc / float(N_SPLITS)
print("\nFeature importances (avg):")
for k, v in sorted(zip(feat_names, fi_avg), key=lambda x: -x[1]):
    print(f"  {k:10s}: {v:.4f}")

# === final train on downsampled full dataset (balanced) ===
final_clf = RandomForestClassifier(
    n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
)
final_clf.fit(X, y)

# save model (backup old)
if MODEL_PATH.exists():
    shutil.copy(str(MODEL_PATH), str(MODEL_PATH) + ".bak")
joblib.dump(final_clf, MODEL_PATH)
print("\nSaved model to", MODEL_PATH)
print("Done.")
