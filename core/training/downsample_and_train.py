#!/usr/bin/env python3
import json, pandas as pd, numpy as np, shutil, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

P = Path("/root/stockbot/core/trading/training_data.json")
if not P.exists():
    print("training_data.json not found:", P); raise SystemExit(1)
d = json.loads(P.read_text())
df = pd.DataFrame(d)
print("original rows:", len(df))
# parameters
MAX_PER_SYMBOL = 12  # cap per symbol (changeable)
RANDOM_SEED = 42

# downsample
np.random.seed(RANDOM_SEED)
groups = df['symbol'].astype(str)
kept_rows = []
for sym, g in df.groupby('symbol'):
    n = len(g)
    if n <= MAX_PER_SYMBOL:
        kept_rows.append(g)
    else:
        kept_rows.append(g.sample(n=MAX_PER_SYMBOL, random_state=RANDOM_SEED))
df_ds = pd.concat(kept_rows, ignore_index=True)
print("downsampled rows:", len(df_ds), f"(max_per_symbol={MAX_PER_SYMBOL})")
print("top symbols after downsample:")
print(df_ds['symbol'].value_counts().head(20).to_dict())

# features
def fe(df_):
    X = pd.DataFrame()
    X['atr_pct'] = (df_['atr'] / df_['entry_price']).replace([np.inf,-np.inf],0).fillna(0) * 100.0
    X['alpha_score'] = df_['alpha_score'].fillna(0)
    X['ema_dev'] = df_['ema_dev'].fillna(0)
    X['rsi'] = df_['rsi'].fillna(50)
    X['vol_ratio'] = df_['vol_ratio'].fillna(1.0)
    X['atr'] = df_['atr'].fillna(0)
    return X

X = fe(df_ds)
y = (df_ds['label'] == 'WIN').astype(int)
groups = df_ds['symbol'].astype(str)

# GroupKFold evaluation
gkf = GroupKFold(n_splits=5)
accs=[]; precs=[]; recs=[]; f1s=[]; fi_accum = np.zeros(X.shape[1])
fold = 0
for train_idx, test_idx in gkf.split(X, y, groups):
    fold += 1
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    accs.append(accuracy_score(yte, yp))
    precs.append(precision_score(yte, yp, zero_division=0))
    recs.append(recall_score(yte, yp, zero_division=0))
    f1s.append(f1_score(yte, yp, zero_division=0))
    fi_accum += clf.feature_importances_
    print(f"Fold {fold}: acc={accs[-1]:.4f} prec={precs[-1]:.4f} rec={recs[-1]:.4f} f1={f1s[-1]:.4f}")
print("GroupKFold summary:", "acc_mean", round(pd.Series(accs).mean(),4), "f1_mean", round(pd.Series(f1s).mean(),4))

# final train on downsampled full dataset with balanced classes
final_clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
final_clf.fit(X, y)

MODEL_PATH = Path("/root/stockbot/core/training/trained_model.pkl")
if MODEL_PATH.exists():
    shutil.copy(str(MODEL_PATH), str(MODEL_PATH)+'.bak')
joblib.dump(final_clf, MODEL_PATH)
print("Saved model to", MODEL_PATH)
fi = dict(zip(X.columns.tolist(), final_clf.feature_importances_))
print("Feature importances (final):")
for k,v in sorted(fi.items(), key=lambda x:-x[1]):
    print(f"  {k:10s}: {v:.4f}")
