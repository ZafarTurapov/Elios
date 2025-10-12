CUTOFF_DATE = "2012-01-01"
import os, io, json, time, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

ROOT = Path("/root/stockbot")
EOD_DIR   = ROOT/"core/data/eod"
TRAIN_DIR = ROOT/"core/data/train"
MODELS    = ROOT/"core/models"
VERSIONS  = MODELS/"versions"; VERSIONS.mkdir(parents=True, exist_ok=True)
REPORTS   = MODELS/"reports";  REPORTS.mkdir(parents=True, exist_ok=True)
LOGS      = ROOT/"logs";       LOGS.mkdir(exist_ok=True)
ACTIVE    = ROOT/"core/trading/candidates_active.json"
CANDS     = ROOT/"core/trading/candidates.json"
FEAT_SPEC = MODELS/"feature_spec.json"
HISTORY   = LOGS/"train_history.json"
LATEST    = MODELS/"xgb_spike4_v1.json"

# --- фичи/таргет ---
RSI_WIN, EMA_WIN, ATR_WIN = 14, 10, 14
VOL_WIN, VOLR_WIN = 30, 10
MOM5_LAG, MOM20_LAG = 5, 20
TARGET_PCT = 0.04
VALID_FRAC = 0.20
RANDOM_SEED = 42
ES_ROUNDS = 200
N_ROUNDS  = 4000
PARAMS = {
    "objective":"binary:logistic","eval_metric":"auc","tree_method":"hist","device":"cpu",
    "eta":0.05,"max_depth":6,"subsample":0.8,"colsample_bytree":0.8,"lambda":1.0,"alpha":0.0,
    "seed":RANDOM_SEED,"nthread":max(1, os.cpu_count()//2)
}
BLOCKLIST = set(["ATVI","SPLK","CS","KSU","DNKN","SAVE","YNDX"])

def tg(msg:str):
    try:
        from core.utils.telegram import send_telegram_message
        send_telegram_message(msg)
    except Exception as e:
        print(f"[WARN] telegram: {e}")

def load_candidates():
    syms = []
    try: syms = json.loads(CANDS.read_text())
    except Exception: pass
    syms = [s.strip().upper() for s in syms if isinstance(s,str) and s.strip()]
    syms = [s for s in dict.fromkeys(syms) if s not in BLOCKLIST]
    return syms

def save_active(syms): ACTIVE.write_text(json.dumps(syms, indent=2))

def try_yf_history(sym:str):
    import yfinance as yf
    tk = yf.Ticker(sym)
    for kw in (
        dict(period="max", interval="1d", auto_adjust=False, actions=False, raise_errors=False),
        dict(period="10y",  interval="1d", auto_adjust=False, actions=False, raise_errors=False),
        dict(start ="2016-01-01", interval="1d", auto_adjust=False, actions=False, raise_errors=False),
    ):
        try:
            df = tk.history(**kw)
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception:
            pass
        time.sleep(0.2)
    return None

def try_stooq_us(sym:str):
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty and "Close" in df.columns:
                df["Adj Close"] = df["Close"]; df["Date"] = pd.to_datetime(df["Date"])
                return df
    except Exception:
        pass
    return None

def eod_update():
    EOD_DIR.mkdir(parents=True, exist_ok=True)
    syms = load_candidates()
    act, fail, skip, dl = [], [], 0, 0
    for s in syms:
        fp = EOD_DIR/f"{s}.csv"
        if fp.exists() and fp.stat().st_size>0:
            act.append(s); skip+=1; continue
        df = try_yf_history(s) or try_stooq_us(s)
        if df is None:
            fail.append(s); continue
        df = df.reset_index()
        if "Adj Close" not in df.columns: df["Adj Close"] = df["Close"]
        df = df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
        df.to_csv(fp, index=False)
        act.append(s); dl+=1
    save_active(act)
    (LOGS/"eod_failed.json").write_text(json.dumps(fail, indent=2, ensure_ascii=False))
    return {"universe":len(syms),"active":len(act),"downloaded":dl,"skipped":skip,"failed":len(fail)}

NUM_COLS = [
    "rsi14","ema_dev_pct","atr_pct","volatility_pct",
    "volume_trend","volume_ratio","gap_up_pct","bullish_body_pct",
    "mom5_pct","mom20_pct","open","close","high","low","volume","open_next","high_next"
]
def _coerce_num(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_features():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    syms = json.loads(ACTIVE.read_text())
    rows, used, skipped = [], 0, 0
    for s in syms:
        f = EOD_DIR/f"{s}.csv"
        if not f.exists(): continue
        raw = pd.read_csv(f)
        if raw.empty: continue
        raw.columns = [c.strip().lower().replace(" ","_") for c in raw.columns]
        if "adj_close" not in raw.columns: raw["adj_close"] = raw.get("close", np.nan)
        for base in ("open","high","low","close","adj_close","volume"):
            if base in raw.columns: raw[base] = pd.to_numeric(raw[base], errors="coerce")
        raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce").dt.tz_localize(None)
        raw = raw.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date").reset_index(drop=True)
        if len(raw) < (VOL_WIN+2): skipped+=1; continue
        o,h,l,c,v = raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"].replace(0, np.nan)

        rsi = RSIIndicator(close=c, window=RSI_WIN).rsi()
        ema = EMAIndicator(close=c, window=EMA_WIN).ema_indicator()
        ema_dev_pct = (c-ema)/ema*100.0
        atr = AverageTrueRange(high=h, low=l, close=c, window=ATR_WIN).average_true_range()
        atr_pct = atr/c*100.0
        vola_pct = c.pct_change().rolling(VOL_WIN).std()*100.0
        vol_ema = EMAIndicator(v.ffill(), window=VOLR_WIN).ema_indicator()
        volume_trend = v/vol_ema
        volume_ratio = v/v.rolling(VOLR_WIN).mean()
        gap_up_pct = (o - c.shift(1))/c.shift(1)*100.0
        bullish_body_pct = (c - o)/o*100.0
        mom5_pct  = c.pct_change(MOM5_LAG)*100.0
        mom20_pct = c.pct_change(MOM20_LAG)*100.0

        o_next, h_next = o.shift(-1), h.shift(-1)
        spike_next_pct = (h_next/o_next - 1.0)*100.0
        target = (spike_next_pct >= (TARGET_PCT*100.0)).astype("Int64")

        feat = pd.DataFrame({
            "symbol": s, "date": raw["date"],
            "rsi14": rsi, "ema_dev_pct": ema_dev_pct, "atr_pct": atr_pct, "volatility_pct": vola_pct,
            "volume_trend": volume_trend, "volume_ratio": volume_ratio,
            "gap_up_pct": gap_up_pct, "bullish_body_pct": bullish_body_pct,
            "mom5_pct": mom5_pct, "mom20_pct": mom20_pct,
            "open":o,"close":c,"high":h,"low":l,"volume":v,
            "open_next":o_next,"high_next":h_next,
            "target_spike4": target
        })
        feat = _coerce_num(feat, NUM_COLS)
        feat = feat.dropna(subset=NUM_COLS+["target_spike4"]).reset_index(drop=True)
        if feat.empty: skipped+=1; continue
        rows.append(feat); used+=1

    if not rows: raise SystemExit("no data for features")
    data = pd.concat(rows, ignore_index=True)
    for c in NUM_COLS: data[c] = data[c].astype("float32")
    ds_path = TRAIN_DIR/"dataset.parquet"
    data.to_parquet(ds_path, index=False)

    FEAT_SPEC.write_text(json.dumps({
        "target":"target_spike4",
        "features":[
            "rsi14","ema_dev_pct","atr_pct","volatility_pct",
            "volume_trend","volume_ratio","gap_up_pct","bullish_body_pct",
            "mom5_pct","mom20_pct"
        ],
        "windows":{"rsi":RSI_WIN,"ema":EMA_WIN,"atr":ATR_WIN,"volatility":VOL_WIN,"volratio":VOLR_WIN,"mom5":MOM5_LAG,"mom20":MOM20_LAG},
        "target_pct":TARGET_PCT
    }, indent=2, ensure_ascii=False))
    pos = int(data["target_spike4"].sum()); rate = pos/len(data)
    return {"symbols_used":used,"rows":len(data),"positives":pos,"pos_rate":rate,"dataset":str(ds_path),"spec":str(FEAT_SPEC)}

def train_and_register():
    spec = json.loads(FEAT_SPEC.read_text())
    feats, target = spec["features"], spec["target"]
    df = pd.read_parquet(TRAIN_DIR/"dataset.parquet").sort_values(["symbol","date"]).reset_index(drop=True)
    df = df[feats+[target,"symbol","date"]].dropna().reset_index(drop=True)

    tr_idx, va_idx = [], []
    for _, g in df.groupby("symbol", sort=False):
        n=len(g); cut=max(1,int(n*(1-VALID_FRAC))); idx=g.index.to_numpy()
        tr_idx.append(idx[:cut]); va_idx.append(idx[cut:])
    tr_idx, va_idx = np.concatenate(tr_idx), np.concatenate(va_idx)
    Xtr = df.loc[tr_idx, feats].astype("float32"); ytr = df.loc[tr_idx, target].astype("int32")
    Xva = df.loc[va_idx, feats].astype("float32"); yva = df.loc[va_idx, target].astype("int32")

    pos = int((ytr==1).sum()); neg = int((ytr==0).sum())
    scale = (neg/pos) if pos>0 else 1.0
    params = dict(PARAMS); params["scale_pos_weight"]=scale

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feats)
    booster = xgb.train(params, dtr, num_boost_round=N_ROUNDS, evals=[(dva,"valid")], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    best_iter = int(booster.best_iteration if booster.best_iteration is not None else booster.num_boosted_rounds()-1)
    proba = booster.predict(dva, iteration_range=(0,best_iter+1))

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support
    auc = float(roc_auc_score(yva, proba)); ap  = float(average_precision_score(yva, proba))
    fpr,tpr,thr = roc_curve(yva, proba); opt = int(np.argmax(tpr-fpr)); thr_opt = float(thr[opt])
    def prf_at(th):
        pred = (proba>=th).astype(int)
        p,r,f,_ = precision_recall_fscore_support(yva, pred, average="binary", zero_division=0)
        return {"threshold":float(th),"precision":float(p),"recall":float(r),"f1":float(f)}
    metrics = {"AUC":auc,"AP":ap,"thresholds":{"t0.50":prf_at(0.50),"t_opt":prf_at(thr_opt)}}

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    ver_path = VERSIONS/f"xgb_spike4_v1_{ts}.json"
    rep_path = REPORTS /f"xgb_spike4_v1_{ts}.json"
    booster.save_model(ver_path)
    report = {
        "timestamp_utc": ts, "params": params, "best_iteration": best_iter,
        "rows": len(df),"train_rows": int(len(Xtr)),"valid_rows": int(len(Xva)),
        "class_balance_train":{"pos":pos,"neg":neg,"pos_rate": float(pos/(pos+neg))},
        "metrics": metrics, "features": feats, "target": target,
        "paths":{"version_model": str(ver_path)}
    }
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # сравнить с предыдущим отчётом (если есть)
    def load_old_report():
        try:
            reps = sorted(REPORTS.glob("xgb_spike4_v1_*.json"))
            prevs = [p for p in reps if p.name != rep_path.name]
            if not prevs: return None
            prev = prevs[-1]
            return json.loads(prev.read_text())
        except Exception:
            return None
    promote, why = True, "better_or_first"
    prev = load_old_report()
    if prev:
        old_auc = float(prev["metrics"]["AUC"]); old_ap = float(prev["metrics"]["AP"])
        if (auc + 1e-9) < (old_auc - 0.01) or (ap + 1e-9) < (old_ap - 0.01):
            promote, why = False, f"kept_old (old AUC={old_auc:.4f}, AP={old_ap:.4f})"
    if promote:
        try: shutil.copy2(ver_path, LATEST)
        except Exception: os.replace(ver_path, LATEST)

    hist = []
    try: hist = json.loads(HISTORY.read_text())
    except Exception: pass
    hist.append({"ts":ts,"model":str(ver_path),"report":str(rep_path),"promoted":promote,"why":why,"AUC":auc,"AP":ap})
    HISTORY.write_text(json.dumps(hist, indent=2, ensure_ascii=False))

    return {"ver":str(ver_path),"rep":str(rep_path),"promoted":promote,"why":why,"AUC":auc,"AP":ap,"best_iter":best_iter}

def main():
    eod = eod_update()
    feats = build_features()
    res = train_and_register()
    msg = (
        "🤖 *Elios — TrainDaily*\n"
        f"📦 EOD: act={eod['active']}/{eod['universe']} | dl+={eod['downloaded']} | fail={eod['failed']}\n"
        f"🧱 Feats: rows={feats['rows']:,} | syms={feats['symbols_used']} | pos={feats['positives']:,} ({feats['pos_rate']:.2%})\n"
        f"🎯 XGB v1: AUC={res['AUC']:.4f} | AP={res['AP']:.4f} | iters={res['best_iter']} | promoted={res['promoted']}\n"
        f"🗂 {res['why']}"
    )
    print(msg); tg(msg)

if __name__ == "__main__":
    main()
