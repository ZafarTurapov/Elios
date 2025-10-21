# -*- coding: utf-8 -*-
"""
Dux Short — MVP фильтров для отбора шорт-кандидатов.

Запуск:
  PYTHONPATH=. python -m core.signals.filters_dux \
    --universe core/trading/candidates.json \
    --out core/trading/signals.json \
    --min-price 2 --max-price 20 --min-gap 5 --min-rvol 2 --max-float-m 50
"""
import os, sys, json, math, argparse, time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yfinance as yf

DEF_UNIVERSE = "core/trading/candidates.json"
DEF_OUT = "core/trading/signals.json"

def load_universe(path: str) -> List[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "tickers" in data:
            return [str(t) for t in data["tickers"]]
        if isinstance(data, list):
            return [str(t) for t in data]
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
    if Path(DEF_UNIVERSE).exists():
        try:
            data = json.loads(Path(DEF_UNIVERSE).read_text(encoding="utf-8"))
            return list(map(str, data.get("tickers", data if isinstance(data, list) else [])))
        except Exception as e:
            print(f"[ERR] fallback read {DEF_UNIVERSE} failed: {e}")
    return []

def load_float_map() -> Dict[str, Optional[float]]:
    candidates = [
        "fundamentals_with_labels.csv",
        "data/fundamentals_with_labels.csv",
        "core/training/fundamentals_with_labels.csv",
    ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                df = pd.read_csv(fp)
                sym_col = next((c for c in df.columns if c.lower() in ("symbol","ticker")), None)
                float_col = next((c for c in df.columns if c in ("float_shares","float","free_float","shares_float")), None)
                if sym_col and float_col:
                    mp: Dict[str, Optional[float]] = {}
                    for _, r in df[[sym_col, float_col]].dropna().iterrows():
                        s = str(r[sym_col]).upper().strip()
                        try:
                            val = float(r[float_col])
                        except Exception:
                            val = None
                        mp[s] = val
                    print(f"[INFO] float map loaded from {fp}: {len(mp)} rows")
                    return mp
            except Exception as e:
                print(f"[WARN] could not load {fp}: {e}")
    return {}

def fetch_ohlc(symbols: List[str], period: str = "90d") -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval="1d", progress=False, auto_adjust=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                data[s] = df
            else:
                print(f"[WARN] empty ohlc for {s}")
        except Exception as e:
            print(f"[WARN] yf download error {s}: {e}")
        time.sleep(0.05)
    return data

def compute_metrics(df: "pd.DataFrame") -> Dict[str, Any]:
    if df is None or len(df) < 21:
        return {}
    df = df.copy()
    df["vol20"] = df["Volume"].rolling(20).mean()
    last = df.iloc[-1]; prev = df.iloc[-2]
    open_ = float(last["Open"]); prev_close = float(prev["Close"])
    volume = float(last["Volume"]); avg20 = float(last["vol20"]) if not math.isnan(last["vol20"]) else None
    price = float(last["Close"]) if not math.isnan(last["Close"]) else open_
    gap_pct = ((open_ / prev_close) - 1.0) * 100.0 if prev_close else None
    rvol = (volume / avg20) if (avg20 and avg20 > 0) else None
    return {
        "price": price, "open": open_, "prev_close": prev_close,
        "volume": volume, "avg_vol20": avg20,
        "gap_pct": gap_pct, "rvol": rvol,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=DEF_UNIVERSE)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--max-price", type=float, default=20.0)
    ap.add_argument("--min-gap", type=float, default=5.0)
    ap.add_argument("--min-rvol", type=float, default=2.0)
    ap.add_argument("--max-float-m", type=float, default=50.0)
    args = ap.parse_args()

    tickers = sorted(set(t.upper().strip() for t in load_universe(args.universe)))
    if not tickers:
        print("[ERR] no tickers in universe"); sys.exit(2)

    float_map = load_float_map()
    ohlc = fetch_ohlc(tickers, period="90d")

    passed, rejected = [], []
    for s in tickers:
        df = ohlc.get(s)
        metrics = compute_metrics(df) if df is not None else {}
        if not metrics:
            rejected.append({"symbol": s, "reason": "insufficient_data"}); continue

        price = metrics["price"]; gap = metrics["gap_pct"]; rvol = metrics["rvol"]
        f_sh = float_map.get(s); f_m = (f_sh / 1e6) if (f_sh and f_sh > 0) else None

        reasons = []
        if price is None or not (args.min_price <= price <= args.max_price):
            reasons.append(f"price_out_of_range:{price}")
        if gap is None or gap < args.min_gap:
            reasons.append(f"gap_lt_{args.min_gap}%:{gap}")
        if rvol is None or rvol < args.min_rvol:
            reasons.append(f"rvol_lt_{args.min_rvol}:{rvol}")
        if f_m is not None and f_m > args.max_float_m:
            reasons.append(f"float_gt_{args.max_float_m}M:{round(f_m,2)}")

        ssr = None; halts = None; dilution_risk = None  # заглушки

        if reasons:
            rejected.append({"symbol": s, "metrics": metrics, "float_m": f_m,
                             "ssr": ssr, "halts": halts, "dilution": dilution_risk,
                             "reason": ";".join(reasons)})
        else:
            passed.append({"symbol": s, "metrics": metrics, "float_m": f_m,
                           "ssr": ssr, "halts": halts, "dilution": dilution_risk,
                           "strategy": "dux_short_mvp", "timestamp": int(time.time())})

    out = {"strategy": "dux_short_mvp", "generated_at": int(time.time()),
           "passed": passed, "rejected": rejected,
           "params": {"min_price": args.min_price, "max_price": args.max_price,
                      "min_gap_pct": args.min_gap, "min_rvol": args.min_rvol,
                      "max_float_m": args.max_float_m}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] total={len(tickers)} passed={len(passed)} rejected={len(rejected)} -> {args.out}")

if __name__ == "__main__":
    main()
