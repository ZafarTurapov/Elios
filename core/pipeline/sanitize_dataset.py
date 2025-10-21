# -*- coding: utf-8 -*-
"""
Elios — Sanitize Dataset (winsorize 0.1/99.9)
- Бэкапит dataset.parquet
- Клиппит выбросы в числовых колонках (OHLC/volume + features из feature_spec.json)
- Сохраняет новый dataset.parquet и отчёт logs/sanitize_dataset_report.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path("/root/stockbot")
DS = ROOT / "core/data/train/dataset.parquet"
SPEC = ROOT / "core/models/feature_spec.json"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
OUTJ = LOGS / "sanitize_dataset_report.json"

Q_LO, Q_HI = 0.1, 99.9
OHLC = ["open", "high", "low", "close", "volume"]
EXCLUDE = {"target_spike4", "y", "label", "target"}  # не трогаем таргеты/ярлыки


def load_features():
    feats = []
    if SPEC.exists():
        try:
            s = json.loads(SPEC.read_text())
            feats = list(s.get("features", []) or [])
        except Exception:
            pass
    # запасной набор
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
    return feats


def winsorize_series(s: pd.Series, lo=Q_LO, hi=Q_HI) -> tuple[pd.Series, dict]:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    ql, qh = np.nanpercentile(x, [lo, hi])
    clipped = x.clip(ql, qh)
    changed = int(np.nansum((x < ql) | (x > qh)))
    return clipped, {"q_lo": float(ql), "q_hi": float(qh), "changed": changed}


def main():
    if not DS.exists():
        print(f"Dataset not found: {DS}")
        sys.exit(2)
    df = pd.read_parquet(DS)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    feats = load_features()
    candidates = [c for c in set(feats + OHLC) if c in df.columns and c not in EXCLUDE]

    # бэкап
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DS.with_suffix(f".parquet.bak-{ts}")
    df.to_parquet(backup, index=False)

    report = {"backup": str(backup), "total_rows": int(len(df)), "columns": {}}

    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if c in {"open", "high", "low", "close"}:
            s[s <= 0] = np.nan
        # volume может быть нулевым, но не отрицательным
        if c == "volume":
            s[s < 0] = np.nan
        clipped, info = winsorize_series(s)
        df[c] = clipped
        report["columns"][c] = info

    # перезаписываем основной датасет
    df.to_parquet(DS, index=False)
    OUTJ.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved sanitized dataset to {DS}")
    print(f"Report: {OUTJ}")
    sys.exit(0)


if __name__ == "__main__":
    main()
