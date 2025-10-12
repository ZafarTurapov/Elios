# -*- coding: utf-8 -*-
"""
Elios — Label Audit (v1.1.1)
Фикс: агрегаты by_month считаем только по строкам с mismatch (mask=True)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

TZ   = ZoneInfo("Asia/Tashkent")
ROOT = Path("/root/stockbot")
DS   = ROOT/"core/data/train/dataset.parquet"
LOGS = ROOT/"logs"; LOGS.mkdir(parents=True, exist_ok=True)
OUTJ = LOGS/"label_audit.json"
OUTM = LOGS/"label_audit.md"
OUTC = LOGS/"label_mismatch_sample.csv"

TARGET = "target_spike4"
THRESH_PCT = 4.0

def main():
    if not DS.exists():
        print(f"Dataset not found: {DS}"); sys.exit(2)
    df = pd.read_parquet(DS)
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    if "date" not in df.columns or "symbol" not in df.columns:
        print("dataset must contain 'date' and 'symbol'"); sys.exit(2)

    # нормализуем дату к US/Eastern (датой следующей сессии у нас уже являются *_next)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("US/Eastern").dt.date

    need = {TARGET, "open_next", "high_next"}
    if not need.issubset(df.columns):
        print(f"dataset must contain: {', '.join(sorted(need))}"); sys.exit(2)

    open_n = pd.to_numeric(df["open_next"], errors="coerce")
    high_n = pd.to_numeric(df["high_next"], errors="coerce")
    calc = ((high_n / open_n - 1.0) * 100.0 >= THRESH_PCT).astype("Int64")
    tgt  = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")

    mask = (calc != tgt) & (~calc.isna()) & (~tgt.isna())
    mism_df = df.loc[mask, ["symbol","date","open_next","high_next", TARGET]].copy()
    mism_df["calc_label"] = calc[mask].astype("Int64")
    mism_df["diff"] = mism_df["calc_label"] - mism_df[TARGET].astype("Int64")

    # агрегаты по год-месяц — считаем ТОЛЬКО на mask=True
    dty = pd.to_datetime(df.loc[mask, "date"])
    year = dty.dt.year.rename("year")
    month = dty.dt.month.rename("month")
    if len(mism_df):
        by_month = (
            mism_df.groupby([year, month]).size()
            .reset_index(name="mismatches")
            .sort_values(["year","month"], ascending=True)
        )
    else:
        by_month = pd.DataFrame(columns=["year","month","mismatches"])

    total_mism = int(mask.sum())
    summary = {
        "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": str(DS),
        "rows": int(len(df)),
        "mismatch_count": total_mism,
        "mismatch_rate": float(total_mism / max(len(df),1)),
        "by_month": by_month.to_dict(orient="records") if not by_month.empty else [],
        "sample_saved": (str(OUTC) if total_mism>0 else None),
    }
    OUTJ.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # MD отчёт
    lines = []
    lines.append(f"# Label Audit — {'OK' if total_mism==0 else 'WARN'}")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- mismatches: **{total_mism}** ({summary['mismatch_rate']:.4%})")
    if summary["by_month"]:
        lines.append("## By month (top 10)")
        top10 = sorted(summary["by_month"], key=lambda r: r["mismatches"], reverse=True)[:10]
        for r in top10:
            lines.append(f"- {r['year']}-{r['month']:02d}: {r['mismatches']}")
    if total_mism>0:
        mism_df.head(1000).to_csv(OUTC, index=False)
        lines.append(f"\nSaved sample: `{OUTC}`")
    OUTM.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0 if total_mism==0 else 2)

if __name__ == "__main__":
    main()
