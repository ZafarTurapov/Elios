# -*- coding: utf-8 -*-
"""
Elios — Data Quality Report (train set audit, v2.1.1)
— фикс: убрать .to_series() при извлечении года (Series.dt.year достаточно)
"""

from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Tashkent")
ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
DS_PARQUET = ROOT/"core/data/train/dataset.parquet"
DS_JSON1   = ROOT/"core/training/training_data.json"
DS_JSON2   = ROOT/"logs/training_data.json"
SPEC       = ROOT/"core/models/feature_spec.json"
OUTJS      = LOGS/"data_quality_report.json"
OUTMD      = LOGS/"data_quality_report.md"

# --- Пороги (переопределяются ENV) ---
MAX_NULL_PCT_CRIT = float(os.getenv("ELIOS_DQ_MAX_NULL_PCT_CRIT", "0.25"))
MAX_NULL_PCT_WARN = float(os.getenv("ELIOS_DQ_MAX_NULL_PCT_WARN", "0.05"))
MAX_DUP_RATE_WARN = float(os.getenv("ELIOS_DQ_MAX_DUP_RATE_WARN", "0.01"))
MIN_RECENT_DAYS_30 = int(os.getenv("ELIOS_DQ_MIN_RECENT_DAYS_30", "18"))
MIN_UNIQ_SYMBOLS   = int(os.getenv("ELIOS_DQ_MIN_UNIQ_SYMBOLS", "250"))
MAX_CLASS_IMBALANCE = float(os.getenv("ELIOS_DQ_MAX_CLASS_IMBALANCE", "0.90"))
PSI_WARN = float(os.getenv("ELIOS_DQ_PSI_WARN", "0.25"))
PSI_FAIL = float(os.getenv("ELIOS_DQ_PSI_FAIL", "0.40"))
TELEGRAM_ON_FAIL = os.getenv("ELIOS_DQ_TG", "1").strip().lower() not in {"0","false","no","off"}

KEY_FEATURES_DEFAULT = [
    "rsi14","ema_dev_pct","atr_pct","volatility_pct",
    "volume_trend","volume_ratio","gap_up_pct","bullish_body_pct",
    "mom5_pct","mom20_pct","alpha_score"
]

def _send_tg(msg: str):
    if not TELEGRAM_ON_FAIL:
        return
    try:
        from core.utils.telegram import send_telegram_message
        send_telegram_message(msg)
    except Exception:
        try:
            import requests
            bot = os.getenv("TELEGRAM_BOT_TOKEN"); chat = os.getenv("TELEGRAM_CHAT_ID")
            if bot and chat:
                requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
                              data={"chat_id": chat, "text": msg, "parse_mode": "Markdown"})
        except Exception:
            pass

def load_spec() -> tuple[list[str], str]:
    if SPEC.exists():
        try:
            s = json.loads(SPEC.read_text())
            feats = s.get("features", []) or KEY_FEATURES_DEFAULT
            target = s.get("target", "target_spike4")
            return feats, target
        except Exception:
            pass
    return KEY_FEATURES_DEFAULT, "target_spike4"

def load_dataset() -> tuple[pd.DataFrame, str]:
    if DS_PARQUET.exists():
        return pd.read_parquet(DS_PARQUET), str(DS_PARQUET)
    for cand in (DS_JSON1, DS_JSON2):
        if cand.exists():
            try:
                df = pd.read_json(cand, lines=True)
                return df, str(cand)
            except ValueError:
                df = pd.DataFrame(json.loads(cand.read_text()))
                return df, str(cand)
    raise FileNotFoundError(f"Не найден датасет: {DS_PARQUET} / {DS_JSON1} / {DS_JSON2}")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "symbol" not in df.columns:
        for alt in ("ticker","sym","code"):
            if alt in df.columns: df["symbol"] = df[alt]; break
    if "date" not in df.columns:
        for alt in ("timestamp","time","datetime"):
            if alt in df.columns: df["date"] = df[alt]; break
    if "date" in df.columns:
        # нормализуем к дате US/Eastern (без времени)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert("US/Eastern").dt.date
    return df

def make_binary_y(df: pd.DataFrame, target_hint: str|None) -> pd.Series:
    candidates = [target_hint, "y", "label", "target", "outcome", "result"]
    candidates = [c for c in candidates if c]
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s.dtype == object:
                m = s.astype(str).str.upper().map({"WIN":1,"LOSS":0,"TRUE":1,"FALSE":0,"1":1,"0":0})
                if m.notna().mean() > 0.7: return m.astype(float)
            try:
                v = pd.to_numeric(s, errors="coerce").astype(float)
                if v.dropna().between(0,1).mean() > 0.9: return v
                return (v > np.nanmedian(v)).astype(float)
            except Exception:
                continue
    return pd.Series([np.nan]*len(df))

def psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    eps = 1e-8
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 200 or len(actual) < 200:
        return np.nan
    qs = np.quantile(expected, np.linspace(0,1,bins+1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    e_hist, _ = np.histogram(expected, bins=qs)
    a_hist, _ = np.histogram(actual,   bins=qs)
    e_rat = e_hist / max(e_hist.sum(), eps)
    a_rat = a_hist / max(a_hist.sum(), eps)
    return float(np.sum((a_rat - e_rat) * np.log((a_rat + eps) / (e_rat + eps))))

def zscore_mad(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad

def main():
    sev = 0  # 0 OK, 1 FAIL, 2 WARN
    issues: list[str] = []

    try:
        df, src = load_dataset()
        feats, target = load_spec()
        df = normalize(df)
        if not {"symbol","date"}.issubset(df.columns):
            issues.append("❌ Нет обязательных колонок: symbol/date")
            OUTJS.write_text(json.dumps({"status":"FAIL","issues":issues}, ensure_ascii=False, indent=2))
            print("\n".join(issues)); sys.exit(1)

        df = df.dropna(subset=["symbol","date"]).reset_index(drop=True)
        symbols = int(df["symbol"].nunique())
        rows    = int(len(df))
        date_min = df["date"].min(); date_max = df["date"].max()

        y = make_binary_y(df, target)
        pos_rate = float((y==1).mean()) if y.notna().any() else float("nan")

        # дубликаты и немонотонные даты
        dup_rate = 0.0; dups = 0
        if len(df) > 0:
            key = df[["symbol","date"]].astype(str).agg("|".join, axis=1)
            dup_rate = float(key.duplicated().mean()); dups = int(key.duplicated().sum())
            if dup_rate > MAX_DUP_RATE_WARN:
                issues.append(f"⚠️ Дубликаты (symbol,date): {dup_rate:.2%} > {MAX_DUP_RATE_WARN:.0%}"); sev = max(sev, 2)

        bad_order = 0
        for _, g in df.groupby("symbol", sort=False):
            s = pd.to_datetime(g["date"]).sort_values().diff().dt.days.fillna(1)
            bad_order += int((s<=0).sum())
        if bad_order > 0:
            issues.append(f"⚠️ Немонотонные даты в ряде символов: {bad_order}"); sev = max(sev, 2)

        # пропуски по фичам
        nan_rates = {c: float(pd.to_numeric(df[c], errors="coerce").isna().mean()) for c in feats if c in df.columns}
        nan_worst = sorted(nan_rates.items(), key=lambda kv: kv[1], reverse=True)[:10]
        bad_crit = [c for c,p in nan_rates.items() if p > MAX_NULL_PCT_CRIT]
        bad_warn = [c for c,p in nan_rates.items() if MAX_NULL_PCT_WARN < p <= MAX_NULL_PCT_CRIT]
        if bad_crit:
            issues.append(f"❌ Высокие пропуски >{int(MAX_NULL_PCT_CRIT*100)}%: {bad_crit[:8]}"); sev = max(sev, 1)
        if bad_warn:
            issues.append(f"⚠️ Пропуски >{int(MAX_NULL_PCT_WARN*100)}%: {bad_warn[:8]}"); sev = max(sev, 2)

        # валидность диапазонов (если есть)
        checks = {
            "rsi14": lambda s: ((s<0) | (s>100)).sum(),
            "open":  lambda s: (s<=0).sum(), "high": lambda s: (s<=0).sum(),
            "low":   lambda s: (s<=0).sum(), "close":lambda s: (s<=0).sum(),
            "volume":lambda s: (s<0).sum(),
            "atr_pct": lambda s: (s<0).sum(), "volatility_pct": lambda s: (s<0).sum(),
        }
        invalid_counts = {}
        for col, fn in checks.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                invalid_counts[col] = int(fn(s))
        if any(v>0 for v in invalid_counts.values()):
            issues.append("⚠️ Невалидные значения (см. отчёт)."); sev = max(sev, 2)

        # согласованность таргета
        mismatch = None
        if {"open_next","high_next"}.issubset(df.columns) and (target in df.columns):
            calc = ((pd.to_numeric(df["high_next"], errors="coerce") /
                     pd.to_numeric(df["open_next"], errors="coerce") - 1.0)*100.0 >= 4.0).astype("Int64")
            mismatch = int((calc != pd.to_numeric(df[target], errors="coerce").astype("Int64")).sum())
            if mismatch > 0:
                issues.append(f"⚠️ Несогласованность таргета с open_next/high_next: {mismatch}"); sev = max(sev, 2)

        # выбросы (MAD z-score)
        outliers = []
        num_cols = [c for c in set(feats) | {"open","high","low","close","volume"} if c in df.columns]
        for c in num_cols:
            z = zscore_mad(pd.to_numeric(df[c], errors="coerce"))
            frac = float((np.abs(z) > 10).mean())
            if frac > 0.005:
                outliers.append((c, frac))
        if outliers:
            issues.append(f"⚠️ Выбросы: {[(c, round(f,4)) for c,f in outliers[:8]]}"); sev = max(sev, 2)

        # свежесть и покрытие последних 30 дней
        recent_days = recent_syms = None
        coverage_cands = None
        if "date" in df.columns:
            last30 = datetime.now(TZ).date() - timedelta(days=30)
            recent_df = df[df["date"] >= last30]
            recent_days = int(recent_df["date"].nunique())
            recent_syms = int(recent_df["symbol"].nunique())
            if recent_days < MIN_RECENT_DAYS_30:
                issues.append(f"⚠️ Мало торговых дат за 30д: {recent_days} < {MIN_RECENT_DAYS_30}"); sev = max(sev, 2)
            cand_path = ROOT/"core/trading"/"candidates.json"
            if cand_path.exists():
                try:
                    cand = pd.read_json(cand_path)
                    if isinstance(cand, pd.DataFrame) and not cand.empty:
                        csyms = set(cand.iloc[:,0].astype(str).unique())
                    else:
                        csyms = set(pd.Series(cand).astype(str).unique())
                    last_syms = set(recent_df["symbol"].astype(str).unique())
                    inter = len(csyms & last_syms)
                    coverage_cands = inter / max(len(csyms), 1)
                    if coverage_cands < 0.60:
                        issues.append(f"⚠️ Покрытие кандидатов за 30д: {coverage_cands:.0%} (пересечение {inter}/{len(csyms)})")
                        sev = max(sev, 2)
                except Exception:
                    pass

        # дисбаланс классов
        if y.notna().any():
            p1 = float((y==1).mean()); p0 = 1.0 - p1
            if max(p0,p1) > MAX_CLASS_IMBALANCE:
                issues.append(f"⚠️ Дисбаланс классов: мажорный={max(p0,p1):.1%} > {MAX_CLASS_IMBALANCE:.0%}"); sev = max(sev, 2)
        else:
            issues.append("⚠️ Не удалось интерпретировать целевую переменную (y/label/target)."); sev = max(sev, 2)

        # PSI-дрифт: (a) ранние vs поздние годы — фикс .dt.year
        psi_early_late = {}
        if "date" in df.columns:
            years = pd.to_datetime(df["date"]).dt.year
            if years.nunique() >= 3:
                early_mask = years <= (int(years.min()) + 1)
                late_mask  = years >= (int(years.max()) - 1)
                early = df.loc[early_mask]
                late  = df.loc[late_mask]
                if len(early)>1000 and len(late)>1000:
                    for c in feats:
                        if c in df.columns:
                            try:
                                psi_early_late[c] = psi_score(pd.to_numeric(early[c], errors="coerce").to_numpy(),
                                                              pd.to_numeric(late[c],  errors="coerce").to_numpy(), bins=20)
                            except Exception:
                                psi_early_late[c] = np.nan

        # PSI-дрифт: (b) база vs последние 30 дней
        psi_last30 = {}
        if "date" in df.columns:
            base = df[df["date"] < (datetime.now(TZ).date() - timedelta(days=30))]
            last = df[df["date"] >= (datetime.now(TZ).date() - timedelta(days=30))]
            if len(base) > 1000 and len(last) > 1000:
                for c in feats:
                    if c in df.columns:
                        try:
                            psi_last30[c] = psi_score(pd.to_numeric(base[c], errors="coerce").to_numpy(),
                                                      pd.to_numeric(last[c], errors="coerce").to_numpy(), bins=20)
                        except Exception:
                            psi_last30[c] = np.nan
        worst_last30 = np.nanmax([v for v in psi_last30.values() if v is not None] or [np.nan])
        if np.isfinite(worst_last30) and worst_last30 >= PSI_FAIL:
            issues.append(f"❌ PSI-дрифт (30д) max={worst_last30:.2f} ≥ {PSI_FAIL}"); sev = max(sev, 1)
        elif np.isfinite(worst_last30) and worst_last30 >= PSI_WARN:
            issues.append(f"⚠️ PSI-дрифт (30д) max={worst_last30:.2f} ≥ {PSI_WARN}"); sev = max(sev, 2)

        # годовые базовые ставки события
        yearly_base = {}
        if y.notna().any() and "date" in df.columns:
            dty = pd.to_datetime(df["date"])
            for yr, g in df.groupby(dty.dt.year):
                try:
                    yearly_base[int(yr)] = float((make_binary_y(g, target)==1).mean())
                except Exception:
                    pass

        status = {0:"OK",1:"FAIL",2:"WARN"}[sev]
        summary = {
            "status": status,
            "source": src,
            "rows": rows,
            "symbols": symbols,
            "date_min": str(date_min) if pd.notna(date_min) else None,
            "date_max": str(date_max) if pd.notna(date_max) else None,
            "positive_rate": (None if math.isnan(pos_rate) else pos_rate),
            "duplicates_symbol_date": {"count": dups, "rate": dup_rate},
            "non_monotonic_date_rows": bad_order,
            "nan_rates_top10": nan_worst,
            "invalid_counts": invalid_counts,
            "target_mismatch": mismatch,
            "recent": {"days_30": recent_days, "uniq_symbols_30": recent_syms},
            "coverage_candidates_30d": coverage_cands,
            "psi_early_late_top5": sorted([(k,v) for k,v in psi_early_late.items() if v is not None],
                                          key=lambda kv: kv[1], reverse=True)[:5],
            "psi_last30_top5": sorted([(k,v) for k,v in psi_last30.items() if v is not None],
                                      key=lambda kv: kv[1], reverse=True)[:5],
            "psi_last30_max": (None if not np.isfinite(worst_last30) else float(worst_last30)),
            "yearly_positive_rate": yearly_base,
            "issues": issues,
            "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        OUTJS.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        # Markdown
        lines = []
        lines += [f"# Data Quality Report — {status}",
                  f"- Source: `{src}`",
                  f"- Period: {summary['date_min']} → {summary['date_max']}",
                  f"- Symbols: **{symbols}**, rows: **{rows}**" + (f", POS-rate: **{pos_rate:.2%}**" if not math.isnan(pos_rate) else ""),
                  f"- Duplicates (symbol,date): {dups} ({dup_rate:.2%})",
                  f"- Non-monotonic dates (rows): {bad_order}",
                  f"- Recent(30d): days={recent_days}, uniq_symbols={recent_syms}",
                  f"- Coverage(candidates,30d): { 'n/a' if coverage_cands is None else f'{coverage_cands:.0%}' }",
                  "## NaN rates (Top-10)",
                  ("\n".join([f"- {c}: {r:.2%}" for c,r in nan_worst]) or "_none_"),
                  "## Invalid values",
                  ("\n".join([f"- {c}: {v}" for c,v in invalid_counts.items() if v]) or "_none_"),
                  "## PSI last30 (Top-5)",
                  ("\n".join([f"- {c}: {v:.3f}" for c,v in summary["psi_last30_top5"]]) or "_insufficient_"),
                  "## PSI early vs late years (Top-5)",
                  ("\n".join([f"- {c}: {v:.3f}" for c,v in summary["psi_early_late_top5"]]) or "_insufficient_"),
                  "## Issues",
                  ("\n".join([f"- {x}" for x in issues]) or "_none_"),
                 ]
        OUTMD.write_text("\n".join(lines), encoding="utf-8")

        print(f"🧪 Data Quality: {status}")
        for it in issues[:12]: print("• " + it)
        print(f"\nSaved:\n- {OUTJS}\n- {OUTMD}")

        if sev == 1:
            _send_tg("*Elios — Data Quality FAIL*\n" + "\n".join("• "+x for x in issues[:8]))

        sys.exit(sev if sev in (0,1,2) else 2)

    except Exception as e:
        msg = f"❌ DQ exception: {type(e).__name__}: {e}"
        print(msg)
        OUTJS.write_text(json.dumps({"status":"FAIL","exception":msg}, ensure_ascii=False, indent=2))
        _send_tg(f"*Elios — Data Quality FAIL*\n{msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
