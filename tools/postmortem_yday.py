# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys, shutil, os, requests
from pathlib import Path
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import pandas as pd, numpy as np, yfinance as yf

ROOT = Path("/root/stockbot")
LOGS = ROOT/"logs"
DS   = ROOT/"core/data/train/dataset.parquet"
SIGLOG = LOGS/"signal_log.json"
MACRO  = LOGS/"macro.json"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"

TZ_ET = ZoneInfo("US/Eastern")
today_et = datetime.now(TZ_ET).date()

def last_business_day(d):
    while d.weekday() >= 5: d -= timedelta(days=1)
    return d

def next_business_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5: d += timedelta(days=1)
    return d

wanted = last_business_day(today_et - timedelta(days=1))
env_d = os.getenv("ELIOS_POSTM_DATE")
if env_d:
    try: wanted = datetime.strptime(env_d, "%Y-%m-%d").date()
    except Exception: pass

def load_ds_slice_exact(date_):
    if not DS.exists():
        return pd.DataFrame(), ["Датасет отсутствует."]
    df = pd.read_parquet(DS)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns or "symbol" not in df.columns:
        return pd.DataFrame(), ["В датасете нет полей date/symbol."]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("US/Eastern").dt.date
    sl = df[df["date"] == date_].copy()
    if sl.empty:
        return pd.DataFrame(), [f"В датасете нет {date_} — использую yfinance/Alpaca fallback."]
    return sl, []

def load_signals_for(d):
    if not SIGLOG.exists(): return []
    try:
        items = json.loads(SIGLOG.read_text()) or []
        day = str(d)
        res = [ {"symbol": str(it.get("symbol","")).upper(), **{k:v for k,v in it.items() if k!="symbol"}}
                for it in items if (it.get("timestamp","")[:10] == day) and it.get("symbol") ]
        seen, uniq = set(), []
        for it in reversed(res):
            s = it["symbol"]
            if s in seen: continue
            uniq.append(it); seen.add(s)
        return list(reversed(uniq))
    except Exception:
        return []

def yf_bar_lookup(sym: str, d: datetime.date):
    start = d - timedelta(days=5); end = d + timedelta(days=7)
    try:
        df = yf.download(sym, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty: return None
        df = df.copy(); df.index = pd.to_datetime(df.index).tz_localize(None); df['date'] = df.index.date
        if d not in set(df['date']): return None
        row_d = df[df['date']==d].iloc[-1]
        nxt = df[df['date']>d]
        if nxt.empty: return {"need_intraday": True}
        row_n = nxt.iloc[0]
        open_next = float(row_n['Open']); high_next = float(row_n['High'])
        prev = df[df['date']<d].iloc[-1] if (df['date']<d).any() else None
        prev_close = float(prev['Close']) if prev is not None else np.nan
        open_d = float(row_d['Open']); close_d = float(row_d['Close'])
        gap_up = (open_d - prev_close)/prev_close*100 if prev_close and prev_close>0 else np.nan
        body = (close_d - open_d)/open_d*100 if open_d else np.nan
        max_gain = (high_next/open_next - 1.0)*100 if open_next>0 and high_next>0 else np.nan
        return {
            "gap_up_pct": None if np.isnan(gap_up) else float(gap_up),
            "bullish_body_pct": None if np.isnan(body) else float(body),
            "open_next": open_next, "high_next": high_next,
            "max_nextday_gain_pct": None if np.isnan(max_gain) else float(max_gain),
            "next_day": str(row_n.name.date())
        }
    except Exception:
        return None

def alpaca_intraday_nextday(sym: str, d: datetime.date):
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY): return None
    nd = next_business_day(d)
    start_dt = datetime.combine(nd, time(9,30), tzinfo=TZ_ET).astimezone(ZoneInfo("UTC"))
    end_dt = min(datetime.now(TZ_ET).astimezone(ZoneInfo("UTC")),
                 datetime.combine(nd, time(20,30), tzinfo=TZ_ET).astimezone(ZoneInfo("UTC")))
    params = {
        "timeframe": "5Min",
        "start": start_dt.isoformat().replace("+00:00","Z"),
        "end": end_dt.isoformat().replace("+00:00","Z"),
        "adjustment": "raw",
        "limit": 10000
    }
    try:
        r = requests.get(f"{ALPACA_DATA_BASE}/stocks/{sym}/bars", params=params,
                         headers={"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}, timeout=15)
        if r.status_code != 200: return None
        bars = (r.json() or {}).get("bars", []) or []
        if not bars: return None
        o_first = float(bars[0].get("o") or 0.0)
        h_max = max(float(b.get("h") or 0.0) for b in bars)
        if o_first <= 0: return None
        max_gain = (h_max/o_first - 1.0)*100.0
        return {"open_next": o_first, "high_next": h_max, "max_nextday_gain_pct": max_gain, "next_day": str(nd)}
    except Exception:
        return None

def main():
    day_df, notes = load_ds_slice_exact(wanted)
    signals = load_signals_for(wanted)

    OUTDIR = LOGS/"postmortem"/str(wanted); OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTJ, OUTM = OUTDIR/"postmortem.json", OUTDIR/"postmortem.md"

    summary = {"date": str(wanted), "wanted_date": str(wanted), "n_signals": len(signals),
               "base_rate": None, "hits": None, "hit_rate": None, "macro": None,
               "notes": [n for n in notes if n], "items": []}

    if MACRO.exists():
        try:
            m = json.loads(MACRO.read_text()) or {}
            if (m.get("timestamp","")[:10] == str(wanted)):
                summary["macro"] = {"regime": m.get("regime"),
                                    "score": m.get("macro") if isinstance(m.get("macro"), (int,float)) else m.get("macro", None),
                                    "vix": (m.get("vix") or {}).get("last") if isinstance(m.get("vix"), dict) else None}
        except Exception: pass

    if not day_df.empty:
        day_df["_SYM"] = day_df["symbol"].astype(str).str.upper()
        target = "target_spike4" if "target_spike4" in day_df.columns else (day_df.filter(like="target").columns.tolist() or [None])[0]
        if target:
            try: summary["base_rate"] = float(pd.to_numeric(day_df[target], errors="coerce").mean())
            except Exception: pass
        hits = 0
        for s in signals:
            sym = s["symbol"]; row = day_df[day_df["_SYM"] == sym]
            if row.empty:
                summary["items"].append({"symbol": sym, "present_in_dataset": False, "accepted_reason": s.get("reason")}); continue
            r = row.iloc[0]
            def f(c):
                try: return float(r.get(c))
                except Exception: return None
            open_next, high_next = f("open_next"), f("high_next")
            max_gain = (high_next/open_next - 1.0)*100 if open_next and high_next and open_next>0 else None
            y = None
            if target in r:
                try: y = int(r[target])
                except Exception:
                    try: y = int(float(r[target] or 0))
                    except Exception: y = None
            hits += (1 if y==1 else 0) if y is not None else 0
            summary["items"].append({"symbol": sym, "accepted_reason": s.get("reason"),
                                     "spike4_hit": (y==1 if y is not None else None),
                                     "max_nextday_gain_pct": (None if max_gain is None else round(max_gain,3)),
                                     "gap_up_pct": f("gap_up_pct"), "bullish_body_pct": f("bullish_body_pct"),
                                     "atr_pct": f("atr_pct"), "volatility_pct": f("volatility_pct"),
                                     "volume_ratio": f("volume_ratio"), "volume_trend": f("volume_trend"),
                                     "alpha_score": f("alpha_score")})
        if signals and target:
            summary["hits"] = int(hits); summary["hit_rate"] = float(hits/len(signals))
    else:
        if not signals:
            summary["notes"].append("Нет сигналов за целевой день.")
        else:
            summary["notes"].append("В датасете нет нужной даты — использую yfinance/Alpaca fallback.")
            HITS = 0
            for s in signals:
                sym = s["symbol"]
                yfm = yf_bar_lookup(sym, wanted)
                item = {"symbol": sym, "accepted_reason": s.get("reason")}
                if yfm is None:
                    item.update({"present_in_dataset": False, "spike4_hit": None, "max_nextday_gain_pct": None})
                elif yfm.get("need_intraday"):
                    alp = alpaca_intraday_nextday(sym, wanted)
                    if alp is None:
                        item.update({"spike4_hit": None, "max_nextday_gain_pct": None})
                    else:
                        hit = (alp["high_next"] >= alp["open_next"] * 1.04)
                        item.update({**alp, "spike4_hit": hit})
                        if hit: HITS += 1
                else:
                    hit = (yfm["high_next"] >= yfm["open_next"] * 1.04) if (yfm["high_next"] and yfm["open_next"]) else None
                    if hit is True: HITS += 1
                    item.update(yfm); item["spike4_hit"] = hit
                summary["items"].append(item)
            summary["hits"] = HITS
            summary["hit_rate"] = float(HITS/len(signals)) if signals else None

    OUTJ.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [f"# Postmortem — {summary['date']}",
             f"- Принято сигналов: **{summary['n_signals']}**"]
    if summary["hit_rate"] is not None:
        lines.append(f"- Hit-rate (spike≥4%): **{summary['hit_rate']:.1%}**  (хитов: {summary['hits']})")
    if summary["base_rate"] is not None:
        lines.append(f"- Базовая частота события: **{summary['base_rate']:.1%}**")
    if summary.get("macro"):
        m = summary["macro"]; lines.append(f"- Macro: **{m.get('regime','?')}**, score={m.get('score','?')}, VIX={m.get('vix','?')}")
    if summary["notes"]: lines.append("- Notes: " + "; ".join(summary["notes"]))
    lines.append("\n## Сигналы и факты (next-day)")
    cols = ["Symbol","Hit","maxGain%","gap%","body%","open_next","high_next","next_day","Reason"]
    lines.append("| " + " | ".join(cols) + " |"); lines.append("|" + "|".join(["---"]*len(cols)) + "|")
    for it in (summary["items"] or []):
        lines.append("| {s} | {hit} | {g} | {gap} | {body} | {on} | {hn} | {nd} | {r} |".format(
            s=it.get("symbol",""),
            hit=("✅" if it.get("spike4_hit") else ("❌" if it.get("spike4_hit") is False else "—")),
            g=("" if it.get("max_nextday_gain_pct") is None else f"{it['max_nextday_gain_pct']:.2f}"),
            gap=("" if it.get("gap_up_pct") is None else f"{it['gap_up_pct']:.2f}"),
            body=("" if it.get("bullish_body_pct") is None else f"{it['bullish_body_pct']:.2f}"),
            on=("" if it.get("open_next") is None else f"{it['open_next']:.2f}"),
            hn=("" if it.get("high_next") is None else f"{it['high_next']:.2f}"),
            nd=(it.get("next_day") or it.get("next_day_price_date") or ""),
            r=(it.get("accepted_reason") or "—"),
        ))
    (LOGS/"postmortem"/str(wanted)/"postmortem.md").write_text("\n".join(lines))

    # Приложения
    extras = [
        "data_quality_report.json","data_quality_report.md",
        "shadow_metrics.json","shadow_metrics.md",
        "calibration_report.json","calibration_report.md",
        "avoidance_grid.json","avoidance_grid.md",
        "train_guard_report.json","rejected.csv","macro.json",
        "signal_log.json"
    ]
    OUTDIR = LOGS/"postmortem"/str(wanted)
    for name in extras:
        p = (LOGS/name).resolve()
        if p.exists():
            try: shutil.copy(p, OUTDIR/p.name)
            except Exception: pass

    print(json.dumps({"out_dir": str(OUTDIR),
                      "picked_date": str(wanted),
                      "wanted_date": str(wanted),
                      "n_signals": summary["n_signals"],
                      "hits": summary.get("hits"),
                      "hit_rate": summary.get("hit_rate"),
                      "base_rate": summary.get("base_rate")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
