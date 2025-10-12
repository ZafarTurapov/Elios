# -*- coding: utf-8 -*-
import json, math
import os
SEND_TG_HEALTH = os.getenv("SEND_TG_HEALTH","0")=="1"
from pathlib import Path
from datetime import datetime, timezone
from core.utils.telegram import send_telegram_message, escape_markdown

ROOT = Path("/root/stockbot")
DQ   = ROOT/"logs/data_quality_report.json"
FAIL = ROOT/"logs/eod_failed.json"
HIST = ROOT/"logs/metrics_history.jsonl"
META = None
for p in sorted((ROOT/"core/models").glob("xgb_spike4_v*.meta.json"), key=lambda x: x.stat().st_mtime):
    META = p

def _load_json(p, default=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return default

def _iter_fail_syms(obj):
    if obj is None: 
        return
    if isinstance(obj, dict):
        for sym in obj.keys():
            yield str(sym)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                sym = item.get("symbol") or item.get("ticker") or item.get("sym") or item.get("Symbol")
                if sym: yield str(sym)

def _last_hist(n=0):
    rows=[]
    try:
        with open(HIST) as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows if n==0 else rows[-n:]

def _delta(cur, prev):
    if cur is None or prev is None: return "n/a"
    try: d = float(cur) - float(prev)
    except: return "n/a"
    return f"{d:+.4f}"

def main():
    dq = _load_json(DQ, {}) or {}
    fail = _load_json(FAIL, None)
    meta = _load_json(META, {}) if META else {}

    auc = meta.get("auc_valid"); ap = meta.get("ap_valid"); iters = meta.get("best_iteration")
    pos_rate = dq.get("positive_rate")
    rows = dq.get("rows"); syms = dq.get("symbols")

    hist = _last_hist(0)
    d_str = w_str = "n/a"
    if len(hist) >= 2:
        d_str = f"AUC {_delta(auc, hist[-2].get('auc'))} | AP {_delta(ap, hist[-2].get('ap'))}"
    if len(hist) >= 8:
        w_str = f"AUC {_delta(auc, hist[-8].get('auc'))} | AP {_delta(ap, hist[-8].get('ap'))}"

    psi_flags = dq.get("psi_flags", {}) or {}
    warn, crit = [], []
    for k, v in psi_flags.items():
        try:
            x = float(v)
            if x >= 1.0: crit.append(f"{k}={x:.2f}")
            elif x >= 0.7: warn.append(f"{k}={x:.2f}")
        except: pass

    fail_syms = []
    seen = set()
    for s in _iter_fail_syms(fail) or []:
        if s not in seen:
            seen.add(s); fail_syms.append(s)
        if len(fail_syms) >= 10: break

    lines = []
    lines.append("🤖 *Elios — Train Healthcheck*")
    if rows is not None and syms is not None:
        lines.append(f"📦 *EOD*: rows={rows} | syms={syms}")
    if pos_rate is not None:
        lines.append(f"🧱 *Feats*: pos={pos_rate:.2%}")
    if auc is not None and ap is not None:
        lines.append(f"🎯 *XGB*: AUC={auc:.4f} | AP={ap:.4f} | iters={iters}")
        lines.append(f"Δd/d: {escape_markdown(d_str)}")
        lines.append(f"Δw/w: {escape_markdown(w_str)}")
    if warn or crit:
        tag=[]
        if warn: tag.append("⚠️ " + ", ".join(warn))
        if crit: tag.append("🛑 " + ", ".join(crit))
        lines.append("🧪 *PSI*: " + "  ".join(tag))
    if fail_syms:
        lines.append("⚠️ fail sample: " + ", ".join(escape_markdown(s) for s in fail_syms))
    lines.append("🕒 " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    try: send_telegram_message("\n".join(lines)) if SEND_TG_HEALTH else print("[healthcheck] Telegram suppressed; set SEND_TG_HEALTH=1 to enable.")
    except Exception as e:
        print("[WARN] telegram:", e)
        print("\n".join(lines))

if __name__ == "__main__":
    main()
