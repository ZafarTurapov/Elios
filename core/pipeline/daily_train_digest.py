# -*- coding: utf-8 -*-
import datetime
import json
import re
from pathlib import Path

try:
    from core.utils.telegram import send_telegram_message
except Exception:

    def send_telegram_message(x):
        print(x)


ROOT = Path("/root/stockbot")


def jload(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_len(x):
    if x is None:
        return 0
    if isinstance(x, dict):
        return len(x)
    if isinstance(x, list):
        return len(x)
    return 0


def latest_meta():
    mdir = ROOT / "core/models"
    metas = sorted(
        mdir.glob("xgb_spike4_v*.meta.json"), key=lambda p: p.stat().st_mtime
    )
    cur = jload(metas[-1], {}) if metas else {}
    prev = jload(metas[-2], {}) if len(metas) > 1 else {}
    v = "?"
    if metas:
        m = re.search(r"v(\d+)", metas[-1].name)
        if m:
            v = m.group(1)
    return v, cur, prev


def main():
    # EOD / активы
    total = safe_len(jload(ROOT / "core/trading/candidates.json", []))
    active = safe_len(jload(ROOT / "core/trading/candidates_active.json", []))
    failed_list = jload(ROOT / "logs/eod_failed.json", []) or []
    failed_n = safe_len(failed_list)

    # Data quality
    dq = jload(ROOT / "logs/data_quality_report.json", {}) or {}
    rows = dq.get("rows")
    syms = dq.get("symbols")
    pos_rate = dq.get("positive_rate")
    psi_flags = dq.get("psi_flags", {}) or {}

    # Модель
    ver, cur, prev = latest_meta()
    auc = cur.get("auc_valid")
    ap = cur.get("ap_valid")
    iters = cur.get("best_iteration")
    ap_prev = prev.get("ap_valid")
    d_ap = (
        (ap - ap_prev)
        if isinstance(ap, (int, float)) and isinstance(ap_prev, (int, float))
        else None
    )

    lines = []
    lines.append("🤖 *Elios — Daily Training Digest*")
    lines.append(f"📦 *EOD*: act={active}/{total} | fail={failed_n}")
    if rows is not None and syms is not None and pos_rate is not None:
        lines.append(
            f"🧱 *Feats*: rows={rows:,} | syms={syms} | pos={pos_rate*100:.2f}%"
        )
    if auc is not None and ap is not None:
        delta = f" | ΔAP={d_ap:+.4f}" if d_ap is not None else ""
        lines.append(
            f"🎯 *XGB v{ver}*: AUC={auc:.4f} | AP={ap:.4f} | iters={iters}{delta}"
        )
    if psi_flags:
        flags = ", ".join([f"{k}={v:.2f}" for k, v in psi_flags.items()])
        lines.append(f"🧪 *PSI flags*: {flags}")
    if failed_n > 0:
        sample = (
            failed_list[:10]
            if isinstance(failed_list, list)
            else list(failed_list)[:10]
        )
        lines.append("⚠️ fail sample: " + ", ".join(sample))
    lines.append(datetime.datetime.now().strftime("🕒 %Y-%m-%d %H:%M"))
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
