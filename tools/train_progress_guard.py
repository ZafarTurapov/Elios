# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo

ROOT = Path("/root/stockbot")
LOGS = ROOT / "logs"
OUT_JSON = LOGS / "train_guard_report.json"
TZ = ZoneInfo("Asia/Tashkent")

MIN_AUC = float(os.getenv("ELIOS_GUARD_MIN_AUC", "0.78"))
MIN_AP = float(os.getenv("ELIOS_GUARD_MIN_AP", "0.20"))
MAX_PSI = float(os.getenv("ELIOS_GUARD_MAX_PSI", "0.60"))
MAX_FAIL_RATE = float(os.getenv("ELIOS_GUARD_MAX_FAIL_RATE", "0.10"))
MAX_CUTOFF_LAG_DAYS = int(os.getenv("ELIOS_GUARD_MAX_CUTOFF_LAG_DAYS", "14"))
SEND_TG = os.getenv("ELIOS_GUARD_TG", "1").strip() not in {"0", "false", "no", "off"}


# optional Telegram util from project
def _noop_send(msg: str) -> None:
    pass


send_telegram_message = _noop_send
try:
    from core.utils.telegram import send_telegram_message as _stm

    send_telegram_message = _stm
except Exception:
    # fallback: allow BOT_TOKEN/CHAT_ID env (опционально)
    import requests

    def send_telegram_message(msg: str) -> None:
        bot = os.getenv("TELEGRAM_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID")
        if not bot or not chat:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                data={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
            )
        except Exception:
            pass


re_auc = re.compile(r"AUC\s*=\s*([0-9.]+)")
re_ap = re.compile(r"AP\s*=\s*([0-9.]+)")
re_psi = re.compile(r"PSI flags:\s*atr_pct=([0-9.]+)\s*,\s*volatility_pct=([0-9.]+)")
RE_ISO_DATE_ANY = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?(?:\s*(?:UTC|[+\-]\d{2}:?\d{2}))?"
)


def tail_lines(p: Path, max_bytes=1_000_000) -> list[str]:
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode(errors="ignore")
        return chunk.splitlines()
    except Exception:
        return []


def parse_dates_from_line(line: str) -> list[date]:
    dates = []
    likely = ("→" in line) or any(
        k in line.lower() for k in ("cutoff", "promote", "model promote", "range")
    )
    if not likely and "AUC" not in line and "AP" not in line:
        return dates
    for m in RE_ISO_DATE_ANY.finditer(line):
        try:
            dates.append(datetime.strptime(m.group(1), "%Y-%m-%d").date())
        except Exception:
            pass
    return dates


def scan_logs() -> dict:
    aucs, aps, psi = [], [], []
    eod = {"act": 0, "tot": 0, "fail": 0}
    cutoff_candidates = []

    files = sorted(
        LOGS.glob("*"),
        key=lambda x: x.stat().st_mtime if x.exists() else 0,
        reverse=True,
    )
    for p in files:
        if not p.is_file() or p.suffix.lower() not in {".log", ".txt", ".json", ".csv"}:
            continue
        for line in tail_lines(p):
            m = re_auc.search(line)
            m and aucs.append(float(m.group(1)))
            m = re_ap.search(line)
            m and aps.append(float(m.group(1)))
            m = re_psi.search(line)
            m and psi.append((float(m.group(1)), float(m.group(2))))
            if "act=" in line and "fail" in line:
                try:
                    part = line.split("act=")[1]
                    act = int(part.split("/")[0].strip())
                    tot = int(part.split("/")[1].split()[0].strip())
                    fail_token = next(t for t in part.split() if "fail" in t)
                    fail = int("".join(ch for ch in fail_token if ch.isdigit()))
                    if act + tot + fail > sum(eod.values()):
                        eod = {"act": act, "tot": tot, "fail": fail}
                except Exception:
                    pass
            cutoff_candidates.extend(parse_dates_from_line(line))

    cutoff_right = max(cutoff_candidates) if cutoff_candidates else None
    return {
        "aucs": aucs,
        "aps": aps,
        "psi": psi,
        "eod": eod,
        "cutoff_right": cutoff_right,
    }


def truthy(x: str | None) -> bool:
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "on"}


def falsy(x: str | None) -> bool:
    if x is None:
        return False
    return str(x).strip().lower() in {"0", "false", "no", "off"}


def detect_train_only():
    notes, sev = [], 0
    sentinel = (ROOT / ".train_only").exists()
    env_use = os.getenv("ELIOS_USE_MODEL_FOR_SIGNALS")
    env_train_only = os.getenv("ELIOS_TRAIN_ONLY")
    if sentinel:
        return True, notes, sev
    if falsy(env_use) or truthy(env_train_only):
        return True, notes, sev
    if env_use is None and env_train_only is None:
        notes.append(
            "⚠️ Не найден ни .train_only, ни env-переменные — не можем подтвердить train-only."
        )
        return None, notes, 2
    notes.append(
        "❌ Среда намекает, что модель может участвовать в сигналах (train-only не подтверждён)."
    )
    return False, notes, 1


SHDW_JSON = LOGS / "shadow_metrics.json"
MIN_P5 = float(os.getenv("ELIOS_SHADOW_MIN_P5", "0.30"))
MIN_P10 = float(os.getenv("ELIOS_SHADOW_MIN_P10", "0.25"))
MIN_P20 = float(os.getenv("ELIOS_SHADOW_MIN_P20", "0.20"))
MAX_BRIER = float(os.getenv("ELIOS_SHADOW_MAX_BRIER", "0.22"))


def read_shadow_metrics():
    try:
        import json

        if not SHDW_JSON.exists():
            return None
        d = json.loads(SHDW_JSON.read_text())
        s = d.get("summary", {})
        return {
            "p5": s.get("p@5"),
            "p10": s.get("p@10"),
            "p20": s.get("p@20"),
            "brier": s.get("brier"),
            "days": s.get("days"),
        }
    except Exception:
        return None


def compute_status(data: dict):
    notes = []
    sev = 0  # 0 OK, 1 FAIL, 2 WARN

    _, tn_notes, tn_sev = detect_train_only()
    notes += tn_notes
    sev = max(sev, tn_sev)

    auc = data["aucs"][0] if data["aucs"] else None
    ap = data["aps"][0] if data["aps"] else None
    if auc is None or ap is None:
        notes.append("❌ Не найдены последние AUC/AP в логах.")
        sev = max(sev, 1)
    else:
        # Shadow thresholds
        sh = read_shadow_metrics()
        if sh:
            if sh.get("p10") is not None and sh["p10"] < MIN_P10:
                notes.append(
                    f"⚠️ Shadow p@10={sh['p10']:.2f} ниже порога {MIN_P10:.2f}."
                )
                sev = max(sev, 2)
            if sh.get("p5") is not None and sh["p5"] < MIN_P5:
                notes.append(f"⚠️ Shadow p@5={sh['p5']:.2f} ниже порога {MIN_P5:.2f}.")
                sev = max(sev, 2)
            if sh.get("p20") is not None and sh["p20"] < MIN_P20:
                notes.append(
                    f"⚠️ Shadow p@20={sh['p20']:.2f} ниже порога {MIN_P20:.2f}."
                )
                sev = max(sev, 2)
            if sh.get("brier") is not None and sh["brier"] > MAX_BRIER:
                notes.append(f"⚠️ Shadow Brier={sh['brier']:.3f} выше {MAX_BRIER:.2f}.")
                sev = max(sev, 2)

        if auc < MIN_AUC:
            notes.append(f"❌ AUC={auc:.4f} ниже порога {MIN_AUC:.2f}.")
            sev = max(sev, 1)
        if ap < MIN_AP:
            notes.append(f"❌ AP={ap:.4f} ниже порога {MIN_AP:.2f}.")
            sev = max(sev, 1)

    if data["psi"]:
        atr, vol = data["psi"][0]
        if atr > MAX_PSI or vol > MAX_PSI:
            notes.append(
                f"⚠️ PSI дрейф: atr={atr:.2f}, vol={vol:.2f} (порог {MAX_PSI:.2f})."
            )
            sev = max(sev, 2)

    e = data["eod"]
    tot = e["tot"]
    if tot:
        fr = e["fail"] / tot
        if fr > MAX_FAIL_RATE:
            notes.append(f"⚠️ EOD fail-rate={fr:.1%} (порог {MAX_FAIL_RATE:.0%}).")
            sev = max(sev, 2)

    cutoff = data["cutoff_right"]
    if cutoff is None:
        notes.append("⚠️ Не удалось определить cutoff (правую дату).")
        sev = max(sev, 2)
        lag = None
    else:
        lag = (datetime.now(TZ).date() - cutoff).days
        if lag > MAX_CUTOFF_LAG_DAYS:
            notes.append(
                f"⚠️ Cutoff отстаёт на {lag} дн. (порог {MAX_CUTOFF_LAG_DAYS})."
            )
            sev = max(sev, 2)
    status = {0: "OK", 1: "FAIL", 2: "WARN"}[sev]
    return status, notes, lag


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    data = scan_logs()
    shadow = read_shadow_metrics()
    status, notes, lag_days = compute_status(data)
    report = {
        "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": status,
        "thresholds": {
            "MIN_AUC": MIN_AUC,
            "MIN_AP": MIN_AP,
            "MAX_PSI": MAX_PSI,
            "MAX_FAIL_RATE": MAX_FAIL_RATE,
            "MAX_CUTOFF_LAG_DAYS": MAX_CUTOFF_LAG_DAYS,
        },
        "latest": {
            "auc": data["aucs"][0] if data["aucs"] else None,
            "ap": data["aps"][0] if data["aps"] else None,
            "psi_atr_vol": data["psi"][0] if data["psi"] else [None, None],
            "eod": data["eod"],
            "cutoff_right_date": (
                data["cutoff_right"].isoformat() if data["cutoff_right"] else None
            ),
            "cutoff_lag_days": lag_days,
        },
        "notes": notes,
    }
    try:
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # консольный вывод
    print(f"🤖 Elios — Training Guard [{report['status']}]")
    latest = report["latest"]
    print(f"🎯 AUC={latest['auc']!s} | AP={latest['ap']!s}")
    atr_vol = latest["psi_atr_vol"]
    if atr_vol and all(x is not None for x in atr_vol):
        print(f"🧪 PSI: atr={atr_vol[0]}, vol={atr_vol[1]}")
    e = latest["eod"]
    if e["tot"]:
        fr = e["fail"] / e["tot"]
        print(f"📦 EOD: act={e['act']}/{e['tot']} | fail={e['fail']} ({fr:.1%})")
    if latest["cutoff_right_date"]:
        print(f"🗓 cutoff→ {latest['cutoff_right_date']}")
    if shadow:
        print(
            f"🔭 Shadow: p@5={shadow['p5']!s} p@10={shadow['p10']!s} p@20={shadow['p20']!s} | Brier={shadow['brier']!s} (days={shadow['days']!s})"
        )
    for n in notes:
        print("• " + n)

    # Telegram alert при WARN/FAIL
    if SEND_TG and status in {"WARN", "FAIL"}:
        try:
            msg = [
                f"*Elios — Training Guard* [{status}]",
                f"AUC={latest['auc']} | AP={latest['ap']}",
            ]
            if atr_vol and all(x is not None for x in atr_vol):
                msg.append(f"PSI: atr={atr_vol[0]}, vol={atr_vol[1]}")
            if latest["cutoff_right_date"]:
                lag = latest["cutoff_lag_days"]
                msg.append(f"cutoff→ {latest['cutoff_right_date']} (lag {lag}d)")
            if e["tot"]:
                fr = e["fail"] / e["tot"]
                msg.append(
                    f"EOD: act={e['act']}/{e['tot']} fail={e['fail']} ({fr:.1%})"
                )
            if notes:
                # берём первые 3 ноты, чтобы не спамить
                msg.append("• " + "\n• ".join(notes[:3]))
            send_telegram_message("\n".join(msg))
        except Exception:
            # не ломаем выходной код из-за Telegram
            traceback.print_exc()

    sys.exit({"OK": 0, "FAIL": 1, "WARN": 2}[status])


if __name__ == "__main__":
    main()
