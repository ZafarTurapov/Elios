# /root/stockbot/core/training/daily_training_update.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ежедневный оркестратор обучения (без привязки к торговому циклу).

Шаги:
 1) Бэкап текущего training_data.json
 2) Сбор свежих данных: core/trading/training_data_builder.py (ALLOW_DIRECT_BUILDER=1)
 3) Слияние с историей без дублей
 4) (опц.) patch_training_atr_pct.py — если присутствует
 5) verify_dataset.py — sanity-check + короткий отчёт
 6) strategy_trainer.py — обучение, бэкап модели, метрики
 7) Telegram-резюме

Страховки:
 - Lock-файл (anti-parallel)
 - daily_training_state.json: не запускаться повторно в тот же день
 - Таймауты подпроцессов
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/stockbot")

# Скрипты пайплайна
BUILDER = ROOT / "core" / "trading" / "training_data_builder.py"
PATCH_ATR = ROOT / "core" / "training" / "patch_training_atr_pct.py"  # optional
VERIFY = ROOT / "core" / "training" / "verify_dataset.py"
TRAINER = ROOT / "core" / "training" / "strategy_trainer.py"

# Артефакты
TRAINING_JSON = ROOT / "core" / "trading" / "training_data.json"
BACKUP_DIR = ROOT / "core" / "training" / "backups"
VERIFY_REPORT = ROOT / "core" / "training" / "verify_report.json"
METRICS_CSV = ROOT / "core" / "training" / "training_metrics.csv"
MODEL_PATH = ROOT / "core" / "training" / "trained_model.pkl"

# Логи/состояния
LOG_PATH = ROOT / "logs" / "daily_training_update.log"
STATE_PATH = ROOT / "core" / "training" / "daily_training_state.json"
LOCK_PATH = ROOT / "core" / "training" / "daily_training_update.lock"

# Python интерпретатор
PREFER_VENV = ROOT / "venv" / "bin" / "python"
PYTHON_BIN = str(PREFER_VENV if PREFER_VENV.exists() else Path("/usr/bin/python3"))

# Telegram (мягкий импорт)
try:
    from core.utils.telegram import escape_markdown, send_telegram_message
except Exception:

    def send_telegram_message(msg: str):  # fallback в лог
        print("[TG MOCK]\n" + msg)

    def escape_markdown(s: str) -> str:
        return s


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def logln(msg: str):
    ts = now_utc().isoformat()
    line = f"{ts} {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_json_safe(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logln(f"[ERROR] load_json {p}: {e}")
        return default


def save_json_safe(p: Path, obj) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2))
        return True
    except Exception as e:
        logln(f"[ERROR] save_json {p}: {e}")
        return False


def make_backup(src: Path) -> Path | None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = now_utc().strftime("%Y%m%dT%H%M%S")
    dst = BACKUP_DIR / f"training_data_{ts}.json.bak"
    if src.exists():
        try:
            shutil.copy2(src, dst)
            logln(f"[BACKUP] saved {dst}")
            return dst
        except Exception as e:
            logln(f"[ERROR] backup {src}: {e}")
    return None


def merge_dedup(old_list: list, new_list: list) -> tuple[list, int]:
    seen = set()
    merged = []

    def key_of(d):
        return (
            d.get("symbol"),
            str(d.get("entry_price")),
            d.get("timestamp") or d.get("timestamp_entry"),
            d.get("timestamp_exit"),
        )

    for d in old_list:
        k = key_of(d)
        if k not in seen:
            seen.add(k)
            merged.append(d)
    appended = 0
    for d in new_list:
        k = key_of(d)
        if k not in seen:
            seen.add(k)
            merged.append(d)
            appended += 1
    return merged, appended


def run_py(
    script: Path, desc: str, timeout_s: int = 1800, env_extra: dict | None = None
) -> subprocess.CompletedProcess:
    if not script.exists():
        raise FileNotFoundError(f"{desc}: {script} not found")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    cmd = [PYTHON_BIN, str(script)]
    logln(f"[RUN] {desc}: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s, env=env
    )
    logln(f"[RET] {desc}: code={proc.returncode}")
    if proc.stdout:
        logln(f"[STDOUT] {proc.stdout.strip()[:4000]}")
    if proc.stderr:
        logln(f"[STDERR] {proc.stderr.strip()[:4000]}")
    if proc.returncode != 0:
        raise RuntimeError(f"{desc} failed with code {proc.returncode}")
    return proc


def already_ran_today() -> bool:
    st = load_json_safe(STATE_PATH, {})
    last = st.get("last_run_utc")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return False
    return last_dt.date() == now_utc().date()


def update_state_ok():
    save_json_safe(STATE_PATH, {"last_run_utc": now_utc().isoformat()})


def main():
    # anti-parallel
    if LOCK_PATH.exists():
        logln("⛔ lock exists, another run in progress. Exit.")
        return
    try:
        LOCK_PATH.write_text(str(os.getpid()))

        # защита от дублей за день
        if already_ran_today():
            logln("ℹ️ already ran today — skipping.")
            return

        logln("🚀 DAILY TRAINING START")

        # 1) backup
        backup_path = make_backup(TRAINING_JSON)

        # 2) build fresh
        try:
            run_py(
                BUILDER, "Build training_data", env_extra={"ALLOW_DIRECT_BUILDER": "1"}
            )
        except Exception as e:
            logln(f"[FATAL] builder failed: {e}")
            send_telegram_message(escape_markdown(f"❌ Builder failed: {e}"))
            return

        # 3) merge old+new
        old_list = load_json_safe(backup_path if backup_path else TRAINING_JSON, [])
        new_list = load_json_safe(TRAINING_JSON, [])
        merged, appended = merge_dedup(old_list, new_list)
        if save_json_safe(TRAINING_JSON, merged):
            merged_bak = make_backup(TRAINING_JSON)
            logln(
                f"[MERGE] merged total={len(merged)}, appended_new={appended}, merged_backup={merged_bak}"
            )
        else:
            logln("[ERROR] failed to save merged training_data")
            send_telegram_message(
                escape_markdown("❌ Failed to save merged training_data")
            )
            return

        # 4) (optional) patch atr_pct
        if PATCH_ATR.exists():
            try:
                run_py(PATCH_ATR, "Patch atr_pct")
            except Exception as e:
                logln(f"[WARN] patch atr_pct failed: {e}")
        else:
            logln("[INFO] patch_training_atr_pct.py not found — skip")

        # 5) verify
        f1_cv = None
        rows = None
        groups = None
        try:
            run_py(VERIFY, "Verify dataset")
            rep = load_json_safe(VERIFY_REPORT, {})
            rows = rep.get("count")
            groups = rep.get("unique_symbols")
            if rep.get("sklearn_cv"):
                f1_cv = float(rep["sklearn_cv"].get("f1_mean") or 0.0)
        except Exception as e:
            logln(f"[WARN] verify failed: {e}")

        # 6) train
        f1_post = None
        folds = None
        feats = None
        try:
            run_py(TRAINER, "Train model")
            # прочитаем последнюю строку метрик
            if METRICS_CSV.exists():
                lines = METRICS_CSV.read_text().strip().splitlines()
                if len(lines) >= 2:
                    hdr = lines[0].split(",")
                    vals = lines[-1].split(",")
                    m = dict(zip(hdr, vals))
                    f1_post = m.get("f1_mean")
                    folds = m.get("folds")
                    feats = m.get("features")
        except Exception as e:
            logln(f"[FATAL] trainer failed: {e}")
            send_telegram_message(escape_markdown(f"❌ Trainer failed: {e}"))
            return

        # 7) Telegram summary
        msg = (
            "🤖 *Daily Training Summary*\n"
            f"🗓️ {escape_markdown(now_utc().isoformat())}\n"
            f"📦 rows={rows} | symbols={groups} | CV_F1(pre)={f1_cv if f1_cv is not None else 'n/a'} | folds={folds or 'n/a'}\n"
            f"🧠 model_f1(post)={f1_post or 'n/a'} | feats={escape_markdown(feats or 'n/a')}\n"
            f"📁 model: {escape_markdown(str(MODEL_PATH))}\n"
        )
        try:
            send_telegram_message(msg)
        except Exception as e:
            logln(f"[WARN] telegram send failed: {e}")

        update_state_ok()
        logln("✅ DAILY TRAINING FINISHED")

    finally:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
