# -*- coding: utf-8 -*-
"""
Elios Full Body Packer (v0.3.9+)
Собирает полный снапшот проекта в архив:
 - все файлы проекта по расширениям (теперь и .sh)
 - явно важные файлы и логи
 - копии systemd unit-файлов (service/timer) с префиксами Elios-цикла
Обновлено: добавлены train-guard (service/timer), sentinel .train_only, скрипты .sh, логи guard.
"""

import os
import zipfile
import shutil
from datetime import datetime
import subprocess
import sys
import re

# === Основные директории проекта ===
PROJECT_ROOT = "/root/stockbot"

# Какие расширения включаем из всего дерева проекта (добавлен .sh)
INCLUDE_EXTENSIONS = (
    ".py",
    ".json",
    ".csv",
    ".env",
    ".env.local",
    ".txt",
    ".parquet",
    ".pkl",
    ".md",
    ".log",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
)

# (Опционально) директории, которые пропускаем при обходе
EXCLUDE_DIR_NAMES = {".git", ".cache", "__pycache__", "venv", ".venv"}

# === Куда складывать копии systemd unit-файлов Elios ===
SYSTEMD_UNITS_TMP = os.path.join(PROJECT_ROOT, "systemd_units")
SYSTEMD_PATH = "/etc/systemd/system"

# === Фильтр по названию для unit-файлов ===
# Захватывает elios-* (включая elios-train-guard.*), loop_launcher*, eod-*, sell-*
ELIOS_UNITS_PREFIXES = ("elios-", "loop_launcher", "eod-", "sell-")

# === Явно добавляемые файлы (вне фильтра по расширению или вне проекта)
# ВАЖНО: всё, что критично для восстановления боевого цикла и обучения
EXTRA_FILES = [
    # --- Торговый цикл (актуальные модули) ---
    "/root/stockbot/core/trading/loop_launcher.py",
    "/root/stockbot/core/trading/trade_executor.py",
    "/root/stockbot/core/trading/signal_engine.py",
    "/root/stockbot/core/trading/positions_sync.py",
    "/root/stockbot/core/trading/pnl_tracker.py",
    "/root/stockbot/core/trading/account_sync.py",
    # --- Нотификатор и EOD-флаттенер ---
    "/root/stockbot/core/trading/sell_notifier.py",
    "/root/stockbot/core/trading/eod_flatten.py",
    # --- Данные торговли / состояние / конфиги адаптации ---
    "/root/stockbot/core/trading/open_positions.json",
    "/root/stockbot/core/trading/signals.json",
    "/root/stockbot/core/trading/rejected.json",
    "/root/stockbot/core/trading/gpt_decisions.json",
    "/root/stockbot/core/trading/adaptive_config.json",
    "/root/stockbot/core/trading/candidates.json",
    "/root/stockbot/core/trading/candidates_active.json",
    # --- Логи (ключевые для анализа) ---
    "/root/stockbot/logs/signal_log.json",
    "/root/stockbot/logs/rejected.csv",
    "/root/stockbot/logs/trade_review.csv",
    "/root/stockbot/logs/training_metrics.csv",
    "/root/stockbot/logs/executor_debug.json",
    "/root/stockbot/logs/eod_failed.json",
    # Новые логи guard:
    "/root/stockbot/logs/train_guard_report.json",
    "/root/stockbot/logs/train_guard_cron.log",
    # --- Обучение (данные и модули) ---
    "/root/stockbot/core/training/training_data.json",
    "/root/stockbot/core/training/training_data_builder.py",
    "/root/stockbot/core/training/strategy_trainer.py",
    "/root/stockbot/core/training/daily_training_update.py",
    "/root/stockbot/core/training/check_model_quality.py",
    "/root/stockbot/core/training/trained_model.pkl",
    "/root/stockbot/core/training/metrics.json",
    # --- Kaggle pipeline ---
    "/root/stockbot/core/training/clean_merged_data.py",
    "/root/stockbot/core/training/merge_with_labels.py",
    "/root/stockbot/core/training/update_fundamentals.py",
    "/root/stockbot/data/merged_sp500_fundamentals.csv",
    "/root/stockbot/data/fundamentals_with_labels.csv",
    "/root/stockbot/.kaggle/kaggle.json",  # если используется
    # --- Утилиты и диагностика ---
    "/root/stockbot/core/utils/telegram.py",
    "/root/stockbot/core/utils/market_calendar.py",
    "/root/stockbot/core/diagnostics/health_check.py",
    # --- Pipeline / модели / данные ---
    "/root/stockbot/core/pipeline/train_daily.py",
    "/root/stockbot/core/pipeline/train_from_dataset.py",
    "/root/stockbot/core/pipeline/enforce_cutoff_dataset.py",
    "/root/stockbot/core/pipeline/data_quality_report.py",
    "/root/stockbot/core/pipeline/model_promote.py",
    "/root/stockbot/core/pipeline/train_healthcheck.py",
    "/root/stockbot/core/models/feature_spec.json",
    "/root/stockbot/core/models/xgb_spike4_v1.json",
    "/root/stockbot/core/models/xgb_spike4_v1.meta.json",
    "/root/stockbot/core/models/xgb_spike4_v2.json",
    "/root/stockbot/core/models/xgb_spike4_v2.meta.json",
    "/root/stockbot/core/data/train/dataset.parquet",
    "/root/stockbot/logs/data_quality_report.json",
    "/root/stockbot/logs/data_quality_report.md",
    "/root/stockbot/logs/train_daily.log",
    "/root/stockbot/logs/train_healthcheck.log",
    # --- Новое: training guard и вспомогательные скрипты ---
    "/root/stockbot/tools/train_progress_guard.py",  # сам guard (явно)
    "/root/stockbot/tools/within_sell_window.sh",  # SELL-guard скрипт
    "/root/stockbot/tools/battle_preflight.sh",  # если есть
    "/root/stockbot/tools/sell_health_probe.py",  # если есть
    "/root/stockbot/.train_only",  # sentinel (если есть)
]

# === Environment snapshot (pip freeze, python version, crontab, systemd) ===
ENV_SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "env_snapshot")


def make_env_snapshot():
    try:
        os.makedirs(ENV_SNAPSHOT_DIR, exist_ok=True)
    except Exception as e:
        print(f"[WARN] cannot create snapshot dir: {e}")
        return
    # pip freeze / python version
    try:
        req = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        open(os.path.join(ENV_SNAPSHOT_DIR, "requirements_freeze.txt"), "w").write(req)
        open(os.path.join(ENV_SNAPSHOT_DIR, "python_version.txt"), "w").write(
            sys.version
        )
    except Exception as e:
        print(f"[WARN] freeze: {e}")
    # crontab
    try:
        crontab = subprocess.check_output(["crontab", "-l"], text=True)
        open(os.path.join(ENV_SNAPSHOT_DIR, "crontab.txt"), "w").write(crontab)
    except Exception as e:
        open(os.path.join(ENV_SNAPSHOT_DIR, "crontab.txt"), "w").write(
            f"[no crontab] {e}"
        )
    # systemd
    try:
        units = subprocess.check_output(
            ["systemctl", "list-units", "--type=service", "--no-pager"], text=True
        )
        open(os.path.join(ENV_SNAPSHOT_DIR, "systemd_services_all.txt"), "w").write(
            units
        )
        try:
            filt = subprocess.check_output(
                [
                    "bash",
                    "-lc",
                    r"systemctl list-units --type=service --no-pager | egrep -i 'elios|stockbot|train|loop|sell|eod'",
                ],
                text=True,
            )
        except Exception:
            filt = ""
        open(
            os.path.join(ENV_SNAPSHOT_DIR, "systemd_services_filtered.txt"), "w"
        ).write(filt)
        timers = subprocess.check_output(
            ["systemctl", "list-timers", "--all", "--no-pager"], text=True
        )
        open(os.path.join(ENV_SNAPSHOT_DIR, "systemd_timers.txt"), "w").write(timers)
    except Exception as e:
        print(f"[WARN] systemd snapshot: {e}")


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


def _copy_dropins_for(unit_filename, dest_dir):
    """Скопировать drop-in override'ы вида /etc/systemd/system/<unit>.d/*.conf"""
    drop_dir = os.path.join(SYSTEMD_PATH, f"{unit_filename}.d")
    if not os.path.isdir(drop_dir):
        return []
    copied = []
    for name in os.listdir(drop_dir):
        if not name.endswith(".conf"):
            continue
        src = os.path.join(drop_dir, name)
        dst = os.path.join(dest_dir, f"{unit_filename}.d", name)
        try:
            _copy(src, dst)
            copied.append(dst)
        except Exception as e:
            print(f"[WARN] cannot copy drop-in {src}: {e}")
    return copied


_ENV_LINE_RE = re.compile(
    r"^\s*EnvironmentFile(?:\s*=\s*|\s*=\s*-)(?P<path>\S+)\s*$", re.IGNORECASE
)


def _copy_environment_files(unit_src_path, dest_dir):
    """Пытаемся вытащить EnvironmentFile=… из unit-файла и скопировать их рядом."""
    copied = []
    try:
        with open(unit_src_path, "r") as f:
            for line in f:
                m = _ENV_LINE_RE.match(line.strip())
                if not m:
                    continue
                env_path = m.group("path")
                env_path = env_path.strip().strip('"').strip("'")
                if not env_path.startswith("/"):
                    continue
                if os.path.exists(env_path):
                    dst = os.path.join(dest_dir, "env", os.path.basename(env_path))
                    try:
                        _copy(env_path, dst)
                        copied.append(dst)
                    except Exception as e:
                        print(f"[WARN] cannot copy env {env_path}: {e}")
    except Exception as e:
        print(f"[WARN] read unit for env: {e}")
    return copied


def collect_systemd_units():
    """
    Копируем подходящие systemd *.service/*.timer в PROJECT_ROOT/systemd_units,
    плюс drop-in override'ы и EnvironmentFile-ы.
    """
    try:
        if os.path.exists(SYSTEMD_UNITS_TMP):
            shutil.rmtree(SYSTEMD_UNITS_TMP)
        os.makedirs(SYSTEMD_UNITS_TMP, exist_ok=True)
    except Exception as e:
        print(f"[WARN] cannot prepare systemd tmp: {e}")

    collected = []
    try:
        for filename in os.listdir(SYSTEMD_PATH):
            if not (filename.endswith(".service") or filename.endswith(".timer")):
                continue
            if not any(filename.startswith(pfx) for pfx in ELIOS_UNITS_PREFIXES):
                continue
            full_path = os.path.join(SYSTEMD_PATH, filename)
            dest_path = os.path.join(SYSTEMD_UNITS_TMP, filename)
            try:
                _copy(full_path, dest_path)
                collected.append((dest_path, os.path.relpath(dest_path, PROJECT_ROOT)))
                # drop-ins
                for dp in _copy_dropins_for(filename, SYSTEMD_UNITS_TMP):
                    collected.append((dp, os.path.relpath(dp, PROJECT_ROOT)))
                # env files
                for ep in _copy_environment_files(full_path, SYSTEMD_UNITS_TMP):
                    collected.append((ep, os.path.relpath(ep, PROJECT_ROOT)))
            except Exception as e:
                print(f"[WARN] cannot copy unit {filename}: {e}")
    except Exception as e:
        print(f"[WARN] cannot list {SYSTEMD_PATH}: {e}")
    return collected


def collect_files():
    """
    1) Собираем все файлы по расширениям из дерева проекта (включая .sh)
    2) Добираем явно указанные EXTRA_FILES (если существуют)
    3) Добавляем скопированные systemd unit-файлы
    """
    all_files = []

    # 1. Все файлы по расширению из проекта
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if file.endswith(INCLUDE_EXTENSIONS) or any(
                    file.endswith(ext) for ext in (".env.local",)
                ):
                    relative_path = os.path.relpath(full_path, PROJECT_ROOT)
                    all_files.append((full_path, relative_path))
            except Exception as e:
                print(f"[WARN] skip {full_path}: {e}")

    # 2. Явно выбранные файлы
    for extra in EXTRA_FILES:
        try:
            if os.path.exists(extra):
                rel_path = os.path.relpath(extra, PROJECT_ROOT)
                all_files.append((extra, rel_path))
            else:
                print(f"⚠️  Пропущен отсутствующий файл: {extra}")
        except Exception as e:
            print(f"[WARN] problem with extra {extra}: {e}")

    # 3. Systemd unit-файлы
    all_files.extend(collect_systemd_units())

    return all_files


def pack_archive():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    archive_name = f"{PROJECT_ROOT}/EliosFullBody_{now}.zip"
    make_env_snapshot()
    files = collect_files()

    # Удаляем дубликаты путей (могут попадать из двух источников)
    seen = set()
    unique_files = []
    for full_path, relative_path in files:
        key = (os.path.abspath(full_path), relative_path)
        if key not in seen:
            seen.add(key)
            unique_files.append((full_path, relative_path))

    with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for full_path, relative_path in unique_files:
            try:
                archive.write(full_path, arcname=relative_path)
            except Exception as e:
                print(f"[WARN] cannot add {full_path}: {e}")

    print(f"📦 Архив создан: {archive_name}")
    print(f"📄 Всего файлов: {len(unique_files)}")


if __name__ == "__main__":
    pack_archive()
