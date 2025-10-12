#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/stockbot
PY="$ROOT/venv/bin/python"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Переопределяемые параметры (можно положить в $ROOT/.calibrate.env)
export ELIOS_CAL_TARGET_PREC=${ELIOS_CAL_TARGET_PREC:-0.62}
export ELIOS_CAL_WINDOW_TRADES=${ELIOS_CAL_WINDOW_TRADES:-80}
export ELIOS_CAL_MIN_TRADES=${ELIOS_CAL_MIN_TRADES:-30}
export ELIOS_HIT_PCT=${ELIOS_HIT_PCT:-1.5}
[ -f "$ROOT/.calibrate.env" ] && source "$ROOT/.calibrate.env"

# Пропуск, если рынок закрыт (мягкая проверка)
"$PY" - <<'PY'
import sys
sys.path.insert(0,"/root/stockbot")
try:
    from core.utils.market_calendar import is_market_open_today
    if not is_market_open_today():
        print("[auto-cal] market closed today -> skip")
        raise SystemExit(0)
except Exception as e:
    # если календаря нет — не мешаемся
    print("[auto-cal] calendar check: soft-fail:", e)
PY

# Бэкфилл (идемпотентно — без дублей)
"$PY" "$ROOT/tools/backfill_signals_from_postmortem.py" || true

# Калибровка (с логом)
ts=$(date -u +'%F_%H-%M-%S')
out="$LOG_DIR/calibrate_$ts.log"
echo "[auto-cal] run calibrate -> $out"
"$PY" "$ROOT/tools/calibrate_thresholds.py" | tee -a "$out"
