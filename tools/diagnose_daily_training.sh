#!/usr/bin/env bash
# Elios — Diagnose Daily Training (fixed cron scan)
set -euo pipefail

ROOT="/root/stockbot"
PYBIN="$ROOT/venv/bin/python"
PYMOD="PYTHONPATH=$ROOT $PYBIN"
LOGDIR="$ROOT/logs/diagnostics/daily_training"
TS="$(date +'%Y-%m-%d_%H-%M-%S')"
OUT="$LOGDIR/run_$TS.log"
mkdir -p "$LOGDIR"

echo "=== Elios Daily Training Diagnose @ $(date) (TZ=$(timedatectl show -p Timezone --value || echo 'n/a')) ===" | tee -a "$OUT"

section() { echo -e "\n--- $1 ---" | tee -a "$OUT"; }

# 1) Systemd timers
section "Systemd timers (elios/*trainer*/training/*data_builder*/quick_train)"
if command -v systemctl >/dev/null 2>&1; then
  systemctl list-timers --all | tee -a "$OUT"
  echo | tee -a "$OUT"
  for pat in "elios" "trainer" "training" "strategy" "data_builder" "quick_train"; do
    echo "[grep: $pat]" | tee -a "$OUT"
    systemctl list-timers --all | grep -Ei "$pat" || echo "(нет совпадений)" | tee -a "$OUT"
  done

  section "Unit files presence & status (candidates)"
  mapfile -t CANDS < <(systemctl list-unit-files --type=service --type=timer | \
    grep -Ei 'elios|trainer|training|strategy|data[_-]builder|quick[_-]train' | awk '{print $1}')
  if [ ${#CANDS[@]} -gt 0 ]; then
    printf "%s\n" "${CANDS[@]}" | tee -a "$OUT"
    for u in "${CANDS[@]}"; do
      echo -e "\n[systemctl status $u]" | tee -a "$OUT"
      systemctl status "$u" --no-pager || true | tee -a "$OUT"
    done
  else
    echo "(юнитов не найдено — проверь *.service/*.timer)" | tee -a "$OUT"
  fi
else
  echo "systemctl недоступен — пропускаю timers/services" | tee -a "$OUT"
fi

# 2) Cron (фикс: nullglob вместо '2>/dev/null' в списке путей)
section "Cron entries (global & cron.d)"
shopt -s nullglob
CRON_FILES=(/etc/crontab /etc/cron.d/* /var/spool/cron/*)
for f in "${CRON_FILES[@]}"; do
  [ -f "$f" ] || continue
  echo -e "\n[$f]" | tee -a "$OUT"
  grep -E 'stockbot|strategy_trainer|training_data_builder|fundamental|quick_train|core/training' "$f" || echo "(нет совпадений)" | tee -a "$OUT"
done
shopt -u nullglob

# 3) Артефакты модели и данных
section "Artifacts (trained_model.pkl, training_data.json/csv, training_metrics.csv)"
ART_MODEL_CANDS=(
  "$ROOT/core/training/trained_model.pkl"
  "$ROOT/core/training/model/trained_model.pkl"
)
ART_DATA_CANDS=(
  "$ROOT/core/training/training_data.json"
  "$ROOT/data/training_data.json"
  "$ROOT/logs/training_data.json"
  "$ROOT/core/training/training_data.csv"
  "$ROOT/data/training_data.csv"
)
ART_METRICS="$ROOT/logs/training_metrics.csv"

echo "[Models]" | tee -a "$OUT"
found_model=""
for p in "${ART_MODEL_CANDS[@]}"; do
  if [ -f "$p" ]; then
    found_model="$p"
    echo "• $p  (mtime=$(date -r "$p" +'%F %T'))  size=$(stat -c%s "$p")" | tee -a "$OUT"
  fi
done
[ -z "$found_model" ] && echo "(модель не найдена ни в одном из известных путей)" | tee -a "$OUT"

echo -e "\n[Training data]" | tee -a "$OUT"
found_data=""
for p in "${ART_DATA_CANDS[@]}"; do
  if [ -f "$p" ]; then
    found_data="$p"
    echo "• $p  (mtime=$(date -r "$p" +'%F %T'))  size=$(stat -c%s "$p")" | tee -a "$OUT"
  fi
done
[ -z "$found_data" ] && echo "(training_data.* не найдено)" | tee -a "$OUT"

echo -e "\n[Metrics CSV]" | tee -a "$OUT"
if [ -f "$ART_METRICS" ]; then
  echo "• $ART_METRICS  (mtime=$(date -r "$ART_METRICS" +'%F %T'))" | tee -a "$OUT"
  tail -n 5 "$ART_METRICS" | sed 's/^/    /' | tee -a "$OUT"
else
  echo "(metrics.csv не найден)" | tee -a "$OUT"
fi

# 4) Быстрая проверка датасета
section "Dataset quick stats"
if [ -n "${found_data:-}" ]; then
  if [[ "$found_data" == *.json ]]; then
    "$PYBIN" - <<'PY' "$found_data" 2>>"$OUT" | tee -a "$OUT"
import pandas as pd, sys
p = sys.argv[1]
try:
    df = pd.read_json(p)
except ValueError:
    df = pd.read_json(p, lines=True)
print(f"path={p}")
print(f"rows={len(df)}, cols={list(df.columns)}")
PY
  else
    "$PYBIN" - <<'PY' "$found_data" 2>>"$OUT" | tee -a "$OUT"
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p)
print(f"path={p}")
print(f"rows={len(df)}, cols={list(df.columns)}")
PY
  fi
else
  echo "(нет датасета — training_data.* отсутствует)" | tee -a "$OUT"
fi

# 5) Журналы служб (последние 3 дня)
section "Journal (last 3 days) — common training services"
if command -v systemctl >/dev/null 2>&1; then
  mapfile -t SVC_CANDS < <(systemctl list-unit-files --type=service | \
    grep -Ei 'trainer|training|strategy|data[_-]builder|fundamental|quick[_-]train' | awk '{print $1}')
  if [ ${#SVC_CANDS[@]} -gt 0 ]; then
    for s in "${SVC_CANDS[@]}"; do
      echo -e "\n[journalctl -u $s --since '3 days ago']" | tee -a "$OUT"
      journalctl -u "$s" --since "3 days ago" --no-pager -n 200 || true | tee -a "$OUT"
      echo "[errors grep]" | tee -a "$OUT"
      (journalctl -u "$s" --since "3 days ago" --no-pager | grep -Ei 'Traceback|Error|Exception|CRITICAL' || true) | sed 's/^/    /' | tee -a "$OUT"
    done
  else
    echo "(подходящих services не найдено)" | tee -a "$OUT"
  fi
else
  echo "systemctl недоступен — пропускаю journalctl" | tee -a "$OUT"
fi

# 6) Ручной прогон (опционально)
if [[ "${1:-}" == "--run-now" ]]; then
  section "RUN-NOW: training_data_builder.py"
  if [ -f "$ROOT/core/training/training_data_builder.py" ]; then
    (cd "$ROOT"; PYTHONPATH="$ROOT" "$PYBIN" core/training/training_data_builder.py) 2>&1 | tee -a "$OUT" || true
  else
    echo "(training_data_builder.py не найден)" | tee -a "$OUT"
  fi

  section "RUN-NOW: strategy_trainer.py"
  if [ -f "$ROOT/core/training/strategy_trainer.py" ]; then
    (cd "$ROOT"; PYTHONPATH="$ROOT" "$PYBIN" core/training/strategy_trainer.py) 2>&1 | tee -a "$OUT" || true
  else
    echo "(strategy_trainer.py не найден)" | tee -a "$OUT"
  fi
fi

# 7) Summary
section "Summary / Возможные причины"
echo "• Включён только quick_train.timer (01:30) — отдельные таймеры trainer/data_builder отключены." | tee -a "$OUT"
echo "• Если training_data.* не обновляется, quick_train тренирует на старых данных — нужно добавить шаг builder." | tee -a "$OUT"
echo "• См. журнал quick_train.service выше — если там Traceback, чинить скрипты." | tee -a "$OUT"
echo -e "\nЛог сохранён: $OUT" | tee -a "$OUT"
