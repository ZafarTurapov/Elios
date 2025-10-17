#!/usr/bin/env bash
set -euo pipefail

# Опции
DRY="${ELIOS_DRY_RUN:-0}"          # 1 — холостой прогон
PYBIN="${PYBIN:-./venv/bin/python}"
PYTHONPATH="${PYTHONPATH:-/root/stockbot}"

export PYTHONPATH

echo "==> [prep] normalize signals"
$PYBIN tools/prep_signals.py || true

echo "==> [exec] trade_executor.py (dry=$DRY)"
if [[ "$DRY" == "1" || "${1:-}" == "--dry-run" ]]; then
  "$PYBIN" -u core/trading/trade_executor.py --dry-run
else
  "$PYBIN" -u core/trading/trade_executor.py
fi
