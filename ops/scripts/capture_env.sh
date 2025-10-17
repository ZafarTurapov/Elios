#!/usr/bin/env bash
set -euo pipefail
cd /root/stockbot

OUT="ops/snapshots/snapshot_$(date +%F_%H%M%S).md"

{
  echo "# Elios — environment snapshot ($(date -Is))"
  echo "## System"
  uname -a || true
  command -v lsb_release >/dev/null && lsb_release -a 2>/dev/null || true
  echo
  echo "## Python"
  ./venv/bin/python -V 2>/dev/null || python -V || true
  echo "### Pip freeze"
  ./venv/bin/pip freeze 2>/dev/null || pip freeze || true
  echo
  echo "## Repo"
  git remote -v
  echo
  echo "### Branch / HEAD"
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  echo
  echo "### Working tree status"
  git status -s
  echo
  echo "## Systemd units (current server)"
  systemctl status elios-executor.timer --no-pager || true
  systemctl status elios-executor.service --no-pager || true
  systemctl status elios-deploy.timer --no-pager || true
  systemctl status elios-deploy.service --no-pager || true
  echo
  echo "## Cron (root)"
  crontab -l 2>/dev/null || echo "(no crontab)"
  echo
  echo "## Project tree (trimmed)"
  echo '```'
  # обрезаем шум
  find . -maxdepth 3 -type d \( -name .git -o -name venv -o -name logs -o -name __pycache__ \) -prune -o -print | sed 's|^\./||'
  echo '```'
} > "$OUT"

echo "✅ Saved: $OUT"
