#!/usr/bin/env bash
set -u
cd /root/stockbot
OUT=""
if OUT=$(/root/stockbot/venv/bin/python -u -m core.trading.account_sync 2>&1); then
  printf "%s\n" "$OUT" | /root/stockbot/venv/bin/python -u tools/pipe_to_telegram.py
  exit 0
else
  printf "❌ Account Sync FAILED\n\n%s\n" "$OUT" | /root/stockbot/venv/bin/python -u tools/pipe_to_telegram.py
  exit 1
fi
