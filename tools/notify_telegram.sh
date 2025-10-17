#!/bin/bash
set -e
cd /root/stockbot

TOKEN="${TELEGRAM_BOT_TOKEN:-$TELEGRAM_TOKEN}"
CHAT_ID="${TELEGRAM_CHAT_ID_MAIN:-$TELEGRAM_CHAT_ID}"
MSG="$1"

if [[ -n "$TOKEN" && -n "$CHAT_ID" && -n "$MSG" ]]; then
  curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
       -d chat_id="${CHAT_ID}" \
       -d text="${MSG}" \
       -d parse_mode="Markdown" >/dev/null || true
fi
