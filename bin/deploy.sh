#!/usr/bin/env bash
set -euo pipefail
cd /root/stockbot

LOG="logs/deploy_$(date +%F).log"
exec >> "$LOG" 2>&1
echo "=== $(date -Is) deploy start ==="

# Защита от локальных правок
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠️  Есть незакоммиченные изменения — авто-деплой пропущен."
  exit 0
fi

git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [[ "$LOCAL" == "$REMOTE" ]]; then
  echo "✅ Уже на последнем коммите ($LOCAL)."
  exit 0
fi

echo "⬇️  Обновление: $LOCAL -> $REMOTE"
git pull --ff-only

# Обновить зависимости при изменении requirements.txt
if git diff --name-only HEAD@{1}..HEAD | grep -q '^requirements\.txt$'; then
  echo "📦 Обновляю зависимости…"
  ./venv/bin/pip install -U -r requirements.txt
fi

# Ничего перезапускать не нужно: наш executor one-shot, таймер сам вызовет свежий код
echo "✅ Деплой завершён."
