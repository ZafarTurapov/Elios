#!/usr/bin/env bash
set -euo pipefail
cd /root/stockbot

branch="${DEPLOY_BRANCH:-main}"
echo "[deploy] pulling origin/${branch} in $(pwd)"

# тянем и жёстко выравниваемся под удалённую ветку
git fetch origin "${branch}"
git reset --hard "origin/${branch}"

# если был предыдущий HEAD, проверим изменился ли requirements.txt
if git rev-parse -q --verify HEAD@{1} >/dev/null 2>&1; then
  if git diff --name-only HEAD@{1} HEAD | grep -qx "requirements.txt"; then
    echo "[deploy] requirements changed → installing…"
    if [[ -x ./venv/bin/pip ]]; then
      ./venv/bin/pip install -r requirements.txt || true
    else
      pip3 install -r requirements.txt || true
    fi
  fi
fi

# телега-уведомление (если есть скрипт)
if [[ -x tools/notify_telegram.sh ]]; then
  tools/notify_telegram.sh "✅ *Elios auto-deploy:* branch ${branch} → $(git rev-parse --short HEAD)"
fi
