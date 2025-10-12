#!/usr/bin/env bash
set -e
ENV="/root/stockbot/.env.local"; sudo touch "$ENV"
upsert(){ k="$1"; v="$2"; grep -qE "^${k}=" "$ENV" \
  && sudo sed -i "s|^${k}=.*|${k}=${v}|" "$ENV" \
  || echo "${k}=${v}" | sudo tee -a "$ENV" >/dev/null; }
upsert GRACE_ENABLED 0
upsert MICRO_HOLD_ENABLED 0
echo "✅ Set: GRACE_ENABLED=0, MICRO_HOLD_ENABLED=0 (TP/SL как прежде, GPT в плюс — без удержаний)"
grep -E '^(GRACE_ENABLED|MICRO_HOLD_ENABLED)=' "$ENV" || true
