#!/usr/bin/env bash
set -euo pipefail

echo "=== TIME ==="; TZ=Asia/Tashkent date

ROOT="/root/stockbot"
VENV="$ROOT/venv/bin/python3"
ENV="$ROOT/.env.local"

echo -e "\n=== SYSTEMD: loop timer & service ==="
systemctl is-enabled elios-loop.timer || true
systemctl status --no-pager --full elios-loop.service | sed -n '1,40p' || true
echo -e "\n-> last 60 lines:"
journalctl -u elios-loop.service -n 60 --no-pager || true

echo -e "\n=== ENV KEYS (.env.local) present? ==="
awk -F= '/^(ALPACA_API_KEY|ALPACA_SECRET_KEY|OPENAI_API_KEY|TELEGRAM_TOKEN)/{print "•",$1,"=", (length($2)>8 ? substr($2,1,4)"…":$2)}' "$ENV" 2>/dev/null || echo "No .env.local"

echo -e "\n=== ALPACA: account & clock ==="
source "$ENV"
for ep in "/v2/account" "/v2/clock"; do
  echo "-- $ep"
  curl -sS -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
           -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
           "${ALPACA_API_BASE_URL}${ep}" | jq .
done

echo -e "\n=== ALPACA: open orders & positions ==="
curl -sS -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
         -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
         "${ALPACA_API_BASE_URL}/v2/orders?status=open&nested=true" | jq 'length as $n | {open_orders:$n}'
curl -sS -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
         -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
         "${ALPACA_API_BASE_URL}/v2/positions" | jq 'length as $n | {positions:$n}'

echo -e "\n=== TELEGRAM quick test ==="
if [ -n "${TELEGRAM_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID_MAIN:-}" ]; then
  curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID_MAIN}" -d text="🔔 Elios: проверка связи (rescue)" | jq .
else
  echo "No TELEGRAM_TOKEN or TELEGRAM_CHAT_ID_MAIN"
fi

echo -e "\n=== FORCE REALIGN: restart loop service ==="
sudo systemctl restart elios-loop.service
sleep 1
journalctl -u elios-loop.service -n 40 --no-pager || true

echo -e "\n=== MANUAL TICK (signals -> executor -> sync) ==="
set -x
PYTHONPATH="$ROOT" "$VENV" -u -m core.trading.signal_engine || true
PYTHONPATH="$ROOT" "$VENV" -u -m core.trading.trade_executor || true
PYTHONPATH="$ROOT" "$VENV" -u -m core.trading.positions_sync || true
set +x

echo -e "\n=== ARTIFACTS: what did signaler produce? ==="
for f in "$ROOT/core/trading/signals.json" "$ROOT/core/trading/rejected.json" "$ROOT/core/trading/gpt_decisions.json"; do
  [ -f "$f" ] && echo "-- $(basename "$f") size=$(wc -c < "$f")" || echo "-- $(basename "$f"): not found"
done
if command -v jq >/dev/null 2>&1 && [ -f "$ROOT/core/trading/signals.json" ]; then
  echo "- signals count:"; jq 'length' "$ROOT/core/trading/signals.json"
  echo "- top 5 tickers:"; jq -r '.[0:5][]?.symbol // .[0:5][]?.ticker // empty' "$ROOT/core/trading/signals.json" | nl
fi

echo -e "\n=== LOOP LIVE LOG ==="
journalctl -u elios-loop.service -n 120 --no-pager || true
