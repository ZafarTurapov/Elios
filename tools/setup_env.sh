#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/stockbot"
ENV_FILE="$ROOT/.env.local"

echo "— Введите ключи (ввод скрыт) —"
read -rsp "ALPACA_API_KEY: " AKEY; echo
read -rsp "ALPACA_SECRET_KEY: " ASEC; echo
read -rsp "OPENAI_API_KEY: " OKEY; echo

install -d -m 700 "$ROOT"
umask 177
cat > "$ENV_FILE" <<EOF
# === Elios env (не коммитить!) ===
ALPACA_API_KEY="$AKEY"
ALPACA_SECRET_KEY="$ASEC"

# Paper по умолчанию; для реала: https://api.alpaca.markets
ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"
# На всякий случай оставим совместимость со старым именем:
ALPACA_BASE_URL="https://paper-api.alpaca.markets"
ALPACA_DATA_BASE="https://data.alpaca.markets/v2"

OPENAI_API_KEY="$OKEY"
EOF
chmod 600 "$ENV_FILE"
echo "✓ $ENV_FILE создан и защищён (600)."

# Подключаем .env.local как EnvironmentFile к сервисам
for SVC in elios-loop.service sell-notifier.service eod-flatten.service; do
  DIR="/etc/systemd/system/$SVC.d"
  sudo mkdir -p "$DIR"
  sudo tee "$DIR/env.conf" >/dev/null <<EOF
[Service]
EnvironmentFile=$ENV_FILE
EOF
  echo "✓ Drop-in подключён: $SVC.d/env.conf"
done

# Перечитываем и убеждаемся, что таймеры активны
sudo systemctl daemon-reload
sudo systemctl reenable --now elios-loop.timer sell-notifier.timer eod-flatten.timer >/dev/null || true

# Прогон префлайта
echo "— Повторный префлайт —"
python3 "$ROOT/tools/elios_preflight.py" || true

echo "Готово. Если в конце 'PRE-FLIGHT: ГОТОВО К ЗАПУСКУ ✅' — всё ок."
