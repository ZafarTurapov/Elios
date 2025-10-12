#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/stockbot"
ENV="$ROOT/.env.local"
TOKEN="7791660338:AAFg6u0B38wYeBVZkwavbQowU-1CibsDUes"

echo "— Пишу TELEGRAM_TOKEN в $ENV"
mkdir -p "$ROOT"
touch "$ENV"
umask 177
# убрать старые строки
grep -vE '^(TELEGRAM_TOKEN|TELEGRAM_CHAT_ID_MAIN|TELEGRAM_CHAT_ID_ALT)=' "$ENV" > "$ENV.tmp" || true
mv "$ENV.tmp" "$ENV"
printf 'TELEGRAM_TOKEN="%s"\n' "$TOKEN" >> "$ENV"
# сохраним ALT как резерв (если надо — поменяешь)
echo 'TELEGRAM_CHAT_ID_ALT="110181893"' >> "$ENV"
chmod 600 "$ENV"

echo "— Проверяю getMe…"
curl -sS "https://api.telegram.org/bot${TOKEN}/getMe" | jq .

echo "— Ищу chat_id через getUpdates (нужен хотя бы один /start боту)…"
CID=$(curl -sS "https://api.telegram.org/bot${TOKEN}/getUpdates?limit=10" \
  | jq -r '[.result[]?|(.message//.channel_post//{})|select(.chat and .chat.id)|.chat.id][-1] // empty')

if [ -n "$CID" ]; then
  echo "✓ Найден chat_id: $CID — сохраняю TELEGRAM_CHAT_ID_MAIN"
  printf 'TELEGRAM_CHAT_ID_MAIN="%s"\n' "$CID" >> "$ENV"
else
  echo "⚠️  Не найден chat_id в getUpdates."
  echo "   Открой в Telegram этого бота и отправь ему /start, затем повтори: sudo /root/stockbot/tools/tg_fix.sh"
fi

source "$ENV"

if [ -n "${TELEGRAM_CHAT_ID_MAIN:-}" ]; then
  echo "— Шлю тест в MAIN ($TELEGRAM_CHAT_ID_MAIN)…"
  curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID_MAIN}" \
    -d text="🔔 Elios: тестовое сообщение (fix)" | jq .
fi

# Честный модуль отправки (OK/FAIL по факту ответа Telegram)
cat > "$ROOT/core/utils/telegram.py" <<'PY'
# -*- coding: utf-8 -*-
import os, json, urllib.request, urllib.parse
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID_MAIN = os.getenv("TELEGRAM_CHAT_ID_MAIN", "")
TELEGRAM_CHAT_ID_ALT  = os.getenv("TELEGRAM_CHAT_ID_ALT", "")
TIMEOUT = 10

def _api(method, params=None, get=False):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    if get:
        if params: url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "EliosTG/1.0"})
    else:
        data = urllib.parse.urlencode(params or {}).encode()
        req = urllib.request.Request(url, data=data, headers={"User-Agent": "EliosTG/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))

def _try_send(cid, text):
    try:
        js = _api("sendMessage", {"chat_id": str(cid), "text": text, "disable_web_page_preview":"1"})
        return bool(js.get("ok")), js
    except Exception as e:
        return False, {"ok": False, "error": str(e)}

def send_message(text, chat_id=None):
    # 1) явный cid -> MAIN -> ALT
    for cid in [chat_id, TELEGRAM_CHAT_ID_MAIN, TELEGRAM_CHAT_ID_ALT]:
        if not cid: continue
        ok, js = _try_send(cid, text)
        if ok:
            print(f"✅ Telegram OK -> {cid}")
            return True
    # 2) autodiscover из getUpdates (если ты написал боту хотя бы /start)
    try:
        upd = _api("getUpdates", {"limit":"10"}, get=True)
        for u in reversed(upd.get("result", [])):
            chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
            if chat and "id" in chat:
                cid = chat["id"]
                ok, js = _try_send(cid, text)
                if ok:
                    # Запишем MAIN на будущее
                    try:
                        env="/root/stockbot/.env.local"
                        content=open(env,"r",encoding="utf-8",errors="ignore").read() if os.path.exists(env) else ""
                        lines=[l for l in content.splitlines() if not l.startswith("TELEGRAM_CHAT_ID_MAIN=")]
                        lines.append(f'TELEGRAM_CHAT_ID_MAIN="{cid}"')
                        open(env,"w",encoding="utf-8").write("\n".join(lines)+"\n")
                        os.chmod(env,0o600)
                    except Exception:
                        pass
                    print(f"✅ Telegram DISCOVERED -> {cid}")
                    return True
    except Exception:
        pass
    print("❌ Telegram FAIL (нет доступного chat_id или бот не получил /start)")
    return False

# совместимость со старым кодом
def notify(text): return send_message(text)
def send(text, chat_id=None): return send_message(text, chat_id=chat_id)
PY

chmod 644 "$ROOT/core/utils/telegram.py"

echo "— Перезапускаю луп и показываю последние логи"
sudo systemctl daemon-reload
sudo systemctl restart elios-loop.service
journalctl -u elios-loop.service -n 80 --no-pager
