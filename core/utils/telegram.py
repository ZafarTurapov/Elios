# -*- coding: utf-8 -*-
import os
import json
import urllib.request
import urllib.parse

# Берём токены/чаты из env_keys, чтобы сохранить единую точку правды
try:
    from core.utils.env_keys import (
        TELEGRAM_TOKEN,
        TELEGRAM_CHAT_ID_MAIN,
        TELEGRAM_CHAT_ID_ALT,
    )
except Exception:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID_MAIN = os.getenv("TELEGRAM_CHAT_ID_MAIN", "")
    TELEGRAM_CHAT_ID_ALT = os.getenv("TELEGRAM_CHAT_ID_ALT", "")

TIMEOUT = 10

# --- MarkdownV2-escape (совместимо с экзекутором/синхом) ---
_MD2_NEED_ESCAPE = r"_\*\[\]\(\)~`>#+\-=|{}.!"


def escape_markdown(text: str | None) -> str:
    """Экранирует спецсимволы для Telegram MarkdownV2."""
    if text is None:
        return ""
    s = str(text)
    out = []
    for ch in s:
        if ch in _MD2_NEED_ESCAPE:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


# --- Низкоуровневый вызов Telegram Bot API ---
def _api(method, params=None, get=False):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    if get:
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "EliosTG/1.0"})
    else:
        data = urllib.parse.urlencode(params or {}).encode()
        req = urllib.request.Request(
            url, data=data, headers={"User-Agent": "EliosTG/1.0"}
        )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def get_me():
    try:
        return _api("getMe", get=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_updates(limit=10):
    try:
        return _api("getUpdates", {"limit": str(limit)}, get=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _try_send(cid, text, parse_mode=None):
    params = {"chat_id": str(cid), "text": text, "disable_web_page_preview": "1"}
    if parse_mode:
        params["parse_mode"] = parse_mode
    try:
        js = _api("sendMessage", params)
        return bool(js.get("ok")), js
    except Exception as e:
        return False, {"ok": False, "error": str(e)}


def _persist_main_chat_id(cid: int):
    env = "/root/stockbot/.env.local"
    try:
        content = ""
        if os.path.exists(env):
            with open(env, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        lines = [
            l
            for l in content.splitlines()
            if not l.startswith("TELEGRAM_CHAT_ID_MAIN=")
        ]
        lines.append(f'TELEGRAM_CHAT_ID_MAIN="{cid}"')
        with open(env, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(env, 0o600)
    except Exception:
        pass


def send_telegram_message(
    text: str,
    chat_id: str | int | None = None,
    parse_mode: str | None = None,
    fallback_discover: bool = True,
) -> bool:
    """
    Порядок: явный chat_id → MAIN → ALT → (опционально) autodiscover через getUpdates.
    Возвращает True/False и печатает понятный лог OK/FAIL.
    """
    candidates = [chat_id, TELEGRAM_CHAT_ID_MAIN, TELEGRAM_CHAT_ID_ALT]
    tried = []

    for cid in candidates:
        if not cid:
            continue
        ok, js = _try_send(cid, text, parse_mode)
        tried.append((cid, ok, js))
        if ok:
            print(f"✅ Telegram OK -> {cid}")
            return True

    if fallback_discover:
        try:
            upd = get_updates(10)
            if upd.get("ok"):
                for u in reversed(upd.get("result", [])):
                    chat = (u.get("message") or u.get("channel_post") or {}).get("chat")
                    if chat and "id" in chat:
                        cid = chat["id"]
                        ok, js = _try_send(cid, text, parse_mode)
                        tried.append((cid, ok, js))
                        if ok:
                            _persist_main_chat_id(cid)
                            print(f"✅ Telegram DISCOVERED -> {cid}")
                            return True
        except Exception:
            pass

    last = tried[-1][2] if tried else {"error": "no chat_id candidates"}
    print(f"❌ Telegram FAIL: {last}")
    return False


# Совместимость со старыми вызовами
def notify(text: str, parse_mode: str | None = None) -> bool:
    return send_telegram_message(text, parse_mode=parse_mode)


def send(
    text: str, chat_id: str | int | None = None, parse_mode: str | None = None
) -> bool:
    return send_telegram_message(text, chat_id=chat_id, parse_mode=parse_mode)
