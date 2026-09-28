import os, json, urllib.parse, urllib.request

_MD2_NEED_ESCAPE = set("_[]()~`>#+-=|{}.!*")


def escape_markdown(text: str | None) -> str:
    """Escape Telegram MarkdownV2 special characters."""
    if text is None:
        return ""
    s = str(text)
    # Fast path: nothing to escape
    if not any(ch in _MD2_NEED_ESCAPE for ch in s):
        return s
    return "".join(("\\" + ch) if ch in _MD2_NEED_ESCAPE else ch for ch in s)

API_BASE = os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org")

def _token():
    return os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or ""

def _chat_ids():
    ids = (
        os.getenv("TELEGRAM_CHAT_IDS")
        or os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("TELEGRAM_CHAT_ID_MAIN")
        or ""
    )
    # поддержим ; и , как разделители
    return [s.strip() for s in ids.replace(";", ",").split(",") if s.strip()]

def send_telegram_message(
    text: str,
    chat_id: str | int | None = None,
    parse_mode: str | None = None,
    fallback_discover: bool = False,
) -> bool:
    tok = _token()
    ids: list[str] = []
    if chat_id:
        ids.append(str(chat_id))
    ids.extend(_chat_ids())
    # сохраняем обратную совместимость с прежним API, где fallback_discover
    # мог добавлять дополнительные chat_id. Здесь просто оставляем заглушку.
    _ = fallback_discover

    if not tok:
        print({"ok": False, "error": "no token"})
        return False
    if not ids:
        print({"ok": False, "error": "no chat_id candidates"})
        return False

    ok_any = False
    for cid in ids:
        url = f"{API_BASE}/bot{tok}/sendMessage"
        payload = {"chat_id": cid, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        data = urllib.parse.urlencode(payload).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                j = json.loads(resp.read().decode("utf-8"))
                ok_any = ok_any or bool(j.get("ok"))
        except Exception as e:
            print({"ok": False, "error": str(e)})
    return ok_any


def notify(text: str, parse_mode: str | None = None) -> bool:
    return send_telegram_message(text, parse_mode=parse_mode)


def send(text: str, chat_id: str | int | None = None, parse_mode: str | None = None) -> bool:
    return send_telegram_message(text, chat_id=chat_id, parse_mode=parse_mode)
