import os, json, urllib.parse, urllib.request

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

def send_telegram_message(text: str) -> bool:
    tok = _token()
    ids = _chat_ids()
    if not tok:
        print({"ok": False, "error": "no token"})
        return False
    if not ids:
        print({"ok": False, "error": "no chat_id candidates"})
        return False

    ok_any = False
    for cid in ids:
        url = f"{API_BASE}/bot{tok}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": cid, "text": text}).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                j = json.loads(resp.read().decode("utf-8"))
                ok_any = ok_any or bool(j.get("ok"))
        except Exception as e:
            print({"ok": False, "error": str(e)})
    return ok_any
