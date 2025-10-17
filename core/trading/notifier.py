# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import os

import requests


def _clean(x: str | None) -> str:
    if not x:
        return ""
    x = x.strip()
    if (x.startswith('"') and x.endswith('"')) or (
        x.startswith("'") and x.endswith("'")
    ):
        x = x[1:-1]
    return x.strip()


def _parse_chats() -> list[str]:
    # поддерживаем TELEGRAM_CHAT_ID и/или TELEGRAM_CHAT_IDS (через запятую/пробел)
    v1 = _clean(os.getenv("TELEGRAM_CHAT_ID"))
    vN = _clean(os.getenv("TELEGRAM_CHAT_IDS"))
    raw = " ".join([v1, vN]).strip()
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    # уникализируем, сохраняем порядок
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _plain(text: str) -> str:
    # жёстко в ASCII, чтобы точно не падало в терминале/телеге
    try:
        return (text or "").encode("ascii", "ignore").decode("ascii")
    except Exception:
        return text or ""


TG_TOKEN = _clean(os.getenv("TELEGRAM_BOT_TOKEN"))
CHATS = _parse_chats()


def send(text: str, html_mode: bool = False) -> bool:
    if not TG_TOKEN:
        print("[TG] skip: empty token")
        return False
    if not CHATS:
        print("[TG] skip: no chats")
        return False
    if not text:
        print("[TG] skip: empty text")
        return False

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    ok_any = False
    for chat in CHATS:
        try:
            data = {
                "chat_id": chat,
                "disable_web_page_preview": True,
            }
            if html_mode:
                data["text"] = html.escape(text, quote=False)
                data["parse_mode"] = "HTML"
            else:
                data["text"] = _plain(text)
            r = requests.post(url, json=data, timeout=12)
            if r.status_code != 200:
                body = r.text[:200].replace("\n", " ")
                print(f"[TG] chat={chat} http {r.status_code}: {body}")
                continue
            ok_any = True
        except Exception as e:
            print(f"[TG] chat={chat} fail: {e}")
    return ok_any


def ping() -> bool:
    if not TG_TOKEN:
        print("[TG] ping: no token")
        return False
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getMe", timeout=8)
        print(f"[TG] getMe: {r.status_code} {r.text[:120].replace(chr(10),' ')}")
    except Exception as e:
        print(f"[TG] getMe fail: {e}")
        return False
    return send("ELIOS: Telegram ready (plain)", html_mode=False)
