# -*- coding: utf-8 -*-
import sys

from core.utils.telegram import send_telegram_message


def chunk(s, n=3900):
    # телега ~4096 симв., оставим запас
    for i in range(0, len(s), n):
        yield s[i : i + n]


def main():
    text = sys.stdin.read().strip()
    if not text:
        return
    header = "📦 Elios — Account Sync Report (forwarded)"
    body = f"{header}\n{text}"
    # если слишком длинно — порежем на части
    parts = list(chunk(body))
    for i, part in enumerate(parts, 1):
        suffix = f"\n— part {i}/{len(parts)}" if len(parts) > 1 else ""
        send_telegram_message(part + suffix)


if __name__ == "__main__":
    main()
