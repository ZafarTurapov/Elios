# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path
from tools.normalize_signals import normalize_payload

SRC = Path("core/trading/signals.json")
DST = Path("core/trading/signals.normalized.json")


def already_ok(payload):
    if not isinstance(payload, dict):
        return False
    rows = payload.get("signals")
    if not isinstance(rows, list):
        return False
    need = {"symbol", "price", "action", "weight", "meta"}
    for r in rows:
        if not isinstance(r, dict) or not need.issubset(r):
            return False
        if not isinstance(r["symbol"], str) or not r["symbol"].isupper():
            return False
        if not isinstance(r["price"], (int, float)) or r["price"] <= 0:
            return False
        if r["action"] not in ("BUY", "SELL"):
            return False
    return True


def main():
    if not SRC.exists():
        print("⚠️ signals.json not found; nothing to prep")
        return 0
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    if already_ok(payload):
        print("✅ signals.json OK (unified schema)")
        return 0
    fixed = normalize_payload(payload)
    DST.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    # подменяем основой файл атомарно
    SRC.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🔧 normalized -> {SRC} (kept copy: {DST})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
