import os
import sys
import json
import time
import pathlib

ROOT = pathlib.Path("/root/stockbot")
HB = ROOT / "logs/sell.heartbeat"
POS = ROOT / "core/trading/open_positions.json"

MAX_AGE_SEC = int(os.getenv("SELL_HB_MAX_AGE", "900"))  # 15 мин по умолчанию


def positions_count():
    try:
        obj = json.loads(POS.read_text(encoding="utf-8"))
        return len(obj or {})
    except Exception:
        return 0


def heartbeat_fresh():
    if not HB.exists():
        return False, "missing"
    try:
        j = json.loads(HB.read_text(encoding="utf-8"))
        ts = float(j.get("ts") or 0)
    except Exception:
        ts = HB.stat().st_mtime
    age = time.time() - ts
    return age <= MAX_AGE_SEC, f"{int(age)}s"


fresh, age_str = heartbeat_fresh()
pos_n = positions_count()

# политика:
# - если позиций нет -> OK даже если давно не было SELL (просто некого продавать)
# - если позиции есть -> OK, если heartbeat свежий; иначе FAIL (эскалация)
if pos_n == 0:
    print(f"OK: no positions; HB_age={age_str}")
    sys.exit(0)
if fresh:
    print(f"OK: positions={pos_n}; HB_age={age_str}")
    sys.exit(0)

print(f"FAIL: positions={pos_n}; stale HB ({age_str} > {MAX_AGE_SEC}s)")
sys.exit(1)
