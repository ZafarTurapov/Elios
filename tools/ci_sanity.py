# -*- coding: utf-8 -*-
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.normalize_signals import normalize_payload

samples = [
    # 1) уже список
    [{"symbol":"aapl","price":190.1,"action":"buy","weight":2}],
    # 2) dict {SYM: {...}}
    {"TSLA":{"price":255.0,"action":"BUY"}},
    # 3) dict {SYM: price}
    {"NVDA": 980.5},
    # 4) завернутый {"signals":[...]}
    {"signals":[{"symbol":"msft","price":410,"action":"BUY"}]},
    # 5) мусор — должен быть отфильтрован
    {"BAD":{"price":0, "action":"BUY"}}
]

total = 0
for s in samples:
    out = normalize_payload(s)
    assert isinstance(out, dict) and "signals" in out
    for row in out["signals"]:
        assert set(("symbol","price","action","weight","meta")).issubset(row)
        assert isinstance(row["symbol"], str) and row["symbol"].isupper()
        assert row["price"] > 0
        assert row["action"] in ("BUY","SELL")
    total += len(out["signals"])

# базовая гарантия: что-то нормализовали
assert total >= 3, f"Too few normalized rows: {total}"
print(f"✅ CI sanity ok: normalized {total} rows across {len(samples)} samples")