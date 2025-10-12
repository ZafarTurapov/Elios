"""Simple smoke tests for signal normalization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running the script directly (``python tools/ci_sanity.py``)
    # without requiring callers to modify ``PYTHONPATH``.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.normalize_signals import normalize_payload  # type: ignore
else:  # pragma: no cover - defensive branch for module execution
    from .normalize_signals import normalize_payload

SAMPLES = [
    # 1) уже список
    [{"symbol": "aapl", "price": 190.1, "action": "buy", "weight": 2}],
    # 2) dict {SYM: {...}}
    {"TSLA": {"price": 255.0, "action": "BUY"}},
    # 3) dict {SYM: price}
    {"NVDA": 980.5},
    # 4) завернутый {"signals":[...]}
    {"signals": [{"symbol": "msft", "price": 410, "action": "BUY"}]},
    # 5) мусор — должен быть отфильтрован
    {"BAD": {"price": 0, "action": "BUY"}},
]


def main() -> None:
    total = 0
    for sample in SAMPLES:
        out = normalize_payload(sample)
        assert isinstance(out, dict) and "signals" in out
        for row in out["signals"]:
            assert set(("symbol", "price", "action", "weight", "meta")).issubset(row)
            assert isinstance(row["symbol"], str) and row["symbol"].isupper()
            assert row["price"] > 0
            assert row["action"] in ("BUY", "SELL")
        total += len(out["signals"])

    # базовая гарантия: что-то нормализовали
    assert total >= 3, f"Too few normalized rows: {total}"
    print(
        f"✅ CI sanity ok: normalized {total} rows across {len(SAMPLES)} samples"
    )


if __name__ == "__main__":
    main()
