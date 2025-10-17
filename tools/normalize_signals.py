# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, List

REQ_FIELDS = ("symbol", "price", "action")


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("signals"), list):
            return payload["signals"]
        out = []
        for k, v in payload.items():
            if k == "signals":
                continue
            if isinstance(v, dict):
                row = {"symbol": k, **v}
            else:
                row = {"symbol": k, "price": float(v or 0), "action": "BUY"}
            out.append(row)
        return out
    elif isinstance(payload, list):
        return payload
    return []


def _norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(row.get("symbol", "")).strip().upper()
    price = float(row.get("price") or row.get("last") or row.get("close") or 0.0)
    action = str(row.get("action", "BUY")).strip().upper() or "BUY"
    weight = float(row.get("weight") or 1.0)
    meta = row.get("meta") or {}
    return {
        "symbol": sym,
        "price": price,
        "action": action,
        "weight": weight,
        "meta": meta,
    }


def normalize_payload(payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    seq = _as_list(payload)
    out, seen = [], set()
    for raw in seq:
        row = _norm_row(raw)
        if not all(row.get(f) for f in REQ_FIELDS):  # symbol/price/action
            continue
        if row["price"] <= 0:
            continue
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        out.append(row)
    return {"signals": out}


def main():
    src = Path("core/trading/signals.json")
    if not src.exists():
        print("❌ core/trading/signals.json not found")
        return
    payload = json.loads(src.read_text(encoding="utf-8"))
    out = normalize_payload(payload)
    Path("core/trading/signals.normalized.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"✅ Normalized: {len(out['signals'])} -> core/trading/signals.normalized.json"
    )


if __name__ == "__main__":
    main()
