# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from pathlib import Path
from core.trading.signals.ohlc_loader import fetch_history_ohlc

CAND_PATHS = [
    Path("candidates.json"),
    Path("core/trading/candidates.json"),
]

OUT_DIR = Path("core/trading")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_FP = OUT_DIR / "signals.json"
REJECTED_FP = OUT_DIR / "rejected.json"


def load_candidates() -> list[str]:
    for p in CAND_PATHS:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # допускаем либо список строк, либо словарь {symbol: ...}
                if isinstance(data, list):
                    return [str(x).strip().upper() for x in data]
                if isinstance(data, dict):
                    return [str(k).strip().upper() for k in data.keys()]
            except Exception:
                pass
    return []


def main():
    print("[SAFE] signal_engine_simple start")
    offline = os.getenv("ELIOS_OFFLINE", "0") == "1"
    print(f"[SAFE] OFFLINE={offline}")

    syms = load_candidates()
    if not syms:
        print("[SAFE] candidates.json не найден или пуст — ничего не делаем.")
        SIGN_FP = OUT_DIR / "signals.json"
        REJ_FP = OUT_DIR / "rejected.json"
        SIGN_FP.write_text("[]", encoding="utf-8")
        REJ_FP.write_text("{}", encoding="utf-8")
        return

    rejected: dict[str, dict] = {}
    signals: list[dict] = []

    print(f"[SAFE] Проверяем {len(syms)} тикеров (dry/compat OHLC)…")
    for i, symbol in enumerate(syms, 1):
        try:
            print(f"🔎 {i:03d}/{len(syms)} {symbol} …", flush=True)
            hist = fetch_history_ohlc(symbol)
            if hist is None or hist.shape[0] < 2:
                rejected[symbol] = {"reason": "data_empty_ohlc"}
                continue
            # Здесь можно добавить простую фичу/фильтр, но цель шага — только стабильность загрузки
        except Exception as e:
            rejected[symbol] = {"reason": f"loader_error:{type(e).__name__}"}

    # Итог: пока безопасно — без сигналов, только диагностика
    SIGNALS_FP.write_text(
        json.dumps(signals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REJECTED_FP.write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[SAFE] Готово. signals={len(signals)}, rejected={len(rejected)}")
    print(f"[SAFE] → {SIGNALS_FP}")
    print(f"[SAFE] → {REJECTED_FP}")


if __name__ == "__main__":
    main()
