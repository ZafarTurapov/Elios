# -*- coding: utf-8 -*-
import json
from pathlib import Path
from core.utils.telegram import send_telegram_message, escape_markdown

ROOT = Path("/root/stockbot")
FAILS = ROOT / "logs/eod_failed.json"  # может быть dict или list
COUNTS = ROOT / "logs/eod_fail_counts.json"
IGNORE = ROOT / "core/data/ignore_symbols.json"
CAND = ROOT / "core/trading/candidates.json"
CAND_ACTIVE = ROOT / "core/trading/candidates_active.json"

PATTS = (
    "no timezone found",
    "symbol may be delisted",
    "http error 404",
)
THRESH = 3  # ≥3 подряд — в ignore


def _load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def _iter_fail_entries(obj):
    """Нормализуем разные возможные схемы в пары (symbol, message)."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for sym, info in obj.items():
            if isinstance(info, (str, int, float)) or info is None:
                msg = "" if info is None else str(info)
            else:
                msg = (
                    info.get("error") or info.get("message") or info.get("reason") or ""
                )
            yield str(sym), str(msg)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                yield item, ""
            elif isinstance(item, dict):
                sym = (
                    item.get("symbol")
                    or item.get("ticker")
                    or item.get("sym")
                    or item.get("Symbol")
                )
                if not sym:
                    # пробуем выдрать символ из текста, если он один
                    txt = json.dumps(item, ensure_ascii=False)
                    # как минимум не упадём
                    continue
                msg = (
                    item.get("error") or item.get("message") or item.get("reason") or ""
                )
                yield str(sym), str(msg)
    else:
        return


def main():
    fails_raw = _load_json(FAILS, None)
    counts = _load_json(COUNTS, {})
    ignore = set(_load_json(IGNORE, []))

    todays = set()
    for sym, msg in _iter_fail_entries(fails_raw):
        mlow = (msg or "").lower()
        hard = (
            any(p in mlow for p in PATTS) or not msg
        )  # если нет текста — считаем как “жёсткий” фейл
        if hard:
            todays.add(sym)
            counts[sym] = counts.get(sym, 0) + 1
        else:
            counts[sym] = max(0, counts.get(sym, 0) - 1)

    # чуть “распадается” счётчик по тем, кого сегодня не было
    for sym in list(counts.keys()):
        if sym not in todays:
            counts[sym] = max(0, counts[sym] - 1)

    newly_ignored = []
    for sym, n in counts.items():
        if n >= THRESH and sym not in ignore:
            ignore.add(sym)
            newly_ignored.append(sym)

    Path(COUNTS).write_text(json.dumps(counts, indent=2))
    Path(IGNORE).write_text(json.dumps(sorted(ignore), indent=2))

    # обновим список активных кандидатов (если есть общий)
    try:
        if Path(CAND).exists():
            all_syms = json.loads(Path(CAND).read_text())
            active = [s for s in all_syms if s not in ignore]
            Path(CAND_ACTIVE).write_text(json.dumps(active, indent=2))
    except Exception:
        pass

    if newly_ignored:
        msg = [
            "🧹 *Auto-ignore* добавлены: "
            + ", ".join(escape_markdown(s) for s in newly_ignored),
            f"Всего в ignore: {len(ignore)}",
        ]
        try:
            send_telegram_message("\n".join(msg))
        except Exception:
            pass


if __name__ == "__main__":
    main()
