# -*- coding: utf-8 -*-
"""
patch_training_atr_pct.py
Добавляет поле atr_pct в training_data.json, если его нет,
используя atr и цену (entry_price/price/close) -> atr_pct = atr/price*100
Сохраняет резервную копию и обновлённый файл.
"""
import json
from pathlib import Path

DATA_PATH = Path("core/trading/training_data.json")
BACKUP_PATH = Path("core/trading/training_data.backup.before_atr_pct.json")

PRICE_KEYS = ["entry_price", "price", "close", "close_at_entry"]


def get_price(row):
    for k in PRICE_KEYS:
        v = row.get(k)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except Exception:
            pass
    return None


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    if not DATA_PATH.exists():
        print(f"❌ Не найден {DATA_PATH}")
        return
    data = json.loads(DATA_PATH.read_text())
    if not isinstance(data, list) or not data:
        print("❌ training_data.json пустой или не список.")
        return

    # бэкап
    try:
        if not BACKUP_PATH.exists():
            BACKUP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"🔒 Backup: {BACKUP_PATH}")
    except Exception as e:
        print(f"[WARN] Backup error: {e}")

    updated = 0
    unchanged = 0
    skipped = 0

    for row in data:
        if "atr_pct" in row and row.get("atr_pct") is not None:
            unchanged += 1
            continue
        atr = to_float(row.get("atr") or row.get("atr_value"))
        price = get_price(row)
        if atr is None or price is None or price <= 0:
            skipped += 1
            continue
        atr_pct = (atr / price) * 100.0
        row["atr_pct"] = round(atr_pct, 4)
        updated += 1

    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(
        f"✅ Готово. Добавлено atr_pct: {updated}, оставлено без изменений: {unchanged}, пропущено: {skipped}"
    )
    print(f"↪️ Файл обновлён: {DATA_PATH}")


if __name__ == "__main__":
    main()
