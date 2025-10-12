# core/training/training_data_cleaner.py
import json
from pathlib import Path

DATA_PATH = Path("core/trading/training_data.json")
OUTPUT_PATH = Path("core/training/training_data_clean.json")

REQUIRED_FIELDS = ["alpha_score", "rsi", "ema_dev", "vol_ratio"]

def clean_training_data():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    cleaned = []
    for trade in data:
        if not all(trade.get(field) not in (None, 0) for field in REQUIRED_FIELDS):
            continue
        if trade.get("change_pct") in (None, 0):
            continue
        cleaned.append(trade)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"✅ Очищено: {len(cleaned)} из {len(data)} сделок сохранено в {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_training_data()
