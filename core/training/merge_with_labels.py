# core/training/merge_with_labels.py

import json

import pandas as pd
from tqdm import tqdm

CLEANED_PATH = "data/merged_sp500_fundamentals_clean.csv"
TRAINING_DATA_PATH = "core/trading/training_data.json"
OUTPUT_PATH = "data/fundamentals_with_labels.csv"


def merge_with_labels():
    print(f"📂 Загружаем боевую историю: {TRAINING_DATA_PATH}...")
    with open(TRAINING_DATA_PATH, "r") as f:
        training_data = json.load(f)

    print(f"📂 Загружаем очищенные данные батчами из {CLEANED_PATH}...")
    merged_rows = []
    for chunk in pd.read_csv(CLEANED_PATH, chunksize=50000):
        for trade in tqdm(training_data[:], desc="📌 Мержим"):
            symbol = trade["symbol"].upper()
            try:
                year = int(trade["timestamp_exit"][:4])
            except Exception:
                continue

            df = chunk.copy()
            df_filtered = df[df["symbol"].str.upper() == symbol]

            # ⛔️ Отключено: фильтрация по году
            # df_filtered = df_filtered[df_filtered["year"] == year]

            if df_filtered.empty:
                continue

            df_filtered = df_filtered.copy()
            df_filtered["label_profit_share"] = trade["change_pct"]
            merged_rows.append(df_filtered)

    if merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)
        print(f"✅ Пакет 1 объединён — {len(merged_df)} строк")
        merged_df.to_csv(OUTPUT_PATH, index=False)
        print(f"🎯 Смерженные данные сохранены: {OUTPUT_PATH}")
    else:
        print(
            "⚠️ Не удалось объединить ни одной строки. Проверь соответствие symbol и year."
        )


if __name__ == "__main__":
    merge_with_labels()
