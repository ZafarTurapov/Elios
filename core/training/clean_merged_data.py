import pandas as pd
import os

RAW_PATH = "/root/stockbot/data/merged_sp500_fundamentals.csv"
STOCKS_PATH = "/root/stockbot/data/sp500_stocks.csv"
CLEANED_PATH = "/root/stockbot/data/merged_sp500_fundamentals_clean.csv"

def clean_merged_data():
    print(f"📂 Загружаем {RAW_PATH} батчами...")

    cleaned_chunks = []
    for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=50000, low_memory=False), start=1):
        print(f"🧹 Обработка пакета {i}, строк: {len(chunk)}")
        
        # Удаляем пустые строки
        chunk.dropna(how="all", inplace=True)

        # Заполняем NaN средними значениями по колонке
        numeric_cols = chunk.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            chunk[col] = chunk[col].fillna(chunk[col].mean())

        cleaned_chunks.append(chunk)

    # Объединяем все пакеты в один DataFrame
    df_clean = pd.concat(cleaned_chunks, ignore_index=True)
    print(f"✅ Всего строк после очистки: {len(df_clean)}")

    # === Добавляем колонку symbol
    print(f"🔗 Добавляем колонку symbol из {STOCKS_PATH}")
    tickers_df = pd.read_csv(STOCKS_PATH)
    unique_symbols = tickers_df["Symbol"].dropna().unique().tolist()

    if len(unique_symbols) == 0:
        raise ValueError("⚠️ В файле sp500_stocks.csv не найдены тикеры.")

    repeated_symbols = (unique_symbols * (len(df_clean) // len(unique_symbols) + 1))[:len(df_clean)]
    df_clean["symbol"] = repeated_symbols

    # === Добавляем колонку year
    YEARS = list(range(2014, 2024))  # 2014–2023
    ROWS_PER_YEAR = 502

    total_rows = len(df_clean)
    repeated_years = []

    for year in YEARS:
        repeated_years.extend([year] * ROWS_PER_YEAR)

    if len(repeated_years) > total_rows:
        repeated_years = repeated_years[:total_rows]
    else:
        repeated_years += [YEARS[-1]] * (total_rows - len(repeated_years))

    df_clean["year"] = repeated_years

    # Сохраняем очищенные данные
    df_clean.to_csv(CLEANED_PATH, index=False)
    print(f"✅ Очищенные данные сохранены: {CLEANED_PATH}")

if __name__ == "__main__":
    clean_merged_data()
