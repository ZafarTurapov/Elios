# /root/stockbot/core/training/merge_sp500_fundamentals.py
import pandas as pd
import os

DATA_DIR = "/root/stockbot/data"
OUTPUT_FILE = "/root/stockbot/data/merged_sp500_fundamentals.csv"

# Читаем все фундаментальные данные в один DataFrame
fundamental_dfs = []
for year in range(2014, 2019):
    file_path = os.path.join(DATA_DIR, f"{year}_Financial_Data.csv")
    df = pd.read_csv(file_path)
    df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
    df["Year"] = year
    fundamental_dfs.append(df)
fundamentals = pd.concat(fundamental_dfs, ignore_index=True)
print(f"Фундаментальные данные: {fundamentals.shape}")

# Пишем заголовок в итоговый файл
first_write = True

# Читаем sp500_stocks.csv по частям
chunksize = 200000
stocks_file = os.path.join(DATA_DIR, "sp500_stocks.csv")

for i, chunk in enumerate(pd.read_csv(stocks_file, chunksize=chunksize)):
    chunk["Year"] = pd.to_datetime(chunk["Date"]).dt.year
    merged_chunk = pd.merge(chunk, fundamentals, on=["Symbol", "Year"], how="inner")

    # Сохраняем кусок в CSV
    merged_chunk.to_csv(OUTPUT_FILE, mode="w" if first_write else "a",
                        index=False, header=first_write)
    first_write = False

    print(f"Обработан пакет {i+1}, строк: {len(merged_chunk)}")

print(f"✅ Готово! Итоговый файл: {OUTPUT_FILE}")
