# -*- coding: utf-8 -*-
import pandas as pd
import os

DATA_DIR = "/root/stockbot/data"


def load_and_preview():
    files = [
        "sp500_stocks.csv",
        "sp500_companies.csv",
        "sp500_index.csv",
        "2014_Financial_Data.csv",
        "2015_Financial_Data.csv",
        "2016_Financial_Data.csv",
        "2017_Financial_Data.csv",
        "2018_Financial_Data.csv",
    ]

    for file in files:
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️ Файл не найден: {file}")
            continue

        try:
            df = pd.read_csv(path)
            print(f"\n📄 {file} — {df.shape[0]} строк × {df.shape[1]} колонок")
            print(df.head(3))
        except Exception as e:
            print(f"❌ Ошибка чтения {file}: {e}")


if __name__ == "__main__":
    load_and_preview()
