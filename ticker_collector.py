import json
import time

import pandas as pd
import yfinance as yf

csv_path = "tickers_nasdaq.csv"
output_path = "/core/trading/candidates.json"

df = pd.read_csv(csv_path)
tickers = df["Symbol"].dropna().unique().tolist()

filtered_tickers = []
print(f"📥 Загружено тикеров из CSV: {len(tickers)}")

for symbol in tickers:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        price = info.get("regularMarketPrice", None)
        volume = info.get("volume", 0)
        name = info.get("shortName", "")

        if price and volume and volume >= 1_000_000:
            filtered_tickers.append(symbol)
            print(f"✅ {symbol} | ${price} | Vol: {volume}")
        else:
            print(f"❌ {symbol} пропущен — низкий объем или нет данных")

        time.sleep(0.6)

    except Exception as e:
        print(f"⚠️ {symbol} ошибка: {e}")

with open(output_path, "w") as f:
    json.dump(filtered_tickers, f, indent=2)

print(f"\n💾 Сохранено: {len(filtered_tickers)} тикеров → {output_path}")
