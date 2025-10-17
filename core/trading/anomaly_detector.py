import yfinance as yf


def detect_anomalies(symbol, period="30d", interval="1d", threshold=2.5):
    data = yf.download(
        symbol, period=period, interval=interval, progress=False, auto_adjust=False
    )

    if data.empty or data.shape[0] < 10:
        return False, "Недостаточно данных"

    volumes = data["Volume"].dropna()
    if volumes.empty or len(volumes) < 10:
        return False, "Недостаточно объёмов"

    mean_vol = volumes.mean().item()
    std_vol = volumes.std().item()
    last_vol = volumes.iloc[-1].item()

    is_anomaly = last_vol > (mean_vol + threshold * std_vol)
    reason = f"Объём: {int(last_vol)} > {int(mean_vol)} + {threshold} * {int(std_vol)}"

    return bool(is_anomaly), reason if is_anomaly else "Аномалий не найдено"


# 🧪 Тест
if __name__ == "__main__":
    result, explanation = detect_anomalies("AAPL")
    print(f"🔍 Аномалия: {result}, причина: {explanation}")
