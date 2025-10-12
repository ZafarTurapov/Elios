from core.utils.paths import TRADE_LOG_PATH
# core/training/from_trade_log_to_training.py

import json
import os
from datetime import datetime

TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "../trading/training_data.json")


def append_to_training_data(entry):
    try:
        # Строгая фильтрация по причине FORCE_CLOSE
        if entry.get("reason") == "FORCE_CLOSE":
            print(f"[SKIP] Пропущена сделка по {entry.get('symbol')} — причина FORCE_CLOSE.")
            return

        if not os.path.exists(TRAINING_DATA_PATH):
            data = []
        else:
            with open(TRAINING_DATA_PATH, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []

        entry["timestamp_saved"] = datetime.utcnow().isoformat()
        data.append(entry)

        with open(TRAINING_DATA_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[📚] Обучающая запись добавлена: {entry.get('symbol')}")
    except Exception as e:
        print(f"[ERROR] Не удалось сохранить обучающие данные: {e}")

def main():
    if not os.path.exists(TRADE_LOG_PATH):
        print("[ERROR] trade_log.json не найден.")
        return

    try:
        with open(TRADE_LOG_PATH, "r") as f:
            trades_by_symbol = json.load(f)

        for symbol, trades in trades_by_symbol.items():
            if not isinstance(trades, list):
                print(f"[WARN] Пропущен тикер {symbol} — не список сделок.")
                continue

            for trade in trades:
                if not isinstance(trade, dict):
                    print(f"[WARN] Пропущена сделка по {symbol} — не словарь.")
                    continue
                append_to_training_data(trade)

    except Exception as e:
        print(f"[ERROR] Ошибка при обработке trade_log.json: {e}")

if __name__ == "__main__":
    main()