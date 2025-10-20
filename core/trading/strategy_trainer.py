# core/trading/strategy_trainer.py

import json
import os

import numpy as np

from core.utils.telegram import send_telegram_message  # ✅ Telegram

# === Пути ===
TRAINING_DATA_PATH = "/root/stockbot/core/trading/training_data.json"
MODEL_PATH = "/root/stockbot/core/trading/strategy_model.json"


def load_training_data():
    if not os.path.exists(TRAINING_DATA_PATH):
        msg = f"❌ Файл не найден: {TRAINING_DATA_PATH}"
        print(msg)
        send_telegram_message("📚 Нет обучающих данных для тренировки.")
        return []
    try:
        with open(TRAINING_DATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки данных: {e}")
        send_telegram_message(f"⛔️ Ошибка чтения обучающих данных: {e}")
        return []


def train_strategy(data):
    win_alpha, win_rsi, win_ema, win_vol = [], [], [], []

    for d in data:
        if d.get("label") != "WIN":
            continue
        if all(
            k in d and d[k] is not None
            for k in ["alpha_score", "rsi", "ema_dev", "vol_ratio"]
        ):
            win_alpha.append(d["alpha_score"])
            win_rsi.append(d["rsi"])
            win_ema.append(d["ema_dev"])
            win_vol.append(d["vol_ratio"])

    if not win_alpha:
        msg = "❌ Недостаточно данных WIN для обучения."
        print(msg)
        send_telegram_message("📚 Недостаточно успешных сделок для обучения.")
        return None

    model = {
        "min_alpha_score": round(float(np.median(win_alpha)), 3),
        "min_rsi": round(float(np.percentile(win_rsi, 25)), 2),
        "min_vol_ratio": round(float(np.percentile(win_vol, 50)), 2),
        "max_ema_dev": round(float(np.percentile(win_ema, 75)), 2),
    }
    return model


def save_model(model):
    try:
        with open(MODEL_PATH, "w") as f:
            json.dump(model, f, indent=2)
        print(f"✅ Модель сохранена: {MODEL_PATH}")
        print(json.dumps(model, indent=2))

        msg = (
            f"🤖 Обучение завершено\n"
            f"🔧 Новые параметры стратегии:\n"
            f"• Alpha ≥ {model['min_alpha_score']}\n"
            f"• RSI ≥ {model['min_rsi']}\n"
            f"• Vol ≥ {model['min_vol_ratio']}\n"
            f"• EMA ≤ {model['max_ema_dev']}"
        )
        send_telegram_message(msg)
    except Exception as e:
        err = f"⛔️ Ошибка при сохранении модели: {e}"
        print(err)
        send_telegram_message(err)


def main():
    try:
        data = load_training_data()
        if not data:
            return
        model = train_strategy(data)
        if model:
            save_model(model)
    except Exception as e:
        err = f"⛔️ Ошибка во время обучения: {e}"
        print(err)
        send_telegram_message(err)


if __name__ == "__main__":
    main()
