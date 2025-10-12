import os
import json
import numpy as np

MODEL_PATH = "/core/trading/strategy_model.json"

def load_strategy_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "r") as f:
            model = json.load(f)
        if "weights" not in model or "bias" not in model:
            print("[WARN] Стратегическая модель повреждена.")
            return None
        return model
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель стратегии: {e}")
        return None

def apply_model(model, percent_change, volume_ratio, rsi, ema_dev):
    try:
        features = np.array([percent_change, volume_ratio, rsi, ema_dev])
        weights = np.array(model["weights"])
        bias = model["bias"]
        score = np.dot(features, weights) + bias
        return 1 / (1 + np.exp(-score))  # сигмоида
    except Exception as e:
        print(f"[ERROR] Ошибка при применении модели: {e}")
        return 0.0
