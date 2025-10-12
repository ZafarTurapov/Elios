import json
import os
from datetime import datetime

TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "../trading/training_data.json")

def append_to_training_data(entry):
    try:
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
