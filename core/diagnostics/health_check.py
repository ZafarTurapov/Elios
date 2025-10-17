import json

# core/diagnostics/health_check.py
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
from openai import OpenAI

from core.connectors.alpaca_connector import get_account_equity
from core.utils.paths import TRADE_LOG_PATH
from core.utils.telegram import send_telegram_message

# === Пути ===
MODEL_PATH = "core/training/trained_model.pkl"
TRAINING_DATA_PATH = "core/trading/training_data.json"

SIGNALS_PATH = "core/trading/signals.json"
POSITIONS_PATH = "core/trading/open_positions.json"
TRAINING_METRICS_PATH = "logs/training_metrics.json"

# === API ключи ===
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)


# === Функции ===
def check_alpaca():
    try:
        equity = get_account_equity()
        return True, f"✅ Alpaca API: OK (${equity})"
    except Exception as e:
        return False, f"❌ Alpaca API: ошибка {str(e)}"


def check_openai():
    try:
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, "✅ OpenAI API: OK"
    except Exception as e:
        return False, f"❌ OpenAI API: {type(e).__name__}: {e}"


def check_model():
    try:
        joblib.load(MODEL_PATH)
        return "✅ Модель: загружена"
    except:
        return "❌ Модель: не загружена"


def check_training_data():
    try:
        with open(TRAINING_DATA_PATH) as f:
            data = json.load(f)
        return f"✅ Данные обучения: {len(data)} записей"
    except:
        return "❌ Ошибка чтения training_data.json"


def check_trades():
    try:
        if not Path(TRADE_LOG_PATH).exists():
            return "⚠️ Журнал сделок отсутствует"
        with open(TRADE_LOG_PATH) as f:
            log = json.load(f)
        today = datetime.now(timezone.utc).date()
        today_trades = sum(
            1
            for trades in log.values()
            for t in trades
            if "timestamp" in t
            and datetime.fromisoformat(t["timestamp"]).date() == today
        )
        if today_trades == 0:
            return "⚠️ Нет сделок за сегодня"
        return f"✅ Сделки за сегодня: {today_trades}"
    except Exception as e:
        return f"⚠️ Ошибка при проверке сделок: {e}"


def check_metrics():
    try:
        with open(TRAINING_METRICS_PATH) as f:
            metrics = json.load(f)
        return f"✅ Метрики: Accuracy {metrics['accuracy']:.4f}, F1 WIN {metrics['f1_win']:.4f}, F1 LOSS {metrics['f1_loss']:.4f}"
    except:
        return "⚠️ Метрики не найдены"


def check_modules():
    msgs = []
    if Path(SIGNALS_PATH).exists():
        with open(SIGNALS_PATH) as f:
            signals = json.load(f)
        msgs.append(f"✅ signal_engine.py — сигналов: {len(signals)}")
    else:
        msgs.append("⚠️ signal_engine.py — файл отсутствует")

    if Path(POSITIONS_PATH).exists():
        mtime = datetime.fromtimestamp(os.path.getmtime(POSITIONS_PATH))
        delta = datetime.now() - mtime
        if delta.total_seconds() < 3600:
            msgs.append("✅ positions_sync.py — позиции обновлены")
        else:
            msgs.append("⚠️ positions_sync.py — файл не обновлялся")

    if Path(TRAINING_DATA_PATH).exists():
        msgs.append("✅ training_data_builder.py — данные обновлены")

    if Path(MODEL_PATH).exists():
        msgs.append("✅ strategy_trainer.py — модель обновлена")

    return msgs


# === Основной блок ===
if __name__ == "__main__":
    print("📡 Elios Диагностика")
    status = []
    ok, msg = check_alpaca()
    print(msg)
    status.append(msg)

    ok2, msg2 = check_openai()
    print(msg2)
    status.append(msg2)

    print(msg_model := check_model())
    print(msg_data := check_training_data())
    print(msg_trades := check_trades())
    print(msg_metrics := check_metrics())

    status += [msg_model, msg_data, msg_trades, msg_metrics]

    print("\n🧩 Проверка модулей:")
    modules = check_modules()
    for m in modules:
        print(m)

    problems = [
        line for line in status if line.startswith("❌") or line.startswith("⚠️")
    ]
    if problems:
        print("\n❌ Обнаружены проблемы:")
        for p in problems:
            print(p)

    # Отправка в Telegram
    full_msg = "\n".join(
        ["📡 Elios Диагностика"] + status + ["", "🧩 Проверка модулей:"] + modules
    )
    if problems:
        full_msg += "\n\n❌ Обнаружены проблемы:\n" + "\n".join(problems)

    send_telegram_message(full_msg)
