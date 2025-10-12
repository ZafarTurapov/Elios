# -*- coding: utf-8 -*-
import os
import shutil
import zipfile
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
from core.training.clean_merged_data import clean_merged_data
from core.training.merge_with_labels import merge_with_labels
from core.training.strategy_trainer import train_strategy_model
from core.utils.telegram import send_telegram_message

DATA_DIR = "/root/stockbot/data"
BACKUP_DIR = f"/root/stockbot/backups/training/{datetime.date.today()}"
KAGGLE_DATASET = "kshitijsaini121/stock-market-prediction-for-july-2025-dataset"

def backup_training_files():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for file_name in ["trained_model.pkl", "fundamentals_with_labels.csv", "training_data.json"]:
        src = os.path.join("/root/stockbot/core/training", file_name) if file_name.endswith(".pkl") else os.path.join(DATA_DIR, file_name)
        dst = os.path.join(BACKUP_DIR, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    send_telegram_message(f"🗂 Бэкап сохранён: {BACKUP_DIR}")

def download_from_kaggle():
    send_telegram_message("📥 Загружаем свежие данные с Kaggle...")
    api = KaggleApi()
    api.authenticate()

    dataset_url = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
    send_telegram_message(f"📦 Датасет Kaggle: {dataset_url}")

    os.makedirs(DATA_DIR, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)

def main():
    backup_training_files()
    download_from_kaggle()
    clean_merged_data()
    merge_with_labels()
    train_strategy_model()
    send_telegram_message("✅ Полный цикл обновления модели завершён.")

if __name__ == "__main__":
    main()
