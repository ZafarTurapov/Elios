# -*- coding: utf-8 -*-
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "data/fundamentals_with_labels.csv"
MODEL_PATH = "core/training/trained_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "label_profit_share" in df.columns and "label" not in df.columns:
        df["label"] = (df["label_profit_share"] > 0).astype(int)

    # Удалим лишние колонки
    drop_cols = [
        "symbol",
        "year",
        "Class",
        "Sector",
        "label_profit_share",
        "Unnamed: 0",
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Удалим строки без label
    df = df.dropna(subset=["label"])

    # Переводим label в 0/1
    df["label"] = (df["label"] > 0).astype(int)

    # Удалим все строки с нечисловыми значениями
    df = df.select_dtypes(include=[np.number]).dropna()

    y = df["label"]
    X = df.drop(columns=["label"])
    return X, y, X.columns.tolist()


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, acc, report


def main():
    X, y, features = load_data()
    model, acc, report = train_model(X, y)

    print("🎯 Accuracy:", round(acc, 4))
    print(report)

    joblib.dump((model, features), MODEL_PATH)
    print(f"✅ Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
