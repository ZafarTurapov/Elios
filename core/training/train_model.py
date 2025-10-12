# /root/stockbot/core/training/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "/root/stockbot/data/merged_with_labels.csv"
MODEL_PATH = "/root/stockbot/core/training/trained_model.pkl"

print(f"📂 Загружаем {DATA_PATH} батчами...")

# Загружаем батчами и объединяем
chunks = pd.read_csv(DATA_PATH, chunksize=50000)
df = pd.concat(chunks, ignore_index=True)

print(f"📊 Всего строк: {len(df)}, колонок: {len(df.columns)}")

# Оставляем только строки с метками
df = df.dropna(subset=["label"])
print(f"✅ После фильтра меток: {len(df)} строк")

# Преобразуем метки в бинарный формат
df["label"] = df["label"].map({"WIN": 1, "LOSS": 0})

# Убираем нечисловые и ненужные колонки
exclude_cols = ["symbol", "year", "label"]
numeric_df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

# Оставляем только числовые
numeric_df = numeric_df.select_dtypes(include=["number"])
X = numeric_df.drop(columns=["label"], errors="ignore")
y = df["label"]

print(f"📈 Фич: {X.shape[1]}, Объектов: {X.shape[0]}")

# Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
print("🤖 Обучаем RandomForest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print("📊 Отчёт по классификации:")
print(classification_report(y_test, y_pred))

# Сохраняем модель
joblib.dump(model, MODEL_PATH)
print(f"💾 Модель сохранена: {MODEL_PATH}")
