import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import yaml  # para leer params.yaml

PROCESSED_PATH = "data/processed/telco_churn_clean.csv"
MODEL_PATH = "models/model_baseline.pkl"
METRICS_PATH = "metrics.json"
PARAMS_PATH = "params.yaml"


def load_params(path=PARAMS_PATH):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def load_data(path=PROCESSED_PATH):
    print("Cargando dataset procesado...")
    df = pd.read_csv(path)
    print(f"Shape dataset limpio: {df.shape}")
    return df


def split_data(df, test_size, random_state):
    print("Preparando features (X) y target (y)...")

    # 1) Sacamos columna ID si existe
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # 2) Separar target
    y = df["churn"]
    X = df.drop(columns=["churn"])

    # 3) Quedarnos solo con columnas numéricas
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X = X[numeric_cols]

    print(f"Cantidad de columnas numéricas usadas: {len(numeric_cols)}")
    print(f"Columnas numéricas: {list(numeric_cols)}")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_model(X_train, y_train, max_iter, penalty, C):
    print("Escalando features numéricas...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Entrenando modelo Logistic Regression (baseline)...")
    model = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
        C=C
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    print("Evaluando modelo...")

    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n📊 Métricas baseline:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return {"accuracy": acc, "f1": f1}


def save_model(model, scaler, path=MODEL_PATH):
    print("Guardando modelo baseline...")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, path)

    print(f"Modelo guardado en: {path}")


def save_metrics(metrics: dict, path=METRICS_PATH):
    print("Guardando métricas en JSON...")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {path}")


def run_all():
    # 1) Cargar hiperparámetros desde params.yaml
    params = load_params()
    test_size = params["test_size"]
    random_state = params["random_state"]
    max_iter = params["max_iter"]
    penalty = params["penalty"]
    C = params["C"]

    # 2) Cargar datos
    df = load_data()

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # 4) Entrenar modelo
    model, scaler = train_model(
        X_train,
        y_train,
        max_iter=max_iter,
        penalty=penalty,
        C=C
    )

    # 5) Evaluar y guardar métricas
    metrics = evaluate(model, scaler, X_test, y_test)
    save_metrics(metrics)

    # 6) Guardar modelo
    save_model(model, scaler)

    print("✔ Entrenamiento baseline completado (con params.yaml y metrics.json).")


if __name__ == "__main__":
    run_all()
