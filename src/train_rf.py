import os
import json
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


PROCESSED_PATH = "data/processed/telco_churn_clean.csv"
MODEL_PATH = "models/model_rf.pkl"
METRICS_PATH = "metrics_rf.json"
PARAMS_PATH = "params.yaml"


def load_params(path=PARAMS_PATH):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params["rf"]      # <<< sección nueva del yaml


def load_data(path=PROCESSED_PATH):
    print("Cargando datos procesados...")
    return pd.read_csv(path)


def prepare_data(df, test_size, random_state):
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    y = df["churn"]
    X = df.drop(columns=["churn"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X = X[numeric_cols]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train, params):
    print("Entrenando RandomForest...")

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }

    print("\n📊 Métricas RandomForest")
    print(metrics)
    return metrics


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def save_metrics(metrics, path=METRICS_PATH):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {path}")


def run():
    params = load_params()
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df, params["test_size"], params["random_state"])

    model = train_model(X_train, y_train, params)
    metrics = evaluate(model, X_test, y_test)

    save_model(model)
    save_metrics(metrics)

    print("✔ Entrenamiento RandomForest completado.")


if __name__ == "__main__":
    run()
