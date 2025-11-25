import joblib
import pandas as pd
import numpy as np
import datetime
import os

MODEL_BASELINE_PATH = "models/model_baseline.pkl"
MODEL_RF_PATH = "models/model_rf.pkl"


def load_model(model_path):
    """Carga un modelo entrenado (baseline o RF)."""
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        # baseline (modelo + scaler)
        return bundle["model"], bundle["scaler"]
    else:
        # random forest (solo modelo)
        return bundle, None


def preprocess_input(df, scaler=None):
    """
    Prepara las features de entrada para el modelo.
    - Si hay scaler, usa exactamente las columnas con las que fue entrenado.
    - Si no hay scaler, se queda con columnas numéricas.
    """
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Las columnas esperadas por el scaler no están en el input: {missing}"
            )
        # Usar exactamente las columnas en el mismo orden
        X = df[expected_cols]
        X_scaled = scaler.transform(X)
        return X_scaled, expected_cols
    else:
        # Caso RF: solo numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]
        return X, numeric_cols

def log_prediction_run(model_path, input_path, n_samples):
    """Registra la ejecución de predicciones para monitoreo."""
    log_path = "monitoring/prediction_log.csv"
    os.makedirs("monitoring", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = f"{timestamp},{os.path.basename(model_path)},{input_path},{n_samples}\n"

    # Si el archivo no existe, escribo encabezado
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,model,input_file,n_samples\n")

    # Agrego la fila
    with open(log_path, "a") as f:
        f.write(row)

    print(f"Log actualizado: {log_path}")


def predict(model_path, input_path):
    print(f"Cargando modelo desde: {model_path}")
    model, scaler = load_model(model_path)

    print(f"Leyendo archivo de entrada: {input_path}")
    df = pd.read_csv(input_path)

    X_processed, feature_cols = preprocess_input(df, scaler)

    print("Generando predicciones...")
    preds = model.predict(X_processed)

    df["prediction"] = preds
    output_path = "predicciones.csv"
    df.to_csv(output_path, index=False)

    print(f"Predicciones guardadas en: {output_path}")
    print(f"Features usadas: {feature_cols}")
    log_prediction_run(model_path, input_path, len(df))
    return output_path



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    predict(args.model, args.input)
