\# Proyecto Telco Churn – MLOps (ISTEA)



Este repositorio contiene el desarrollo de un pipeline de Machine Learning para predecir la deserción de clientes (churn) de una compañía de telecomunicaciones ficticia.

Las primeras etapas del proyecto se enfocan en buenas prácticas de MLOps: estructura del proyecto, versionado de datos, reproducibilidad y entrenamiento de un modelo baseline.



---



\## 1. Objetivo del proyecto



\- Dataset: `telco\\\_churn.csv` (~10.000 registros)

\- Target: `churn` (0 = no churn, 1 = churn)

\- Alcance de las etapas 1 a 3:

  - Setup del proyecto (Git + DVC + DagsHub)

  - Versionado del dataset crudo

  - Limpieza y generación del dataset procesado

  - Entrenamiento de un modelo baseline (Logistic Regression)



---



\## 2. Estructura del repositorio



istea-telco-churn-mlops/

├── data/

│ ├── raw/

│ │ └── telco\_churn.csv.dvc

│ └── processed/

│ └── telco\_churn\_clean.csv.dvc

├── models/

│ └── model\_baseline.pkl.dvc

├── src/

│ ├── data\_prep.py

│ └── train.py

├── .dvc/

├── .gitignore

├── .dvcignore

├── requirements.txt

└── README.md







⚠️ Los archivos `.csv` y `.pkl` reales \*\*NO están en Git\*\*.

Se descargan automáticamente desde DVC con:





dvc pull

3\. Instalación del entorno

Crear entorno





conda create -n telco\_churn\_env python=3.11 -y

conda activate telco\_churn\_env

Instalar dependencias





pip install -r requirements.txt

4\. Traer datos y modelo (DVC)

Antes de ejecutar cualquier script:







dvc pull

Esto recupera desde el remote de DagsHub:



data/raw/telco\_churn.csv



data/processed/telco\_churn\_clean.csv



models/model\_baseline.pkl



5\. Pipeline (Etapas 1 a 3)

⭐ ETAPA 1 — Setup

Inicialización de Git y DVC



Subida del dataset crudo con DVC



Integración con DagsHub



⭐ ETAPA 2 — Limpieza de datos

Script: src/data\_prep.py







python src/data\_prep.py

Este script:



Carga el dataset crudo



Limpia columnas, tipos y nulos



Convierte variables numéricas



Asegura que la variable churn sea 0/1



Genera data/processed/telco\_churn\_clean.csv (también versionado con DVC)



⭐ ETAPA 3 — Modelo baseline

Script: src/train.py







python src/train.py

Pipeline del modelo:



Carga dataset procesado



Elimina customer\_id



Usa solo features numéricas



Divide en train/test (80/20, stratify)



Escala features con StandardScaler



Entrena Logistic Regression (max\_iter=200)



Calcula Accuracy y F1



Guarda el modelo en: models/model\_baseline.pkl (versionado con DVC)



6\. Resultados del modelo baseline

Modelo: Logistic Regression

Features usadas: solo numéricas

Métricas obtenidas:



Accuracy: 0.XX



F1-score: 0.XX





7\. Cómo reproducir de cero





git clone https://github.com/nahuelbouzon/istea-telco-churn-mlops.git

cd istea-telco-churn-mlops



conda activate telco\_churn\_env

pip install -r requirements.txt



\# Descargar datos y modelos desde DVC/DagsHub

dvc pull



\# Regenerar dataset limpio

python src/data\_prep.py



\# Entrenar el modelo baseline

python src/train.py

8\. Repositorios

GitHub: https://github.com/nahuelbouzon/istea-telco-churn-mlops



DagsHub: https://dagshub.com/nahuel.bouzon/istea-telco-churn-mlops





En DagsHub se visualizan:



datos versionados

modelos

historial de experimentos (en etapas posteriores)

teting