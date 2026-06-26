import os
import pickle

import mlflow
import pandas as pd

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "credit-risk-cox"

COVARIATES = [
    "dias_sin_pagar",
    "cuotas_atrasadas",
    "pago_promedio_diario",
    "dias_con_abono",
    "intensidad_recuperacion",
    "ratio_atraso",
    "saldo_relativo",
    "total_credito",
    "valor_cuota",
    "cuotas_pactadas",
]


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.c_index DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id

    model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/cox_model.pkl")
    scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, run_id


def prepare_features(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{col: data[col] for col in COVARIATES}])


def predict(model, scaler, features: pd.DataFrame) -> dict:
    features_scaled = features.copy()
    features_scaled[COVARIATES] = scaler.transform(features_scaled[COVARIATES])

    hazard = float(model.predict_partial_hazard(features_scaled).iloc[0])

    if hazard < 0.85:
        segment = "bajo"
    elif hazard < 1.2:
        segment = "medio"
    else:
        segment = "alto"

    return {
        "partial_hazard": round(hazard, 4),
        "risk_segment": segment,
    }
