import os
import pickle
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "taxi-duration-pipeline"
MODEL_ALIAS = "champion"


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run_id = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).run_id
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    dv_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/dv.pkl")
    with open(dv_path, "rb") as f:
        dv = pickle.load(f)
    return model, dv


def prepare_features(pu_location_id: str, do_location_id: str, trip_distance: float) -> dict:
    return {
        "PU_DO": f"{pu_location_id}_{do_location_id}",
        "trip_distance": trip_distance,
    }


def predict(model, dv, features: dict) -> float:
    X = dv.transform([features])
    duration = model.predict(X)[0]
    return round(float(duration), 2)
