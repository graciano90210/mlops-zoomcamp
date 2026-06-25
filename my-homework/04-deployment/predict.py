import os
import pickle
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "taxi-duration-pipeline"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.tracking.MlflowClient()
run_id = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).run_id

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

dv_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/dv.pkl")
with open(dv_path, "rb") as f:
    dv = pickle.load(f)

app = FastAPI()


class TripFeatures(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: TripFeatures):
    features = {
        "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
        "trip_distance": trip.trip_distance,
    }
    X = dv.transform([features])
    duration = model.predict(X)[0]
    return {"duration_minutes": round(float(duration), 2)}
