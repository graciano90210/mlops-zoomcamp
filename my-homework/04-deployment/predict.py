import pickle
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "taxi-duration-pipeline"
MODEL_ALIAS = "champion"
DV_PATH = "../03-orchestration/dags/data/dv.pkl"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

with open(DV_PATH, "rb") as f:
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
