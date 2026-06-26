from fastapi import FastAPI
from pydantic import BaseModel

from model import load_model, prepare_features, predict as model_predict

model, dv = load_model()

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
    features = prepare_features(trip.PULocationID, trip.DOLocationID, trip.trip_distance)
    duration = model_predict(model, dv, features)
    return {"duration_minutes": duration}
