from fastapi import FastAPI
from pydantic import BaseModel

from model import load_model, prepare_features, predict as model_predict

model, scaler, run_id = load_model()
app = FastAPI(title="Credit Risk API", version="1.0")


class CreditFeatures(BaseModel):
    dias_sin_pagar: float
    cuotas_atrasadas: float
    pago_promedio_diario: float
    dias_con_abono: float
    intensidad_recuperacion: float
    ratio_atraso: float
    saldo_relativo: float
    total_credito: float
    valor_cuota: float
    cuotas_pactadas: float


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": run_id}


@app.post("/predict")
def predict(credit: CreditFeatures):
    features = prepare_features(credit.model_dump())
    result = model_predict(model, scaler, features)
    return result
