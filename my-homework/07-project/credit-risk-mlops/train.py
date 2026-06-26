import os
import pandas as pd
import mlflow
import mlflow.sklearn
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = "credit-risk-cox"
DATA_PATH = "data/survival_dataset.csv"

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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=COVARIATES + ["t_start", "t_stop", "evento"])
    df = df[df["t_stop"] > df["t_start"]]

    scaler = StandardScaler()
    df[COVARIATES] = scaler.fit_transform(df[COVARIATES])

    return df, scaler


def train(df: pd.DataFrame, scaler):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        params = {"penalizer": 0.01, "l1_ratio": 0.0}
        mlflow.log_params(params)

        ctv = CoxTimeVaryingFitter(**params)
        ctv.fit(
            df,
            id_col="CÓDIGO CRÉDITO",
            start_col="t_start",
            stop_col="t_stop",
            event_col="evento",
            formula=" + ".join(COVARIATES),
        )

        # C-index manual — CoxTimeVaryingFitter no lo calcula automáticamente
        last_obs = df.sort_values("t_stop").groupby("CÓDIGO CRÉDITO").last().reset_index()
        partial_hazard = ctv.predict_partial_hazard(last_obs)
        c_index_val = concordance_index(
            last_obs["t_stop"],
            -partial_hazard,
            last_obs["evento"]
        )
        mlflow.log_metric("c_index", c_index_val)
        print(f"C-index: {c_index_val:.4f}")

        ctv.print_summary()

        import pickle, tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "cox_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(ctv, f)
            mlflow.log_artifact(model_path, artifact_path="model")
            scaler_path = os.path.join(tmp, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, artifact_path="model")
        print("Modelo guardado en MLflow.")

    return ctv


if __name__ == "__main__":
    print("Cargando datos...")
    df, scaler = load_data(DATA_PATH)
    print(f"Dataset: {len(df):,} intervalos, {df['CÓDIGO CRÉDITO'].nunique():,} créditos")
    print(f"Eventos: {df['evento'].sum()} ({df['evento'].mean()*100:.2f}%)")
    print("Entrenando modelo Cox...")
    train(df, scaler)
