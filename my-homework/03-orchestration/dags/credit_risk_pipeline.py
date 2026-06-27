import os
import pickle

import pandas as pd
from airflow.decorators import dag, task
from datetime import datetime

DATA_PATH = "/opt/airflow/dags/data/survival_dataset.csv"
MLFLOW_TRACKING_URI = "http://172.18.0.1:5000"
MLFLOW_EXPERIMENT = "credit-risk-cox"

COVARIATES = [
    "dias_sin_pagar", "cuotas_atrasadas", "pago_promedio_diario",
    "dias_con_abono", "intensidad_recuperacion", "ratio_atraso",
    "saldo_relativo", "total_credito", "valor_cuota", "cuotas_pactadas",
]


@dag(
    dag_id="credit_risk_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@monthly",
    catchup=False,
    params={"penalizer": 0.01},
)
def credit_risk_pipeline():

    @task
    def load_and_preprocess() -> str:
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=COVARIATES + ["t_start", "t_stop", "evento"])
        df = df[df["t_stop"] > df["t_start"]]

        scaler = StandardScaler()
        df[COVARIATES] = scaler.fit_transform(df[COVARIATES])

        out_dir = "/opt/airflow/dags/data"
        df.to_parquet(f"{out_dir}/credit_risk_processed.parquet", index=False)
        with open(f"{out_dir}/credit_risk_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print(f"Dataset: {len(df):,} intervalos, {df['CÓDIGO CRÉDITO'].nunique():,} créditos")
        print(f"Eventos: {df['evento'].sum()} ({df['evento'].mean()*100:.2f}%)")
        return out_dir

    @task
    def train(data_dir: str, penalizer: float) -> float:
        import mlflow
        from lifelines import CoxTimeVaryingFitter
        from lifelines.utils import concordance_index

        df = pd.read_parquet(f"{data_dir}/credit_risk_processed.parquet")
        with open(f"{data_dir}/credit_risk_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            params = {"penalizer": float(penalizer), "l1_ratio": 0.0}
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

            last_obs = df.sort_values("t_stop").groupby("CÓDIGO CRÉDITO").last().reset_index()
            partial_hazard = ctv.predict_partial_hazard(last_obs)
            c_index = concordance_index(last_obs["t_stop"], -partial_hazard, last_obs["evento"])
            mlflow.log_metric("c_index", c_index)
            print(f"C-index: {c_index:.4f}")

            with open(f"{data_dir}/cox_model.pkl", "wb") as f:
                pickle.dump(ctv, f)
            mlflow.log_artifact(f"{data_dir}/cox_model.pkl", artifact_path="model")
            mlflow.log_artifact(f"{data_dir}/credit_risk_scaler.pkl", artifact_path="model")

        return c_index

    data_dir = load_and_preprocess()
    train(data_dir=data_dir, penalizer="{{ params.penalizer }}")


credit_risk_pipeline()
