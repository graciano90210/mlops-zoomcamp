from airflow.decorators import dag, task
from datetime import datetime
import os
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

DATA_DIR = "/opt/airflow/dags/data"
MLFLOW_TRACKING_URI = "http://172.18.0.1:5000"
MLFLOW_EXPERIMENT = "homework-03-orchestration"


@dag(
    dag_id="homework_dag",
    start_date=datetime(2023, 3, 1),
    schedule=None,
    catchup=False,
)
def homework_dag():

    @task
    def download_data() -> str:
        import requests
        os.makedirs(DATA_DIR, exist_ok=True)
        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
        filepath = f"{DATA_DIR}/yellow_tripdata_2023-03.parquet"
        if not os.path.exists(filepath):
            print(f"Downloading {url} ...")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
        df = pd.read_parquet(filepath)
        print(f"Q3 - Records loaded: {len(df)}")
        return filepath

    @task
    def prepare_data(filepath: str) -> str:
        df = pd.read_parquet(filepath)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        print(f"Q4 - Records after filtering: {len(df)}")

        output_path = f"{DATA_DIR}/yellow_2023_03_prepared.parquet"
        df.to_parquet(output_path, index=False)
        return output_path

    @task
    def train_model(data_path: str) -> float:
        import mlflow
        import mlflow.sklearn

        df = pd.read_parquet(data_path)

        categorical = ['PULocationID', 'DOLocationID']
        dicts = df[categorical].to_dict(orient='records')

        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        y = df['duration'].values

        lr = LinearRegression()
        lr.fit(X, y)

        print(f"Q5 - Model intercept: {lr.intercept_:.2f}")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("intercept", lr.intercept_)
            mlflow.sklearn.log_model(lr, artifact_path="model")
            print("Q6 - Check MLflow Artifacts > model > MLmodel for model_size_bytes")

        with open(f"{DATA_DIR}/dv_homework.pkl", "wb") as f:
            pickle.dump(dv, f)

        return float(lr.intercept_)

    filepath = download_data()
    data_path = prepare_data(filepath)
    train_model(data_path)


homework_dag()
