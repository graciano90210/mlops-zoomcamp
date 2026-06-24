from airflow.decorators import dag, task
from datetime import datetime
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

DATA_DIR = "/opt/airflow/dags/data"
TAXI_COLOR = "green"


def _download_file(year: int, month: int) -> str:
    import requests
    os.makedirs(DATA_DIR, exist_ok=True)
    url = (
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"{TAXI_COLOR}_tripdata_{year}-{month:02d}.parquet"
    )
    filepath = f"{DATA_DIR}/{TAXI_COLOR}_tripdata_{year}-{month:02d}.parquet"
    if not os.path.exists(filepath):
        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
    print(f"File ready: {filepath}")
    return filepath


def _read_dataframe(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


@dag(
    dag_id="taxi_duration_pipeline",
    start_date=datetime(2023, 3, 1),
    schedule="@monthly",
    catchup=False,
    params={"year": 2023, "train_month": 1, "val_month": 2},
)
def taxi_duration_pipeline():

    @task
    def download_train_data(year: str, month: str) -> str:
        return _download_file(int(year), int(month))

    @task
    def download_val_data(year: str, month: str) -> str:
        return _download_file(int(year), int(month))

    @task
    def preprocess(train_path: str, val_path: str) -> str:
        df_train = _read_dataframe(train_path)
        df_val = _read_dataframe(val_path)

        features = ["PU_DO", "trip_distance"]
        dv = DictVectorizer()

        X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
        y_train = df_train["duration"].values

        X_val = dv.transform(df_val[features].to_dict(orient="records"))
        y_val = df_val["duration"].values

        with open(f"{DATA_DIR}/train.pkl", "wb") as f:
            pickle.dump((X_train, y_train), f)
        with open(f"{DATA_DIR}/val.pkl", "wb") as f:
            pickle.dump((X_val, y_val), f)
        with open(f"{DATA_DIR}/dv.pkl", "wb") as f:
            pickle.dump(dv, f)

        return DATA_DIR

    @task
    def train(data_dir: str) -> float:
        with open(f"{data_dir}/train.pkl", "rb") as f:
            X_train, y_train = pickle.load(f)
        with open(f"{data_dir}/val.pkl", "rb") as f:
            X_val, y_val = pickle.load(f)

        rf = RandomForestRegressor(max_depth=10, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)

        rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        print(f"Validation RMSE: {rmse:.4f}")

        with open(f"{data_dir}/model.pkl", "wb") as f:
            pickle.dump(rf, f)

        return rmse

    train_path = download_train_data(year="{{ params.year }}", month="{{ params.train_month }}")
    val_path = download_val_data(year="{{ params.year }}", month="{{ params.val_month }}")
    data_dir = preprocess(train_path=train_path, val_path=val_path)
    train(data_dir=data_dir)


taxi_duration_pipeline()
