import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

CLUSTER_COLUMNS = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Loaded {len(df)} rows from file.csv")
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string


def profile_training_data(data_b64: str):
    """
    Checks the training data before preprocessing and returns a small profile.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    missing_columns = [column for column in CLUSTER_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": {
            column: int(df[column].isna().sum())
            for column in CLUSTER_COLUMNS
        },
        "means": {
            column: round(float(df[column].mean()), 2)
            for column in CLUSTER_COLUMNS
        },
    }

    print(f"Training rows: {profile['rows']}")
    print(f"Training columns: {profile['columns']}")
    print(f"Missing values in model columns: {profile['missing_values']}")
    print(f"Mean values for model columns: {profile['means']}")
    return profile


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[CLUSTER_COLUMNS]
    print(f"Prepared {len(clustering_data)} rows with columns: {', '.join(CLUSTER_COLUMNS)}")

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Saved KMeans model to {output_path}")

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns predictions for all rows in test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))[CLUSTER_COLUMNS]
    predictions = loaded_model.predict(df)
    prediction_list = [int(pred) for pred in predictions]
    print(f"Predictions for test.csv rows: {prediction_list}")

    return {
        "test_rows": int(len(df)),
        "predictions": prediction_list,
    }
