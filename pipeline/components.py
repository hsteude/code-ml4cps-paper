from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact
from typing import Dict, List, Optional
import toml

# load config
with open("config.toml", "r") as f:
    config = toml.load(f)


@dsl.container_component
def split_parquet_file(
    file_path: str,
    target_path: str,
    timestamp_col: str,
    rowgroup_limit: Optional[int] = None,
    rowgroups_per_file: int = 10,
):
    """Kubeflow pipeline component to split a large Parquet file into smaller files"""
    return dsl.ContainerSpec(
        image=f'{config["images"]["eclss-ad-image"]}:commit-c2f04ede',
        command=["python", "container_component_src/main.py"],
        args=[
            "split_parquet_file",
            "--file-path",
            file_path,
            "--target-path",
            target_path,
            "--timestamp-col",
            timestamp_col,
            "--rowgroup-limit",
            rowgroup_limit,
            "--rowgroups-per-file",
            rowgroups_per_file,
        ],
    )


@dsl.container_component
def run_dask_preprocessing(
    partitioned_telemetry_paths: str,
    sample_frac: float,
    preprocessed_df: Output[Dataset],
    timestamp_col: str,
    minio_endpoint: str,
    dask_worker_image: str,
    num_dask_workers: int,
):
    """Kubeflow pipeline component for Dask preprocessing"""
    return dsl.ContainerSpec(
        image=f'{config["images"]["eclss-ad-image"]}:commit-0a2239b9',
        command=["python", "container_component_src/main.py"],
        args=[
            "run_dask_preprocessing",
            "--partitioned-telemetry-paths",
            partitioned_telemetry_paths,
            "--sample-frac",
            str(sample_frac),
            "--df-out-path",
            preprocessed_df.path,
            "--timestamp-col",
            timestamp_col,
            "--minio-endpoint",
            minio_endpoint,
            "--dask-worker-image",
            dask_worker_image,
            "--num-dask-workers",
            str(num_dask_workers),
        ],
    )


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "xlrd"], base_image="python:3.9"
)
def get_label_series(
    df_telemetry_in: Input[Dataset],
    df_label_in: Input[Dataset],
    anomaly_start_col: str,
    anomaly_end_col: str,
    ar_col: str,
    labels_series: Output[Dataset],
) -> None:
    """
    Assigns an anomaly label to each row in the telemetry dataframe based on anomaly periods.

    Args:
        df_telemetry: The telemetry dataframe with timestamps as the index.
        df_label: The dataframe containing anomaly periods.
        anomaly_start_col: The column name in df_label indicating the start of an anomaly.
        anomaly_end_col: The column name in df_label indicating the end of an anomaly.

    Returns:
        Updated telemetry dataframe with an 'anomaly_label' column.
    """
    import pandas as pd

    # Read dataset
    df_telemetry = pd.read_parquet(df_telemetry_in.path)
    df_label = pd.read_excel(df_label_in.path)

    # Convert anomaly start and end times to datetime
    df_label[anomaly_start_col] = pd.to_datetime(
        df_label[anomaly_start_col], format="%Y/%j/%H:%M:%S", utc=True
    )
    df_label[anomaly_end_col] = pd.to_datetime(
        df_label[anomaly_end_col], format="%Y/%j/%H:%M:%S", utc=True
    )

    # Function to check if a timestamp is within any interval
    def is_in_anomaly_interval(ts: pd.Timestamp, intervals: list) -> bool:
        return any(start <= ts < end for start, end in intervals)

    # Create a list of anomaly intervals
    anomaly_intervals = list(
        zip(df_label[anomaly_start_col], df_label[anomaly_end_col])
    )

    # Assign anomaly labels
    df_telemetry[ar_col] = df_telemetry.index.to_series().apply(
        lambda ts: 1 if is_in_anomaly_interval(ts, anomaly_intervals) else 0
    )

    # Save labels series as df
    df_telemetry[[ar_col]].to_parquet(labels_series.path)


@dsl.component(packages_to_install=["pyarrow", "pandas"], base_image="python:3.9")
def create_train_dev_test_split(
    preproc_df_in: Input[Dataset],
    anomaly_df_in: Input[Dataset],
    window_hours: float,
    label_col: str,
    df_train: Output[Dataset],
    df_val: Output[Dataset],
    df_test: Output[Dataset],
    train_split: float = 0.8,
) -> None:
    """
    Splits the data into train, dev, and test sets. Test set will contain only the data points within the specified
    time window around each anomaly and a column indicating actual anomalies.

    Args:
        preproc_df: KFP Dataset for preprocessed multivariate time series data.
        anomaly_df: KFP Dataset DataFrame with anomaly labels.
        label_col: Column name for the anomaly labels
        window_hours: Size of the time window around each anomaly in hours.
        train_split: Proportion of non-test data to use for training.

    Returns:
        tuple: Three DataFrames corresponding to the train, dev, and test sets.
    """
    import pandas as pd
    from datetime import timedelta

    # read in data
    anomaly_df = pd.read_parquet(anomaly_df_in.path)
    preproc_df = pd.read_parquet(preproc_df_in.path)

    window_size = timedelta(hours=window_hours)

    # Identifying the timestamps of anomalies
    anomaly_timestamps = anomaly_df.index[anomaly_df["Anomaly"] == 1]

    # Creating a mask for the test set
    test_mask = pd.Series(False, index=preproc_df.index)
    for timestamp in anomaly_timestamps:
        start_window = timestamp - window_size
        end_window = timestamp + window_size
        test_mask.loc[start_window:end_window] = True

    # Splitting the dataframes and adding anomaly column to test set
    test_df = preproc_df[test_mask].copy()
    test_df[label_col] = anomaly_df[label_col][test_mask]

    non_test_df = preproc_df[~test_mask]

    # Randomly splitting the non-test data into train and dev sets
    train_df = non_test_df.sample(frac=train_split, random_state=42)
    val_df = non_test_df.drop(train_df.index)

    # write out
    train_df.to_parquet(df_train.path)
    val_df.to_parquet(df_val.path)
    test_df.to_parquet(df_test.path)


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "scikit-learn==1.3.2"],
    base_image="python:3.9",
)
def fit_scaler(
    df_in: Input[Dataset],
    scaler_type: str,
    fitted_scaler: Output[Artifact],
) -> None:
    """
    Scales the columns of a DataFrame using Min-Max scaling or Standard scaling.
    Fits the scaler on a DataFrame using either Min-Max or Standard scaling.

    Args:
        df: The KFP Dataset Input for the dataframe the scaler should be fitted on.
        scaler_type: Type of scaling to apply ('minmax' or 'standard').
        scaler_object_path: KFP Output Artifact for the fitted scaler object.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import joblib

    # read df
    df = pd.read_parquet(df_in.path)

    # Choose the appropriate scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Choose 'minmax' or 'standard'.")

    # Fit and transform the DataFrame
    scaler = scaler.fit(df)

    # Save the scaled DataFrame and scaler object
    joblib.dump(scaler, fitted_scaler.path)


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "scikit-learn==1.3.2"],
    base_image="python:3.9",
)
def scale_dataframes(
    train_df_in: Input[Dataset],
    val_df_in: Input[Dataset],
    test_df_in: Input[Dataset],
    scaler_in: Input[Artifact],
    label_column: str,
    train_df_scaled: Output[Dataset],
    val_df_scaled: Output[Dataset],
    test_df_scaled: Output[Dataset],
) -> None:
    """
    Scales the given train, validation, and test dataframes using the provided scaler.
    Assumes that the test dataframe contains an additional column for labels.

    Args:
    train_df: KFP Dataset for DataFrame containing training data.
    val_df: KFP Dataset for DataFrame containing validation data.
    test_df: KFP Dataset for DataFrame containing test data, with an additional column for labels.
    scaler: KFP Artifact for Scikit-Learn Scaler.
    label_column: The name of the label column in the test dataframe.

    Returns:
    None: Scaled dataframes are written out as parquet files.
    """
    import pandas as pd
    import joblib

    # Read DataFrames and load the scaler
    train_df, val_df, test_df = [
        pd.read_parquet(ds.path) for ds in [train_df_in, val_df_in, test_df_in]
    ]
    scaler = joblib.load(scaler_in.path)

    # Scale the training data
    train_df_sc = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    train_df_sc.to_parquet(train_df_scaled.path)

    # Scale the validation data
    val_df_sc = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
    val_df_sc.to_parquet(val_df_scaled.path)

    # Scale the test data while preserving the label column
    test_labels = test_df.pop(label_column)
    test_df_sc = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    test_df_sc[label_column] = test_labels
    test_df_sc.to_parquet(test_df_scaled.path)
