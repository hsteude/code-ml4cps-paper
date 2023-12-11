from kfp import dsl
from kfp.dsl import Input, Output, Dataset
from typing import Dict, List, Optional
import toml

# load config
with open("config.toml", "r") as f:
    config = toml.load(f)





 
    # @staticmethod
    # def _create_split(
    #     df: pd.DataFrame, ar_col: str
    # ) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     df_train = df[~df[ar_col]]
    #     df_val = df[df[ar_col]]
    #     return df_train, df_val

    # logger.debug(f"Rows before na removal: {len(df)}")
    # df = df.dropna(how="any", axis=0)
    # logger.debug(f"Rows befor na removal: {len(df)}")
    # df = self._assign_label_col_to_df(
    #     df=df,
    #     df_ar=df_ar,
    #     ar_start_col=ar_col_dct["ar_start_ts_col"],
    #     ar_end_col=ar_col_dct["ar_end_ts_col"],
    #     ar_col=ar_col_dct["ar_col"],
    # )
    # df_train, df_val = self._create_split(df, ar_col=ar_col_dct["ar_col"])
    # return df_train, df_val


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


@dsl.component(packages_to_install=["pyarrow", "pandas", "xlrd" ], base_image="python:3.9")
def get_label_series(
    df_telemetry_in: Input[Dataset],
    df_label_in: Input[Dataset],
    anomaly_start_col: str,
    anomaly_end_col: str,
    ar_col: str,
    labels_series: Output[Dataset]
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
