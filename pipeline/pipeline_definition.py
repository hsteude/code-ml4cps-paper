from kfp import dsl
from typing import List
import os
import toml
from pipeline.components import (
    split_parquet_file,
    run_dask_preprocessing,
    get_label_series,
    create_train_dev_test_split,
    fit_scaler,
)
from container_component_src.utils import create_s3_client

# load config
with open("config.toml", "r") as f:
    config = toml.load(f)


def add_minio_env_vars_to_tasks(task_list: List[dsl.PipelineTask]) -> None:
    """Adds environment variables for minio to the tasks"""
    for task in task_list:
        task.set_env_variable(
            "AWS_SECRET_ACCESS_KEY", os.environ["AWS_SECRET_ACCESS_KEY"]
        ).set_env_variable(
            "AWS_ACCESS_KEY_ID", os.environ["AWS_ACCESS_KEY_ID"]
        ).set_env_variable(
            "S3_ENDPOINT", "minio.minio"
        )


# define pipeline
@dsl.pipeline
def split_parquet_files_sub_pipeline():
    raw_files_paths = create_s3_client().ls(config["paths"]["telemetry_data_directory"])
    for file in raw_files_paths:
        split_parquet_file_task = split_parquet_file(
            file_path=file,
            target_path="eclss-data/telemetry/partitioned-telemetry",
            timestamp_col="normalizedTime",
            rowgroup_limit=100_000,
            rowgroups_per_file=10,
        )
        add_minio_env_vars_to_tasks([split_parquet_file_task])


# define pipeline
@dsl.pipeline
def columbus_eclss_ad_pipeline():
    split_parquet_files_task = split_parquet_files_sub_pipeline()
    dask_preprocessing_task = run_dask_preprocessing(
        partitioned_telemetry_paths=config["paths"]["partitioned_telemetry_path"],
        sample_frac=0.001,
        timestamp_col="normalizedTime",
        minio_endpoint="http://minio.minio",
        dask_worker_image=f'{config["images"]["eclss-ad-image"]}:commit-0a2239b9',
        num_dask_workers=4,
    )
    dask_preprocessing_task.after(split_parquet_files_task)
    add_minio_env_vars_to_tasks([dask_preprocessing_task])

    import_label_xls_task = dsl.importer(
        artifact_uri=config["paths"]["labels_xls_artifact_uri"],
        artifact_class=dsl.Dataset,
    )

    get_label_series_task = get_label_series(
        df_telemetry_in=dask_preprocessing_task.outputs["preprocessed_df"],
        df_label_in=import_label_xls_task.output,
        ar_col=config["col-names"]["ar_col"],
        anomaly_start_col=config["col-names"]["ar_start_ts_col"],
        anomaly_end_col=config["col-names"]["ar_end_ts_col"],
    )

    split_data_task = create_train_dev_test_split(
        preproc_df_in=dask_preprocessing_task.outputs["preprocessed_df"],
        anomaly_df_in=get_label_series_task.outputs["labels_series"],
        window_hours=250.0,
        train_split=0.8,
    )

    fit_scaler_task = fit_scaler(
        df_in=split_data_task.outputs["df_val"],
        scaler_type="standard",
    )
