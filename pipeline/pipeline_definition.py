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
    scale_dataframes,
    visualize_split,
    run_katib_experiment,
    run_pytorch_training_job,
    run_evaluation,
    visualize_results,
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
def split_parquet_files_sub_pipeline(rowgroups_per_file: int):
    raw_files_paths = create_s3_client().ls(config["paths"]["telemetry_data_directory"])
    for file in raw_files_paths:
        split_parquet_file_task = split_parquet_file(
            file_path=file,
            target_path=config["paths"]["telemetry_data_directory_splitted"],
            timestamp_col=config["col-names"]["timestamp_col"],
            rowgroup_limit=100_000,
            rowgroups_per_file=rowgroups_per_file,
        )
        add_minio_env_vars_to_tasks([split_parquet_file_task])


# define pipeline
@dsl.pipeline
def columbus_eclss_ad_pipeline(
    rowgroups_per_file: int = 10,
    dask_preproc_sample_frac: float = 0.001,
    num_dask_workers: int = 4,
    split_window_hours: float = 250.0,
    train_split: float = 0.8,
    viz_sample_fraction: float = 0.1,
    katib_max_epochs: int = 100,
    katib_max_trials: int = 5,
    katib_batch_size_list: List = ["32", "64", "128", "256"],
    katib_beta_list: List = ["0.001", "0.0001", "0.0001", "0.00001", "0.000001"],
    katib_learning_rate_list: List = ["0.0005", "0.0001", "0.00005"],
    latent_dim: int = 18,
    pytorchjob_num_dl_workers: int = 12,
    pytorchjob_max_epochs: int = 1000,
    pytorchjob_early_stopping_patience: int = 30,
    pytorchjob_num_gpu_nodes: int = 3,
    eval_batch_size: int = 1024,
    eval_threshold_min: int = -200,
    eval_threshold_max: int = -100,
    eval_number_thresholds: int = 100,
):
    split_parquet_files_task = split_parquet_files_sub_pipeline(
        rowgroups_per_file=rowgroups_per_file
    )
    dask_preprocessing_task = run_dask_preprocessing(
        partitioned_telemetry_paths=config["paths"]["partitioned_telemetry_path"],
        sample_frac=dask_preproc_sample_frac,
        timestamp_col=config["col-names"]["timestamp_col"],
        minio_endpoint=config["platform"]["minio_endpoint"],
        dask_worker_image=f'{config["images"]["eclss-ad-image"]}:commit-58f7cd50',
        num_dask_workers=num_dask_workers,
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
        label_col=config["col-names"]["ar_col"],
        window_hours=split_window_hours,
        train_split=train_split,
    )

    fit_scaler_task = fit_scaler(
        df_in=split_data_task.outputs["df_train"],
        scaler_type="standard",
    )

    scale_data_task = scale_dataframes(
        train_df_in=split_data_task.outputs["df_train"],
        val_df_in=split_data_task.outputs["df_val"],
        test_df_in=split_data_task.outputs["df_test"],
        scaler_in=fit_scaler_task.outputs["fitted_scaler"],
        label_column=config["col-names"]["ar_col"],
    )

    # visualize_split_task = visualize_split(
    #     train_df_in=scale_data_task.outputs["train_df_scaled"],
    #     test_df_in=scale_data_task.outputs["test_df_scaled"],
    #     column_name="AFS2_Cab_Air_Massflow_MVD",
    #     sample_fraction=viz_sample_fraction,
    # )

    katib_task = run_katib_experiment(
        df_train=scale_data_task.outputs["train_df_scaled"],
        df_val=scale_data_task.outputs["val_df_scaled"],
        experiment_name="columbus-anomaly-detection-ml4cps",
        image=f'{config["images"]["eclss-ad-image"]}:commit-58f7cd50',
        namespace=config["platform"]["namespace"],
        max_epochs=katib_max_epochs,
        max_trials=katib_max_trials,
        batch_size_list=katib_batch_size_list,
        beta_list=katib_beta_list,
        learning_rate_list=katib_learning_rate_list,
        latent_dim=latent_dim,
    )

    train_model_task = run_pytorch_training_job(
        train_df_in=scale_data_task.outputs["train_df_scaled"],
        val_df_in=scale_data_task.outputs["val_df_scaled"],
        minio_model_bucket=config["paths"]["minio_model_bucket"],
        training_image=f'{config["images"]["eclss-ad-image"]}:commit-58f7cd50',
        namespace=config["platform"]["namespace"],
        tuning_param_dct=katib_task.output,
        num_dl_workers=pytorchjob_num_dl_workers,
        max_epochs=pytorchjob_max_epochs,
        early_stopping_patience=pytorchjob_early_stopping_patience,
        latent_dim=latent_dim,
        num_gpu_nodes=pytorchjob_num_gpu_nodes,
        seed=42,
    )

    evaluation_task = run_evaluation(
        model_path=train_model_task.output,
        val_df_in=scale_data_task.outputs["val_df_scaled"],
        test_df_in=scale_data_task.outputs["test_df_scaled"],
        label_col_name=config["col-names"]["ar_col"],
        device="cuda",
        batch_size=eval_batch_size,
        threshold_min=eval_threshold_min,
        threshold_max=eval_threshold_max,
        number_thresholds=eval_number_thresholds,
    )
    add_minio_env_vars_to_tasks([evaluation_task])

    visualize_results_task = visualize_results(
        result_df_in=evaluation_task.outputs["result_df"],
        metrics_json=evaluation_task.outputs["metrics_dict"],
        sample_fraction=viz_sample_fraction,
        label_col_name=config["col-names"]["ar_col"],
    )
