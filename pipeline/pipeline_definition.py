from kfp import dsl
from kfp import kubernetes
from typing import List
import os
import toml
from pipeline.components import (
    split_parquet_file,
    run_dask_preprocessing,
    get_label,
    get_label_series,
    create_train_dev_test_split,
    fit_scaler,
    scale_dataframes,
    run_katib_experiment,
    run_pytorch_training_job,
    evaluate_model,
    visualize_results,
    extract_composite_f1,
    extract_scaler_path,
    deploy_to_kserve
)
from container_component_src.utils import create_s3_client

# load config
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../config.toml", "r") as f:
    config = toml.load(f)


def add_minio_env_vars_to_tasks(task_list: List[dsl.PipelineTask]) -> None:
    """Adds environment variables for MinIO to the tasks"""
    for task in task_list:
        kubernetes.use_secret_as_env(
            task,
            secret_name="s3creds",
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            },
        )
            # Add S3_ENDPOINT from environment variable
        s3_endpoint = os.environ.get("S3_ENDPOINT")
        if s3_endpoint:
            task.set_env_variable(name="S3_ENDPOINT", value=os.environ.get("S3_ENDPOINT"))


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
    threshold: float = 0.7,
):
    split_parquet_files_task = split_parquet_files_sub_pipeline(
        rowgroups_per_file=rowgroups_per_file
    )
    dask_preprocessing_task = run_dask_preprocessing(
        partitioned_telemetry_paths=config["paths"]["partitioned_telemetry_path"],
        sample_frac=dask_preproc_sample_frac,
        timestamp_col=config["col-names"]["timestamp_col"],
        minio_endpoint=config["platform"]["minio_endpoint"],
        dask_worker_image=config["images"]["dask-worker"],
        namespace=config["platform"]["namespace"],
        num_dask_workers=num_dask_workers,
    )
    dask_preprocessing_task.after(split_parquet_files_task)
    get_labels_task = get_label(labels_xls_path=config["paths"]["labels_xls_artifact_uri"])
    add_minio_env_vars_to_tasks([dask_preprocessing_task, get_labels_task])

    get_label_series_task = get_label_series(
        df_telemetry_in=dask_preprocessing_task.outputs["preprocessed_df"],
        df_label_in=get_labels_task.outputs['df_labels'],
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

    katib_task = run_katib_experiment(
        df_train=scale_data_task.outputs["train_df_scaled"],
        df_val=scale_data_task.outputs["val_df_scaled"],
        experiment_name=f"columbus-anomaly-detection-ml4cps",
        image=config["images"]["tuning"],
        namespace=config["platform"]["namespace"],
        max_epochs=katib_max_epochs,
        max_trials=katib_max_trials,
        batch_size_list=katib_batch_size_list,
        beta_list=katib_beta_list,
        learning_rate_list=katib_learning_rate_list,
        latent_dim=latent_dim,
        mlflow_uri=config["platform"]["mlflow_uri"],
        mlflow_experiment_name=config["platform"]["mlflow_experiment_name"],
        minio_endpoint_url=config["platform"]["minio_endpoint"],
    )
 
    train_model_task = run_pytorch_training_job(
        train_df_in=scale_data_task.outputs["train_df_scaled"],
        val_df_in=scale_data_task.outputs["val_df_scaled"],
        training_image=config["images"]["training"],
        namespace=config["platform"]["namespace"],
        tuning_param_dct=katib_task.output,
        num_dl_workers=pytorchjob_num_dl_workers,
        max_epochs=pytorchjob_max_epochs,
        early_stopping_patience=pytorchjob_early_stopping_patience,
        latent_dim=latent_dim,
        num_gpu_nodes=pytorchjob_num_gpu_nodes,
        seed=42,
        mlflow_uri=config["platform"]["mlflow_uri"],
        mlflow_experiment_name=config["platform"]["mlflow_experiment_name"],
        minio_endpoint_url=config["platform"]["minio_endpoint"],
    )

    # Modell-Evaluierungsschritt
    evaluation_task = evaluate_model(
        val_df=scale_data_task.outputs["val_df_scaled"],
        test_df=scale_data_task.outputs["test_df_scaled"],
        train_out_dict=train_model_task.output,
        mlflow_uri=config["platform"]["mlflow_uri"],
        mlflow_experiment_name=config["platform"]["mlflow_experiment_name"],
        minio_endpoint_url=config["platform"]["minio_endpoint"],
        label_col_name="Anomaly",
        device="cpu",
        batch_size=eval_batch_size,
        threshold_min=eval_threshold_min,
        threshold_max=eval_threshold_max,
        number_thresholds=eval_number_thresholds,
    )
    add_minio_env_vars_to_tasks([evaluation_task])

    visualize_results_task = visualize_results(
        result_df_in=evaluation_task.outputs["results_df"],
        metrics_dict=evaluation_task.outputs["Output"],
        sample_fraction=viz_sample_fraction,
        label_col_name=config["col-names"]["ar_col"],
        scatter_y_min=-4000,
        scatter_y_max=500
    )

    composite_f1_task = extract_composite_f1(metrics_dict=evaluation_task.outputs['Output'])
    scaler_path_task = extract_scaler_path(scaler=fit_scaler_task.output)
    with dsl.If(composite_f1_task.output > threshold):
        # Deployment-Schritt
        deploy_task = deploy_to_kserve(
            train_out_dict=train_model_task.output,
            mlflow_uri=config["platform"]["mlflow_uri"],
            mlflow_experiment_name=config["platform"]["mlflow_experiment_name"],
            minio_endpoint_url=config["platform"]["minio_endpoint"],
            namespace=config["platform"]["namespace"],
            service_name="eclss-point-vae",
            device="gpu",
            gpu_count=1,
            runtime_version="22.12-py3",
            mlflow_bucket=config["platform"]["mlflow_bucket"],
        )
