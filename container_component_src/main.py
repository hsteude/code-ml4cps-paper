import click
import json
import os
from typing import Optional, List
import pytorch_lightning as pl
import re
import pyarrow.parquet as pq
from loguru import logger
from container_component_src.utils import (
    create_s3_client,
    read_data_from_minio,
    upload_file_to_minio_bucket,
)
from container_component_src.dask_preprocessor.preprocessor import DaskPreprocessor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins.environments import KubeflowEnvironment
import toml
import pandas as pd
import numpy as np
from pytorch_lightning import seed_everything
import torch
from container_component_src.model.datamodule import TimeStampDataModule
from container_component_src.model.lightning_module import TimeStampVAE
from container_component_src.model.callbacks import (
    StdOutLoggerCallback,
    ResetLogVarCallback,
)
from container_component_src.eval.evaluator import ModelEvaluator
import mlflow
import tempfile
import shutil

# load config
with open("config.toml", "r") as f:
    config = toml.load(f)


@click.group()
def cli():
    pass


@cli.command("split_parquet_file")
@click.option("--file-path", required=True, type=str)
@click.option("--target-path", required=True, type=str)
@click.option("--timestamp-col", default="timestamp", type=str)
@click.option("--rowgroup-limit", default=None, type=int)
@click.option(
    "--rowgroups-per-file",
    default=10,
    type=int,
    help="Number of row groups to combine in each output file.",
)
def split_parquet_file(
    file_path: str,
    target_path: str,
    timestamp_col: str,
    rowgroup_limit: Optional[int],
    rowgroups_per_file: int,
) -> None:
    """
    Splits a large Parquet file into multiple smaller files, each containing a specified number of row groups.

    Args:
        file_path: The path to the large Parquet file.
        target_path: The bucket or path where the new files will be stored.
        timestamp_col: Name of the timestamp column.
        rowgroup_limit: The maximum number of row groups to process. If None, all row groups will be processed.
        rowgroups_per_file: The number of row groups to combine in each output file.
    """

    year_match = re.search(r"\d{4}", file_path)
    if not year_match:
        logger.error("Year not found in the file path.")
        return
    year = year_match.group()

    bucket_name = f"{target_path}/{year}"
    s3 = create_s3_client()
    try:
        if not s3.exists(bucket_name):
            s3.mkdir(bucket_name)
    except Exception as e:
        logger.error(f"Error accessing or creating bucket {bucket_name}: {e}")
        return

    try:
        file = s3.open(file_path)
        parquet_file = pq.ParquetFile(file)
        total_row_groups = parquet_file.num_row_groups
        if rowgroup_limit is not None:
            total_row_groups = min(total_row_groups, rowgroup_limit)

        for start_index in range(0, total_row_groups, rowgroups_per_file):
            end_index = min(start_index + rowgroups_per_file, total_row_groups)
            df = pd.concat(
                [
                    parquet_file.read_row_group(i).to_pandas()
                    for i in range(start_index, end_index)
                ]
            )
            df.set_index(timestamp_col, inplace=True)
            output_file_path = (
                f"{bucket_name}/row_groups_{start_index:05d}_to_{end_index:05d}.parquet"
            )
            with s3.open(output_file_path, "wb") as f:
                df.to_parquet(f)

            if start_index % 100 == 0:
                logger.info(
                    f"Progress: Completed {start_index} to {end_index} of {total_row_groups} row groups"
                )
    except Exception as e:
        logger.error(f"Error processing Parquet file {file_path}: {e}")

    logger.info(f"Completed splitting {file_path} into multiple files in {bucket_name}")


@cli.command("run_dask_preprocessing")
@click.option("--partitioned-telemetry-paths", type=str)
@click.option("--sample-frac", type=float)
@click.option("--df-out-path", type=str)
@click.option("--timestamp-col", type=str)
@click.option("--minio-endpoint", type=str)
@click.option("--dask-worker-image", type=str)
@click.option("--num-dask-workers", type=int)
@click.option("--namespace", type=str)
def run_dask_preprocessing(
    partitioned_telemetry_paths: List[str],
    sample_frac: float,
    df_out_path: str,
    timestamp_col: str,
    minio_endpoint: str,
    dask_worker_image: str,
    num_dask_workers: int,
    namespace: str,
) -> None:
    """
    Runs the Dask preprocessing pipeline, which mainly does random sampling.

    Parameters:
        partitioned_telemetry_paths: String of paths (comma separated) to the partitioned telemetry data.
        sample_frac: Fraction of data to sample.
        df_out_path: Path to save the output DataFrame.
        timestamp_col: Name of the timestamp column.
        minio_endpoint: Endpoint URL of MinIO.
        dask_worker_image: Docker image to use for Dask workers.
        num_dask_workers: Number of workers to use in the Dask cluster.
        namespace: Namespace for the Dask cluster.
    """
    minio_storage_option_dct = {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {"endpoint_url": minio_endpoint},
    }

    prc = DaskPreprocessor(
        num_dask_workers=num_dask_workers,
        image=dask_worker_image,
        storage_options=minio_storage_option_dct,
        sample_frac=sample_frac,
        namespace=namespace
    )
    df = prc.run(path_list=partitioned_telemetry_paths, timestamp_col=timestamp_col)
    df.to_parquet(df_out_path)

@cli.command("run_training")
@click.option("--train-df-path", type=str, required=True)
@click.option("--val-df-path", type=str, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--latent-dim", type=int, required=True)
@click.option("--hidden-dims", type=int, required=True)
@click.option("--beta", type=float, required=True)
@click.option("--lr", type=float, required=True)
@click.option("--early-stopping-patience", type=int, required=True)
@click.option("--max-epochs", type=int, required=True)
@click.option("--num-gpu-nodes", type=int, required=True)
@click.option("--num-dl-workers", type=int, required=True)
@click.option("--run-as-pytorchjob", type=bool, required=True)
@click.option("--model-output-file", type=str, required=False)  # Optional now
@click.option("--likelihood-mse-mixing-factor", type=float, required=False)
@click.option("--mlflow-uri", type=str, required=True)
@click.option("--mlflow-experiment-name", type=str, required=True)
@click.option("--minio-endpoint-url", type=str, required=True)
@click.option("--export-torchscript", type=bool, default=False, required=False)  # Added option for TorchScript
@click.option("--run-name", type=str, required=False, help="MLflow run name to use")
def run_training(
    train_df_path: str,
    val_df_path: str,
    seed: int,
    batch_size: int,
    latent_dim: int,
    hidden_dims: int,
    beta: float,
    lr: float,
    early_stopping_patience: int,
    max_epochs: int,
    num_gpu_nodes: int,
    num_dl_workers: int,
    run_as_pytorchjob: bool,
    model_output_file: Optional[str],
    likelihood_mse_mixing_factor: Optional[float],
    mlflow_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint_url: str,
    export_torchscript: bool = False,  # Default to False
    run_name: Optional[str] = None,
):
    """
    Starts the training of the model.

    Args:
    train_df_path: Path to the trained DataFrame.
    val_df_path: Path to the validated DataFrame.
    seed: Seed for random number generators.
    batch_size: Batch size for training.
    latent_dim: Dimension of the latent space.
    hidden_dims: Dimension of the hidden layers.
    beta: Weighting factor for KL divergence in the VAE loss.
    lr: Learning rate for the optimizer.
    early_stopping_patience: Patience for early stopping of training.
    max_epochs: Maximum number of epochs.
    num_gpu_nodes: Number of GPU nodes to use.
    num_dl_workers: Number of workers to use for the data loader.
    run_as_pytorchjob: Indicates whether to run the training as a PyTorch job.
    model_output_file: Optional file name under which the trained model checkpoint will be saved.
    likelihood_mse_mixing_factor: Factor for mixing MSE into the likelihood calculation.
    mlflow_uri: URI of the MLflow tracking server.
    mlflow_experiment_name: Name of the MLflow experiment.
    minio_endpoint_url: URL of the MinIO endpoint for artifact storage.
    export_torchscript: Whether to export a TorchScript version of the model.
    """
    # Set up random seeds
    seed_everything(seed)
    np.random.seed(seed)
    logger.info(f"Random seed in training script set to {seed}")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    # MLflow configuration
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    os.environ["AWS_ENDPOINT_URL"] = minio_endpoint_url

    # load dataset and initiate data module
    train_df = read_data_from_minio(train_df_path)
    val_df = read_data_from_minio(val_df_path)
    dm = TimeStampDataModule(
        train_df=train_df,
        val_df=val_df,
        batch_size=batch_size,
        num_workers=num_dl_workers,
    )
    dm.setup()

    # initiate model
    model = TimeStampVAE(
        input_dim=len(train_df.columns),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
        lr=lr,
        likelihood_mse_mixing_factor=likelihood_mse_mixing_factor,
    )

    # early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )

    # Determine if we need to save checkpoints locally
    callbacks = [
        early_stop_callback,
        StdOutLoggerCallback(),
        # ResetLogVarCallback(reset_epochs=5, reset_value=-2),
    ]
    
    # Add checkpoint callback only if model_output_file is provided
    if model_output_file:
        os.makedirs("data", exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="data/",
            filename=model_output_file,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        plugins=[KubeflowEnvironment()] if run_as_pytorchjob else [],
        devices=1,
        callbacks=callbacks,
        num_nodes=num_gpu_nodes,
        strategy="ddp" if run_as_pytorchjob else "auto",
        enable_progress_bar=False,
    )

    # Log relevant trainer attributes
    if run_as_pytorchjob:
        logger.debug(f"Trainer accelerator: {trainer.accelerator}")
        logger.debug(f"Trainer strategy: {trainer.strategy}")
        logger.debug(f"Trainer global_rank: {trainer.global_rank}")
        logger.debug(f"Trainer local_rank: {trainer.local_rank}")

    # Only initialize MLflow on the master process
    if not run_as_pytorchjob or trainer.global_rank == 0:
        # MLflow configuration
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        os.environ["AWS_ENDPOINT_URL"] = minio_endpoint_url
        
        # Enable MLflow autologging
        mlflow.pytorch.autolog()
        
        # Set run name if provided
        if run_name:
            mlflow.set_tag("mlflow.runName", run_name)
    
    # Train the model - this will automatically log to MLflow from the master process
    trainer.fit(model=model, datamodule=dm)
        
    # Only perform post-training logging on the master process
    if not run_as_pytorchjob or trainer.global_rank == 0:
        # Get the current run to construct return values
        active_run = mlflow.active_run()
        
        # Initialize triton_model_uri as None
        triton_model_uri = None
        
        if active_run:
            run_id = active_run.info.run_id
            
            # Export TorchScript model if requested (only from master process)
            if export_torchscript:
                logger.info("Exporting TorchScript model")
                try:
                    # Get a batch for tracing
                    batch = next(iter(dm.val_dataloader()))
                    
                    # Create and save TorchScript model
                    model.eval()
                    script_model = model.to_torchscript(method="script", example_inputs=batch)
                    torchscript_path = "model_torchscript.pt"
                    torch.jit.save(script_model, torchscript_path)
                    
                    # Create a temporary directory with Triton model structure
                    
                    # Model name for Triton
                    model_name = "eclss-vae"  # Adjust to your model name
                    
                    # Model properties - use the actual shape from your batch
                    input_dim = batch.shape[1]  # Input dimension from batch (should be 181)
                    
                    # Create temporary directory with Triton structure
                    triton_dir = tempfile.mkdtemp(prefix="triton_model_")

                    # Create Triton directory structure
                    model_dir = os.path.join(triton_dir, model_name)
                    version_dir = os.path.join(model_dir, "1")
                    os.makedirs(version_dir, exist_ok=True)
                    
                    # Create COMPLETE config.pbtxt with correct dimensions
                    config_content = f"""name: "{model_name}"
platform: "pytorch_libtorch"
max_batch_size: 64

input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {input_dim} ]  # Just the feature dimension (181), batch is handled separately
  }}
]

output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {input_dim} ]  # Same as input for VAE reconstruction
  }}
]
"""
                    # Save config.pbtxt
                    config_path = os.path.join(model_dir, "config.pbtxt")
                    with open(config_path, "w") as f:
                        f.write(config_content)
                    
                    # Copy TorchScript model to Triton structure
                    shutil.copy(torchscript_path, os.path.join(version_dir, "model.pt"))
                    
                    # Log the entire Triton model directory as an artifact
                    mlflow.log_artifact(model_dir, "triton_model")
                    
                    # Get the S3 path for the Triton model
                    # Extract the experiment ID and run ID to construct the S3 path
                    experiment_id = active_run.info.experiment_id
                    
                    logger.info(f"Successfully exported TorchScript model with Triton configuration")

                            
                except Exception as e:
                    logger.error(f"Failed to export TorchScript model: {e}")
            
            # Construct the return dictionary
            run_info = {
                "run_id": run_id,
                "run_name": run_name,
                "model_uri": f"runs:/{run_id}/model",
                "hyperparameters": {
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "beta": beta,
                    "latent_dim": latent_dim
                },
                "status": "success"
            }
                
            # Don't end the run - autolog will handle this
            return run_info
        else:
            # Fallback if no active run found
            return {
                "status": "completed without active MLflow run",
                "run_name": run_name
            }


@cli.command("run_evaluation")
@click.option("--run-name", type=str, required=True, help="MLflow run name to load the model from")
@click.option("--val-df-path", type=str, required=True, help="Path to validation dataframe")
@click.option("--test-df-path", type=str, required=True, help="Path to test dataframe")
@click.option("--mlflow-uri", type=str, required=True, help="URI to MLflow tracking server")
@click.option("--minio-endpoint-url", type=str, required=True, help="URL to Minio/S3 endpoint")
@click.option("--label-col-name", type=str, required=True, help="Column name for labels")
@click.option("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
@click.option("--batch-size", type=int, default=512, help="Batch size for data loading")
@click.option("--result-df-path", type=str, required=True, help="Path to save result dataframe")
@click.option("--metrics-dict-path", type=str, required=True, help="Path to save metrics dictionary")
@click.option("--threshold-min", type=int, default=-1500, help="Minimum threshold value")
@click.option("--threshold-max", type=int, default=0, help="Maximum threshold value")
@click.option("--number-thresholds", type=int, default=500, help="Number of thresholds to evaluate")
def run_evaluation(
    val_df_path: str,
    test_df_path: str,
    run_name: str,
    mlflow_uri: str,
    minio_endpoint_url: str,
    label_col_name: str,
    device: str,
    batch_size: int,
    result_df_path: str,
    metrics_dict_path: str,
    threshold_min: int,
    threshold_max: int,
    number_thresholds: int,
):
    """Evaluate model performance using a model stored in MLflow"""

    me = ModelEvaluator(
        val_df_path=val_df_path,
        test_df_path=test_df_path,
        run_name=run_name,
        mlflow_uri=mlflow_uri,
        minio_endpoint_url=minio_endpoint_url,
        label_col_name=label_col_name,
        batch_size=batch_size,
        device=device,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        num_thresholds=number_thresholds,
    )
    result_df, metrics_dict = me.run()
    result_df.to_parquet(result_df_path)
    with open(metrics_dict_path, 'w') as file:
        json.dump(metrics_dict, file)



if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cli()
