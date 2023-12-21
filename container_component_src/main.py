import click
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
def run_dask_preprocessing(
    partitioned_telemetry_paths: List[str],
    sample_frac: float,
    df_out_path: str,
    timestamp_col: str,
    minio_endpoint: str,
    dask_worker_image: str,
    num_dask_workers: int,
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
@click.option("--model-output-file", type=str, required=True)
@click.option("--minio-model-bucket", type=str, required=False)
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
    model_output_file: str,
    minio_model_bucket: Optional[str],
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
    model_output_file: File name under which the trained model will be saved.
    minio_model_bucket: Name of the MinIO bucket to store the model.
    """
    seed_everything(seed)
    np.random.seed(seed)
    logger.info(f"Random seed in training script set to {seed}")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

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
    )

    # early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )

    # saves top-K checkpoints based on "val_loss" metric
    os.makedirs("data", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="data/",
        filename=model_output_file,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        plugins=[KubeflowEnvironment()] if run_as_pytorchjob else [],
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            StdOutLoggerCallback(),
            ResetLogVarCallback(reset_epochs=5, reset_value=-2),
        ],
        num_nodes=num_gpu_nodes,
        strategy="ddp" if run_as_pytorchjob else "auto",
    )

    # Log relevant trainer attributes
    if run_as_pytorchjob:
        logger.debug(f"Trainer accelerator: {trainer.accelerator}")
        logger.debug(f"Trainer strategy: {trainer.strategy}")
        logger.debug(f"Trainer global_rank: {trainer.global_rank}")
        logger.debug(f"Trainer local_rank: {trainer.local_rank}")

    trainer.fit(model=model, datamodule=dm)
    if minio_model_bucket:
        upload_file_to_minio_bucket(minio_model_bucket, model_output_file)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cli()
