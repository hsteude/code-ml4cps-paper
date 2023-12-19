import click
import os
from typing import Optional, List
import re
import pyarrow.parquet as pq
from loguru import logger
from container_component_src.utils import create_s3_client
from container_component_src.dask_preprocessor.preprocessor import DaskPreprocessor
import toml
import pandas as pd

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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cli()