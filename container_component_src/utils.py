import s3fs
import os
import pandas as pd
from loguru import logger


def create_s3_client() -> s3fs.S3FileSystem:
    """
    Initializes the S3 Client using specific environment variables.

    This function expects the following environment variables to be set:
    - 'S3_ENDPOINT': The endpoint URL for the S3 service.
    - 'AWS_ACCESS_KEY_ID': The AWS access key ID.
    - 'AWS_SECRET_ACCESS_KEY': The AWS secret access key.

    Returns:
        An instance of the S3FileSystem client.
    """
    return s3fs.S3FileSystem(
        anon=False,
        use_ssl=False,
        client_kwargs={
            "endpoint_url": f'http://{os.environ["S3_ENDPOINT"]}',
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        },
    )


def read_data_from_minio(df_path: str, minio_endpoint: str) -> pd.DataFrame:
    """
    Read data from MinIO and return it as a pandas DataFrame.

    Parameters:
    df_path -- Path to the data on MinIO.

    Returns:
    The loaded data.
    """
    storage_options = {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {"endpoint_url": minio_endpoint},
    }

    if df_path.startswith("minio://"):
        df_path = df_path.replace("minio://", "s3://")
    elif df_path.startswith("/minio/"):
        df_path = df_path.replace("/minio/", "s3://")
    elif df_path.startswith("minio/"):
        df_path = df_path.replace("minio/", "s3://")

    try:
        return pd.read_parquet(df_path, storage_options=storage_options)
    except Exception as e:
        logger.error(f"Error while reading from MinIO: {str(e)}")
        raise e


def upload_file_to_minio_bucket(bucket: str, file: str):
    """Uploads the file to the bucket ..."""
    fs = create_s3_client()
    if bucket not in fs.ls("/"):
        logger.info(f"Creating bucket {bucket}, cause it did not exist")
        fs.mkdir(f"/{bucket}")
    if os.path.exists(file):
        logger.info(
            f'Attempting to upload this file "{file}" to this bucket "{bucket}"'
        )
        fs.put(file, f"{bucket}/{os.path.basename(file)}")
    else:
        logger.error(
            f'File "{file}" does not exist and cannot be uploaded to bucket "{bucket}"'
        )
