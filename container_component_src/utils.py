import s3fs
import os

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

