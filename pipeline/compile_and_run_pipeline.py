import os
import click
from kfp.client import Client
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    logger.info("dotenv package not found. Only existing environment variables will be used")


@click.command()
@click.option(
    "--remote",
    is_flag=True,
    help="Set this flag to run the pipeline remotely.",
)
def run(remote):
    """Compile and run the Kubeflow pipeline.

    This script needs a bunch of env variables (or .env file with dotenv package installed):

    KUBEFLOW_ENDPOINT
    KUBEFLOW_USERNAME
    KUBEFLOW_PASSWORD
    KUBEFLOW_NAMESPACE (optional inferred from KUBEFLOW_USERNAME if not provided)
    S3_ENDPOINT
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY

    """
    from pipeline.pipeline_definition import columbus_eclss_ad_pipeline
    from pipeline.auth_session import get_istio_auth_session


    if remote:
        # Infer namespace from username if no namespace provided
        namespace = os.environ.get('KUBEFLOW_NAMESPACE', None) or \
            os.environ['KUBEFLOW_USERNAME'].split("@")[0].replace(".", "-")

        auth_session = get_istio_auth_session(
            url=os.environ["KUBEFLOW_ENDPOINT"],
            username=os.environ["KUBEFLOW_USERNAME"],
            password=os.environ["KUBEFLOW_PASSWORD"],
        )

        client = Client(
            host=f"{os.environ['KUBEFLOW_ENDPOINT']}/pipeline",
            namespace=namespace,
            cookies=auth_session["session_cookie"],
            verify_ssl=False,
        )
    else:
        client = Client()

    # Pipeline arguments
    args = dict(
        rowgroups_per_file=10,
        dask_preproc_sample_frac=0.001,
        num_dask_workers=4,
        split_window_hours=250.0,
        train_split=0.8,
        viz_sample_fraction=0.01,
        katib_max_epochs=100,
        katib_max_trials=15,
        katib_batch_size_list=["32", "64", "128", "256"],
        katib_beta_list=["0.001", "0.0001", "0.0001", "0.00001"],
        katib_learning_rate_list=["0.0005", "0.0001", "0.00005"],
        latent_dim=18,
        pytorchjob_num_dl_workers=12,
        pytorchjob_max_epochs=100,
        pytorchjob_early_stopping_patience=10,
        pytorchjob_num_gpu_nodes=3,
        eval_batch_size=1024,
        eval_threshold_min=-200,
        eval_threshold_max=-100,
        eval_number_thresholds=100,
        threshold=0.7
    )

    # Compile and run the pipeline
    client.create_run_from_pipeline_func(
        columbus_eclss_ad_pipeline,
        arguments=args,
        experiment_name="ml4cps-eclss-ad-pipeline",
        enable_caching=True,
    )
    click.echo("Pipeline execution started.")


if __name__ == "__main__":
    run()
