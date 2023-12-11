import os
from dotenv import load_dotenv
import click
from kfp.client import Client


@click.command()
@click.option(
    "--remote",
    is_flag=True,
    help="Set this flag to run the pipeline remotely.",
)
def run(remote):
    load_dotenv()
    from pipeline.pipeline_definition import columbus_eclss_ad_pipeline
    from pipeline.auth_session import get_istio_auth_session

    """ Compile and run the Kubeflow pipeline. """
    if remote:
        auth_session = get_istio_auth_session(
            url=os.environ["KUBEFLOW_ENDPOINT"],
            username=os.environ["KUBEFLOW_USERNAME"],
            password=os.environ["KUBEFLOW_PASSWORD"],
        )

        client = Client(
            host=f"{os.environ['KUBEFLOW_ENDPOINT']}/pipeline",
            namespace=os.environ["KUBEFLOW_NAMESPACE"],
            cookies=auth_session["session_cookie"],
            verify_ssl=False,
        )
    else:
        client = Client()

    # Pipeline arguments
    args = dict()

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
