from kfp.client import Client
from kfp import dsl, compiler
from pipeline.components import extract_composite_f1, serve_model
from pipeline.auth_session import get_istio_auth_session
from typing import List
import os
import datetime

SERVING_IMAGE = 'gitlab.kiss.space.unibw-hamburg.de:4567/kiss/code-ml4cps-paper/columbus-ad:commit-2f7cfe03'

auth_session = get_istio_auth_session(
    url=os.environ['KUBEFLOW_ENDPOINT'],
    username=os.environ['KUBEFLOW_USERNAME'],
    password=os.environ['KUBEFLOW_PASSWORD']
)

namespace = os.environ.get('KUBEFLOW_NAMESPACE', None) or \
            os.environ['KUBEFLOW_USERNAME'].split("@")[0].replace(".", "-")


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

@dsl.pipeline
def serving_pipeline(
        threshold: float
):
    """A short pipeline that takes existing artefacts and serves them if composite f1 of the model is above threshold"""
    importer_metrics = dsl.importer(
        artifact_uri='minio://mlpipeline/v2/artifacts/columbus-eclss-ad-pipeline/dd2114e2-f7f4-407e-8e26-182abfcf2a17/run-evaluation/metrics_dict',
        artifact_class=dsl.Dataset,
        reimport=True,
    )

    composite_f1 = extract_composite_f1(metrics_json=importer_metrics.output)
    # Using pipeline's flow control to decide wheter to deploy a model or not
    with dsl.If(composite_f1.output > threshold):
        importer_scaler = dsl.importer(
            artifact_uri='minio://mlpipeline/v2/artifacts/columbus-eclss-ad-pipeline/627c04f8-f2e8-4086-ac17-d25a1c629f1e/fit-scaler/fitted_scaler',
            artifact_class=dsl.Model,
            reimport=False,
        )
        serve_task = serve_model(
            model_path='minio://eclss-model-bucket/pytorch-job_20240110_095510.pt',
            scaler=importer_scaler.output,
            prod_path='minio://prod-models/eclss-vae/',
            serving_image=SERVING_IMAGE)
        add_minio_env_vars_to_tasks([serve_task])

if __name__ == "__main__":
    client = Client(host=f"{os.environ['KUBEFLOW_ENDPOINT']}/pipeline", namespace=namespace,
                    cookies=auth_session["session_cookie"], verify_ssl=False)

    compiler.Compiler().compile(serving_pipeline, 'serving_pipeline.yaml')

    run = client.create_run_from_pipeline_package(
        'serving_pipeline.yaml',
        enable_caching=False,
        arguments={
            "threshold": 0.7  # <= pipeline arguments. Also exposed via UI
        },
        run_name=f'pipeline_if_test {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )
