# Code for ML4CPS Paper on Kubeflow

This repository contains the code for our paper titled "Integrating Multiple Kubeflow Features for Anomaly Detection on ISS Telemetry Data".
The Abstract will be something like this:

Kubeflow integrates a suite of powerful tools for ML software development and deployment.
While often showcased independently in demos, this paper focuses on their collective effectiveness within a unified end-to-end workflow.
We conduct a case study on anomaly detection using telemetry data from the International Space Station (ISS) to discuss the effectiveness of integrating various tools—Dask, Katib, PyTorch Operator, and KServe—in a single Kubeflow Pipelines (KFP) framework.

## Project Structure

All necessary components for defining, compiling, and executing the pipeline are contained within a single Python package named `pipeline`.
The `container_component_src` directory contains Python code for custom container images used for various tasks, including Dask workers, PyTorch training jobs, Katib hyperparameter tuning, and more.
The idea is to use a single image for all these applications, defining individual commands for different tasks.
This setup includes a single `Dockerfile` at the root of this directory and a `.gitlab-ci.yml` file for automated builds in GitLab.

A `config.toml` file is used to store constants and path configurations.
The project employs `Poetry` for package management.
All Jupyter notebooks, such as those for data exploration and analysis, are located in the `notebooks` directory.
The `local_data` directory, while empty, is intended for local development.

The repository structure is outlined below:

```text
.
├── config.toml
├── container_component_src
│   ├── dask_preprocessor
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── datamodule.py
│   │   └── lightning_module.py
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── Dockerfile
├── LICENSE
├── local_data
├── notebooks
│   └── 01_exploration.ipynb
├── pipeline
│   ├── auth_session.py
│   ├── compile_and_run_pipeline.py
│   ├── components.py
│   ├── __init__.py
│   ├── pipeline_definition.py
├── poetry.lock
├── pyproject.toml
├── pyproject.toml
└── README.md
```

## Installation

To install the Python project locally, execute the following command:

```sh
poetry install
```

This command will set up the project with all necessary dependencies as defined in the pyproject.toml file.

## Building the Image

The Docker image for this project is set up to build automatically via GitLab's CI/CD pipeline upon pushing your code.
The CI pipeline builds the Docker image with two tags: latest and a short hash of the corresponding commit.
The registry URL for the image is specified in the config.toml file.

You can also build the image locally using the following command:

```sh
docker build -t <example-image-name> .
```

After building the image locally, push it to your registry:

```sh
docker login
docker push <example-image-name>
```

## Pipeline Execution

There are multiple ways to execute the Kubeflow pipeline:

### From a Kubeflow Notebook (KF-Notebook)

Within a Kubeflow notebook environment in the cluster, all necessary environment variables and permissions should already be set. After building and pushing the image to the specified registry (as noted in config.toml), you can run the pipeline with:

```sh
poetry run python pipeline/compile_and_run_pipeline.py
```

### Running Remotely

You can also execute the pipeline from your local development machine.
To do so, ensure to forward the MinIO port to your local machine.
This can be done using the following kubectl command:

```sh
kubectl port-forward -n minio svc/defaulttenant-hl 9000:9000
```

Alternatively, tools like k9s or Lens can be used for port forwarding.
If you lack the required permissions, please contact your system administrator.

To run the pipeline locally (from a remote machine outside the cluster), create a `.env` file in the root of this project with the following content:

```toml
KUBEFLOW_ENDPOINT='https://example-url'
KUBEFLOW_USERNAME='example.user@prokube.ai'
KUBEFLOW_PASSWORD='examplepassword'
KUBEFLOW_NAMESPACE='example-user'
AWS_SECRET_ACCESS_KEY='example-minio-password'
AWS_ACCESS_KEY_ID='example-minio-user'
S3_USE_HTTPS=0
S3_ENDPOINT='localhost:9000'
S3_VERIFY_SSL=0
```

Replace the placeholder values with your actual credentials and URLs.
Also add the option `--remote` the python command like so:

```bash
poetry run python pipeline/compile_and_run_pipeline.py --remote
```

### From the Kubeflow UI

The pipeline can be executed directly from the Kubeflow UI.
This method is convenient but requires either that the pipeline has been previously run in the same namespace, or that you have access to the compiled pipeline YAML file.

## How to Contribute

1. **Type Hinting**: Always use type hinting in Python code.
2. **Docstrings**: Lets try to follow [Google's style guide for Python docstrings](https://google.github.io/styleguide/pyguide.html).
3. **Branching and Commits**:
   - Create a new branch for your contribution.
   - Clean up your commits to maintain a clear history.
   - Rebase your branch to the latest master before submitting a pull request.

### Random collection of commands:

Run Dask preprocessing locally (from within a kf-notebook), e.g. for debugging.

```bash
poetry run python container_component_src/main.py run_dask_preprocessing \
--partitioned-telemetry-paths "s3://eclss-data/telemetry/partitioned-telemetry/*/*" \
--sample-frac 0.001 \
--df-out-path "./local_data/df_dask_out.parquet" \
--timestamp-col "normalizedTime" \
--minio-endpoint "http://minio.minio" \
--dask-worker-image "gitlab.kiss.space.unibw-hamburg.de:4567/kiss/code-ml4cps-paper/columbus-ad:commit-c2f04ede" \
--num-dask-workers 4
```

Install ipykernel for this virtual environment:

```sh
poetry run ipython kernel install --name "ml4cps" --user
```

Run Pytorch trianing locally in a notebook pod
```sh
poetry run python main.py run_training \
  --train-df-path "minio://mlpipeline/v2/artifacts/columbus-eclss-ad-pipeline/af29dcf9-e43e-4e95-951a-83120beb60dc/scale-dataframes/train_df_scaled" \
  --val-df-path "minio://mlpipeline/v2/artifacts/columbus-eclss-ad-pipeline/af29dcf9-e43e-4e95-951a-83120beb60dc/scale-dataframes/val_df_scaled" \
  --seed 42 \
  --batch-size 32 \
  --latent-dim 10 \
  --hidden-dims 100 \
  --beta 0.5 \
  --lr 0.001 \
  --early-stopping-patience 30 \
  --max-epochs 10 \
  --num-gpu-nodes 1 \
  --run-as-pytorchjob False \
  --model-output-file "local_test_model"

```
