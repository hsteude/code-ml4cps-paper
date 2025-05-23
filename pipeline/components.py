import os
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact, HTML, Metrics, Model
from typing import Dict, List, Optional
import toml

# load config
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../config.toml", "r") as f:
    config = toml.load(f)


@dsl.container_component
def split_parquet_file(
    file_path: str,
    target_path: str,
    timestamp_col: str,
    rowgroup_limit: Optional[int] = None,
    rowgroups_per_file: int = 10,
):
    """Kubeflow pipeline component to split a large Parquet file into smaller files"""
    return dsl.ContainerSpec(
        image=f'{config["images"]["split-parquet-file"]}',
        command=["python", "container_component_src/main.py"],
        args=[
            "split_parquet_file",
            "--file-path",
            file_path,
            "--target-path",
            target_path,
            "--timestamp-col",
            timestamp_col,
            "--rowgroup-limit",
            rowgroup_limit,
            "--rowgroups-per-file",
            rowgroups_per_file,
        ],
    )


@dsl.container_component
def run_dask_preprocessing(
    partitioned_telemetry_paths: str,
    sample_frac: float,
    preprocessed_df: Output[Dataset],
    timestamp_col: str,
    minio_endpoint: str,
    dask_worker_image: str,
    num_dask_workers: int,
    namespace: str,
):
    """Kubeflow pipeline component for Dask preprocessing"""
    return dsl.ContainerSpec(
        image=f'{config["images"]["dask-component"]}',
        command=["python", "container_component_src/main.py"],
        args=[
            "run_dask_preprocessing",
            "--partitioned-telemetry-paths",
            partitioned_telemetry_paths,
            "--sample-frac",
            str(sample_frac),
            "--df-out-path",
            preprocessed_df.path,
            "--timestamp-col",
            timestamp_col,
            "--minio-endpoint",
            minio_endpoint,
            "--dask-worker-image",
            dask_worker_image,
            "--num-dask-workers",
            str(num_dask_workers),
            "--namespace",
            namespace,
        ],
    )


@dsl.component(
    base_image=f'{config["images"]["dask-component"]}',
)
def get_label(labels_xls_path: str, df_labels: Output[Dataset]):
    """Reads an Excel file containing labels and converts it to parquet format.

    Args:
        labels_xls_path: Path to the Excel file containing labels
        df_labels: KFP Output Dataset for the converted parquet file
    """
    import pandas as pd
    import os

    storage_options = {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {"endpoint_url": f'http://{os.environ["S3_ENDPOINT"]}'},
    }

    df_label = pd.read_excel(labels_xls_path, storage_options=storage_options)
    df_label.to_parquet(df_labels.path)


@dsl.component(
    base_image=f'{config["images"]["dask-component"]}',
)
def get_label_series(
    df_telemetry_in: Input[Dataset],
    df_label_in: Input[Dataset],
    anomaly_start_col: str,
    anomaly_end_col: str,
    ar_col: str,
    labels_series: Output[Dataset],
) -> None:
    """
    Assigns an anomaly label to each row in the telemetry dataframe based on anomaly periods.

    Args:
        df_telemetry: The telemetry dataframe with timestamps as the index.
        df_label: The dataframe containing anomaly periods.
        anomaly_start_col: The column name in df_label indicating the start of an anomaly.
        anomaly_end_col: The column name in df_label indicating the end of an anomaly.

    Returns:
        Updated telemetry dataframe with an 'anomaly_label' column.
    """
    import pandas as pd

    # Read dataset
    df_telemetry = pd.read_parquet(df_telemetry_in.path)
    df_label = pd.read_parquet(df_label_in.path)

    # Convert anomaly start and end times to datetime
    df_label[anomaly_start_col] = pd.to_datetime(
        df_label[anomaly_start_col], format="%Y/%j/%H:%M:%S", utc=True
    )
    df_label[anomaly_end_col] = pd.to_datetime(
        df_label[anomaly_end_col], format="%Y/%j/%H:%M:%S", utc=True
    )

    # Function to check if a timestamp is within any interval
    def is_in_anomaly_interval(ts: pd.Timestamp, intervals: list) -> bool:
        return any(start <= ts < end for start, end in intervals)

    # Create a list of anomaly intervals
    anomaly_intervals = list(
        zip(df_label[anomaly_start_col], df_label[anomaly_end_col])
    )

    # Assign anomaly labels
    df_telemetry[ar_col] = df_telemetry.index.to_series().apply(
        lambda ts: 1 if is_in_anomaly_interval(ts, anomaly_intervals) else 0
    )

    # Save labels series as df
    df_telemetry[[ar_col]].to_parquet(labels_series.path)


@dsl.component(packages_to_install=["pyarrow", "pandas"], base_image="python:3.9")
def create_train_dev_test_split(
    preproc_df_in: Input[Dataset],
    anomaly_df_in: Input[Dataset],
    window_hours: float,
    label_col: str,
    df_train: Output[Dataset],
    df_val: Output[Dataset],
    df_test: Output[Dataset],
    train_split: float = 0.8,
) -> None:
    """
    Splits the data into train, dev, and test sets. Test set will contain only the data points within the specified
    time window around each anomaly and a column indicating actual anomalies.

    Args:
        preproc_df: KFP Dataset for preprocessed multivariate time series data.
        anomaly_df: KFP Dataset DataFrame with anomaly labels.
        label_col: Column name for the anomaly labels
        window_hours: Size of the time window around each anomaly in hours.
        train_split: Proportion of non-test data to use for training.

    Returns:
        tuple: Three DataFrames corresponding to the train, dev, and test sets.
    """
    import pandas as pd
    from datetime import timedelta

    # read in data
    anomaly_df = pd.read_parquet(anomaly_df_in.path)
    preproc_df = pd.read_parquet(preproc_df_in.path)

    window_size = timedelta(hours=window_hours)

    # Identifying the timestamps of anomalies
    anomaly_timestamps = anomaly_df.index[anomaly_df["Anomaly"] == 1]

    # Creating a mask for the test set
    test_mask = pd.Series(False, index=preproc_df.index)
    for timestamp in anomaly_timestamps:
        start_window = timestamp - window_size
        end_window = timestamp + window_size
        test_mask.loc[start_window:end_window] = True

    # Splitting the dataframes and adding anomaly column to test set
    test_df = preproc_df[test_mask].copy()
    test_df[label_col] = anomaly_df[label_col][test_mask]

    non_test_df = preproc_df[~test_mask]

    # Randomly splitting the non-test data into train and dev sets
    train_df = non_test_df.sample(frac=train_split, random_state=42)
    val_df = non_test_df.drop(train_df.index)

    # write out
    train_df.to_parquet(df_train.path)
    val_df.to_parquet(df_val.path)
    test_df.to_parquet(df_test.path)


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "scikit-learn==1.3.2"],
    base_image="python:3.9",
)
def fit_scaler(
    df_in: Input[Dataset],
    scaler_type: str,
    fitted_scaler: Output[Artifact],
) -> None:
    """
    Scales the columns of a DataFrame using Min-Max scaling or Standard scaling.
    Fits the scaler on a DataFrame using either Min-Max or Standard scaling.

    Args:
        df: The KFP Dataset Input for the dataframe the scaler should be fitted on.
        scaler_type: Type of scaling to apply ('minmax' or 'standard').
        scaler_object_path: KFP Output Artifact for the fitted scaler object.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import joblib

    # read df
    df = pd.read_parquet(df_in.path)

    # Choose the appropriate scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Choose 'minmax' or 'standard'.")

    # Fit and transform the DataFrame
    scaler = scaler.fit(df)

    # Save the scaled DataFrame and scaler object
    joblib.dump(scaler, fitted_scaler.path)


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "scikit-learn==1.3.2"],
    base_image="python:3.9",
)
def scale_dataframes(
    train_df_in: Input[Dataset],
    val_df_in: Input[Dataset],
    test_df_in: Input[Dataset],
    scaler_in: Input[Artifact],
    label_column: str,
    train_df_scaled: Output[Dataset],
    val_df_scaled: Output[Dataset],
    test_df_scaled: Output[Dataset],
) -> None:
    """
    Scales the given train, validation, and test dataframes using the provided scaler.
    Assumes that the test dataframe contains an additional column for labels.
    """
    import pandas as pd
    import joblib

    # Read DataFrames and load the scaler
    train_df, val_df, test_df = [
        pd.read_parquet(ds.path) for ds in [train_df_in, val_df_in, test_df_in]
    ]
    scaler = joblib.load(scaler_in.path)

    # Scale the training data while preserving the index
    train_df_sc = pd.DataFrame(
        scaler.transform(train_df), columns=train_df.columns, index=train_df.index
    )
    train_df_sc.to_parquet(train_df_scaled.path)

    # Scale the validation data while preserving the index
    val_df_sc = pd.DataFrame(
        scaler.transform(val_df), columns=val_df.columns, index=val_df.index
    )
    val_df_sc.to_parquet(val_df_scaled.path)

    # Scale the test data while preserving the label column and index
    test_labels = test_df.pop(label_column)
    test_df_sc = pd.DataFrame(
        scaler.transform(test_df), columns=test_df.columns, index=test_df.index
    )
    test_df_sc[label_column] = test_labels
    test_df_sc.to_parquet(test_df_scaled.path)


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "plotly==5.3.1"],
    base_image="python:3.9",
)
def visualize_split(
    train_df_in: Input[Dataset],
    test_df_in: Input[Dataset],
    column_name: str,
    plot_html: Output[HTML],
    sample_fraction: float = 0.1,
) -> None:
    """
    Creates a Plotly plot visualizing the specified column from sampled train and test dataframes,
    and the anomaly column from the test dataframe, and saves it as an HTML file.

    Args:
        train_df_in: KFP Dataset Input for the training dataframe.
        test_df_in: KFP Dataset Input for the testing dataframe.
        column_name: Name of the column to be plotted.
        plot_html: KFP Output Artifact for the Plotly plot HTML.
        sample_fraction: Fraction of data to sample for plotting (default 0.1).
    """
    import pandas as pd
    import plotly.graph_objects as go

    # Read DataFrames
    train_df = pd.read_parquet(train_df_in.path)
    test_df = pd.read_parquet(test_df_in.path)

    # Sample the dataframes
    train_df_sampled = train_df.sample(
        frac=sample_fraction, random_state=42
    ).sort_index()
    test_df_sampled = test_df.sample(frac=sample_fraction, random_state=42).sort_index()

    # Create Plotly figure
    fig = go.Figure()

    # Scatter plot for the sampled train dataframe
    fig.add_trace(
        go.Scatter(
            x=train_df_sampled.index,
            y=train_df_sampled[column_name],
            mode="markers",
            name="Train Data",
            marker=dict(size=3),
        )
    )

    # Scatter plot for the sampled test dataframe
    fig.add_trace(
        go.Scatter(
            x=test_df_sampled.index,
            y=test_df_sampled[column_name],
            mode="markers",
            name="Test Data",
            marker=dict(size=3),
        )
    )

    # Line plot for the anomalies in the test dataframe
    fig.add_trace(
        go.Scatter(
            x=test_df_sampled.index,
            y=test_df_sampled["Anomaly"],
            mode="lines",
            name="Anomalies",
        )
    )

    fig.update_layout(
        title=f"Visualization of train test split trough random telemetry parameter",
        xaxis_title="Timestamp",
        yaxis_title="Random telemetry parameter",
    )

    # Save the plot as HTML
    fig.write_html(plot_html.path)


@dsl.component(
    packages_to_install=["kubernetes", "loguru"],
    base_image="python:3.9",
)
def run_katib_experiment(
    df_train: Input[Dataset],
    df_val: Input[Dataset],
    experiment_name: str,
    image: str,
    namespace: str,
    max_epochs: int,
    max_trials: int,
    batch_size_list: List[str],
    beta_list: List[str],
    learning_rate_list: List[str],
    latent_dim: int,
    mlflow_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint_url: str,
) -> Dict:
    import time
    from kubernetes import client, config
    from loguru import logger
    import os

    # Appending argo node id to experiment name so experiments always get unique names
    experiment_name = experiment_name + "-" + os.environ["ARGO_NODE_ID"].split("-")[-1]
    logger.info(f"Starting experiment: {experiment_name}")
    group = "kubeflow.org"
    version = "v1beta1"
    plural = "experiments"
    # Command List
    command_list = [
        "python",
        "container_component_src/main.py",
        "run_training",
        f"--train-df-path={df_train.path}",
        f"--val-df-path={df_val.path}",
        "--seed=42",
        "--batch-size=${trialParameters.batchSize}",
        f"--latent-dim={latent_dim}",
        "--hidden-dims=500",
        "--beta=${trialParameters.beta}",
        "--lr=${trialParameters.learningRate}",
        "--early-stopping-patience=30",
        f"--max-epochs={max_epochs}",
        "--num-gpu-nodes=1",
        "--num-dl-workers=12",
        "--run-as-pytorchjob=False",
        "--model-output-file=local_test_model",
        "--likelihood-mse-mixing-factor=0.01",
        f"--mlflow-uri={mlflow_uri}",
        f"--mlflow-experiment-name={mlflow_experiment_name}",
        f"--minio-endpoint-url={minio_endpoint_url}",
    ]

    # Environment Dictionary
    env_dict = [
        {
            "name": "AWS_ACCESS_KEY_ID",
            "valueFrom": {
                "secretKeyRef": {"name": "s3creds", "key": "AWS_ACCESS_KEY_ID"}
            },
        },
        {
            "name": "AWS_SECRET_ACCESS_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": "s3creds", "key": "AWS_SECRET_ACCESS_KEY"}
            },
        },
        {"name": "S3_ENDPOINT", "value": "minio.minio"},
        {"name": "S3_USE_HTTPS", "value": "0"},
        {"name": "S3_VERIFY_SSL", "value": "0"},
    ]

    # Spec Dictionary
    spec_dict = {
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "training-container",
                        "image": image,
                        "resources": {"limits": {"nvidia.com/gpu": 1}},
                        "command": command_list,
                        "env": env_dict,
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    }
                ],
                "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
                "restartPolicy": "Never",
            },
        }
    }

    # Trial Template Dictionary
    trial_template_dict = {
        "primaryContainerName": "training-container",
        "trialParameters": [
            {
                "name": "learningRate",
                "description": "Learning rate for the training model",
                "reference": "learning_rate",
            },
            {
                "name": "batchSize",
                "description": "Batch size for training",
                "reference": "batch_size",
            },
            {
                "name": "beta",
                "description": "beta in beta vae loss function",
                "reference": "beta",
            },
        ],
        "trialSpec": {"apiVersion": "batch/v1", "kind": "Job", "spec": spec_dict},
    }

    experiment_config = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": experiment_name, "namespace": namespace},
        "spec": {
            "parallelTrialCount": 3,
            "maxTrialCount": max_trials,
            "maxFailedTrialCount": 3,
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "objective": {
                "type": "minimize",
                "goal": -100_000,
                "objectiveMetricName": "val_loss",
                "additionalMetricNames": ["train_loss"],
            },
            "algorithm": {"algorithmName": "random"},
            "parameters": [
                {
                    "name": "learning_rate",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": learning_rate_list},
                },
                {
                    "name": "batch_size",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": batch_size_list},
                },
                {
                    "name": "beta",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": beta_list},
                },
            ],
            "trialTemplate": trial_template_dict,
        },
    }

    config.load_incluster_config()

    k8s_client = client.ApiClient()

    katib_api_instance = client.CustomObjectsApi(k8s_client)
    katib_api_instance.create_namespaced_custom_object(
        group, version, namespace, plural, experiment_config
    )
    time.sleep(60)
    logger.info(f"Experiment {experiment_name} submitted. Waiting for completion...")

    while True:
        response = katib_api_instance.get_namespaced_custom_object(
            group, version, namespace, plural, experiment_name
        )

        status = response["status"]["conditions"][-1]["type"]
        if status == "Succeeded":
            result = response
            break
        elif status == "Failed":
            raise Exception(f"Experiment {experiment_name} failed!")

        logger.info(f"Experiment {experiment_name} running. Waiting for completion...")
        time.sleep(60)

    best_trial = result["status"]["currentOptimalTrial"]["parameterAssignments"]
    return {b["name"]: b["value"] for b in best_trial}


@dsl.component(
    packages_to_install=["kubernetes", "loguru"],
    base_image="python:3.9",
)
def run_pytorch_training_job(
    train_df_in: Input[Dataset],
    val_df_in: Input[Dataset],
    training_image: str,
    namespace: str,
    num_dl_workers: int,
    tuning_param_dct: Dict,
    max_epochs: int,
    early_stopping_patience: int,
    latent_dim: int,
    seed: int,
    num_gpu_nodes: int,
    mlflow_uri: str,
    mlflow_experiment_name: str,
    minio_endpoint_url: str,
    export_torchscript: bool = True,
    model_output_file: str = None,
) -> Dict:
    """Initiates a PyTorch training job with MLflow tracking.

    Parameters:
    train_df_in: KFP input for the training dataframe.
    val_df_in: KFP input for the validation dataframe.
    training_image: Docker image containing training code.
    namespace: Kubernetes namespace to run the training job in.
    num_dl_workers: Number of data loader processes.
    tuning_param_dct: Dictionary containing tuned hyperparameters (batch_size, learning_rate, beta).
    max_epochs: Maximum epochs to train.
    early_stopping_patience: Number of epochs to wait for improved validation loss before early stopping.
    latent_dim: Number of latent variables.
    seed: Seed for random number generation to ensure reproducibility.
    num_gpu_nodes: Number of GPU nodes to utilize during training.
    mlflow_uri: URI of the MLflow tracking server.
    mlflow_experiment_name: Name of the MLflow experiment.
    minio_endpoint_url: URL of the MinIO endpoint for artifact storage.
    export_torchscript: Whether to export a TorchScript version of the model.
    model_output_file: Optional file name under which the trained model checkpoint will be saved.

    Returns:
    A dictionary containing model run information including MLflow URIs
    """
    import time
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    from datetime import datetime
    from loguru import logger
    import uuid

    # Extract hyperparameters from the tuning dictionary
    batch_size, learning_rate, beta = [
        tuning_param_dct[k] for k in ("batch_size", "learning_rate", "beta")
    ]

    pytorchjob_name = "pytorch-job"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate model_output_file name if not provided
    if not model_output_file:
        model_output_file = f"{pytorchjob_name}_{current_time}"

    # Generate a unique run name for MLflow
    run_name = f"pytorch-job-{current_time}-{str(uuid.uuid4())[:8]}"
    logger.info(f"Generated MLflow run name: {run_name}")

    # Create the command for the PyTorch training job
    command = [
        "python",
        "container_component_src/main.py",
        "run_training",
        f"--train-df-path={train_df_in.path}",
        f"--val-df-path={val_df_in.path}",
        f"--seed={seed}",
        f"--batch-size={batch_size}",
        f"--latent-dim={latent_dim}",
        f"--num-dl-workers={num_dl_workers}",
        "--hidden-dims=100",
        f"--beta={beta}",
        f"--lr={learning_rate}",
        f"--early-stopping-patience={early_stopping_patience}",
        f"--max-epochs={max_epochs}",
        f"--num-gpu-nodes={num_gpu_nodes}",
        "--run-as-pytorchjob=True",
        f"--model-output-file={model_output_file}",
        f"--likelihood-mse-mixing-factor=0.1",
        f"--mlflow-uri={mlflow_uri}",
        f"--mlflow-experiment-name={mlflow_experiment_name}",
        f"--minio-endpoint-url={minio_endpoint_url}",
        f"--export-torchscript={'True' if export_torchscript else 'False'}",
        f"--run-name={run_name}",  # Pass the generated run name
    ]

    # Define the Kubernetes pod template with GPU resources
    template = {
        "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
        "spec": {
            "containers": [
                {
                    "name": "pytorch",
                    "image": training_image,
                    "imagePullPolicy": "Always",
                    "command": command,
                    "resources": {
                        "limits": {"nvidia.com/gpu": 1}  # Request 1 GPU per pod
                    },
                    "env": [
                        {
                            "name": "AWS_ACCESS_KEY_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "s3creds",
                                    "key": "AWS_ACCESS_KEY_ID",
                                }
                            },
                        },
                        {
                            "name": "AWS_SECRET_ACCESS_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "s3creds",
                                    "key": "AWS_SECRET_ACCESS_KEY",
                                }
                            },
                        },
                        {"name": "S3_ENDPOINT", "value": minio_endpoint_url},
                        {"name": "S3_USE_HTTPS", "value": "0"},
                        {"name": "S3_VERIFY_SSL", "value": "0"},
                    ],
                    "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                }
            ],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        },
    }

    # Define the PyTorchJob manifest
    training_job_manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": f"{pytorchjob_name}", "namespace": f"{namespace}"},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": template,
                },
                "Worker": {
                    "replicas": num_gpu_nodes - 1,
                    "restartPolicy": "OnFailure",
                    "template": template,
                },
            }
        },
    }

    # Initialize Kubernetes API clients
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api_client = client.ApiClient()
    custom_api = client.CustomObjectsApi(api_client)

    # Common arguments for API calls
    args_dict = {
        "group": "kubeflow.org",
        "version": "v1",
        "namespace": namespace,
        "plural": "pytorchjobs",
    }

    def start_pytorchjob():
        """Starts the PyTorchJob by calling create_namespaced_custom_object."""
        try:
            custom_api.create_namespaced_custom_object(
                body=training_job_manifest, **args_dict
            )
            logger.info(
                f"Job {pytorchjob_name} started with MLflow run name {run_name}"
            )
        except ApiException as e:
            logger.error(f"Error starting job: {e}")
            raise

    def get_pytorchjob_status():
        """Retrieves and returns the last status of the PyTorchJob."""
        try:
            job = custom_api.get_namespaced_custom_object(
                name=pytorchjob_name, **args_dict
            )
            return job.get("status", {}).get("conditions", [])[-1].get("type", "")
        except ApiException as e:
            logger.error(f"Error retrieving job status: {e}")
            return None

    def delete_pytorchjob():
        """Deletes the PyTorchJob once it's completed."""
        try:
            custom_api.delete_namespaced_custom_object(
                name=pytorchjob_name, **args_dict
            )
            logger.info(f"Job {pytorchjob_name} deleted.")
        except ApiException as e:
            logger.error(f"Error deleting job: {e}")

    # Start the job and wait for a while
    start_pytorchjob()
    time.sleep(20)

    # Periodically check the job status
    while True:
        status = get_pytorchjob_status()
        logger.info(f"Current job status: {status}")

        if status == "Succeeded":
            logger.info("Job succeeded!")

            # Create run info with just the run name
            run_info = {
                "status": "success",
                "run_name": run_name,
                "model_uri": f"runs:/name:{run_name}/model",  # Special format for referring to run by name
                "hyperparameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "beta": beta,
                    "latent_dim": latent_dim,
                },
            }

            if export_torchscript:
                run_info["torchscript_uri"] = (
                    f"runs:/name:{run_name}/artifacts/torchscript/model_torchscript.pt"
                )

            delete_pytorchjob()
            return run_info

        elif status in ["Failed", "Error"]:
            logger.error("Job failed or encountered an error.")
            delete_pytorchjob()
            raise RuntimeError(f"PyTorch job {pytorchjob_name} failed")

        time.sleep(10)


from kfp import dsl
from kfp.dsl import Input, Output, Dataset, component
import json
import numpy as np
import pandas as pd


@dsl.component(
    base_image=f'{config["images"]["evaluation"]}',
    packages_to_install=["pandas", "numpy", "pyarrow", "mlflow"],
)
def evaluate_model(
    train_out_dict: dict,
    val_df: Input[Dataset],
    test_df: Input[Dataset],
    mlflow_uri: str,
    minio_endpoint_url: str,
    mlflow_experiment_name: str,
    results_df: Output[Dataset],
    label_col_name: str = "Anomaly",
    device: str = "cuda",
    batch_size: int = 512,
    threshold_min: int = -1500,
    threshold_max: int = 0,
    number_thresholds: int = 500,
) -> dict:
    """
    Komponente zur Evaluierung eines trainierten Modells aus MLflow.

    Args:
        train_out_dict: Dictionary mit MLflow Run-Informationen (muss 'run_name' enthalten)
        val_df_path: Pfad zum Validierungs-Dataframe
        test_df_path: Pfad zum Test-Dataframe
        mlflow_uri: URI zum MLflow Tracking Server
        minio_endpoint_url: URL zum Minio/S3 Endpoint
        results_df: KFP Output Dataset für den Ergebnis-DataFrame
        label_col_name: Name der Spalte mit den Labels
        device: Gerät für die Berechnung (cuda/cpu)
        batch_size: Batch-Größe für Datenlader
        threshold_min: Minimaler Threshold-Wert
        threshold_max: Maximaler Threshold-Wert
        number_thresholds: Anzahl der zu evaluierenden Thresholds

    Returns:
        Dictionary mit Evaluierungsmetriken
    """
    # Import ModelEvaluator from your package
    from container_component_src.eval.evaluator import ModelEvaluator
    import os

    # Set AWS/Minio environment variables
    os.environ["AWS_ENDPOINT_URL"] = minio_endpoint_url

    # Extract run name from dictionary
    run_name = train_out_dict.get("run_name")
    if not run_name:
        raise ValueError("train_out_dict muss einen 'run_name' enthalten")

    # Initialize and run ModelEvaluator
    evaluator = ModelEvaluator(
        val_df_path=val_df.path,
        test_df_path=test_df.path,
        run_name=run_name,
        mlflow_uri=mlflow_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        minio_endpoint_url=minio_endpoint_url,
        label_col_name=label_col_name,
        batch_size=batch_size,
        device=device,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        num_thresholds=number_thresholds,
    )

    # run eval
    result_df, metrics_dict = evaluator.run()

    result_df.to_parquet(results_df.path)

    return metrics_dict


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "plotly==5.3.1"],
    base_image="python:3.9",
)
def visualize_results(
    result_df_in: Input[Dataset],
    metrics_dict: Dict,
    result_viz: Output[HTML],
    metrics: Output[Metrics],
    sample_fraction: float = 0.1,
    label_col_name: str = "Anomaly",
    scatter_y_min: int = -4000,
    scatter_y_max: int = 500,
) -> None:
    """Creates output plot and logs metrics"""

    import plotly.graph_objs as go
    import pandas as pd
    import json

    for k, v in metrics_dict.items():
        metrics.log_metric(metric=k, value=v)

    def get_intervals(df, column, value):
        intervals = []
        start = None
        for i, row in df.iterrows():
            if row[column] == value and start is None:
                start = i
            elif row[column] != value and start is not None:
                intervals.append((start, i))
                start = None
        if start is not None:
            intervals.append((start, df.index[-1]))
        return intervals

    result_df = pd.read_parquet(result_df_in.path)

    fig = go.Figure()
    plot_df = result_df.sample(frac=sample_fraction).sort_index()

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=-plot_df.neg_log_likelihood,
            mode="markers",
            name="Log Likelihood",
            marker=dict(size=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=-plot_df.smoothed_neg_log_likelihood,
            mode="lines",
            name="Log Likelihood smoothed",
            marker=dict(size=3),
        )
    )

    for start, end in get_intervals(result_df, "dataset", "test"):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="Orange",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    for start, end in get_intervals(result_df, label_col_name, 1):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="Red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title="Log Likelihood and Anomalies Over Time",
        xaxis_title="Time",
        yaxis_title="Values",
        yaxis=dict(range=[scatter_y_min, scatter_y_max]),
    )

    fig.write_html(result_viz.path)


@dsl.component(
    base_image="python:3.9",
)
def extract_composite_f1(
    metrics_dict: Dict,
) -> float:
    """Extracts composite F1 score from metrics"""
    return float(metrics_dict["composite_f1"])


@dsl.component(
    base_image="python:3.9",
)
def extract_scaler_path(scaler: Input[Model]) -> str:
    """Extracts scaler path because passing artifacts to nested pipeline is not yet implemented"""

    return scaler.uri


@dsl.component(
    base_image=f'{config["images"]["serving"]}',
    packages_to_install=["kubernetes==24.2.0", "mlflow", "boto3"],
)
def deploy_to_kserve(
    train_out_dict: Dict,
    mlflow_uri: str,
    minio_endpoint_url: str,
    namespace: str,
    service_name: str,
    mlflow_bucket: str,
    mlflow_experiment_name: str,
    device: str = "gpu",
    gpu_count: int = 1,
    runtime_version: str = "22.12-py3",
) -> Dict:
    """
    Deploys a PyTorch model to KServe using Triton Inference Server.

    TODO: in practice we would add a kserve transformer that applies
    the fitted scaler

    Args:
        train_out_dict: Dictionary with training output information
        mlflow_uri: URI to MLflow tracking server
        minio_endpoint_url: URL to MinIO endpoint
        namespace: Kubernetes namespace to deploy to
        service_name: Name for the InferenceService (defaults to run_name if not provided)
        device: "gpu" or "cpu"
        gpu_count: Number of GPUs to allocate (if device is "gpu")
        runtime_version: Triton runtime version

    Returns:
        Dictionary with deployment information
    """
    import os
    import mlflow
    from kubernetes import client, config
    import time
    import json

    # Configure MLflow and Minio
    mlflow.set_tracking_uri(mlflow_uri)
    os.environ["AWS_ENDPOINT_URL"] = minio_endpoint_url

    # Extract run_name from dictionary
    run_name = train_out_dict.get("run_name")
    if not run_name:
        raise ValueError("train_out_dict must contain a run_name")

    # Find the run_id using the run_name
    mlflow.set_experiment(mlflow_experiment_name)
    run_info = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'")
    if run_info.empty:
        raise ValueError(f"No run found with name {run_name}")

    run_id = run_info.iloc[0].run_id
    experiment_id = run_info.iloc[0].experiment_id

    print(f"Found run with ID: {run_id}, experiment ID: {experiment_id}")

    # Construct the S3 path to the triton_model artifact
    triton_model_uri = (
        f"s3://{mlflow_bucket}/{experiment_id}/{run_id}/artifacts/triton_model"
    )

    # Load Kubernetes configuration
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # Create the InferenceService manifest
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": service_name, "namespace": namespace},
        "spec": {
            "predictor": {
                "triton": {
                    "storageUri": triton_model_uri,
                    "runtimeVersion": runtime_version,
                    "env": [{"name": "OMP_NUM_THREADS", "value": "1"}],
                }
            }
        },
    }

    # Add GPU resources if requested
    if device.lower() == "gpu" and gpu_count > 0:
        inference_service["spec"]["predictor"]["triton"]["resources"] = {
            "limits": {"nvidia.com/gpu": gpu_count},
            "requests": {"nvidia.com/gpu": gpu_count},
        }

    # Create or update the InferenceService
    api_instance = client.CustomObjectsApi()

    try:
        # Check if service already exists
        try:
            existing_service = api_instance.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
            )
            print(f"Service {service_name} already exists, updating...")

            # Update the existing service
            response = api_instance.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
                body=inference_service,
            )
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Service doesn't exist, create it
                print(f"Creating new InferenceService {service_name}...")
                response = api_instance.create_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    body=inference_service,
                )
            else:
                raise e

        print(f"Successfully deployed/updated InferenceService '{service_name}'")

        # Wait for the service to be ready (maximum of 2 minutes)
        print("Waiting for service to be ready...")
        max_retries = 24  # 24 retries * 5 seconds = 2 minutes
        retries = 0
        is_ready = False

        while retries < max_retries and not is_ready:
            time.sleep(5)
            service = api_instance.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
            )

            status = service.get("status", {})
            conditions = status.get("conditions", [])

            for condition in conditions:
                if (
                    condition.get("type") == "Ready"
                    and condition.get("status") == "True"
                ):
                    is_ready = True
                    print(f"Service {service_name} is ready!")
                    break

            if not is_ready:
                print(f"Waiting for service to be ready ({retries+1}/{max_retries})...")
                retries += 1

        # Construct service URLs
        internal_url = f"http://{service_name}.{namespace}.svc.cluster.local"
        service_hostname = service.get("status", {}).get("url", "")
        if service_hostname:
            # Format may vary depending on your ingress setup
            external_url = f"https://$DOMAIN/serving/{namespace}/{service_name}"
        else:
            external_url = "Not available"

        # Return result
        return {
            "status": "deployed" if is_ready else "pending",
            "service_name": service_name,
            "namespace": namespace,
            "triton_model_uri": triton_model_uri,
            "internal_url": internal_url,
            "external_url": external_url,
            "run_id": run_id,
            "run_name": run_name,
        }

    except Exception as e:
        print(f"Error deploying InferenceService: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "service_name": service_name,
            "namespace": namespace,
            "run_id": run_id,
            "run_name": run_name,
        }
