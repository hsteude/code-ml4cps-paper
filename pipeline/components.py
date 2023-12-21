from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact, HTML
from typing import Dict, List, Optional
import toml

# load config
with open("config.toml", "r") as f:
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
        image=f'{config["images"]["eclss-ad-image"]}:commit-c2f04ede',
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
):
    """Kubeflow pipeline component for Dask preprocessing"""
    return dsl.ContainerSpec(
        image=f'{config["images"]["eclss-ad-image"]}:commit-0a2239b9',
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
        ],
    )


@dsl.component(
    packages_to_install=["pyarrow", "pandas", "xlrd"], base_image="python:3.9"
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
    df_label = pd.read_excel(df_label_in.path)

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
        scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index
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
) -> Dict:
    import time
    from kubernetes import client, config
    from loguru import logger

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
        "--hidden-dims=100",
        "--beta=${trialParameters.beta}",
        "--lr=${trialParameters.learningRate}",
        "--early-stopping-patience=30",
        f"--max-epochs={max_epochs}",
        "--num-gpu-nodes=1",
        "--run-as-pytorchjob=False",
        "--model-output-file=local_test_model",
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
            "algorithm": {"algorithmName": "bayesianoptimization"},
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
