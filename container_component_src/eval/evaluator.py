from container_component_src.utils import read_data_from_minio
from sklearn.metrics import precision_score, recall_score
from container_component_src.model.datamodule import TimeStampDataModule
from container_component_src.model.lightning_module import TimeStampVAE
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Union, Dict, List
import pandas as pd
import numpy as np
import mlflow
import os


class ModelEvaluator:
    """
    A class for evaluating a machine learning model on validation and test datasets using MLflow.
    """

    def __init__(
        self,
        val_df_path: str,
        test_df_path: str,
        run_name: str,
        mlflow_uri: str,
        mlflow_experiment_name: str,
        minio_endpoint_url: str,
        label_col_name: str = "Anomaly",
        batch_size: int = 512,
        device: str = "cuda",
        threshold_min: int = -1500,
        threshold_max: int = 0,
        num_thresholds: int = 500,
    ) -> None:
        # Setup MLflow
        os.environ["AWS_ENDPOINT_URL"] = minio_endpoint_url
        mlflow.set_tracking_uri(uri=mlflow_uri)
        
        self.val_df = pd.read_parquet(val_df_path)
        self.test_df = pd.read_parquet(test_df_path)
        self.run_name = run_name
        self.batch_size = batch_size
        self.label_col_name = label_col_name
        self.device = device
        self.mlflow_experiment_name = mlflow_experiment_name
        self.threshold_range = np.linspace(
            start=threshold_min, stop=threshold_max, num=num_thresholds
        )
        
        self.model = self._load_model()
        self.val_data_loader, self.test_data_loader = self._create_data_loaders()

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates data loaders for the validation and test datasets.
        """
        datamodule = TimeStampDataModule(
            train_df=self.val_df,
            val_df=self.val_df,
            test_df=self.test_df,
            batch_size=self.batch_size,
            label_col_name=self.label_col_name,
        )
        datamodule.setup()
        val_data_loader = datamodule.val_dataloader()
        test_data_loader = datamodule.test_dataloader()
        return val_data_loader, test_data_loader

    def _load_model(self):
        """
        Loads the model checkpoint directly using MLflow's load_checkpoint function.
        """
        # Find the run ID based on run name
        mlflow.set_experiment(self.mlflow_experiment_name)
        run = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{self.run_name}'")
        if run.empty:
            raise ValueError(f"Run with name '{self.run_name}' not found")
        
        run_id = run.iloc[0].run_id
        
        # Load the model directly using MLflow's load_checkpoint
        model = mlflow.pytorch.load_checkpoint(
            TimeStampVAE,
            run_id=run_id,
        ).to(self.device)
        
        return model

    def _infer_model_on_dataloader(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs inference on the data loaded by the given data loader.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Initialization of lists to store the results
        (
            mean_z_list,
            log_var_z_list,
            mean_x_list,
            log_var_x_list,
            neg_log_likelihood_list,
        ) = ([], [], [], [], [])

        for batch in data_loader:
            batch = batch.to(device)
            mean_z, log_var_z, mean_x, log_var_x, neg_log_likelihood = self.model.infer(
                batch
            )

            mean_z_list.append(mean_z.detach())
            log_var_z_list.append(log_var_z.detach())
            mean_x_list.append(mean_x.detach())
            log_var_x_list.append(log_var_x.detach())
            neg_log_likelihood_list.append(neg_log_likelihood.detach())

        # Concatenating results from all batches
        mean_z_tensor = torch.cat(mean_z_list, dim=0).cpu().numpy()
        log_var_z_tensor = torch.cat(log_var_z_list, dim=0).cpu().numpy()
        mean_x_tensor = torch.cat(mean_x_list, dim=0).cpu().numpy()
        log_var_x_tensor = torch.cat(log_var_x_list, dim=0).cpu().numpy()
        neg_log_likelihood_tensor = (
            torch.cat(neg_log_likelihood_list, dim=0).cpu().numpy()
        )

        return (
            mean_z_tensor,
            log_var_z_tensor,
            mean_x_tensor,
            log_var_x_tensor,
            neg_log_likelihood_tensor,
        )

    def _prepare_result_df(
        self,
        test_neg_log_likelihood: np.ndarray,
        val_neg_log_likelihood: np.ndarray,
        test_mean_z: np.ndarray,
        val_mean_z: np.ndarray,
        test_log_var_z: np.ndarray,
        val_log_var_z: np.ndarray,
    ) -> pd.DataFrame:
        """
        Prepares a DataFrame containing the results of the model evaluation.
        """
        test_results_df = pd.DataFrame(
            {
                "neg_log_likelihood": test_neg_log_likelihood,
                "dataset": "test",
                self.label_col_name: self.test_df[self.label_col_name],
            },
            index=self.test_df.index,
        )
        for i in range(test_mean_z.shape[1]):
            test_results_df[f"mean_z_{i}"] = test_mean_z[:, i]
            test_results_df[f"log_var_z_{i}"] = test_log_var_z[:, i]

        # Concatenating results from validation and test datasets
        val_results_df = pd.DataFrame(
            {
                "neg_log_likelihood": val_neg_log_likelihood,
                "dataset": "validation",
                self.label_col_name: 0,
            },
            index=self.val_df.index,
        )
        for i in range(val_mean_z.shape[1]):
            val_results_df[f"mean_z_{i}"] = val_mean_z[:, i]
            val_results_df[f"log_var_z_{i}"] = val_log_var_z[:, i]
        results_df = pd.concat([test_results_df, val_results_df]).sort_index()

        # Smoothing the negative log-likelihood
        results_df[
            "smoothed_neg_log_likelihood"
        ] = results_df.neg_log_likelihood.rolling(window=100, min_periods=1).median()
        return results_df

    def _get_intervals(
        self, df: pd.DataFrame, column: str, value: Union[str, int, float]
    ) -> list:
        """
        Identifies intervals in the DataFrame where the specified column equals the given value.
        """
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

    def _calculate_metrics(
        self, result_df: pd.DataFrame, threshold: float, label_intervals: List
    ) -> dict:
        """
        Calculates point-wise precision, event-wise recall, and composite F1 score.
        """
        # Generate point-wise predictions based on the threshold
        result_df["prediction_point"] = (
            -result_df["smoothed_neg_log_likelihood"] < threshold
        ).astype(int)

        # Generate event-wise predictions
        result_df["prediction_event"] = result_df["prediction_point"]
        for start, end in label_intervals:
            if result_df.loc[start:end, "prediction_point"].any():
                result_df.loc[start:end, "prediction_event"] = 1

        # Calculate point-wise precision
        point_precision = precision_score(
            result_df[self.label_col_name], result_df["prediction_point"]
        )

        # Calculate event-wise recall
        event_recall = recall_score(
            result_df[self.label_col_name], result_df["prediction_event"]
        )

        # Calculate composite F1 score
        composite_f1 = (
            (2 * point_precision * event_recall) / (point_precision + event_recall)
            if (point_precision + event_recall) > 0
            else 0
        )

        return {
            "point_precision": point_precision,
            "event_recall": event_recall,
            "composite_f1": composite_f1,
        }

    def _find_optimal_threshold(
        self,
        df: pd.DataFrame,
        neg_log_likelihood_column: str,
        threshold_range: np.ndarray,
        label_intervals: List,
    ) -> Tuple[Dict, float]:
        """
        Finds the threshold that maximizes the composite F1 score.
        """
        best_threshold = threshold_range[0]
        best_composite_f1 = 0
        best_metrics = None

        for threshold in threshold_range:
            metrics = self._calculate_metrics(df, threshold, label_intervals)
            if metrics["composite_f1"] > best_composite_f1:
                best_composite_f1 = metrics["composite_f1"]
                best_threshold = threshold
                best_metrics = metrics

        return best_metrics, best_threshold

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Runs the model evaluation process.
        """
        val_outputs = self._infer_model_on_dataloader(self.val_data_loader)
        test_outputs = self._infer_model_on_dataloader(self.test_data_loader)

        # Unpacking the outputs
        (
            val_mean_z,
            val_log_var_z,
            val_mean_x,
            val_log_var_x,
            val_neg_log_likelihood,
        ) = val_outputs
        (
            test_mean_z,
            test_log_var_z,
            test_mean_x,
            test_log_var_x,
            test_neg_log_likelihood,
        ) = test_outputs

        # Preparing the result DataFrame
        result_df = self._prepare_result_df(
            test_neg_log_likelihood=test_neg_log_likelihood,
            val_neg_log_likelihood=val_neg_log_likelihood,
            test_mean_z=test_mean_z,
            val_mean_z=val_mean_z,
            test_log_var_z=test_log_var_z,
            val_log_var_z=val_log_var_z,
        )
        label_intervals = self._get_intervals(result_df, self.label_col_name, 1)
        metrics, best_threshold = self._find_optimal_threshold(
            df=result_df,
            neg_log_likelihood_column="smoothed_neg_log_likelihood",
            threshold_range=self.threshold_range,
            label_intervals=label_intervals,
        )
        metrics.update(dict(best_threshold=best_threshold))
        return result_df, metrics
