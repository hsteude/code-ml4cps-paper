from container_component_src.utils import read_data_from_minio, create_s3_client
from sklearn.metrics import precision_score, recall_score
from container_component_src.model.datamodule import TimeStampDataModule
from container_component_src.model.lightning_module import TimeStampVAE
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Union, Dict, List
import pandas as pd
import numpy as np


class ModelEvaluator:
    """
    A class for evaluating a machine learning model on validation and test datasets.

    Attributes:
        val_df_path (str): Path to the validation dataset.
        test_df_path (str): Path to the test dataset.
        model_path (str): Path to the trained model.
        label_col_name (str): Name of the column in the dataframe representing the labels.
        batch_size (int): Batch size for loading data.
        device (str): Device to use for computations (e.g., 'cuda' or 'cpu').
    """

    def __init__(
        self,
        val_df_path: str,
        test_df_path: str,
        model_path: str,
        label_col_name: str = "Anomaly",
        batch_size: int = 512,
        device: str = "cuda",
        threshold_min: int = -1500,
        threshold_max: int = 0,
        num_thresholds: int = 500,
    ) -> None:
        self.val_df = read_data_from_minio(val_df_path)
        self.test_df = read_data_from_minio(test_df_path)
        self.model_path = model_path
        self.batch_size = batch_size
        self.label_col_name = label_col_name
        self.val_data_loader, self.test_data_loader = self._create_data_loaders()
        self.device = device
        self.model = self._load_model()
        self.threshold_range = np.linspace(
            start=threshold_min, stop=threshold_max, num=num_thresholds
        )

    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates data loaders for the validation and test datasets.

        Returns:
            Tuple[DataLoader, DataLoader]: Data loaders for validation and test datasets.
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
        Loads the trained model from a specified path.

        Returns:
            The loaded model.
        """
        s3_client = create_s3_client()
        with s3_client.open(self.model_path, "rb") as f:
            model = TimeStampVAE.load_from_checkpoint(f).to(self.device)
        return model

    def _infer_model_on_dataloader(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs inference on the data loaded by the given data loader.

        Args:
            data_loader (DataLoader): DataLoader for the dataset on which to perform inference.

        Returns:
            Tuple: Tensors containing mean and log-variance in latent and data spaces, and negative log-likelihood.
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

        Args:
            test_neg_log_likelihood (np.ndarray): Array of negative log-likelihood values for the test dataset.
            val_neg_log_likelihood (np.ndarray): Array of negative log-likelihood values for the validation dataset.
            test_mean_z (np.ndarray): Array of means of p(z|x).
            val_mean_z (np.ndarray): Array of means of p(z|x).
            test_log_var_z (np.ndarray): Array of log-variance of p(z|x).
            val_log_var_z (np.ndarray): Array of log-variance of p(z|x).

        Returns:
            pd.DataFrame: DataFrame containing the results.
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

        Args:
            df (pd.DataFrame): DataFrame to search for intervals.
            column (str): Column name to look for the value.
            value (Union[str, int, float]): Value to identify intervals for.

        Returns:
            list: List of tuples representing start and end of each interval.
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
        Calculates point-wise precision, event-wise recall, and composite F1 score based on a given threshold.

        Args:
            threshold (float): Threshold for determining an anomaly based on negative log-likelihood.
            results_df (pd.DataFrame): Dataframe with likelihood and smoothed likelihood

        Returns:
            dict: A dictionary containing the metrics.
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

        Args:
            df (pd.DataFrame): DataFrame containing the true labels and negative log-likelihood values.
            neg_log_likelihood_column (str): Name of the column containing the negative log-likelihood values.
            threshold_range (np.ndarray): An array of threshold values to test.

        Returns:
            float: The threshold value that maximizes the composite F1 score.
        """
        best_threshold = threshold_range[0]
        best_composite_f1 = 0

        for threshold in threshold_range:
            metrics = self._calculate_metrics(df, threshold, label_intervals)
            if metrics["composite_f1"] > best_composite_f1:
                best_composite_f1 = metrics["composite_f1"]
                best_threshold = threshold

        return metrics, best_threshold

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Runs the model evaluation process.

        Returns:
            pd.DataFrame: DataFrame containing the results of the evaluation.
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
