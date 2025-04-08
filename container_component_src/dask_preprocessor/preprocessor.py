import dask.dataframe as dd
from dask_kubernetes.operator import KubeCluster
from dask.distributed import Client
import pandas as pd
from loguru import logger


class DaskPreprocessor:
    def __init__(
        self,
        num_dask_workers: int,
        image: str,
        namespace: str,
        storage_options: dict,
        sample_frac: float,
    ) -> None:
        """
        Initializes the DaskPreprocessor with the specified configuration.

        Parameters:
            num_dask_workers: The number of workers for the Dask cluster.
            image: The Docker image to use for the Dask workers.
            storage_options: A dictionary containing storage configuration for reading data.
            sample_frac: The fraction of data to sample.
        """
        self._create_dask_cluster(num_worker=num_dask_workers, image=image, namespace=namespace)
        self.minio_storage_options = storage_options
        self.sample_frac = sample_frac

    def _create_dask_cluster(self, num_worker: int, image: str, namespace: str) -> None:
        """
        Creates a Dask cluster and connects a Dask client to it.

        Parameters:
            num_worker: The number of workers to scale the cluster to.
            image: The Docker image to use for the Dask workers.
        """
        logger.info(f"Creating dask cluster")
        self.cluster = KubeCluster(name="preproc", image=image, namespace=namespace)
        self.cluster.scale(num_worker)
        self.client = Client(self.cluster)

    def _close_dask_cluster(self) -> None:
        """
        Closes the Dask cluster and the associated client.
        """
        logger.info(f"Closing dask cluster")
        self.client.close()
        self.cluster.close()

    def _read_dask_ddf(self, path_list: list[str], ts_col_name: str) -> dd.DataFrame:
        """
        Reads the specified parquet files into a Dask DataFrame.

        Parameters:
            path_list: A list of file paths to read.
            ts_col_name: The name of the column to use as the timestamp index.
        """
        logger.info(f"Reading these files {path_list}")
        ddf = dd.read_parquet(
            path_list,
            engine="pyarrow",
            sorted=True,
            index=ts_col_name,
            calculate_divisions=False,
            storage_options=self.minio_storage_options,
        )
        return ddf

    def _select_columns(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Selects columns with float data types from the Dask DataFrame.

        Parameters:
            ddf: The Dask DataFrame to process.
        """
        float_columns = [
            col for col, dtype in ddf.dtypes.items() if dtype in ["float32", "float64"]
        ]
        ddf_sel = ddf[float_columns]
        return ddf_sel

    def run(self, path_list: list[str], timestamp_col: str) -> pd.DataFrame:
        """
        Runs the preprocessing steps on the specified dataset and returns a Pandas DataFrame.

        Parameters:
            path_list: A list of file paths to process.
            timestamp_col: The name of the column to use as the timestamp index.

        Returns a Pandas DataFrame after preprocessing.
        """
        ddf = self._read_dask_ddf(path_list, ts_col_name=timestamp_col)
        ddf = self._select_columns(ddf)
        ddf = ddf.sample(frac=self.sample_frac, random_state=42)
        logger.info(f"Running dask computation")
        df = ddf.compute()
        self._close_dask_cluster()
        return df.sort_index()
