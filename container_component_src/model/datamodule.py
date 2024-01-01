import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Optional


class TimeStampDataset(Dataset):
    """
    A custom Dataset for loading time series data into PyTorch.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the time series data.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class TimeStampDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling time series data for training, validation, and testing.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        val_df (pd.DataFrame): DataFrame containing the validation data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        batch_size (int): Batch size for data loading.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label_col_name: Optional[str] = '',
        test_df: Optional[pd.DataFrame] = None,
        batch_size: int = 32,
        num_workers: Optional[int] = 0
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_col_name = label_col_name

    def setup(self, stage: str = None) -> None:
        self.train_dataset = TimeStampDataset(self.train_df)
        self.val_dataset = TimeStampDataset(self.val_df)
        if self.test_df is not None:
            self.test_dataset = TimeStampDataset(self.test_df.drop(columns=[self.label_col_name]))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_df is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        return None
