import pytorch_lightning as pl
from pytorch_lightning import Callback
from datetime import datetime, timezone
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from loguru import logger


class StdOutLoggerCallback(Callback):
    """
    Callback for logging training and validation loss to standard output.

    This callback is designed for use with Katib hyperparameter tuning.
    It prints the training and validation loss metrics in a way that Katib
    can parse from the logs to understand the objective metric.
    """

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        if "train_loss" in metrics and "val_loss" in metrics:
            # Get the current time and format it as a string
            now = datetime.now()
            timestamp = (
                now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )

            print(f"\nepoch {epoch}:")
            print(f"{timestamp} train_loss={metrics['train_loss'].item():.4f}")
            # print(f"{timestamp} train_recon_loss={metrics['train_recon_loss'].item():.4f}")
            # print(f"{timestamp} train_kl_div={metrics['train_kl_div'].item():.4f}")
            print(f"{timestamp} val_loss={metrics['val_loss'].item():.4f}")
            # print(f"{timestamp} val_recon_loss={metrics['val_recon_loss'].item():.4f}")
            # print(f"{timestamp} val_kl_div={metrics['val_kl_div'].item():.4f}")
            #


class ResetLogVarCallback(Callback):
    """
    A callback to reset the log_var_x parameter to a small value after each batch for a specified number of epochs.
    This is done to maintain a reasonable level of variance in the output distribution during the initial training epochs.

    Attributes:
        reset_epochs (int): The number of epochs for which log_var_x will be reset after each batch.
        reset_value (float): The value to which log_var_x will be reset.
    """

    def __init__(self, reset_epochs: int, reset_value: float = -2.0):
        self.reset_epochs = reset_epochs
        self.reset_value = reset_value

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        unused=0,
    ) -> None:
        """
        Called at the end of the training batch.
        """
        current_epoch = trainer.current_epoch
        if current_epoch < self.reset_epochs:
            pl_module.log_var_x.data.fill_(self.reset_value)

    def on_validation_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        current_epoch = trainer.current_epoch
        if current_epoch == self.reset_epochs:
            logger.info("Starting to fit variance")
