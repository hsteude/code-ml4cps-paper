import pytorch_lightning as pl
from pytorch_lightning import Callback
from datetime import datetime, timezone


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
            print(f"{timestamp} train_recon_loss={metrics['train_recon_loss'].item():.4f}")
            print(f"{timestamp} train_kl_div={metrics['train_kl_div'].item():.4f}")
            print(f"{timestamp} val_loss={metrics['val_loss'].item():.4f}")
            print(f"{timestamp} val_recon_loss={metrics['val_recon_loss'].item():.4f}")
            print(f"{timestamp} val_kl_div={metrics['val_kl_div'].item():.4f}")
