import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class TimeStampVAE(pl.LightningModule):
    """
    A Variational Autoencoder (VAE) implemented using PyTorch Lightning.

    Args:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space.
        hidden_dims (int): Dimension of the hidden layers.
        beta (float): Weight for the KL divergence in the loss function.
        lr (float): Learning rate for the optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: int = 400,
        beta: float = 1.0,
        lr: float = 0.001,
    ):
        super().__init__()
        self.beta = beta
        self.lr = lr

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dim * 2),
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        mean, log_var = z.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return recon_x, mean, log_var

    def step(self, batch: torch.Tensor) -> tuple:
        x = batch
        recon_x, mean, log_var = self.forward(x)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + self.beta * kl_div
        return loss, recon_loss, kl_div

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, recon_loss, kl_div = self.step(batch)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_div", kl_div)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, recon_loss, kl_div = self.step(batch)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_div", kl_div)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
