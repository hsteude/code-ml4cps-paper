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
            nn.Linear(hidden_dims, latent_dim * 2),  # latent_dim for mean and std
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim * 2),
        )
        self.log_var_x = nn.Parameter(torch.full((input_dim,), -2.0))

    def encode(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        mean, log_var = z.chunk(2, dim=-1)
        log_var = F.softplus(log_var)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> tuple:
        x = self.decoder(z)
        mean, log_var = x.chunk(2, dim=-1)
        log_var = F.softplus(log_var)
        return mean, log_var

    def negative_log_likelihood(
        self,
        mean_x: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp(0.5 * self.log_var_x)
        dist = torch.distributions.Normal(mean_x, scale)
        return -dist.log_prob(x).sum()

    def step(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch
        mean_z, log_var_z = self.encode(x)
        z = self.reparameterize(mean_z, log_var_z)
        mean_x, log_var_x = self.decode(z)
        neg_log_likelihood = self.negative_log_likelihood(mean_x, x)
        kl_div = -0.5 * torch.sum(1 + log_var_z - mean_z.pow(2) - log_var_z.exp())
        loss = neg_log_likelihood + self.beta * kl_div
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        loss = self.step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
