import pytorch_lightning as pl
import torch
from torch import nn
import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Tuple


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
        likelihood_mse_mixing_factor: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.lr = lr
        self.likelihood_mse_mixing_factor = likelihood_mse_mixing_factor

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dim * 2),  # latent_dim for mean and std
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim),
        )
        self.log_var_x = nn.Parameter(torch.full((input_dim,), 0.0))

    def encode(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        mean, log_var = z.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        mean_x = self.decoder(z)
        return mean_x

    def infer(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch
        mean_z, log_var_z = self.encode(x)
        # note: we deliberately go for the mean here during inference
        z = mean_z
        mean_x = self.decode(z)
        scale = torch.exp(0.5 * self.log_var_x)
        dist = MultivariateNormal(mean_x, scale_tril=torch.diag(scale))
        neg_log_likelihood = -dist.log_prob(x)
        return mean_z, log_var_z, mean_x, self.log_var_x, neg_log_likelihood

    def negative_log_likelihood(
        self,
        mean_x: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp(0.5 * self.log_var_x) + 1e-30
        dist = MultivariateNormal(mean_x, scale_tril=torch.diag(scale))
        return nn.MSELoss()(x, mean_x) - 1e-5 * dist.log_prob(x).mean()

    def loss_function(
        self,
        x: torch.Tensor,
        mean_z: torch.Tensor,
        log_var_z: torch.Tensor,
        mean_x: torch.Tensor,
    ):
        kl_div = -0.5 * torch.sum(1 + log_var_z - mean_z.pow(2) - log_var_z.exp())
        neg_log_likelihood = self.negative_log_likelihood(mean_x, x)
        mse_loss = nn.MSELoss()(x, mean_x)
        loss = (
            mse_loss
            + self.likelihood_mse_mixing_factor * neg_log_likelihood
            + self.beta * kl_div
        )
        return loss

    def step(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch
        mean_z, log_var_z = self.encode(x)
        z = self.reparameterize(mean_z, log_var_z)
        mean_x = self.decode(z)
        loss = self.loss_function(
            x=x, mean_z=mean_z, log_var_z=log_var_z, mean_x=mean_x
        )
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
