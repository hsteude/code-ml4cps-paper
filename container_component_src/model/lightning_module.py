import pytorch_lightning as pl
import torch
from torch import nn
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Tuple, List
import math


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
        self.likelihood_mse_mixing_factor = likelihood_mse_mixing_factor if likelihood_mse_mixing_factor else 1

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        dist = Normal(loc=mean_x, scale=scale)
        neg_log_likelihood = -dist.log_prob(x).sum(dim=1)
        return mean_z, log_var_z, mean_x, self.log_var_x, neg_log_likelihood

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, _, _, _, neg_log_likelihood = self.infer(batch)
        return neg_log_likelihood

    def custom_negative_log_likelihood(
        self,
        mean_x: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp(0.5 * self.log_var_x) + 1e-30
        dist = Normal(loc=mean_x, scale=scale)
        log_probs = dist.log_prob(x).sum(dim=1)
        return nn.MSELoss()(x, mean_x) - 1e-5 * log_probs.mean()

    def loss_function(
        self,
        x: torch.Tensor,
        mean_z: torch.Tensor,
        log_var_z: torch.Tensor,
        mean_x: torch.Tensor,
    ):
        kl_div = -0.5 * torch.sum(1 + log_var_z - mean_z.pow(2) - log_var_z.exp())
        neg_log_likelihood = self.custom_negative_log_likelihood(mean_x, x)
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


class Normal:
    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        resized_ = torch.broadcast_tensors(loc, scale)
        self.loc = resized_[0]
        self.scale = resized_[1]
        self._batch_shape = list(self.loc.size())

    def _extended_shape(self, sample_shape: List[int]) -> List[int]:
        return sample_shape + self._batch_shape

    def sample(self, sample_shape: List[int]) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape: List[int]) -> torch.Tensor:
        shape: List[int] = self._extended_shape(sample_shape)
        eps = torch.normal(
            torch.zeros(shape, device=self.loc.device),
            torch.ones(shape, device=self.scale.device),
        )
        return self.loc + eps * self.scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        var = self.scale**2
        log_scale = self.scale.log()
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def entropy(self) -> torch.Tensor:
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
