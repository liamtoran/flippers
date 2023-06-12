"""Implements a VAE to learn a latent parameter generating the weak label."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from flippers._typing import ListLike, MatrixLike
from flippers.models._base import _Model

torch.manual_seed(0)


class WeakLabelVAE(nn.Module, _Model):
    """A label model implementation for weak supervision based on a VAE."""

    def __init__(
        self,
        polarities: ListLike,
        class_balances: ListLike = [],
        latent_dim: int = 4,
    ):
        """Initializes a SnorkelModel instance with the given configuration
        options.

        Parameters
        ----------
        polarities:
            List that maps weak labels to polarities, size n_weak.

        class_balances: ListLike, optional
            List specifying class balance prior for each possible class, size n_classes.

            Defaults to balanced classes prior.

        Example
        -------
        >>> polarities = [1, 0, 1, 1]
        >>> class_balances = [0.7, 0.3]
        >>> snorkel_model = SnorkelModel(polarities, cardinality, class_balances)
        """
        nn.Module.__init__(self)
        self.cardinality = 2
        _Model.__init__(self, polarities, self.cardinality)

        self.Polarities = nn.Parameter(torch.Tensor(polarities), requires_grad=False)
        self.n_weak = len(polarities)
        self.latent_dim = latent_dim if latent_dim else self.n_weak // 2

        if not class_balances:
            class_balances = 1 / self.cardinality * np.ones(int(self.cardinality))
        self.class_balances = class_balances
        self.Balances = nn.Parameter(torch.Tensor(class_balances), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_weak, self.n_weak),
            nn.ReLU(),
            nn.Linear(self.n_weak, 2 * self.latent_dim + 1),
        )

        self.decoder_mu_true = nn.Sequential(
            nn.Linear(self.latent_dim + 1, self.n_weak),
            nn.ReLU(),
            nn.Linear(self.n_weak, self.n_weak),
            nn.Sigmoid(),
        )

        self.decoder_mu_false = nn.Sequential(
            nn.Linear(self.latent_dim + 1, self.n_weak),
            nn.ReLU(),
            nn.Linear(self.n_weak, self.n_weak),
            nn.Sigmoid(),
        )

        self.loss_history = []

    @property
    def device(self):
        return next(self.parameters()).device

    def _convert_L(self, L: MatrixLike) -> torch.Tensor:
        """Convert input L to binary tensor."""
        L = np.array(L)
        L = L > 0.5
        L = torch.tensor(L).to(torch.float)
        L = L.to(self.device)
        return L

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps * std

    def reparameterize_bool(self, p, eps=1e-8):
        U = torch.rand_like(p, device=self.device)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        noisy_logit = p + gumbel_noise
        return torch.sigmoid(noisy_logit)

    def forward(self, L: torch.Tensor):
        mu_z, logvar_z, p = self.encoder(L).split(
            (self.latent_dim, self.latent_dim, 1), dim=1
        )
        z = self.reparameterize(mu_z, logvar_z)
        y = self.reparameterize_bool(p)

        latents = torch.cat((z, y), dim=1)

        mu_true = self.decoder_mu_true(latents)
        mu_false = self.decoder_mu_false(latents)

        weak_reconstructed = (
            self.Polarities * y + (1 - self.Polarities) * (1 - y)
        ) * mu_true + (
            (1 - self.Polarities) * (y) + (self.Polarities) * (1 - y)
        ) * mu_false
        weak_reconstructed = weak_reconstructed.clamp(1e-8, 1 - 1e-8)

        return weak_reconstructed, mu_z, logvar_z, p, mu_true, mu_false

    def predict_proba(self, L):
        L = self._convert_L(L)

        with torch.no_grad():
            mu_z, logvar_z, p = self.encoder(L).split(
                (self.latent_dim, self.latent_dim, 1), dim=1
            )

            p = torch.sigmoid(p)

            p = torch.cat((1 - p, p), dim=1)

        p = p.detach().cpu().numpy()

        return p

    def loss(self, L, outputs, kld_weight, nudge, capacity):
        L_reconstructed, mu_z, logvar_z, p, mu_true, mu_false = outputs

        loss_L = (
            nn.functional.binary_cross_entropy(L_reconstructed, L, reduction="none")
            .mean(dim=1)
            .sum()
        )

        kl_z = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).mean(dim=1).sum()

        # Nudge from 0.5
        # Force mean to be class balance
        p = torch.sigmoid(p)
        p_mean = p.mean(dim=0)
        y = self.class_balances[1]
        kl_p = y * torch.log(y / p_mean) + (1 - y) * torch.log((1 - y) / (1 - p_mean))
        kl_p = (kl_p - capacity).relu()
        kl_p = kl_p * L.shape[0]

        # mu_false/mu_true should be small
        mu_ratio = mu_false / mu_true
        mu_ratio = mu_ratio.pow(2).mean(dim=1).sum()

        loss = loss_L + kld_weight * kl_z + nudge * kl_p + mu_ratio
        return loss

    def _get_dataloader(self, L, batch_size):
        class CustomDataset(Dataset):
            def __init__(self, L):
                self.L = L

            def __len__(self):
                return len(self.L)

            def __getitem__(self, idx):
                return self.L[idx]

        dataloader = DataLoader(
            CustomDataset(L), batch_size=batch_size, shuffle=True, drop_last=True
        )

        return dataloader

    def fit(
        self,
        L: MatrixLike,
        learning_rate: float = 1e-3,
        num_batches: int = 5000,
        batch_size: int = 32,
        weight_decay: float = 1e-3,
        kld_weight: float = 50,
        nudge: float = 1e-1,
        capacity=1e-1,
        verbose: bool = True,
        **_,
    ) -> None:
        """Train the Snorkel model on the given weak label matrix L.

        Parameters
        ----------
        L : MatrixLike
            Weak Label matrix of shape (num_samples, n_weak)
        learning_rate : float, optional, default: 1e-3
            Learning rate for the optimizer.
        num_batches : int, optional, default: 5000
            Number of batches to train the model
        weight_decay : float, optional, default: 0
            Weight decay (L2 penalty) for the optimizer
        kld_weight : float, optional, default: 0
            Weight of class blance loss term.

            This term penalizes the model for predicting a class on the train set
            differently to its specified balance
        device : str, optional, default: "cpu"
            Device to use for training, either "cpu" or "cuda"
        verbose : bool, optional, default: False
            If True, displays training progress

        Returns
        -------
        None

        Example
        -------
        >>> L = [
        ...     [1, 0, 1, 1],
        ...     [0, 1, 0, 1],
        ...     [1, 0, 1, 0]
        ... ]
        >>> model.fit(
        ...     L,
        ...     learning_rate=1e-2,
        ...     num_batches=3000,
        ...     prec_init=0.7,
        ...     k=5e-3,
        ...     device="cpu",
        ...     verbose=True
        ... )
        """

        # Convert L to binary tensor
        L = self._convert_L(L)

        optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        num_epochs = num_batches * batch_size // L.shape[0] + 1

        epoch_range = trange(num_epochs) if verbose else range(num_epochs)

        dataloader = self._get_dataloader(L, batch_size)

        for epoch in epoch_range:
            epoch_loss = 0
            for batch_L in dataloader:
                # Zero the gradients
                optimizer.zero_grad()

                batch_L = batch_L.to(self.device)

                outputs = self(batch_L)
                loss = self.loss(
                    batch_L,
                    outputs,
                    kld_weight=kld_weight,
                    nudge=nudge,
                    capacity=capacity,
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                epoch_loss /= len(dataloader)
                epoch_range.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                description = {}
                # Log loss_history
                self.loss_history.append(epoch_loss)
                description["Loss"] = f"{epoch_loss:.1f}"
                epoch_range.set_postfix(description)
        self.eval()
