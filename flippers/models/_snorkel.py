"""Implements the snorkel library label model.

See (link library)
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from .._typing import ListLike, MatrixLike
from ._base import _BaseModel


class SnorkelModel(nn.Module, _BaseModel):
    """A label model implementation for weak supervision based on a generative
    approach.

    This implementation is based on the Snorkel library's label model.

    Like its snorkel library counterpart assumes
    that the labeling functions are independent,
    similar to a naive Bayes assumption.

    However, good results can also be observed in practice for correlated LFs.

    See the following link[] for more information on how to use this model and
    a comparison with the Snorkel library's implementation.
    """

    def __init__(
        self,
        polarities: ListLike,
        cardinality: int = 0,
        names: ListLike = [],
        class_balances: ListLike = [],
    ):
        """Initializes a SnorkelModel instance with the given configuration
        options.

        {0}
        """.format(
            _BaseModel.__doc__
        )
        nn.Module.__init__(self)
        _BaseModel.__init__(self, polarities, cardinality, names)

        self.Polarities = torch.Tensor(self.polarities_matrix)
        if not class_balances:
            class_balances = 1 / self.cardinality * np.ones(self.cardinality)
        self.class_balances = class_balances
        self.Balances = torch.Tensor(class_balances)

    def _convert_L(self, L: MatrixLike) -> torch.Tensor:
        """Convert input L to binary tensor."""
        L = np.array(L)
        L = (L > 0.5).astype(float)
        L = torch.tensor(L)
        return L

    def fit(
        self,
        L: MatrixLike,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        prec_init: float = 0.7,
        weight_decay: float = 0,
        mu_init_l2_penalty: float = 0,
    ) -> None:
        """Train the Snorkel model."""
        self.num_samples = len(L)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_history = []

        # Convert to tensor astype float
        self.mu_init_l2_penalty = mu_init_l2_penalty
        self.L2 = mu_init_l2_penalty * torch.ones(self.n_weak)

        # Deal with multiple types
        self.prec_init = prec_init
        self.Prec_init = torch.ones(self.n_weak) * prec_init

        # Convert L to binary tensor
        L = self._convert_L(L)

        # Gram matrix of L gives overlaps
        Overlaps = L.T @ L / self.num_samples
        Coverage = torch.diag(Overlaps)  # Overlaps with itself = coverage
        self.Coverage = Coverage

        # mu(i, j) = P(L_i = 1 | Y = j)
        # Bayes theorem
        self.mu_init = (
            self.Polarities * (self.Prec_init * Coverage).view(-1, 1) / self.Balances
        ).float()
        self.mu = nn.Parameter(self.mu_init.clone())  # * np.random.random())

        # This will be changed when dealing with cliques the right way
        mask = torch.ones(self.n_weak, self.n_weak)
        mask = mask - torch.diag(torch.diag(mask))
        mask = mask.bool()

        optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0,
        )

        for epoch in trange(self.num_epochs):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward
            # Calculate loss
            loss_overlap = torch.norm(
                (Overlaps - (self.mu * self.Balances) @ self.mu.t())[mask]
            )

            # mu * class_balance = precision * coverage
            # so if for precision ~ 1, we want mu * class_balance - coverage ~ 0
            # this term of the loss is trying to maximize the precision of
            # the input weak labeler.
            loss_coverage = torch.norm(self.mu @ self.Balances - Coverage)

            loss_mu_init_l2 = torch.norm(
                self.mu_init_l2_penalty * (self.mu - self.mu_init)
            )

            loss = loss_overlap**2 + loss_coverage**2 + loss_mu_init_l2**2

            # Backward
            loss.backward()

            # Log loss_history
            self.loss_history.append(loss.item())

            # Optimizer Step
            optimizer.step()

            # Scheduler step

        # Clamp mu
        self.mu.data = self.mu.clamp(1e-6, 1 - 1e-6)

        # Break permutation symetry in case class balance has non unique values
        # TODO

        # Eval mode so predict_proba doesnt calculate gradients
        self.eval()

    def predict_proba(self, L: MatrixLike, prior_update: str = "all") -> np.ndarray:
        """Predicts the probabilities of the classes by updating the prior
        using the learned parameter mu as posteriors.

        Parameters
        ----------
        L : MatrixLike
            Weak Label matrix
        prior_update : str, optional
            Prior update method. There are two options:
            - "all" (default): updates using both votes and abstains.
            - "ignore_abstains": updates using only votes and ignores abstains.


        Returns
        -------
        numpy.ndarray
            An array of predicted probabilities of shape (num_samples, num_classes).
        """
        L = self._convert_L(L).float()
        mu = self.mu.cpu().detach().float()

        # Computing trick :
        # exp(L @ log(mu))[i, k] = Product of mu[j, k] for j / L[i, j] = 1
        with torch.no_grad():
            if prior_update == "all":
                # This assumes the labeling functions are independent
                # Which is most likely not respected in real world use
                Likelihood_votes = (L @ mu.log()).exp()
                Likelihood_abstains = ((1 - L) @ (1 - mu).log()).exp()

                # Update prior
                Proba = (Likelihood_votes * Likelihood_abstains) * self.Balances
            elif prior_update == "ignore_abstains":
                # This is the prior_update used by the Snorkel library's label model
                # It does not count abstains or in the prior update
                Likelihood_votes = (L @ mu.log()).exp()
                Proba = Likelihood_votes * self.Balances

        proba = Proba.detach().cpu().numpy()

        # Normalize the outputs row wise so they sum to one.
        proba = self._normalize_preds(proba)

        return proba
