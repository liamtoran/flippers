"""Implements the snorkel library label model."""


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

    Like its snorkel library counterpart assumes that the labeling
    functions are independent conditionally to Y, similar to a naive
    Bayes assumption.

    However, good results can also be observed in practice for
    correlated LFs.

    See the following link[] for more information on how to use this
    model and a comparison with the Snorkel library's implementation.
    """

    def __init__(
        self,
        polarities: ListLike,
        cardinality: int = 0,
        class_balances: ListLike = [],
    ):
        """Initializes a SnorkelModel instance with the given configuration
        options.

        Parameters
        ----------
        polarities:
            List that maps weak labels to polarities, size n_weak.
        cardinality: int, optional
            Number of possible label values.

            If unspecified, it will be inferred from the maximum value in polarities.
        class_balances: ListLike, optional
            List specifying class balance prior for each possible class, size n_classes.

            Defaults to balanced classes prior.

        Example
        -------
        >>> polarities = [1, 0, 1, 1]
        >>> cardinality = 2
        >>> class_balances = [0.7, 0.3]
        >>> snorkel_model = SnorkelModel(polarities, cardinality, class_balances)
        >>> # Change device
        >>> snorkel_model.to("cuda")
        """
        nn.Module.__init__(self)
        _BaseModel.__init__(self, polarities, cardinality)

        self.Polarities = nn.Parameter(
            torch.Tensor(self.polarities_matrix), requires_grad=False
        )

        if not class_balances:
            class_balances = 1 / self.cardinality * np.ones(int(self.cardinality))
        self.class_balances = class_balances
        self.Balances = nn.Parameter(torch.Tensor(class_balances), requires_grad=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def _convert_L(self, L: MatrixLike) -> torch.Tensor:
        """Convert input L to binary tensor."""
        L = np.array(L)
        L = L > 0.5
        L = torch.tensor(L).to(torch.float)
        return L

    def fit(
        self,
        L: MatrixLike,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        prec_init: float = 0.7,
        weight_decay: float = 0,
        k: float = 0,
        verbose: bool = False,
        *_,
    ) -> None:
        """Train the Snorkel model on the given weak label matrix L.

        Parameters
        ----------
        L : MatrixLike
            Weak Label matrix of shape (num_samples, n_weak)
        learning_rate : float, optional, default: 1e-3
            Learning rate for the optimizer.
        num_epochs : int, optional, default: 50
            Number of epochs to train the model
        prec_init : float, optional, default: 0.7
            Initial value for precision

            Can be of shape (n_weak) to set precision for each LF.
        weight_decay : float, optional, default: 0
            Weight decay (L2 penalty) for the optimizer
        k : float, optional, default: 0
            Weight of class blance loss term.

            This term penalizes the model for predicting a class on the train set
            differently to its specified balance
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
        >>> snorkel_model.fit(
        ...     L,
        ...     learning_rate=1e-2,
        ...     num_epochs=10,
        ...     prec_init=0.7,
        ...     k=5e-3,
        ...     verbose=True
        ... )
        """

        self.num_samples = len(L)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_history = []

        # Deal with multiple types
        self.prec_init = prec_init
        self.Prec_init = torch.ones(self.n_weak).to(self.device) * prec_init

        # Convert L to binary tensor
        L = self._convert_L(L)
        L = L.to(self.device)

        # Gram matrix of L gives overlaps
        Overlaps = L.T @ L / self.num_samples
        Coverage = torch.diag(Overlaps)  # Overlaps with itself = coverage
        Overlaps, Coverage = Overlaps.to(self.device), Coverage.to(self.device)

        # mu(i, j) = P(L_i = 1 | Y = j)
        # Bayes theorem
        self.mu_init = (
            self.Polarities * (self.Prec_init * Coverage).view(-1, 1) / self.Balances
        ).float()
        self.mu = nn.Parameter(self.mu_init.clone())  # * np.random.random())

        # This will be changed when dealing with cliques the right way
        mask_overlap = torch.ones(self.n_weak, self.n_weak).to(self.device).int()
        mask_overlap = mask_overlap - torch.diag(torch.diag(mask_overlap))
        mask_overlap = mask_overlap.bool()

        optimizer = optim.AdamW(
            [x for x in self.parameters() if x.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if verbose:
            epoch_range = trange(self.num_epochs)
        else:
            epoch_range = range(self.num_epochs)

        for epoch in epoch_range:
            # Zero the gradients
            optimizer.zero_grad()
            self.mu.data = self.mu.clamp(1e-6, 1 - 1e-6)

            # Forward
            # Calculate loss
            loss_overlap = torch.norm(
                (Overlaps - (self.mu * self.Balances) @ self.mu.t())[mask_overlap]
            )

            # mu * class_balance = precision * coverage
            # so if for precision ~ 1, we want mu * class_balance - coverage ~ 0
            # this term of the loss is trying to maximize the precision of
            # the input weak labeler.
            loss_coverage = torch.norm(self.mu @ self.Balances - Coverage)

            loss = loss_overlap**2 + loss_coverage**2
            if k > 0:
                loss_prediction = torch.norm(
                    self.Balances - self._predict_proba_tensor(L).mean(0)
                )
                loss = loss + k * loss_prediction**2

            # Backward
            loss.backward()

            # Log loss_history
            self.loss_history.append(loss.item())

            # Optimizer Step
            optimizer.step()

            # Scheduler step
            # TODO

        # Clamp mu
        self.mu.data = self.mu.clamp(1e-6, 1 - 1e-6)

        # Break permutation symetry in case class balance has non unique values
        # TODO

        # Eval mode so predict_proba doesnt calculate gradients
        self.eval()
        self.to("cpu")

    def _predict_proba_tensor(
        self, L: torch.tensor, ignore_abstains: bool = False
    ) -> torch.tensor:
        # Computing trick :
        # exp(L @ log(mu))[i, k] = Product of mu[j, k] for j / L[i, j] = 1
        Log_likelihood = L @ self.mu.log()
        if not ignore_abstains:
            # Include likelihood of abstains
            Log_likelihood = Log_likelihood + ((1 - L) @ (1 - self.mu).log())

        # Update prior
        Proba = Log_likelihood.exp() * self.Balances
        Proba = Proba / Proba.sum(1).view(-1, 1)
        return Proba

    def predict_proba(self, L: MatrixLike, ignore_abstains: bool = False) -> np.ndarray:
        """Predicts the probabilities of the classes by updating the prior
        using the learned parameter mu as posteriors.

        Parameters
        ----------
        L : MatrixLike
            Weak Label matrix
        ignore_abstains : bool, optional
            Whether to ignore abstains in the prior update:

            - False (default), using both votes and abstains. This helps leveraging
            information gained from knowing which labeling function abstained.

            - True, updates prior only using non abstained votes


        Returns
        -------
        numpy.ndarray
            An array of predicted probabilities of shape (num_samples, num_classes).

        Example
        -------
        >>> L = [
        ...     [1, 0, 1, 1],
        ...     [0, 1, 0, 1],
        ...     [1, 0, 1, 0]
        ... ]
        >>> proba = snorkel_model.predict_proba(L)
        >>> # proba.shape = (len(L), cardinality)
        """
        L = self._convert_L(L).float()

        with torch.no_grad():
            Proba = self._predict_proba_tensor(L, ignore_abstains)

        proba = Proba.detach().cpu().numpy()

        return proba
