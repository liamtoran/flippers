"""Groups basic generative models."""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from .._core import _WeakLabelInfo
from .._typing import ListLike, MatrixLike


class _BaseModel(_WeakLabelInfo, ABC):
    """Create a Model object."""

    def __init__(
        self,
        polarities: ListLike,
        cardinality: int = 0,
        names: ListLike = [],
    ):
        """
        Parameters
        ----------
        polarities:
            List that maps weak labels to polarities, shape n_weak.
        cardinality:
            Number of possible label values.

            If unspecified, it will be inferred from the maximum value in polarities.
        names:
            List of names for the different weak labels, shape n_weak.
        """
        ABC.__init__(self)
        _WeakLabelInfo.__init__(self, polarities, cardinality, names)

    @abstractmethod
    def predict_proba(self, L: MatrixLike) -> np.ndarray:
        """Predict probabilites for the given weak label matrix.

        Parameters
        ----------
        L:
            Weak label dataframe.

            Shape: (n_samples, n_weak)

        Returns
        -------
            Array of predicted probabilities of shape (len(L), cardinality)
        """
        pass

    def predict(self, L: MatrixLike, strategy: str = "majority") -> np.ndarray:
        """Predict labels for the given weak label matrix using the specified
        strategy.

        Parameters
        ----------
        L
            Weak label dataframe.

            Shape: (n_samples, n_weak)
        strategy
            Prediction strategy to use. Supported values: majority, probability.

            Controls how labels are predicted from the predicted probabilites.

            - majority: Predict the label with the highest number of votes.

            - probability: Predict label j with probability proba[i, j].

              This can be useful to enforce specific class_balances in the predictions.

            Default is "majority".

            If there are no votes for a sample, will predict -1.

        Returns
        -------
            1-D array of predicted labels of size `len(L)`
        """
        proba = self.predict_proba(L)
        unlabeled = proba.sum(axis=1) == 0
        if strategy == "majority":
            # Predict -1 if no votes were cast for all categories
            # Else predict the majority
            predictions = np.where(unlabeled, -1, proba.argmax(axis=1))
        elif strategy == "probability":
            # Predict -1 if no votes were cast for all categories
            # Else use probability matching / Thompson sampling
            filled_proba = proba.copy()
            filled_proba[unlabeled] = 1 / self.cardinality
            predictions = np.apply_along_axis(
                lambda x: np.random.choice(self.cardinality, p=x),
                axis=1,
                arr=filled_proba,
            )
            predictions = np.where(unlabeled, -1, predictions)
        else:
            raise ValueError('strategy should be "majority" or "probability"')

        return predictions

    def _get_votes(self, L: MatrixLike) -> np.ndarray:
        """Compute the sum of votes for each label based on the given weak
        label matrix.

        Parameters
        ----------
        L : pd.DataFrame
            Weak label dataframe.

            Shape: (n_samples, n_weak)

        Returns
        -------
        votes : np.ndarray
            2-D array where votes[i.j] = sum votes for class j for sample i.

            Shape: (n_samples, n_classes)
        """
        L = np.array(L)
        votes = L @ self.polarities_matrix
        return votes

    def _normalize_preds(self, preds: np.ndarray) -> np.ndarray:
        # Normalize votes per row
        row_wise_sum = preds.sum(axis=1)
        # In case there were no votes for this row, no need to renormalize
        row_wise_sum = np.where(row_wise_sum == 0, 1, row_wise_sum).reshape(-1, 1)
        proba = preds / row_wise_sum
        return proba


class Voter(_BaseModel):
    """Basic model that bases its decisions on the sum of votes for each
    class."""

    def fit(self) -> None:
        """Voter model does not require fitting."""
        warnings.warn("Voter object does not need to be fitted", category=UserWarning)

    def predict_proba(self, L: MatrixLike) -> np.ndarray:
        """Predict probabilities using sum of votes.

        Parameters
        ----------
        L : pd.DataFrame
            Weak label dataframe.

        Returns
        -------
            Array of predicted probabilities of shape (len(L), cardinality)
        """

        votes = self._get_votes(L)
        proba = self._normalize_preds(votes)
        return proba


class BalancedVoter(_BaseModel):
    """Basic model that bases its decisions on a weighted sum of votes for each
    class.

    The weights are computed during fitting so the sum of votes over
    training matches the given class balance.
    """

    def fit(self, L: MatrixLike, class_balances: ListLike = []) -> None:
        """Fit the BalancedMajorityVoter model.

        Parameters
        ----------
        L
            Weak label dataframe.
        class_balances
            Numpy array of shape cardinality giving a weight to each class.

            By default, assumes all classes are equally likely.
        """
        if not class_balances:
            class_balances = np.ones(self.cardinality)
        class_balances = np.array(class_balances)
        self.class_balances = class_balances
        self.normalized_class_balances = self.class_balances / self.class_balances.sum()

        # Learn votes weights per class
        # vote weights is the factor that will reweigh the predicted probabilites

        # Compute the product between the weak label matrix and the polarities matrix
        # To get the sum of votes per label
        votes = self._get_votes(L)

        # Mean votes per class
        votes_mean = votes.mean(axis=0)
        # Deal with the case where there are no votes for a class
        no_votes = np.where(votes_mean == 0)[0]
        if no_votes.size > 0:
            for i in no_votes:
                warnings.warn(
                    (
                        f"No votes were given to {self.names[i]},"
                        "assuming its balance is correct which is likely wrong"
                    ),
                    UserWarning,
                )
            votes_mean[no_votes] = self.normalized_class_balances[no_votes]
        votes_weights = self.normalized_class_balances / votes_mean
        self.votes_weights = votes_weights

    def predict_proba(self, L: MatrixLike) -> np.ndarray:
        """Predict probabilites using weighted voting.

        Parameters
        ----------
        L : pd.DataFrame
            Weak label dataframe.

        Returns
        -------
            Array of predicted probabilities of shape (len(L), cardinality)
        """
        votes = self._get_votes(L)
        weighted_votes = votes * self.votes_weights
        proba = self._normalize_preds(weighted_votes)
        return proba
