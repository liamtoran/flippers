import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class _WeakLabelInfo:
    """Collects information about the weak labelers."""

    def __init__(
        self,
        polarities: Union[np.ndarray, List],
        cardinality: int = 0,
        names: List = [],
    ):
        """
        Parameters
        ----------
        polarities
            List that maps weak labels to polarities.
        cardinality
            Number of classes

            If not specified, it will be inferred from the maximum value in polarities.
        names
            List of names for the different weak labels.
        """
        # Set polarities
        if isinstance(polarities, list):
            self.polarities = np.array(polarities)
        elif isinstance(polarities, np.ndarray):
            self.polarities = polarities
        else:
            ValueError("Input polaritiy is not a list or NumPy array")

        # Set cardinality
        if not cardinality:
            # Infer the number of categories from the maximum value of polarities
            cardinality = max(self.polarities) + 1
            print(f"No cardinality given, assuming it is equal to {cardinality}")
        self.cardinality = cardinality

        # Set names
        if not names:
            names = [f"weak_label_{i}" for i in range(len(self.polarities))]
        self.names = names

        # Validate inputs work
        self.__validate_init__()

        self.polarities_matrix = self._polarities_to_matrix()

    def __validate_init__(self):
        """Assert whether the input data is valid."""
        # Check more than one class
        assert self.cardinality > 1

        # Check no polarities outside of cardinality
        assert self.polarities.max() <= self.cardinality - 1
        assert self.polarities.min() >= 0

        # Check names is the same size as polarities
        assert len(self.names) == len(self.polarities)

    def _polarities_to_matrix(self):
        """Convert polarities to a binary matrix.

        Returns
        -------
            Binary matrix of shape (n_weak, cardinality) where:

            mask[i,j] = polarity[i] == j
        """
        # Use broadcasting to create a boolean mask of the elements that
        # match each possible value of polarities
        mask = self.polarities[:, np.newaxis] == np.arange(self.cardinality)

        return mask


class _BaseModel(_WeakLabelInfo, ABC):
    """Create a Model object."""

    def __init__(
        self,
        polarities: Union[List, np.ndarray],
        cardinality: int = 0,
        names: List = [],
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
    def predict_proba(self, L: pd.DataFrame) -> np.ndarray:
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

    def predict(self, L: pd.DataFrame, strategy: str = "majority") -> np.ndarray:
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

              This can be useful to enforce specific balances in the predictions.

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
            raise ValueError("strategy should be majority or probability")

        return predictions

    def _get_votes(self, L: pd.DataFrame) -> np.ndarray:
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
        votes = L.values @ self.polarities_matrix
        return votes

    def _normalize_preds(self, preds: np.ndarray) -> np.ndarray:
        # Normalize votes per row
        row_wise_sum = preds.sum(axis=1)
        # In case there were no votes for this row, no need to renormalize
        row_wise_sum = np.where(row_wise_sum == 0, 1, row_wise_sum).reshape(-1, 1)
        proba = preds / row_wise_sum
        return proba


class Voter(_BaseModel):
    """Basic model that bases its decisions on the sum of votes for each class."""

    def fit(self) -> None:
        """Voter model does not require fitting."""
        warnings.warn("Voter object does not need to be fitted", category=UserWarning)

    def predict_proba(self, L: pd.DataFrame) -> np.ndarray:
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
    """Basic model that bases its decisions on a weighted sum of votes for each class.

    The weights are computed during fitting so the sum of votes over training matches
    the given class balance.
    """

    def fit(
        self, L: pd.DataFrame, balances: Optional[Union[List, np.ndarray]] = None
    ) -> None:
        """Fit the BalancedMajorityVoter model.

        Parameters
        ----------
        L
            Weak label dataframe.
        balances
            Numpy array of shape cardinality giving a weight to each class.

            By default, assumes all classes are equally likely.
        """
        if balances is None:
            balances = np.ones(self.cardinality)
        elif isinstance(balances, list):
            balances = np.array(balances)
        self.balances = balances
        self.normalized_balances = self.balances / self.balances.sum()

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
            votes_mean[no_votes] = self.normalized_balances[no_votes]
        votes_weights = self.normalized_balances / votes_mean
        self.votes_weights = votes_weights

    def predict_proba(self, L: pd.DataFrame) -> np.ndarray:
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
