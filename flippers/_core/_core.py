from typing import List, Union

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
        self.cardinality = cardinality

        # Set names
        if not names:
            names = [f"weak_label_{i}" for i in range(len(self.polarities))]
        self.names = names

        # Validate inputs work
        self.__validate_init__()

        self.polarities_matrix = self._get_polarity_matrix()
        self.overlap_matrix = self._get_overlap_matrix()

    def __validate_init__(self):
        """Assert whether the input data is valid."""
        # Check more than one class
        assert self.cardinality > 1

        # Check no polarities outside of cardinality
        assert self.polarities.max() <= self.cardinality - 1
        assert self.polarities.min() >= 0

        # Check names is the same size as polarities
        assert len(self.names) == len(self.polarities)

    def _get_polarity_matrix(self):
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

    def _get_overlap_matrix(self):
        polarity_overlap = self.polarities_matrix @ self.polarities_matrix.T
        return polarity_overlap


def is_labeled(L: pd.DataFrame) -> pd.Series:
    """Check if any labels exist in the given label matrix L.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Series of size n_samples indicating whether a sample is labeled or not.
    """
    return L.sum(axis=1) > 0


def filter_labeled(L: pd.DataFrame) -> pd.DataFrame:
    """Filter out unlabeled samples from the given label matrix.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Returns a filtered label matrix of shape (n_labeled_samples, n_weak).
    Sliced on the condition that the first one is labeled.
    """
    labeled = is_labeled(L)
    return L.loc[labeled]


def coverage(L: pd.DataFrame) -> pd.Series:
    """Calculate the average of samples labeled per weak labeler.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Series of size n_weak with average of samples labeled per weak labeler.
    """
    return (L > 0).mean()


def confidence(L: pd.DataFrame) -> pd.Series:
    """Calculate the average confidence level per weak labeler.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Series of size n_weak with average confidence level per weak labeler.
    """
    return L[L > 0].mean()


# overlap (all)


def num_overlapped(
    L: pd.DataFrame, polarities: Union[np.ndarray, List], sign: str = "all"
) -> pd.DataFrame:
    """Calculate the number of overlaps for each sample and weak label.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).
    polarities:
        Array or list of size n_weak containing the polarity of each weak label.
    sign:
        String specifying which overlaps to include.
    Valid values are
    - "all" (default) to include both positive and negative overlaps,
    - "match" to include matching overlaps only,
    - "conflict" to include negative overlaps only.

    Returns
    -------
    DataFrame of shape (n_samples, n_weak) where the (i, j)-th element represents
    the number of weak labels assigned to i that have an overlap with L[i,j].

    If sign is "positive", only positive overlaps are included;

    if sign is "negative",only negative overlaps are included;

    if sign is "all", all overlaps are included.
    """
    overlap_matrix = _WeakLabelInfo(polarities).overlap_matrix
    if sign == "all":
        # No masking if no sign provided
        overlap_matrix[:] = 1
    elif sign == "conflict":
        # overlap matrix is 1 if positive overlap
        overlap_matrix = ~overlap_matrix

    # Convert the DataFrame to a numeric type to avoid issues with bool values
    L_numeric = (L).astype(float)

    # Compute the number of overlaps from other weak labels with same polarities
    overlap = (L_numeric @ overlap_matrix).values - L_numeric

    # overlaps only happen when labeled
    overlap = L_numeric * overlap

    return overlap  # type: ignore


def overlapped_ratio(
    L: pd.DataFrame, polarities: Union[np.ndarray, List], sign=None
) -> pd.Series:
    """Calculate the ratio of weakly labeled samples that have an overlap.
    (positive and negatve)

    Parameters
    ----------
    L :
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Proportion of assigned labels that have overlapping labels per weak labeler.
    """
    return (num_overlapped(L, polarities, sign) > 0).sum() / L.sum()


def summary(
    L: pd.DataFrame, polarities: Union[np.ndarray, List], digits: int = 3
) -> pd.DataFrame:
    """Calculate summary statistics for the given weak label matrix and
    polarities.

    Parameters
    ----------
    L:
        Weak label DataFrame of shape (n_samples, n_weak).
    polarities:
        1D array or list of size n_weak containing the polarities of each weak label.
    digits:
        Number of digits to round the output statistics to. Default 3.

    Returns
    -------
    DataFrame of shape (n_weak, n_summaries) containing the following columns
        - "polarity": The polarity of each weak label.
        - "coverage": The average ratio of samples that are assigned each weak label.
        - "confidence": The average confidence level of the assigned weak labels.
        - "overlapped_ratio": The ratio of assigned labels that have overlapping labels.
        - "matched_ratio": The ratio of assigned labels that have other matching labels.
        - "conflicted_ratio": The ratio of assigned labels that have conflicting labels.
    """
    descriptions = pd.DataFrame(index=L.columns)
    descriptions["polarity"] = polarities
    descriptions["coverage"] = coverage(L)
    descriptions["confidence"] = confidence(L)
    descriptions["overlapped_ratio"] = overlapped_ratio(L, polarities, sign="all")
    descriptions["matched_ratio"] = overlapped_ratio(L, polarities, sign="positive")
    descriptions["conflicted_ratio"] = overlapped_ratio(L, polarities, sign="negative")
    return descriptions.round(digits)
