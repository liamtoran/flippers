import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .._typing import ListLike, MatrixLike


class _WeakLabelInfo:
    """Collects information about the weak labelers."""

    def __init__(
        self, polarities: ListLike, cardinality: int = 0, names: ListLike = []
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
        # Converts arrays to lists
        names = list(names)
        self.names = names

        # Validate inputs work
        self.__validate_init__()

        self.n_weak = len(self.polarities)
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


def multipolar_to_monopolar(
    L: MatrixLike, polarities_mapping: Dict[str, List[int]] = {}
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[int]]]:
    """Convert a pandas DataFrame of weak labels in multipolar representation
    to monopolar representation.

    Parameters
    ----------
    L :
        The input DataFrame of weak labels.

        Each column represents a weak labeler and each row represents a data point.
    polarities_mapping :
        A dictionary specifying the possible polarities for each weak labeler.

        The keys are the column names of L and the values are lists of integers
        representing the possible polarities.

        If not specified, the function
        will attempt to infer the polarities by examining the unique values in
        each column of L.

    Returns
    -------
    L_monopolar, polarities, polarities_mapping
        A tuple containing the following elements:

        - A pandas DataFrame of monopolar weak labels.

        - A 1D numpy array containing the polarities

        - A dictionary specifying the original polarities for each weak labeler.

        The keys are the column names of the input and the values are lists of integers
        representing the possible polarities.
    """
    L = pd.DataFrame(L)

    L.columns = L.columns.astype(str)
    new_columns = {}
    polarities = []

    # Create polarity mapping if not furnished
    if not polarities_mapping:
        warnings.warn(
            (
                "Polarity mapping if not furnished."
                "\nMake sure L captures all possible values of each weak labelers."
            )
        )
        polarities_mapping = {}

        for col in L.columns:
            values = L[col].unique()
            values = list(values[values >= 0])
            values.sort()

            if len(values) == 0:
                warnings.warn(
                    f"Column {col} is never labeled, assuming its polarity is 0"
                )
                values = [0]

            polarities_mapping[str(col)] = values

            for polarity in values:
                new_col = (L[col] == polarity).astype(int)
                new_columns[col + (len(values) > 1) * f"__{polarity}"] = new_col
                polarities.append(polarity)

    # Use polarity mapping
    # This is more robust but requires using a polarities mapping dict
    else:
        for col, values in polarities_mapping.items():
            for polarity in values:
                polarities.append(polarity)
                new_col = (L[col] == polarity).astype(int)
                new_columns[col + (len(values) > 1) * f"__{polarity}"] = new_col

    new_L = pd.DataFrame.from_dict(new_columns)
    polarities = np.array(polarities)

    return new_L, polarities, polarities_mapping


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
        Weak label DataFrame of shape (n_samples, n_weak)1.

    Returns
    -------
    Series of size n_weak with average confidence level per weak labeler.
    """
    return L[L > 0].mean()


def overlaps(L: pd.DataFrame, polarities: ListLike, sign: str = "all") -> pd.DataFrame:
    """Calculate the number of overlaps for each sample and weak label.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).
    polarities : Union[list, np.ndarray]
        Array or list of size n_weak containing the polarity of each weak label.
    sign : str, optional (default="all")
        String specifying which overlaps to include. Valid values are:

        - "all" (default) to include both positive and negative overlaps,

        - "match" to include matching overlaps only,

        - "conflict" to include negative overlaps only.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_samples, n_weak) where the (i, j)-th element represents
        the number of weak labels assigned to i that have an overlap with L[i,j].
    """
    L = pd.DataFrame(L)

    mask = _WeakLabelInfo(polarities).overlap_matrix
    match_overlap = L @ mask * L - L
    all_overlap = L.values.sum(axis=1).reshape(-1, 1) * L - L
    if sign == "match":
        overlap = match_overlap
    if sign == "all":
        overlap = all_overlap
    elif sign == "conflict":
        overlap = all_overlap - match_overlap

    overlap = overlap > 0
    overlap = overlap.sum(axis=0)
    overlap = overlap / len(L)

    return overlap  # type: ignore


def summary(
    L: MatrixLike, polarities: ListLike, digits: int = 3, normalize: int = False
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
    normalize:
        When True, shows overlaps/matched/conflicted as a ratio of coverage.

    Returns
    -------
    DataFrame of shape (n_weak, n_summaries) containing the following columns
        - "polarity": The polarity of each weak label.
        - "coverage": The average ratio of samples that are assigned each weak label.
        - "confidence": The average confidence level of the assigned weak labels.
        - "overlaps": The ratio of assigned labels that have overlapping labels.
        - "matched": The ratio of assigned labels that have other matching labels.
        - "conflicted": The ratio of assigned labels that have conflicting labels.
    """
    L = pd.DataFrame(L)

    descriptions = pd.DataFrame(index=L.columns)
    descriptions["polarity"] = polarities
    descriptions["coverage"] = coverage(L)
    descriptions["confidence"] = confidence(L)
    descriptions["overlaps"] = overlaps(L, polarities, sign="all")
    descriptions["matched"] = overlaps(L, polarities, sign="match")
    descriptions["conflicted"] = overlaps(L, polarities, sign="conflict")
    if normalize:
        descriptions["overlaps"] /= descriptions["coverage"]
        descriptions["matched"] /= descriptions["coverage"]
        descriptions["conflicted"] /= descriptions["coverage"]
    descriptions = descriptions.round(digits)

    return descriptions


assert summary(
    np.array(
        [
            [1, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ]
    ),
    [0, 1, 1, 0],
).to_dict() == {
    "polarity": {0: 0, 1: 1, 2: 1, 3: 0},
    "coverage": {0: 0.214, 1: 0.5, 2: 0.214, 3: 0.714},
    "confidence": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    "overlaps": {0: 0.214, 1: 0.429, 2: 0.071, 3: 0.429},
    "matched": {0: 0.143, 1: 0.0, 2: 0.0, 3: 0.143},
    "conflicted": {0: 0.214, 1: 0.429, 2: 0.071, 3: 0.429},
}
