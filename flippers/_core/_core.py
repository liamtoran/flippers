import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .._typing import ListLike, MatrixLike


class _WeakLabelInfo:
    """Collects information about the weak labelers."""

    def __init__(self, polarities: ListLike, cardinality: int = 0):
        """
        Parameters
        ----------
        polarities
            List that maps weak labels to polarities.
        cardinality: int, optional
            Number of classes

            If not specified, it will be inferred from the maximum value in polarities.

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
    L : pd.DataFrame
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

    Example
    -------
    Its always better to use an hand written ``polairites_mapping``.
    ``polarities_mapping`` lists possible polarities each labeling function can have.

    >>> import pandas as pd
    >>> import flippers
    >>> multipolar = pd.DataFrame([[-1, 1, 2], [0, -1, 0], [-1, -1, 2]])
    >>> polarities_mapping = {'0': [0], '1': [1], '2': [0, 2]}
    >>> L, polarities, _ = flippers.multipolar_to_monopolar(
                                                multipolar, polarities_mapping
                                                )
    >>> L
       0  1  2__0  2__2
    0  0  1     0     1
    1  1  0     1     0
    2  0  0     0     1

    If you dont want to create the mapping, the function can infer one.
    This is potentially breaking if multipolar does not contain all possible outputs.

    >>> L, polarities, polarities_mapping = flippers.multipolar_to_monopolar(multipolar)
    >>> L
       0  1  2__0  2__2
    0  0  1     0     1
    1  1  0     1     0
    2  0  0     0     1
    >>> polarities # output L polarities
    array([0, 1, 0, 2], dtype=int64)
    >>> polarities_mapping # input multipolar polarities
    {'0': [0], '1': [1], '2': [0, 2]}
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
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Series of size n_samples indicating whether a sample is labeled or not.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> flippers.is_labeled(L)
    0     True
    1     True
    2    False
    3     True
    dtype: bool
    """
    return L.sum(axis=1) > 0


def total_coverage(L: pd.DataFrame) -> float:
    """Calculate the total proportion of labeled samples in the given label
    matrix.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    float
        Total coverage, ranging from 0 to 1, indicating the proportion of
        labeled samples in the label matrix.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> flippers.total_coverage(L)
    0.75
    """
    return is_labeled(L).mean()


def filter_labeled(L: pd.DataFrame) -> pd.DataFrame:
    """Filter out unlabeled samples from the given label matrix.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Returns a filtered label matrix of shape (n_labeled_samples, n_weak).
    Sliced on the condition that the first one is labeled.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> flippers.filter_labeled(L)
       0  1  2
    0  0  1  0
    2  1  0  1
    """
    labeled = is_labeled(L)
    return L.loc[labeled]


def coverage(L: pd.DataFrame) -> pd.Series:
    """Calculate the average of samples labeled per weak labeler.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).

    Returns
    -------
    Series of size n_weak with average of samples labeled per weak labeler.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> flippers.coverage(L)
    0    0.5
    1    0.5
    2    0.5
    dtype: float64
    """
    return (L > 0).mean()


def confidence(L: pd.DataFrame) -> pd.Series:
    """Calculate the average confidence level per weak labeler.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak)1.

    Returns
    -------
    Series of size n_weak with average confidence level per weak labeler.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, .1, 0], [1, 0, .5], [0, 0, 0], [.7, .1, .2]])
    >>> flippers.confidence(L)
    0    0.85
    1    0.10
    2    0.35
    dtype: float64
    """
    return L[L > 0].mean()


def overlaps(L: pd.DataFrame, polarities: ListLike, sign: str = "all") -> pd.Series:
    """Calculate the number of overlaps for each sample and weak label.

    Parameters
    ----------
    L : pd.DataFrame
        Weak label DataFrame of shape (n_samples, n_weak).
    polarities : Union[list, np.ndarray]
        Array or list of size n_weak containing the polarity of each weak label.
    sign : str, optional (default="all", options "all", "match", "conflict")
        String specifying which overlaps to include. Valid values are:

        - "all" (default) to include both positive and negative overlaps,

        - "match" to include matching overlaps only,

        - "conflict" to include negative overlaps only.

    Returns
    -------
    pd.Series
        Series of length n_weak indicating the fraction of
        annotated samples with other annotations for each LF.

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> # All overlaps
    >>> flippers.overlaps(L, polarities)
    0    0.50
    1    0.25
    2    0.50
    dtype: float64
    >>> # Only overlaps with matching assigned label
    >>> flippers.overlaps(L, polarities, sign="match")
    0    0.00
    1    0.25
    2    0.25
    dtype: float64
    >>> # Only overlaps with conflicting assigned label
    >>> flippers.overlaps(L, polarities, sign="conflict")
    0    0.50
    1    0.25
    2    0.50
    dtype: float64
    """
    L_pd = pd.DataFrame(L)
    L = np.array(L)

    mask = _WeakLabelInfo(polarities).overlap_matrix
    match_overlap = L @ mask * L - L
    all_overlap = L.sum(axis=1).reshape(-1, 1) * L - L
    if sign == "match":
        overlap = match_overlap
    if sign == "all":
        overlap = all_overlap
    elif sign == "conflict":
        overlap = all_overlap - match_overlap

    overlap = overlap > 0
    overlap = overlap.sum(axis=0)
    overlap = overlap / len(L)

    overlap = pd.Series(overlap, index=L_pd.columns)
    return overlap  # type: ignore


def summary(
    L: MatrixLike, polarities: ListLike, digits: int = 3, normalize: int = False
) -> pd.DataFrame:
    """Calculate summary statistics for the given weak label matrix and
    polarities.

    Parameters
    ----------
    L : pd.DataFrame
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

    Example
    -------
    >>> import pandas as pd
    >>> import flippers
    >>> L = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> polarities = [0, 1, 1]
    >>> flippers.summary(L, polarities)
        polarity  coverage  confidence  overlaps  matched  conflicted
    0         0       0.5         1.0      0.50     0.00        0.50
    1         1       0.5         1.0      0.25     0.25        0.25
    2         1       0.5         1.0      0.50     0.25        0.50
    >>> flippers.summary(L, polarities, normalize=True)
        polarity  coverage  confidence  overlaps  matched  conflicted
    0         0       0.5         1.0       1.0      0.0         1.0
    1         1       0.5         1.0       0.5      0.5         0.5
    2         1       0.5         1.0       1.0      0.5         1.0
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
