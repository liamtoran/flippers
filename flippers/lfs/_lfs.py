import numpy as np
import pandas as pd


class LFS:
    """A container for managing labeling functions for data labeling.

    This class allows for the easy addition of labeling functions and
    their associated polarities.

    It also provides functionality to
    create a labeling matrix from a given DataFrame.

    Example
    -------
    >>> lfs = flippers.lfs.LFS()
    >>>
    >>> # Add a labeling function with polarity 1
    >>> @lfs.add(polarity=1)
    >>> def lf(df):
    >>>     return df["column_name"] > 0
    >>>
    >>> # Create a labeling matrix from a DataFrame
    >>> L = lfs.create_matrix(df)
    """

    def __init__(self) -> None:
        self.lfs = []
        self.polarities = []

    def add(self, polarity: int):
        """Adds a labeling function with its associated polarity.

        Parameters
        ----------
        polarity: The polarity associated with the labeling function.

        Example
        -------
        >>> @lfs.add(polarity=1)
        >>> def lf_happy(df):
        >>>     return df["text"].str.contains("happy")
        >>>
        >>> @lfs.add(polarity=0)
        >>> def lf_angry(df):
        >>>     return df["text"].str.contains("angry")
        """
        self.polarities.append(polarity)

        def decorator(lf):
            self.lfs.append(lf)
            return lf

        return decorator

    def create_matrix(self, df: pd.DataFrame):
        """Creates a labeling matrix from a given DataFrame.

        Each labeling function is applied to the DataFrame, and the
        results are stored in a matrix. The matrix is then returned as a
        DataFrame with the labeling function names as column names.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame:
            Labeling matrix.

        Example
        -------
        >>> L_train = lfs.create_matrix(df_train)
        """
        L = np.zeros((len(df), len(self.lfs)))
        for i, lf in enumerate(self.lfs):
            L[:, i] = lf(df)
        column_names = [lf.__name__ for lf in self.lfs]

        L = pd.DataFrame(L, columns=column_names, index=df.index)
        return L
