import numpy as np
import pandas as pd


# List of labeling functions
class LF_List:
    def __init__(self) -> None:
        self.lfs = []
        self.polarities = []

    # Add labeling function
    def add(self, polarity):
        self.polarities.append(polarity)

        def decorator(lf):
            self.lfs.append(lf)
            return lf

        return decorator

    # Create labeling matrix
    def create_matrix(self, df):
        L = np.zeros((len(df), len(self.lfs)))
        for i, lf in enumerate(self.lfs):
            L[:, i] = lf(df)
        column_names = [lf.__name__ for lf in self.lfs]

        L = pd.DataFrame(L, columns=column_names, index=df.index)
        return L
