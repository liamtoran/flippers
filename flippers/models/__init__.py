"""Groups label generative models."""

from ._base import Voter
from ._snorkel import SnorkelModel

__all__ = ["Voter", "SnorkelModel"]
