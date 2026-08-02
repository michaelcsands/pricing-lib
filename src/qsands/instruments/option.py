from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import numpy as np

class Instrument(ABC):
    """Base class for all priceable instruments."""

    is_path_dependent: bool = False

    @abstractmethod
    def payoff(self, terminal: dict, paths: Optional[dict] = None, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the (undiscounted) payoff per path, shape (n_paths,).
        
        terminal: dict[str, ndarray]
            Each state variable's value at T (always provided).
        paths, time_grid: provided only when is_path_dependent=True and
            the engine is called with store_paths=True otherwise None.
        """
        ...

@dataclass
class EuropeanOption(Instrument):
    """A European call or put, path-independent by construction."""
    strike: float
    expiry: date
    option_type: Literal["call", "put"] = "call"
    is_path_dependent: bool = False

    def payoff(self, terminal: dict, paths: Optional[dict] = None, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        S_T = terminal["S"]
        if self.option_type == "call":
            return np.maximum(S_T - self.strike, 0.0)
        return np.maximum(self.strike - S_T, 0.0)