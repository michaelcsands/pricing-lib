from __future__ import annotations

from dataclasses import dataclass 
from typing import Optional

@dataclass
class MarketData:
    r: float
    sigma: Optional[float] = None