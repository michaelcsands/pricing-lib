from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from qsands.market.process import SimulationResult

@dataclass
class PricingResult:
    price: float
    method: str
    stderr: Optional[float] = None
    simulation: Optional[SimulationResult] = None
    diagnostic: dict = field(default_factory=dict)
    