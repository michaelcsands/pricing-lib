from __future__ import annotations

from datetime import date

import numpy as np

from qsands.instruments.option import Instrument
from qsands.market.data import MarketData
from qsands.market.process import StochasticProcess
from qsands.results import PricingResult
from qsands.time.daycount import year_fraction


class MonteCarloEngine:
    def __init__(
            self,
            process: StochasticProcess,
            valuation_date: date,
            n_paths: int = 10000,
            n_steps: int = 252,
            seed: int | None = None,
            keep_paths_in_result: bool = False,
    ):
        self.process = process
        self.valuation_date = valuation_date
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
        self.keep_paths_in_result = keep_paths_in_result

    def __repr__(self):
            params = vars(self)
            args = ", ".join(f"{k}={v}" for k, v in params.items())
            return f"{self.__class__.__name__}({args})"
        
    def calculate(self, instrument: Instrument, market: MarketData) -> PricingResult:
        T = year_fraction(self.valuation_date, instrument.expiry)

        store_path = instrument.is_path_dependent or self.keep_paths_in_result

        sim = self.process.simulate(
            n_paths = self.n_paths,
            n_steps = self.n_steps,
            T = T,
            rng = self.rng,
            store_path = store_path,
        )

        payoffs = instrument.payoff(sim.terminal, sim.paths, sim.time_grid)
        discount = np.exp(-market.r * T)
        price = discount * payoffs.mean()
        stderr = discount * payoffs.std(ddof=1) / np.sqrt(self.n_paths)

        return PricingResult(
            price = float(price),
            method = "monte_carlo",
            stderr = float(stderr),
            simulation = sim if self.keep_paths_in_result else None,
            diagnostic = {"n_paths": self.n_paths, "n_steps": self.n_steps}
        )