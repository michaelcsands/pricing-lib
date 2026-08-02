"""
General stochastic-process abstraction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

@dataclass
class SimulationResult:
    """
    Output of StochasticProcess.simulate()
    
    terminal: dict[str, ndarray]
        Each state variable's value at T, shape (n_paths,)
    paths: dict[str, ndarray] | None
        Each state variable's full trajectory shape (n_steps + 1, n_paths)
        None if store_paths=False
    time_grid: ndarray | None
        Shape (n_steps + 1,)
    n_paths: int
    n_steps: int
    """

    terminal: dict
    paths: Optional[dict] = None
    time_grid: Optional[np.ndarray] = None
    n_paths: int = 0
    n_steps: int = 0


class StochasticProcess(ABC):
    """
    Base class for asset-price dynamics used by Monte Carlo engines. 
    
    Subclasses must define:
    - state_names: the variables carried through the simulation (must include S)
    - driver_names: the independent Brownian sources
    - correlation: driver_names x driver_names correlation matrix
    - initial_state(n_paths): starting values, broadcast across paths
    - step(state, dt, dW): one time-step transition (vectorized over paths)
    """

    #: names of state variables carried through the simulation (eg: S, V)
    state_names: tuple

    #: names of independent Brownian drivers (eg: W_S, W_V)
    driver_names: tuple

    @abstractmethod
    def initial_state(self, n_paths: int) -> dict:
        """ Return {name: ndarray of shape (n_paths,)} at t=0 """
        ...
 

    @abstractmethod
    def correlation(self) -> np.ndarray:
        """ Return the (n_drivers, n_drivers) correlation matrix """
        ...

    @abstractmethod
    def step(self, state: dict, dt: float, dW: dict) -> dict:
        """ Advance state by one step of size dt, given correlated increments dW.
        Must return a new dict with the same keys as state_names, all shape 
        (n_paths,). Vectorized over paths not time"""
        ...

    def __repr__(self):
        params = vars(self)
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({args})"
    
    def simulate(self, n_paths: int, n_steps: int, T: float, rng: np.random.Generator, store_path: bool = True) -> SimulationResult:
        """ 
        Simulate n_paths independent trajectories over [0, T] in n_steps steps.
        
        store_path = False skips path allocation (O(n_paths) instead of O(n_steps*n_paths))
        """
        dt = T / n_steps
        chol = np.linalg.cholesky(self.correlation())

        state = self.initial_state(n_paths)
        for name in self.state_names:
            state[name] = np.broadcast_to(state[name], (n_paths,)).astype(float).copy() # initialize n_paths for each state_name

        time_grid = np.linspace(0.0, T, n_steps + 1) if store_path else None
        paths = None
        if store_path:
            paths = {name: np.empty((n_steps + 1, n_paths)) for name in self.state_names}
            for name in self.state_names:
                paths[name][0] = state[name]

        n_drivers = len(self.driver_names)
        for i in range(n_steps):
            Z = rng.standard_normal(size=(n_drivers, n_paths)) 
            correlated = chol @ Z * np.sqrt(dt)
            dW = {name: correlated[j] for j, name in enumerate(self.driver_names)}

            state = self.step(state, dt, dW)

            if store_path:
                for name in self.state_names:
                    paths[name][i+1] = state[name]

        return SimulationResult(
            terminal=state,
            paths=paths,
            time_grid=time_grid,
            n_paths=n_paths,
            n_steps=n_steps,
        )


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion
    
    The asset follows
    dS = mu * S * dt + sigma * S * dW
    """

    state_names = ("S",) 
    driver_names = ("W",)

    def __init__(self, S0: float, mu: float, sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def initial_state(self, n_paths: int) -> dict:
        return {"S": np.full(n_paths, self.S0, dtype = float)}

    def correlation(self) -> np.ndarray:
        return np.array([[1.0]])

    def step(self, state: dict, dt: float, dW: dict) -> dict:
        log_S = np.log(state["S"]) + (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW["W"]
        return {"S": np.exp(log_S)}
    
class HestonModel(StochasticProcess):
    """
    Heston stochastic volatility model
    """

    state_names = ("S", "V")
    driver_names = ("W_S", "W_V")

    def __init__(self, S0: float, V0: float, r: float, kappa: float, theta: float, xi: float, rho: float):
        self.S0, self.V0, self.r, self.kappa = S0, V0, r, kappa
        self.theta, self.xi, self.rho = theta, xi, rho

    def initial_state(self, n_paths: int) -> dict:
        return {
            "S": np.full(n_paths, self.S0, dtype=float),
            "V": np.full(n_paths, self.V0, dtype=float),
        }

    def correlation(self) -> np.ndarray:
        return np.array([[1.0, self.rho], [self.rho, 1.0]])

    def step(self, state:dict, dt: float, dW: dict) -> dict:
        V_plus = np.maximum(state["V"], 0.0) 
        log_S = np.log(state["S"]) + (self.r - 0.5 * V_plus) * dt + np.sqrt(V_plus) * dW["W_S"]
        V_next = (
            state["V"] + self.kappa * (self.theta - V_plus) * dt
            + self.xi * np.sqrt(V_plus) * dW["W_V"]
            + 0.25 * self.xi**2 * (dW["W_V"]**2 - dt)
        )
        return {"S": np.exp(log_S), "V": V_next}
