"""Convergence diagnostics: Raftery-Lewis test and Wald confidence interval."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WaldCI:
    estimate: float
    lower: float
    upper: float
    margin: float
    z: float = 1.96  # 95% CI

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @classmethod
    def from_proportion(cls, p: float, n: int, confidence: float = 0.95) -> "WaldCI":
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        se = np.sqrt(p * (1 - p) / max(n, 1))
        margin = z * se
        return cls(estimate=p, lower=max(0.0, p - margin), upper=min(1.0, p + margin), margin=margin, z=z)


@dataclass
class RafteryLewisResult:
    """Result of the Raftery-Lewis convergence diagnostic.

    Estimates burn-in and total MCMC chain length needed for reliable quantile estimation.
    Reference: Raftery & Lewis (1992), MCMC in Practice, Chapter 7.
    """
    converged: bool
    burn_in: int
    n_required: int
    n_current: int
    dependence_factor: float  # I = (n_required × p(1-p)) / (thin × r(1-r)); I > 5 signals autocorrelation
    message: str


def raftery_lewis(
    chain: np.ndarray,
    q: float = 0.025,
    r: float = 0.005,
    s: float = 0.95,
) -> RafteryLewisResult:
    """Raftery-Lewis diagnostic for a scalar MCMC chain.

    Args:
        chain: 1-D array of MCMC estimates (one per sampling step).
        q: quantile of interest (default 0.025 — lower tail).
        r: desired accuracy (|estimated quantile − true quantile| < r with probability s).
        s: desired probability for the accuracy guarantee.
    """
    n = len(chain)
    if n < 20:
        return RafteryLewisResult(
            converged=False,
            burn_in=0,
            n_required=100,
            n_current=n,
            dependence_factor=0.0,
            message="Chain too short for Raftery-Lewis",
        )

    # Dichotomize chain at empirical quantile q
    threshold = float(np.quantile(chain, q))
    binary = (chain <= threshold).astype(int)

    # Estimate transition matrix of the two-state chain
    n00 = n01 = n10 = n11 = 0
    for i in range(n - 1):
        a, b = binary[i], binary[i + 1]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1

    alpha = n01 / max(n00 + n01, 1)  # P(0→1)
    beta = n10 / max(n10 + n11, 1)   # P(1→0)

    if alpha + beta == 0:
        return RafteryLewisResult(
            converged=True, burn_in=0, n_required=n, n_current=n,
            dependence_factor=1.0, message="Chain constant — no mixing"
        )

    # Stationary probability
    p_star = beta / (alpha + beta)

    # Required chain length (from Raftery-Lewis formula)
    from scipy import stats as _stats
    z_s = _stats.norm.ppf((1 + s) / 2)
    z_r = _stats.norm.ppf(0.5 + r / 2)

    numerator = 2 * alpha * beta * (1 - (alpha + beta)) * z_s ** 2
    denominator = ((alpha + beta) ** 3) * r ** 2
    n_ind = numerator / max(denominator, 1e-12)

    # Dependence factor I
    I_factor = (n_ind * p_star * (1 - p_star)) / max(p_star * (1 - p_star), 1e-9)
    n_required = max(int(I_factor * n_ind), 20)
    burn_in = max(int(np.log(r / p_star) / np.log(abs(1 - alpha - beta) + 1e-9)), 0)

    converged = n >= n_required

    return RafteryLewisResult(
        converged=converged,
        burn_in=burn_in,
        n_required=n_required,
        n_current=n,
        dependence_factor=I_factor,
        message="OK" if converged else f"Need {n_required - n} more steps",
    )


class ConvergenceChecker:
    """Tracks MCMC chain of quality estimates and checks convergence."""

    def __init__(self, window: int = 50):
        self.window = window
        self._chains: Dict[str, List[float]] = {}

    def update(self, profile_dict: Dict[str, float]) -> None:
        for key, val in profile_dict.items():
            self._chains.setdefault(key, []).append(val)

    def check(self) -> Dict[str, RafteryLewisResult]:
        results = {}
        for key, chain in self._chains.items():
            if len(chain) >= 20:
                results[key] = raftery_lewis(np.array(chain))
        return results

    def all_converged(self) -> bool:
        results = self.check()
        if not results:
            return False
        return all(r.converged for r in results.values())

    def wald_cis(self, n_total: int) -> Dict[str, WaldCI]:
        cis = {}
        for key, chain in self._chains.items():
            if chain:
                p = chain[-1]
                cis[key] = WaldCI.from_proportion(p, n_total)
        return cis
