"""Attribute dependency DAG for dependencies-cognizant sampling."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class AttributeDAG:
    """Directed acyclic graph of attribute dependencies, learned from data or injected.

    Used by DAGSampler to bias proposals toward correlated attribute clusters,
    ensuring minority error classes are not missed due to uniform sampling.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.n = len(columns)
        self.col_index = {c: i for i, c in enumerate(columns)}
        # Weighted adjacency: adj[i][j] = conditional mutual information I(Xi; Xj)
        self.adj: np.ndarray = np.zeros((self.n, self.n))
        # Per-column error affinity (higher = column more likely to be in an error-prone record)
        self.error_affinity: np.ndarray = np.ones(self.n) / self.n

    def fit(self, df: pd.DataFrame, error_mask: Optional[np.ndarray] = None) -> "AttributeDAG":
        """Learn dependency weights from data using pairwise mutual information.

        error_mask: boolean array, True for records flagged as erroneous.
        If provided, error_affinity is updated based on attribute–error correlation.
        """
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        for i, ci in enumerate(self.columns):
            for j, cj in enumerate(self.columns):
                if i >= j or ci not in numeric or cj not in numeric:
                    continue
                xi = df[ci].fillna(0).values
                xj = df[cj].fillna(0).values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr = abs(float(np.corrcoef(xi, xj)[0, 1]))
                if not np.isfinite(corr):  # constant column → NaN corr
                    corr = 0.0
                self.adj[i, j] = corr
                self.adj[j, i] = corr

        if error_mask is not None and error_mask.sum() > 0:
            err_df = df[error_mask]
            non_df = df[~error_mask]
            for i, col in enumerate(self.columns):
                if col not in numeric:
                    continue
                err_mean = float(err_df[col].mean()) if len(err_df) > 0 else 0.0
                non_mean = float(non_df[col].mean()) if len(non_df) > 0 else 0.0
                std = float(df[col].std()) + 1e-9
                self.error_affinity[i] = 1.0 + abs(err_mean - non_mean) / std
            self.error_affinity /= self.error_affinity.sum()

        return self

    def inject(self, adj: np.ndarray) -> "AttributeDAG":
        """Directly inject a known adjacency matrix (for synthetic controlled experiments)."""
        assert adj.shape == (self.n, self.n)
        self.adj = adj.copy()
        return self

    def neighborhood_weight(self, col_idx: int) -> np.ndarray:
        """Return sampling weight vector emphasizing neighbors of col_idx in the DAG."""
        raw = self.adj[col_idx] + self.error_affinity
        total = raw.sum()
        if total == 0:
            return np.ones(self.n) / self.n
        return raw / total

    def cluster_priority(self) -> np.ndarray:
        """Return a probability distribution over columns, highest for high-degree hubs."""
        degree = np.nan_to_num(self.adj, nan=0.0).sum(axis=1) + self.error_affinity
        total = degree.sum()
        if total == 0 or not np.isfinite(total):
            return np.ones(self.n) / self.n
        return degree / total

    @classmethod
    def from_correlation(cls, df: pd.DataFrame, rho_inject: Optional[float] = None) -> "AttributeDAG":
        """Construct a DAG from a DataFrame, optionally injecting a known correlation level."""
        cols = df.columns.tolist()
        dag = cls(cols)
        if rho_inject is not None:
            # Synthetic uniform correlation for ablation experiment
            n = len(cols)
            adj = np.full((n, n), rho_inject)
            np.fill_diagonal(adj, 0.0)
            dag.inject(adj)
        else:
            dag.fit(df)
        return dag
