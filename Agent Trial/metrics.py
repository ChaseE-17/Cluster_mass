"""Evaluation metrics for the Y-M relation project.

The primary loss (per AGENTS.md) is the *unweighted* MSE of the
relative residual ``M_pred / M_true - 1`` on the test set. We also
report:

  * the M-weighted version of the same MSE (for diagnostic purposes),
  * the relative scatter ``std(M_pred/M_true - 1)`` in the seven
    log-mass bins ``np.logspace(3.6, 4.7, 7)`` used by MTNG_copy.py,
  * the bin-wise mean residual (bias).

A small dataclass ``Metrics`` packs all of this for tabular printing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Same log-mass bin edges as MTNG_copy.py (line 389): the binned scatter
# / bias we report below uses these exact bins. Units are 1e10 Msun/h.
BIN_EDGES = np.logspace(3.6, 4.7, num=7)


@dataclass
class Metrics:
    mse: float                  # primary metric (unweighted MSE of M_pred/M_true - 1)
    mse_weighted: float         # M-weighted MSE of same residual
    bin_scatter: np.ndarray     # std of relative residual in each log-M bin
    bin_bias: np.ndarray        # mean of relative residual in each log-M bin
    bin_counts: np.ndarray      # halos per bin (for filtering low-count bins)
    bin_centers: np.ndarray     # geometric mean of bin edges
    max_bin_scatter: float      # max scatter over bins with >=30 halos
    max_abs_bin_bias: float     # max |bias| over bins with >=30 halos


def relative_residual(m_pred: np.ndarray, m_true: np.ndarray) -> np.ndarray:
    """Return ``M_pred / M_true - 1`` element-wise."""
    return m_pred / m_true - 1.0


def evaluate(
    m_pred: np.ndarray,
    m_true: np.ndarray,
    bin_edges: np.ndarray = BIN_EDGES,
    min_count: int = 30,
) -> Metrics:
    """Compute the full Metrics bundle for a single set of predictions."""
    res = relative_residual(m_pred, m_true)
    mse = float(np.mean(res ** 2))
    w = m_true
    mse_w = float(np.sum(w * res ** 2) / np.sum(w))

    n_bins = len(bin_edges) - 1
    scatter = np.full(n_bins, np.nan)
    bias = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = (m_true >= bin_edges[i]) & (m_true <= bin_edges[i + 1])
        counts[i] = int(sel.sum())
        if counts[i] >= 2:
            scatter[i] = float(np.std(res[sel]))
            bias[i] = float(np.mean(res[sel]))

    keep = counts >= min_count
    max_scat = float(np.nanmax(scatter[keep])) if keep.any() else float("nan")
    max_bias = float(np.nanmax(np.abs(bias[keep]))) if keep.any() else float("nan")

    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return Metrics(
        mse=mse,
        mse_weighted=mse_w,
        bin_scatter=scatter,
        bin_bias=bias,
        bin_counts=counts,
        bin_centers=centers,
        max_bin_scatter=max_scat,
        max_abs_bin_bias=max_bias,
    )


def fit_sigma(m_true_train: np.ndarray) -> np.ndarray:
    """Return the ``sigma`` array for ``scipy.optimize.curve_fit``.

    AGENTS.md prescribes linear-in-M weighting (matching the
    Random-Forest benchmark). curve_fit minimizes ``sum((y-f)/sigma)^2``;
    we want ``sum( w * (M_pred/M_true - 1)^2 )`` with w = M_true, i.e.
    ``sum((M_pred - M_true) * sqrt(w) / M_true)^2``. Setting
    ``sigma = M_true / sqrt(w) = sqrt(M_true)`` achieves exactly that.
    """
    return np.sqrt(m_true_train)
