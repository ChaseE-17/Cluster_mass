"""Residual-structure inspection for the *current best* candidate.

Numerically reports the residual--vs--feature trends so the next
candidate can be motivated by data, not by guesswork.

Per AGENTS.md the inspection must use the *training set only* for
choosing features / functional forms. We therefore evaluate the trends
on the train slice, not the test slice.

Outputs (per allowed feature):
  * Pearson correlation (residual vs. feature, residual vs. log-feature)
  * Spearman rank correlation (robust to monotonic non-linearities)
  * Mean residual in 6 quantile bins of the feature
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

from data_loader import load_dataset
from candidates import CANDIDATES, pack_features, Candidate
from run_candidates import fit_candidate
import metrics as M


def feature_dict(ds, mask):
    """Return a dict of feature name -> 1-D array on the masked slice.

    All allowed features per AGENTS.md (ex. profile-derived scalars,
    which we add only when explicitly proposed in candidates.py).
    """
    return {
        "Y200":            ds.Y200H[mask],
        "YConc":           ds.YConcH[mask],
        "GasConc":         ds.GasConcH[mask],
        "mGas":            ds.mGasH[mask],
        "mStar":           ds.mStarH[mask],
        "mStar/mGas":      (ds.mStarH / ds.mGasH)[mask],
    }


def quantile_bin_means(x: np.ndarray, r: np.ndarray, n_bins: int = 6):
    """Return (bin_centers, mean_residual_per_bin) using equal-count bins."""
    order = np.argsort(x)
    xs = x[order]
    rs = r[order]
    n = len(xs)
    edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
    centers = np.empty(n_bins)
    means = np.empty(n_bins)
    for i in range(n_bins):
        sl = slice(edges_idx[i], edges_idx[i + 1])
        centers[i] = np.median(xs[sl])
        means[i] = np.mean(rs[sl])
    return centers, means


def analyze(cand: Candidate):
    ds = load_dataset()
    mask_train = ~ds.mask_test

    popt, status = fit_candidate(cand, ds, mask_train)
    X_train = pack_features(ds, cand.feature_names)[:, mask_train]
    pred_train = cand.func(X_train, *popt)
    res = pred_train / ds.m200H[mask_train] - 1.0

    print(f"Candidate: {cand.name}  (params: {cand.param_names} = {popt})")
    print(f"  fit via {status}")
    print(f"  train MSE of residual: {np.mean(res ** 2):.4e}")
    print()

    feats = feature_dict(ds, mask_train)
    feats["m200"] = ds.m200H[mask_train]  # diagnostic only - not allowed in candidates
    print(f"{'feature':<14} {'pearson':>10} {'pearson(log)':>14} {'spearman':>10}")
    print("-" * 50)
    for name, x in feats.items():
        # log version only well-defined for strictly positive features
        pe_lin = float(np.corrcoef(x, res)[0, 1])
        if (x > 0).all():
            pe_log = float(np.corrcoef(np.log(x), res)[0, 1])
        else:
            pe_log = float("nan")
        sp = float(spearmanr(x, res).statistic)
        print(f"{name:<14} {pe_lin:>10.4f} {pe_log:>14.4f} {sp:>10.4f}")

    print("\nQuantile-binned mean residual per feature:")
    for name, x in feats.items():
        if name == "m200":
            continue
        centers, means = quantile_bin_means(x, res, n_bins=6)
        cells = "  ".join(f"{c:.3g}:{m:+.4f}" for c, m in zip(centers, means))
        print(f"  {name:<14} {cells}")


if __name__ == "__main__":
    # Default to the most-recent candidate in CANDIDATES (last entry).
    cand = CANDIDATES[-1]
    analyze(cand)
