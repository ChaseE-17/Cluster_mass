"""Analytic candidate equations for the Y-M relation.

Every candidate is a *pure function* of the form

    cand_NN(X, *theta) -> M_pred  [units: 1e10 Msun/h]

where ``X`` is a 2-D ndarray of shape ``(n_features, n_halos)`` whose
rows are stacked in a candidate-specific order.  The driver
(``run_candidates.py``) packs the rows in the order specified by each
candidate's ``feature_names`` attribute, fits the parameters with
``scipy.optimize.curve_fit`` (with sigma from
``metrics.fit_sigma``), and reports test metrics next to a freshly
re-fitted power-law benchmark.

Each candidate ships with:

  * ``feature_names``  -> tuple of feature names (rows of X, in order)
  * ``param_names``    -> tuple of parameter names (matches *theta order)
  * ``p0``             -> initial guess for curve_fit
  * ``expression``     -> human-readable expression
  * ``token_count``    -> rough token count (see AGENTS.md complexity budget)
  * ``motivation``     -> one-line motivation string

A registry list ``CANDIDATES`` collects them in proposal order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class Candidate:
    name: str
    func: Callable[..., np.ndarray]
    feature_names: Sequence[str]
    param_names: Sequence[str]
    p0: Sequence[float]
    expression: str
    token_count: int
    motivation: str


# ---------------------------------------------------------------------------
# Round 0 - Benchmark: pure power law M = A * Y^alpha
# ---------------------------------------------------------------------------
def cand_benchmark(X: np.ndarray, A: float, alpha: float) -> np.ndarray:
    """``M_pred = A * Y200**alpha``  (the canonical power-law benchmark)."""
    Y = X[0]
    return A * Y ** alpha


BENCHMARK = Candidate(
    name="cand_benchmark",
    func=cand_benchmark,
    feature_names=("Y200",),
    param_names=("A", "alpha"),
    # A in units of 1e10 Msun/h (since m200H is in those units). The
    # MTNG_copy.py value 1.97e17 Msun/h corresponds to A = 1.97e7
    # in these units.
    p0=(2.0e7, 0.6),
    expression="A * Y200**alpha",
    token_count=4,  # A, *, Y200, **alpha
    motivation="Refit of the canonical power-law benchmark.",
)


# ---------------------------------------------------------------------------
# Round 1 - Add a YConc correction:  M = A * Y^alpha * (1 - b*YConc)
# ---------------------------------------------------------------------------
def cand_yconc_linear(
    X: np.ndarray, A: float, alpha: float, b: float
) -> np.ndarray:
    """``M_pred = A * Y200**alpha * (1 - b * YConc)``.

    Y-concentration is the simplest physical correction: at fixed
    integrated Y, more centrally-concentrated profiles correspond to
    cooler / disturbed cores at low M, and to relaxed cool-core systems
    at high M. The MTNG_copy.py paper figure showed a clear residual
    trend in YConc; this candidate is the same form they used.
    """
    Y, cY = X
    return A * Y ** alpha * (1.0 - b * cY)


CAND_YCONC = Candidate(
    name="cand_yconc_linear",
    func=cand_yconc_linear,
    feature_names=("Y200", "YConc"),
    param_names=("A", "alpha", "b"),
    p0=(2.7e7, 0.61, 0.4),
    expression="A * Y200**alpha * (1 - b*YConc)",
    token_count=8,  # A, *, Y200, **alpha, *, (1, -, b*YConc)
    motivation=(
        "Reproduce the MTNG_copy.py 'YConc correction' as a baseline "
        "improvement over the pure power law."
    ),
)


# ---------------------------------------------------------------------------
# Round 2 - Add an mStar/mGas correction:
#   M = A * Y^alpha * (1 - b*YConc + c*(mStar/mGas))
# ---------------------------------------------------------------------------
def cand_yconc_fstar(
    X: np.ndarray, A: float, alpha: float, b: float, c: float
) -> np.ndarray:
    """``M = A * Y200**alpha * (1 - b*YConc + c*(mStar/mGas))``.

    Round-1 residuals showed a strong, monotonic trend with the
    stellar-to-gas ratio (Pearson -0.39, Spearman -0.40, ~7% swing
    across the feature range) -- the largest remaining correlation by a
    factor of ~3 over any other allowed feature. We absorb it as an
    additive linear correction inside the same bracket as the YConc
    term (rather than as a separate factor) to keep the token count
    near the AGENTS.md budget.
    """
    Y, cY, fStar = X
    return A * Y ** alpha * (1.0 - b * cY + c * fStar)


CAND_YCONC_FSTAR = Candidate(
    name="cand_yconc_fstar",
    func=cand_yconc_fstar,
    feature_names=("Y200", "YConc", "mStar_over_mGas"),
    param_names=("A", "alpha", "b", "c"),
    p0=(2.7e7, 0.61, 0.4, 1.0),
    expression="A * Y200**alpha * (1 - b*YConc + c*(mStar/mGas))",
    token_count=11,  # ~10 budget; +1 token for the additive fStar term
    motivation=(
        "Round-1 residual analysis: mStar/mGas has by far the strongest "
        "remaining residual correlation (Pearson -0.39, ~3x bigger than "
        "any other feature). Add it as a linear correction."
    ),
)


# ---------------------------------------------------------------------------
# Round 3 - Replace the linear bracket with an exponential:
#   M = A * Y^alpha * exp(-b*YConc + c*(mStar/mGas))
# ---------------------------------------------------------------------------
def cand_yconc_fstar_exp(
    X: np.ndarray, A: float, alpha: float, b: float, c: float
) -> np.ndarray:
    """``M = A * Y200**alpha * exp(-b*YConc + c*(mStar/mGas))``.

    Round-2 residuals showed two pieces of non-linear structure that
    a *linear* correction cannot capture:
      * mild curvature in the binned residual vs. Y200 (peak-to-peak
        ~0.025, U-shaped),
      * a small "U" shape in mStar/mGas (peak-to-peak ~0.014).
    Swapping ``(1 + x)`` for ``exp(x)`` adds Taylor-order-2 (and
    higher) corrections in YConc and fStar at *zero parameter cost*
    and the same token budget, while preserving the round-2 first-
    order behaviour (since exp(x) ~ 1+x for small x).
    """
    Y, cY, fStar = X
    return A * Y ** alpha * np.exp(-b * cY + c * fStar)


CAND_YCONC_FSTAR_EXP = Candidate(
    name="cand_yconc_fstar_exp",
    func=cand_yconc_fstar_exp,
    feature_names=("Y200", "YConc", "mStar_over_mGas"),
    param_names=("A", "alpha", "b", "c"),
    p0=(2.5e7, 0.62, 0.39, 1.28),  # match round-2 fit so curve_fit lands fast
    expression="A * Y200**alpha * exp(-b*YConc + c*(mStar/mGas))",
    token_count=11,
    motivation=(
        "Round-2 residuals show mild non-linear curvature in YConc and "
        "fStar; replace (1 - b*YConc + c*fStar) with exp(...) to absorb "
        "second-order corrections at zero parameter cost."
    ),
)


# ---------------------------------------------------------------------------
# Round 4 - Add a GasConc correction:
#   M = A * Y^alpha * (1 - b*YConc + c*fStar - d*GasConc)
# ---------------------------------------------------------------------------
def cand_yconc_fstar_gasconc(
    X: np.ndarray, A: float, alpha: float, b: float, c: float, d: float
) -> np.ndarray:
    """``M = A * Y200**alpha * (1 - b*YConc + c*fStar - d*GasConc)``.

    Round-2 residuals had only weak structure in GasConc (Pearson
    -0.024) but a non-trivial bin pattern. GasConc traces a different
    physical aspect (gas-mass concentration) than YConc (pressure
    concentration); this adds it as one more linear correction inside
    the same bracket to test whether dual-concentration info reduces
    scatter.
    """
    Y, cY, fStar, cGas = X
    return A * Y ** alpha * (1.0 - b * cY + c * fStar - d * cGas)


CAND_YCONC_FSTAR_GASCONC = Candidate(
    name="cand_yconc_fstar_gasconc",
    func=cand_yconc_fstar_gasconc,
    feature_names=("Y200", "YConc", "mStar_over_mGas", "GasConc"),
    param_names=("A", "alpha", "b", "c", "d"),
    p0=(2.5e7, 0.62, 0.39, 1.28, 0.05),
    expression="A * Y200**alpha * (1 - b*YConc + c*fStar - d*GasConc)",
    token_count=13,  # +2 over round 2; justified by adding an independent feature
    motivation=(
        "Test whether GasConc carries information independent of YConc. "
        "Single new feature, single new parameter, single bracket -- "
        "stays close to the AGENTS.md token budget."
    ),
)


# Registry: round 0 (benchmark) first, then candidates in proposal order.
# The driver appends further rounds programmatically.
CANDIDATES: list[Candidate] = [
    BENCHMARK,
    CAND_YCONC,
    CAND_YCONC_FSTAR,
    CAND_YCONC_FSTAR_EXP,
    CAND_YCONC_FSTAR_GASCONC,
]


def pack_features(ds, feature_names: Sequence[str]) -> np.ndarray:
    """Stack the feature arrays from ``Dataset`` in the order requested.

    Recognised feature names:

      Y200, YConc, mGas, GasConc, mStar, mStar_over_mGas
    """
    rows = []
    for name in feature_names:
        if name == "Y200":
            rows.append(ds.Y200H)
        elif name == "YConc":
            rows.append(ds.YConcH)
        elif name == "mGas":
            rows.append(ds.mGasH)
        elif name == "GasConc":
            rows.append(ds.GasConcH)
        elif name == "mStar":
            rows.append(ds.mStarH)
        elif name == "mStar_over_mGas":
            rows.append(ds.mStarH / ds.mGasH)
        else:
            raise KeyError(f"Unknown feature: {name!r}")
    return np.vstack(rows)
