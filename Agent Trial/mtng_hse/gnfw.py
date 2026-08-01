"""gNFW pressure-profile model, its analytic logarithmic slope, and fitting.

Three deliberate departures from the original notebook implementation:

1. **Fit in log-log with unconstrained parameters.**  The old fit used bounded
   ``(P_0, x_c, beta)`` with ``P_0`` spanning [0.05, 1e4].  Parameters that span
   five decades make the Jacobian badly scaled and the bounds occasionally bind.
   Fitting ``(ln P0, ln xc, beta)`` unconstrained is better conditioned and lets
   the returned covariance mean something.

2. **The log-slope is analytic.**  The old code numerically differentiated the
   *fitted* profile with ``np.gradient`` even though a closed form exists.  That
   adds a discretisation error which is largest where the log grid is coarsest,
   i.e. in the outskirts -- exactly where M_HSE is evaluated.

3. **Fit quality is recorded, not discarded.**  ``rms_log_resid`` is both a
   quality cut and a physically meaningful feature: a halo whose pressure
   profile is poorly described by a smooth gNFW is a disturbed halo, and
   disturbance is the leading driver of hydrostatic bias.

The model, with ``u = x / x_c``:

    ln P(x) = ln P0 + gamma * ln u - beta * ln(1 + u^alpha)
    dlnP/dlnx = gamma - alpha * beta * u^alpha / (1 + u^alpha)

``alpha`` and ``gamma`` are held fixed (see ``config``); only
``(ln P0, ln xc, beta)`` are free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from . import config as C

__all__ = [
    "GNFWFit",
    "log_pressure",
    "log_slope",
    "fit_gnfw",
    "fit_gnfw_many",
    "local_log_slope",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def log_pressure(
    ln_x: np.ndarray,
    ln_P0: float,
    ln_xc: float,
    beta: float,
    alpha: float = C.GNFW_ALPHA,
    gamma: float = C.GNFW_GAMMA,
) -> np.ndarray:
    """Natural log of the gNFW pressure at ``ln_x = ln(r/R200c)``."""
    ln_u = ln_x - ln_xc
    # log1p(exp(alpha*ln_u)) computed stably for large u
    a_ln_u = alpha * ln_u
    ln_1pu = np.where(a_ln_u > 30.0, a_ln_u, np.log1p(np.exp(np.minimum(a_ln_u, 30.0))))
    return ln_P0 + gamma * ln_u - beta * ln_1pu


def log_slope(
    ln_x: np.ndarray | float,
    ln_xc: float,
    beta: float,
    alpha: float = C.GNFW_ALPHA,
    gamma: float = C.GNFW_GAMMA,
) -> np.ndarray | float:
    """Analytic ``dlnP/dlnr`` of the gNFW model. Always negative for beta > 0."""
    a_ln_u = alpha * (np.asarray(ln_x, dtype=float) - ln_xc)
    # u^alpha / (1 + u^alpha) = sigmoid(alpha * ln u), stable at both extremes
    frac = 1.0 / (1.0 + np.exp(-np.clip(a_ln_u, -700, 700)))
    return gamma - alpha * beta * frac


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------
@dataclass
class GNFWFit:
    """Outcome of fitting one halo's pressure profile."""

    ln_P0: float = np.nan
    ln_xc: float = np.nan
    beta: float = np.nan
    cov: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))
    rms_log_resid: float = np.nan     # RMS of ln(P_data) - ln(P_fit)
    max_log_resid: float = np.nan
    n_points: int = 0
    converged: bool = False

    @property
    def P0(self) -> float:
        return float(np.exp(self.ln_P0))

    @property
    def xc(self) -> float:
        return float(np.exp(self.ln_xc))

    def slope_at(self, r_over_r200: float | np.ndarray):
        """Analytic dlnP/dlnr at one or more radii."""
        return log_slope(np.log(r_over_r200), self.ln_xc, self.beta)

    def log_pressure_at(self, r_over_r200: float | np.ndarray):
        return log_pressure(np.log(r_over_r200), self.ln_P0, self.ln_xc, self.beta)

    def slope_sigma_at(self, r_over_r200: float) -> float:
        """1-sigma uncertainty on the log-slope, propagated from ``cov``.

        This is the number that shows *why* raw ``(x_c, beta)`` should not be
        used as features: they are strongly anti-correlated, but the slope --
        the combination that M_HSE actually depends on -- is well determined.
        """
        if not np.all(np.isfinite(self.cov)):
            return np.nan
        ln_x = np.log(r_over_r200)
        a = C.GNFW_ALPHA
        u_a = np.exp(a * (ln_x - self.ln_xc))
        f = u_a / (1.0 + u_a)
        # d(slope)/d(ln_P0) = 0 ; d/d(ln_xc) = +a^2*beta*f*(1-f) ; d/d(beta) = -a*f
        jac = np.array([0.0, a * a * self.beta * f * (1.0 - f), -a * f])
        var = float(jac @ self.cov @ jac)
        return float(np.sqrt(var)) if var > 0 else np.nan


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _initial_guess(ln_x: np.ndarray, ln_P: np.ndarray) -> tuple[float, float, float]:
    """Cheap, robust starting point from the end-to-end slope."""
    slope = (ln_P[-1] - ln_P[0]) / (ln_x[-1] - ln_x[0])
    beta0 = float(np.clip(-(slope - C.GNFW_GAMMA), 0.5, 20.0))
    ln_xc0 = float(np.log(0.5))
    # ln_P0 such that the model passes through the middle datum
    mid = len(ln_x) // 2
    ln_P0_0 = float(
        ln_P[mid] - (log_pressure(ln_x[mid], 0.0, ln_xc0, beta0))
    )
    return ln_P0_0, ln_xc0, beta0


def fit_gnfw(
    r_over_r200: np.ndarray,
    pressure: np.ndarray,
    fit_range: tuple[float, float] = C.FIT_RANGE,
    sigma: np.ndarray | None = None,
) -> GNFWFit:
    """Fit one halo's pressure profile.

    Parameters
    ----------
    r_over_r200
        Radii at which ``pressure`` is sampled.  Pass ``config.R_EFF`` (the
        volume-weighted shell centres), **not** the outer bin edges.
    pressure
        Pressure in any consistent units; only the shape matters for the slope,
        and ``ln_P0`` carries the normalisation.
    fit_range
        Radial window used for the fit.
    sigma
        Optional per-point uncertainty on ``ln P``.  Defaults to uniform.
    """
    r = np.asarray(r_over_r200, dtype=float)
    p = np.asarray(pressure, dtype=float)

    ok = np.isfinite(p) & (p > 0) & np.isfinite(r) & (r > 0)
    ok &= (r >= fit_range[0]) & (r <= fit_range[1])
    if ok.sum() < 5:
        return GNFWFit(n_points=int(ok.sum()))

    ln_x = np.log(r[ok])
    ln_P = np.log(p[ok])
    s = None if sigma is None else np.asarray(sigma, dtype=float)[ok]

    p0 = _initial_guess(ln_x, ln_P)
    try:
        popt, pcov = curve_fit(
            log_pressure, ln_x, ln_P, p0=p0, sigma=s,
            absolute_sigma=False, maxfev=20000,
        )
    except (RuntimeError, ValueError, TypeError):
        return GNFWFit(n_points=int(ok.sum()))

    resid = ln_P - log_pressure(ln_x, *popt)
    return GNFWFit(
        ln_P0=float(popt[0]),
        ln_xc=float(popt[1]),
        beta=float(popt[2]),
        cov=np.asarray(pcov, dtype=float),
        rms_log_resid=float(np.sqrt(np.mean(resid**2))),
        max_log_resid=float(np.max(np.abs(resid))),
        n_points=int(ok.sum()),
        converged=bool(np.all(np.isfinite(popt)) and popt[2] > 0),
    )


def fit_gnfw_many(
    r_over_r200: np.ndarray,
    pressure_profiles: np.ndarray,
    fit_range: tuple[float, float] = C.FIT_RANGE,
    progress: bool = False,
) -> list[GNFWFit]:
    """Fit every row of ``pressure_profiles`` (shape ``(n_halos, n_bins)``)."""
    out: list[GNFWFit] = []
    n = len(pressure_profiles)
    for i, prof in enumerate(pressure_profiles):
        out.append(fit_gnfw(r_over_r200, prof, fit_range=fit_range))
        if progress and (i % 250 == 0):
            print(f"  gNFW fit {i}/{n}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Model-free slope
# ---------------------------------------------------------------------------
def local_log_slope(
    r_over_r200: np.ndarray,
    pressure: np.ndarray,
    r_target: float = 1.0,
    window: tuple[float, float] = C.LOCAL_SLOPE_WINDOW,
) -> float:
    """Least-squares ``dlnP/dlnr`` from the data alone, in a radial window.

    Independent of the gNFW parameterisation.  Comparing this with
    ``GNFWFit.slope_at(r_target)`` separates "the pressure profile really is
    this steep" from "the three-parameter model says it is".  The difference
    between the two is a genuine systematic on b and should be reported as one.
    """
    r = np.asarray(r_over_r200, dtype=float)
    p = np.asarray(pressure, dtype=float)
    ok = np.isfinite(p) & (p > 0) & (r >= window[0]) & (r <= window[1])
    if ok.sum() < 3:
        return np.nan
    ln_x = np.log(r[ok] / r_target)
    ln_P = np.log(p[ok])
    slope, _ = np.polyfit(ln_x, ln_P, 1)
    return float(slope)
