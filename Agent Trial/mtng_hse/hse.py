"""Hydrostatic mass and hydrostatic mass bias.

The hydrostatic equilibrium relation, written in the log-slope form actually
used here:

    M_HSE(r) = - r * P_th(r) / (G * rho_gas(r)) * dlnP/dlnr |_r

Everything on the right is measurable.  That is the point: **the hydrostatic
mass is not a black box**, it is "gas temperature at r" times "how steeply the
pressure is falling at r", and the whole modelling problem is how those two
factors relate to the true mass.

Bias convention
---------------
This module uses the literature definition

    M_HSE = (1 - b) * M_true      =>      b = 1 - M_HSE / M_true

so b is **positive** when the hydrostatic mass under-estimates the truth, which
is the usual case.  The original notebook used ``M_HSE/M_true - 1`` (negative),
which is the same information with the opposite sign; mixing the two across a
project is a reliable way to get a sign error into a fitted coefficient, so the
convention is pinned here and asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .gnfw import GNFWFit, fit_gnfw, local_log_slope
from .profiles import interp_profile

__all__ = ["HSEResult", "hse_mass_cgs", "hse_at_radius", "bias_from_identity"]


@dataclass
class HSEResult:
    """Per-halo hydrostatic quantities evaluated at one radius."""

    r_over_r200: float = np.nan
    m_hse_msun: float = np.nan
    m_true_msun: float = np.nan
    bias: float = np.nan              # b = 1 - M_HSE / M_true
    slope_fit: float = np.nan         # dlnP/dlnr from the gNFW fit
    slope_local: float = np.nan       # dlnP/dlnr from a local power-law fit
    slope_sigma: float = np.nan
    P_th_cgs: float = np.nan
    rho_gas_cgs: float = np.nan
    kT_keV: float = np.nan            # mu * m_p * P_th / rho_gas
    rms_log_resid: float = np.nan
    beta: float = np.nan
    xc: float = np.nan
    ok: bool = False


def hse_mass_cgs(
    r_cm: float | np.ndarray,
    P_th_cgs: float | np.ndarray,
    rho_gas_cgs: float | np.ndarray,
    dlnP_dlnr: float | np.ndarray,
) -> float | np.ndarray:
    """M_HSE in grams. Single place where the HSE algebra lives."""
    return -(r_cm * P_th_cgs / (C.G_CGS * rho_gas_cgs)) * dlnP_dlnr


def bias_from_identity(
    r200_mpc_h: float,
    P_th_cgs: float,
    rho_gas_cgs: float,
    dlnP_dlnr: float,
    m_true_msun: float,
) -> float:
    """b computed straight from the four measurable numbers.

    Used by ``tests/test_pipeline.py::test_hse_identity`` as an independent
    re-derivation of the pipeline's answer.  If the pipeline and this function
    disagree, the pipeline has a bug -- there is nothing else it could be.
    """
    m_hse_g = hse_mass_cgs(
        r200_mpc_h * C.TNG_TO_CM, P_th_cgs, rho_gas_cgs, dlnP_dlnr
    )
    return 1.0 - (m_hse_g / C.MSUN_G) / m_true_msun


def hse_at_radius(
    p_electron_profile: np.ndarray,
    rho_gas_profile: np.ndarray,
    r200_mpc_h: float,
    m200_tng: float,
    r_over_r200: float = 1.0,
    fit: GNFWFit | None = None,
    fit_range: tuple[float, float] = C.FIT_RANGE,
    slope_source: str = "fit",
    p_source: str = "data",
) -> HSEResult:
    """Evaluate M_HSE and b for one halo at ``r_over_r200``.

    Parameters
    ----------
    p_electron_profile
        Electron pressure per shell, CGS (barye), sampled at ``config.R_EFF``.
    rho_gas_profile
        Gas mass density per shell, CGS (g/cm^3), sampled at ``config.R_EFF``.
    r200_mpc_h, m200_tng
        R200c in Mpc/h and M200c in 1e10 Msun/h.
    slope_source
        ``"fit"`` uses the analytic gNFW slope; ``"local"`` uses a model-free
        power-law fit in a window around the evaluation radius.  Running both
        and differencing gives you the model systematic for free.
    p_source
        ``"data"`` interpolates the measured pressure (recommended -- the fit
        then only supplies the derivative); ``"fit"`` reproduces the original
        notebook behaviour, where the fitted profile supplied both.

    Notes
    -----
    Evaluation happens at *exactly* ``r_over_r200``, by interpolation. No bin
    index is involved, so pressure, density and radius are guaranteed to refer
    to the same place -- which was not true before.
    """
    res = HSEResult(r_over_r200=float(r_over_r200))

    p_e = np.asarray(p_electron_profile, dtype=float)
    rho = np.asarray(rho_gas_profile, dtype=float)

    if fit is None:
        fit = fit_gnfw(C.R_EFF, p_e, fit_range=fit_range)
    res.rms_log_resid = fit.rms_log_resid
    res.beta = fit.beta
    res.xc = fit.xc
    if not fit.converged:
        return res

    res.slope_fit = float(fit.slope_at(r_over_r200))
    res.slope_sigma = fit.slope_sigma_at(r_over_r200)
    res.slope_local = local_log_slope(C.R_EFF, p_e, r_target=r_over_r200)
    slope = res.slope_fit if slope_source == "fit" else res.slope_local

    if p_source == "data":
        p_e_at_r = interp_profile(p_e, r_over_r200)
    else:
        p_e_at_r = float(np.exp(fit.log_pressure_at(r_over_r200)))

    rho_at_r = interp_profile(rho, r_over_r200)
    if not (np.isfinite(p_e_at_r) and np.isfinite(rho_at_r) and rho_at_r > 0):
        return res

    # Total thermal pressure from the electron pressure the profiles store.
    res.P_th_cgs = float(p_e_at_r * C.PTH_OVER_PE)
    res.rho_gas_cgs = float(rho_at_r)

    r_cm = r_over_r200 * r200_mpc_h * C.TNG_TO_CM
    m_hse_g = hse_mass_cgs(r_cm, res.P_th_cgs, res.rho_gas_cgs, slope)

    res.m_hse_msun = float(m_hse_g / C.MSUN_G)
    res.m_true_msun = float(m200_tng * C.TNG_TO_MSUN)
    res.bias = 1.0 - res.m_hse_msun / res.m_true_msun

    # kT = mu * m_p * P_th / rho_gas -- the unit canary.
    res.kT_keV = float(
        C.MU * C.PROTON_G * res.P_th_cgs / res.rho_gas_cgs / C.KEV_ERG
    )

    res.ok = bool(
        np.isfinite(res.bias)
        and res.m_hse_msun > 0
        and fit.rms_log_resid < C.MAX_LOG_RESID
    )
    return res
