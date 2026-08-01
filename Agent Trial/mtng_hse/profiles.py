"""Radial-profile utilities: cumulative integration and aperture extraction.

The single most consequential change here is that **apertures are specified by
radius, not by bin index**.

In the original notebook, ``R200_BIN_IDX = 101`` was used to index cumulative
profiles.  But ``cumulative_profile()[:, i]`` is the quantity inside the
*outer edge* of bin ``i``, which is ``RBIN_EDGES[i+1]``.  On this grid
``RBIN_EDGES[102] = 1.011``, so every quantity labelled "inside R200c" was
actually measured inside 1.011 R200c.  Meanwhile the *density* profiles indexed
at 101 refer to a shell centred on 0.993 R200c.  M_HSE mixes both, so the same
expression evaluated ρ at 0.993 and r at 1.011 -- a ~1.8% inconsistency in a
quantity whose signal (the bias) is ~15%.

Specifying the radius and interpolating removes the whole class of error.
"""

from __future__ import annotations

import numpy as np

from . import config as C

__all__ = [
    "cumulative_profile",
    "aperture_value",
    "interp_profile",
    "concentration",
]


def cumulative_profile(
    density_profiles: np.ndarray,
    r200: np.ndarray,
    vol_bin: np.ndarray = C.VOL_BIN,
) -> np.ndarray:
    """Integrate shell densities into enclosed totals.

    Parameters
    ----------
    density_profiles
        Shape ``(n_halos, n_bins)``.  Volume density averaged over each shell,
        in simulation units per (Mpc/h)^3.
    r200
        Shape ``(n_halos,)``, in Mpc/h.  The grid is dimensionless (r/R200c),
        so physical shell volumes are ``r200^3 * vol_bin``.

    Returns
    -------
    ndarray
        ``out[:, i]`` = total inside ``config.RBIN_OUTER[i]``.

    Notes
    -----
    Pure: the input array is never modified.  The original implementation used
    ``profiles *= ...`` on a copy inside the function but the *callers* did
    ``neProf *= rho_scaling`` at module scope, which double-applies on cell
    re-execution.  Everything here takes and returns fresh arrays.
    """
    prof = np.asarray(density_profiles, dtype=float)
    shell_totals = prof * (np.asarray(r200, dtype=float)[:, None] ** 3) * vol_bin[None, :]
    return np.cumsum(shell_totals, axis=1)


def aperture_value(
    cumulative: np.ndarray,
    r_target: float,
    r_outer: np.ndarray = C.RBIN_OUTER,
) -> np.ndarray:
    """Enclosed quantity inside ``r_target`` (in r/R200c), log-log interpolated.

    Works on a single profile (1-D) or a stack (2-D, one halo per row).
    """
    cum = np.atleast_2d(np.asarray(cumulative, dtype=float))
    ln_r = np.log(r_outer)
    ln_t = np.log(r_target)

    out = np.full(cum.shape[0], np.nan)
    for i, row in enumerate(cum):
        good = np.isfinite(row) & (row > 0)
        if good.sum() < 2:
            continue
        out[i] = np.exp(np.interp(ln_t, ln_r[good], np.log(row[good])))
    return out if np.ndim(cumulative) == 2 else out[0]


def interp_profile(
    profile: np.ndarray,
    r_target: float | np.ndarray,
    r_eff: np.ndarray = C.R_EFF,
) -> float | np.ndarray:
    """Log-log interpolate a *density-like* profile to ``r_target``.

    Uses the volume-weighted shell centres, so a value is never silently
    associated with a bin edge.
    """
    prof = np.asarray(profile, dtype=float)
    good = np.isfinite(prof) & (prof > 0) & np.isfinite(r_eff) & (r_eff > 0)
    if good.sum() < 2:
        return np.nan if np.isscalar(r_target) else np.full(np.shape(r_target), np.nan)
    val = np.exp(
        np.interp(np.log(r_target), np.log(r_eff[good]), np.log(prof[good]))
    )
    return float(val) if np.isscalar(r_target) else val


def concentration(
    cumulative: np.ndarray,
    r_inner: float,
    r_outer_ap: float = 1.0,
) -> np.ndarray:
    """Aperture concentration ``Q(<r_inner) / Q(<r_outer_ap)``.

    This is the ``c_Y`` / ``c_gas`` family, now with both apertures named by
    radius so that e.g. ``c_gas(0.15)`` and ``c_gas(0.5)`` are unambiguous.
    """
    inner = aperture_value(cumulative, r_inner)
    outer = aperture_value(cumulative, r_outer_ap)
    with np.errstate(divide="ignore", invalid="ignore"):
        return inner / outer
