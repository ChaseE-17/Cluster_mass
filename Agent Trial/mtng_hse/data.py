"""Loading and per-halo table construction.

Everything here is a pure function of ``(data_dir, snapshot)``.  No module-level
array is mutated, so re-executing a cell or re-importing cannot double-apply a
unit conversion -- the failure mode behind ``neProf *= rho_scaling`` in the
original notebook, which silently changed every result on a second run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import config as C
from .gnfw import fit_gnfw
from .hse import hse_at_radius
from .profiles import aperture_value, concentration, cumulative_profile

__all__ = ["Snapshot", "load_snapshot", "build_table"]


@dataclass
class Snapshot:
    """Raw + scaled arrays for one redshift. All arrays share row order."""

    m200: np.ndarray        # 1e10 Msun/h
    r200: np.ndarray        # Mpc/h
    P_e: np.ndarray         # electron pressure per shell, CGS barye
    rho_gas: np.ndarray     # gas mass density per shell, CGS g/cm^3
    y_prof: np.ndarray      # raw Compton-y density profile
    mstar_prof: np.ndarray  # raw stellar density profile
    redshift: float

    def __len__(self) -> int:
        return len(self.m200)


def load_snapshot(data_dir: str, snapshot: str = "z=0.0", redshift: float = 0.0) -> Snapshot:
    """Load one snapshot and apply unit scalings to *copies*."""
    base = os.path.join(data_dir, snapshot)

    def _load(name):
        return np.load(os.path.join(base, name))

    m200 = _load("M200c.npy").astype(float)
    r200 = _load("R200c.npy").astype(float)
    y_prof = _load("y_profiles.npy").astype(float)
    ne_prof = _load("ne_profiles.npy").astype(float)
    mstar_prof = _load("mStar_profiles.npy").astype(float)

    # NOTE: no E(z)^{2/5} factor is applied. At z=0 it is 1 anyway, and baking
    # it into the mass makes the target redshift-calibrated rather than
    # self-similar. If z=0.5 is added, handle it explicitly at that point.
    P_e = y_prof * C.P_SCALING
    rho_gas = ne_prof * C.RHO_SCALING * C.TNG_TO_G / C.TNG_TO_CM**3

    return Snapshot(
        m200=m200, r200=r200, P_e=P_e, rho_gas=rho_gas,
        y_prof=y_prof, mstar_prof=mstar_prof, redshift=redshift,
    )


def build_table(
    snap: Snapshot,
    mass_cut: float = C.MASS_CUT,
    r_eval: float = 1.0,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Per-halo table of the target and every registered feature.

    Returns a dict of 1-D arrays, all the same length, restricted to halos
    above ``mass_cut``.  Rows with a failed gNFW fit are kept but flagged via
    ``ok`` so that the failure rate is visible rather than silently dropped.
    """
    sel = np.where(snap.m200 > mass_cut)[0]
    n = len(sel)
    if progress:
        print(f"{n} halos above M200c = {mass_cut:.3g} (1e10 Msun/h)")

    # --- aperture quantities ------------------------------------------------
    cum_Y = cumulative_profile(snap.y_prof[sel] * C.Y_SCALING, snap.r200[sel])
    cum_gas = cumulative_profile(snap.rho_gas[sel] * C.TNG_TO_CM**3 / C.TNG_TO_G,
                                 snap.r200[sel])
    cum_star = cumulative_profile(snap.mstar_prof[sel], snap.r200[sel])

    m_gas = aperture_value(cum_gas, 1.0)
    m_star = aperture_value(cum_star, 1.0)

    out: dict[str, np.ndarray] = {
        "halo_id": sel.astype(float),
        "M200c": snap.m200[sel] * C.TNG_TO_MSUN,
        "R200c": snap.r200[sel],
        "c_Y_0p50": concentration(cum_Y, 0.50),
        "c_Y_0p15": concentration(cum_Y, 0.15),
        "c_gas_0p50": concentration(cum_gas, 0.50),
        "c_gas_0p15": concentration(cum_gas, 0.15),
        "Y_ratio_0p65": concentration(cum_Y, 0.65),
        "Mstar_over_Mgas": m_star / m_gas,
        "M_gas": m_gas * C.TNG_TO_MSUN,
        "M_star": m_star * C.TNG_TO_MSUN,
    }
    out["f_gas_true"] = m_gas / snap.m200[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.clip(out["c_Y_0p50"], 1e-6, 1 - 1e-6)
        out["logit_c_Y_0p50"] = np.log(c / (1 - c))

    # --- HSE quantities -----------------------------------------------------
    keys = ["M_hse", "bias", "slope_R200", "slope_local_R200", "slope_sigma",
            "kT_R200", "gnfw_rms_resid", "beta", "xc", "ok"]
    for k in keys:
        out[k] = np.full(n, np.nan)

    for j, i in enumerate(sel):
        if progress and j % 250 == 0:
            print(f"  HSE {j}/{n}", flush=True)
        r = hse_at_radius(
            snap.P_e[i], snap.rho_gas[i], snap.r200[i], snap.m200[i],
            r_over_r200=r_eval,
        )
        out["M_hse"][j] = r.m_hse_msun
        out["bias"][j] = r.bias
        out["slope_R200"][j] = r.slope_fit
        out["slope_local_R200"][j] = r.slope_local
        out["slope_sigma"][j] = r.slope_sigma
        out["kT_R200"][j] = r.kT_keV
        out["gnfw_rms_resid"][j] = r.rms_log_resid
        out["beta"][j] = r.beta
        out["xc"][j] = r.xc
        out["ok"][j] = float(r.ok)

    out["f_gas_hse"] = out["M_gas"] / out["M_hse"]

    if progress:
        frac = np.nanmean(out["ok"])
        print(f"gNFW fit + HSE succeeded for {frac:.1%} of halos")
    return out
