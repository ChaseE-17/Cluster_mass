"""Reproduces the data-loading machinery of MTNG_copy.py verbatim.

Per AGENTS.md we must use the *same* H-suffixed arrays and the *same*
50/50 train/test split (`np.random.default_rng(0)`) as the reference
script. We re-implement (rather than import) because MTNG_copy.py is a
percent-formatted notebook with side-effecting top-level code; this
keeps the deliverables runnable as plain Python while staying byte-for-
byte equivalent on the relevant arrays.

Returns a single namespace object with all derived quantities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---- Constants identical to MTNG_copy.py ---------------------------------

DIRE_Z05 = "/mnt/c/Users/cenlo/Desktop/scatter_proj/MTNG_data/z=0.5/"
DIRE_Z00 = "/mnt/c/Users/cenlo/Desktop/scatter_proj/MTNG_data/z=0.0/"

REDSHIFT_2ND = 0.5
OMEGA_M = 0.3089

# Bin convention from MTNG_copy.py
RBIN_EDGES = np.append(0, np.geomspace(0.03, 2.5, num=128))
R_BINS = 0.5 * (RBIN_EDGES[1:] + RBIN_EDGES[:-1])
VOL_BIN = (4.0 / 3.0) * np.pi * (RBIN_EDGES[1:] ** 3 - RBIN_EDGES[:-1] ** 3)
INNER_BIN_IDX = 81
R200_BIN_IDX = 101

XH = 0.76
MASS_CUT = 5e3

Y_SCALING = 1e-6 * (6.65e-25) / (9.1e-28 * 3e10 ** 2) * 1.98e43 / 0.6774 / (3.154e16) ** 2
RHO_SCALING = 2 * 1.6726219e-24 / (1.98e43 / 0.6774) / (1 + XH)


def cumulative_profile(profiles_in: np.ndarray, r200: np.ndarray) -> np.ndarray:
    """Cumulative integral of a density profile (verbatim from MTNG_copy.py)."""
    profiles = profiles_in.copy()
    num_halos, num_bins = profiles.shape
    profiles *= (r200[:, np.newaxis] ** 3) * VOL_BIN[np.newaxis, :]
    cum = np.zeros_like(profiles)
    cum[:, 0] = profiles[:, 0]
    for i in range(1, num_bins):
        cum[:, i] = cum[:, i - 1] + profiles[:, i]
    return cum


@dataclass
class Dataset:
    """Bundle of arrays after the mass cut and split.

    All H-suffixed arrays are aligned 1:1.  ``mask_test`` is a boolean
    array of the same length.  Profiles are stored *un-cut* alongside a
    boolean ``mass_mask`` so that any new profile-derived scalar can be
    rebuilt easily without re-loading from disk.
    """

    # Halo-level scalars after mass cut
    m200H: np.ndarray
    Y200H: np.ndarray
    mGasH: np.ndarray
    GasConcH: np.ndarray
    YConcH: np.ndarray
    mStarH: np.ndarray
    r200H: np.ndarray
    zH: np.ndarray
    # Profiles after mass cut (each row = one halo); already physically scaled
    yProfH: np.ndarray   # Compton-y density profile
    neProfH: np.ndarray  # gas density profile
    mStarProfH: np.ndarray  # stellar density profile
    # Cumulative-mass / Y-aperture profiles after mass cut
    Y_rH: np.ndarray
    mGas_rH: np.ndarray
    mStar_rH: np.ndarray
    # Radial bin metadata
    r_bins: np.ndarray
    inner_bin_idx: int
    r200_bin_idx: int
    # 50/50 train/test split (length = number of halos after mass cut)
    mask_test: np.ndarray


def load_dataset() -> Dataset:
    """Load + concatenate z=0.5 and z=0 catalogues exactly as MTNG_copy.py does."""
    # ---- z=0.5 catalogue ------------------------------------------------
    Y200 = np.load(os.path.join(DIRE_Z05, "Y200c.npy"))
    m200 = np.load(os.path.join(DIRE_Z05, "M200c.npy"))
    r200 = np.load(os.path.join(DIRE_Z05, "R200c.npy"))
    yProf_original = np.load(os.path.join(DIRE_Z05, "y_profiles.npy"))
    neProf = np.load(os.path.join(DIRE_Z05, "ne_profiles.npy"))
    mStarProf = np.load(os.path.join(DIRE_Z05, "mStar_profiles.npy"))

    # E(z) array bookkeeping (verbatim from MTNG_copy.py).
    Ez = np.sqrt(OMEGA_M * (1 + REDSHIFT_2ND) ** 3 + 1.0 - OMEGA_M)
    n_total = 37371 + 31649
    mask = np.arange(n_total) < 31649
    Ez = np.ones(n_total) * Ez
    Ez[~mask] = 1.0
    z = np.zeros_like(Ez)
    z[mask] = REDSHIFT_2ND

    # ---- z=0 catalogue (concatenated onto z=0.5 arrays) -----------------
    m200 = np.concatenate((m200, np.load(os.path.join(DIRE_Z00, "M200c.npy"))))
    m200 = m200 * Ez ** (2.0 / 5.0)  # absorbs E(z)^{2/5} into the mass target
    r200 = np.concatenate((r200, np.load(os.path.join(DIRE_Z00, "R200c.npy"))))
    Y200 = np.concatenate((Y200, np.load(os.path.join(DIRE_Z00, "Y200c.npy"))))
    neProf = np.concatenate((neProf, np.load(os.path.join(DIRE_Z00, "ne_profiles.npy"))))
    mStarProf = np.concatenate((mStarProf, np.load(os.path.join(DIRE_Z00, "mStar_profiles.npy"))))
    yProf_original = np.concatenate(
        (yProf_original, np.load(os.path.join(DIRE_Z00, "y_profiles.npy")))
    )

    # ---- Physical scaling ----------------------------------------------
    # MTNG_copy.py applies the scaling to yProf and neProf *before*
    # cumulative_profile, and never re-multiplies. We replicate that.
    Y200 = Y200 * Y_SCALING
    neProf = neProf * RHO_SCALING
    yProf_scaled = yProf_original * Y_SCALING

    # ---- Cumulative profiles -------------------------------------------
    Y_r = cumulative_profile(yProf_scaled, r200)
    mGas_r = cumulative_profile(neProf, r200)
    mStar_r = cumulative_profile(mStarProf, r200)

    GasConc = mGas_r[:, INNER_BIN_IDX] / mGas_r[:, R200_BIN_IDX]
    mGas = mGas_r[:, R200_BIN_IDX]
    mStar = mStar_r[:, R200_BIN_IDX]
    YConc = Y_r[:, INNER_BIN_IDX] / Y_r[:, R200_BIN_IDX]

    # ---- Mass cut -------------------------------------------------------
    mass_mask = m200 > MASS_CUT
    m200H = m200[mass_mask]
    Y200H = Y200[mass_mask]
    mGasH = mGas[mass_mask]
    GasConcH = GasConc[mass_mask]
    YConcH = YConc[mass_mask]
    mStarH = mStar[mass_mask]
    r200H = r200[mass_mask]
    zH = z[mass_mask]
    yProfH = yProf_scaled[mass_mask]
    neProfH = neProf[mass_mask]
    mStarProfH = mStarProf[mass_mask]
    Y_rH = Y_r[mass_mask]
    mGas_rH = mGas_r[mass_mask]
    mStar_rH = mStar_r[mass_mask]

    # ---- Train/test split (identical to MTNG_copy.py) -------------------
    rng = np.random.default_rng(0)
    mask_test = rng.choice([True, False], size=len(m200H), p=[0.5, 0.5])

    return Dataset(
        m200H=m200H,
        Y200H=Y200H,
        mGasH=mGasH,
        GasConcH=GasConcH,
        YConcH=YConcH,
        mStarH=mStarH,
        r200H=r200H,
        zH=zH,
        yProfH=yProfH,
        neProfH=neProfH,
        mStarProfH=mStarProfH,
        Y_rH=Y_rH,
        mGas_rH=mGas_rH,
        mStar_rH=mStar_rH,
        r_bins=R_BINS,
        inner_bin_idx=INNER_BIN_IDX,
        r200_bin_idx=R200_BIN_IDX,
        mask_test=mask_test,
    )


if __name__ == "__main__":
    ds = load_dataset()
    n_total = ds.m200H.size
    n_train = (~ds.mask_test).sum()
    n_test = ds.mask_test.sum()
    print(f"Halos after mass cut (m200 > {MASS_CUT:g}): {n_total}")
    print(f"  train: {n_train}, test: {n_test}")
    print(f"  m200H range: [{ds.m200H.min():.3g}, {ds.m200H.max():.3g}] (1e10 Msun/h)")
    print(f"  Y200H range: [{ds.Y200H.min():.3g}, {ds.Y200H.max():.3g}]")
    print(f"  YConcH range: [{ds.YConcH.min():.3g}, {ds.YConcH.max():.3g}]")
    print(f"  GasConcH range: [{ds.GasConcH.min():.3g}, {ds.GasConcH.max():.3g}]")
