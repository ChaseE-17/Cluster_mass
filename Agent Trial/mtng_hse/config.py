"""Single source of truth for constants, units and grid geometry.

Every magic number in the old notebook lives here, named, with the reasoning
attached.  Nothing else in the package is allowed to hard-code a constant, a
bin index, or a unit conversion.

Unit conventions
----------------
Simulation ("TNG") units, as produced by Leander Thiele's ``group_particles``
profile code:

    mass    : 1e10 Msun / h
    length  : Mpc / h
    time    : Gyr / h

The radial profile grid is expressed in units of ``r / R200c`` (dimensionless),
NOT in Mpc -- ``cumulative_profile`` multiplies by ``r200**3``, which only makes
sense if the bin volumes are dimensionless.  The comment in the original
notebook ("[Mpc] - Radial bin edges") is wrong; this is the fix.

The raw ``y_profiles.npy`` arrays carry an extra (kpc/Mpc)^2 = 1e-6 factor
inherited from the ``Y_scaling`` convention, which is why ``P_SCALING`` below
has a 1e-6 in it.  This is exactly the kind of factor that a fitted normalisation
would silently absorb, so ``tests/test_pipeline.py::test_physical_ranges``
checks it against the implied gas temperature instead of trusting it.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cosmology (MillenniumTNG)
# ---------------------------------------------------------------------------
H_LITTLE = 0.6774          # h
OMEGA_M = 0.3089
XH = 0.76                  # hydrogen mass fraction

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
G_CGS = 6.67430e-8         # cm^3 g^-1 s^-2
PROTON_G = 1.6726219e-24   # g
MSUN_G = 1.98892e33        # g
KEV_ERG = 1.602176634e-9   # erg per keV

# Mean molecular weights for a fully ionised H/He plasma
MU_E = 2.0 / (1.0 + XH)            # per free electron        ~1.136
MU = 4.0 / (3.0 + 5.0 * XH)        # per particle             ~0.588
PTH_OVER_PE = MU_E / MU            # P_thermal / P_electron   ~1.932

# ---------------------------------------------------------------------------
# Simulation unit -> CGS
# ---------------------------------------------------------------------------
TNG_TO_MSUN = 1e10 / H_LITTLE               # 1e10 Msun/h -> Msun
TNG_TO_G = TNG_TO_MSUN * MSUN_G             # 1e10 Msun/h -> g
TNG_TO_CM = 3.0857e24 / H_LITTLE            # Mpc/h       -> cm
TNG_TO_S = 3.15576e16 / H_LITTLE            # Gyr/h       -> s

PROTON_TO_TNG = PROTON_G / TNG_TO_G

# ne_profiles.npy -> gas mass density in simulation units
# (n_e -> rho_gas via rho = n_e * mu_e * m_p)
RHO_SCALING = MU_E * PROTON_TO_TNG

# y_profiles.npy -> electron pressure in CGS (g cm^-1 s^-2 = barye)
# [1e10 Msun/h][Gyr/h]^-2[Mpc/h]^-1 * (kpc/Mpc)^2, hence the 1e-6.
P_SCALING = TNG_TO_G / TNG_TO_S**2 / TNG_TO_CM * 1e-6

# Compton-y integrated flux scaling (unchanged from the reference pipeline)
Y_SCALING = (
    1e-6 * 6.65e-25 / (9.1e-28 * 3e10**2) * 1.98e43 / H_LITTLE / (3.154e16) ** 2
)

# ---------------------------------------------------------------------------
# Radial grid  (Leander's 128-bin convention)
# ---------------------------------------------------------------------------
N_BINS = 128
R_MIN_GRID = 0.03          # r / R200c
R_MAX_GRID = 2.5           # r / R200c

#: Bin *edges*, length N_BINS + 1.  Edge 0 is exactly 0, so bin 0 is a sphere
#: of radius 0.03 R200c rather than a shell -- it is excluded everywhere.
RBIN_EDGES = np.append(0.0, np.geomspace(R_MIN_GRID, R_MAX_GRID, num=N_BINS))

#: Outer edge of each bin, length N_BINS.  ``cumulative_profile()[:, i]`` is the
#: integrated quantity inside ``RBIN_OUTER[i]``.  Using this array (rather than
#: a hard-coded index) removes the off-by-one in the original aperture
#: definitions, where ``mGas_r[:, 101]`` was labelled "R200c" but actually
#: measured the mass inside 1.011 R200c.
RBIN_OUTER = RBIN_EDGES[1:]

#: Shell volumes in units of R200c^3.
VOL_BIN = (4.0 / 3.0) * np.pi * (RBIN_EDGES[1:] ** 3 - RBIN_EDGES[:-1] ** 3)


def effective_radii(edges: np.ndarray = RBIN_EDGES) -> np.ndarray:
    """Volume-weighted mean radius of each shell, in units of R200c.

    A profile array stores <rho> averaged over the shell, so the radius it
    should be plotted (and fitted) against is the volume-weighted mean

        r_eff = (3/4) (r2^4 - r1^4) / (r2^3 - r1^3),

    not the outer edge.  The original notebook fitted the profile values
    against ``RBIN_EDGES[1:]`` (the outer edges), a systematic ~1.7% inward
    radial offset on this grid.  Small, but it propagates straight into x_c,
    into the log-slope at R200c, and therefore into M_HSE.

    For the 3.5%-wide log bins used here the choice of centre convention
    (volume-weighted / geometric / arithmetic) matters at the 1e-4 level;
    ``tests`` quantifies this so the claim is checked rather than asserted.
    """
    r1, r2 = edges[:-1], edges[1:]
    return 0.75 * (r2**4 - r1**4) / (r2**3 - r1**3)


R_EFF = effective_radii()

# ---------------------------------------------------------------------------
# gNFW pressure model
# ---------------------------------------------------------------------------
# P(x) = P0 (x/xc)^GAMMA [1 + (x/xc)^ALPHA]^(-beta)
# so the inner log-slope is GAMMA and the outer log-slope is GAMMA - ALPHA*beta.
GNFW_ALPHA = 1.0
GNFW_GAMMA = -0.3

#: Radial range used for the gNFW fit, in r/R200c.  The original fit used the
#: full 0.03-2.5 grid, which lets the AGN-shaped core (where a 3-parameter gNFW
#: has no business fitting) drive the outer slope that M_HSE actually depends
#: on.  Restricting the range is a *choice*; ``sweep_fit_range()`` in the
#: notebook quantifies how much b moves when you change it.
FIT_RANGE = (0.10, 2.00)

#: Window for the model-free local log-slope estimator at R200c.
LOCAL_SLOPE_WINDOW = (0.70, 1.40)

# ---------------------------------------------------------------------------
# Apertures used for summary scalars, in r/R200c
# ---------------------------------------------------------------------------
APERTURES = {
    "0p15": 0.15,
    "0p25": 0.25,
    "0p50": 0.50,
    "0p65": 0.65,   # stand-in for R500c until real R500c is available
    "0p75": 0.75,
    "1p00": 1.00,
}

# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
MASS_CUT = 1e4             # 1e10 Msun/h  ->  1e14 Msun/h
MAX_LOG_RESID = 0.25       # reject gNFW fits worse than this RMS in ln P

# ---------------------------------------------------------------------------
# Physical plausibility ranges, asserted by the test suite.  If a unit
# conversion is wrong, one of these will fail loudly instead of being absorbed
# into a fitted normalisation.
# ---------------------------------------------------------------------------
SANITY = {
    "kT_R200_keV": (0.5, 12.0),    # gas temperature at R200c
    "f_gas": (0.02, 0.20),         # M_gas(<R200c) / M200c
    "f_star": (0.002, 0.05),       # M_star(<R200c) / M200c
    "bias": (-0.15, 0.55),         # b = 1 - M_HSE/M200c
    "slope_R200": (-6.0, -1.5),    # dlnP/dlnr at R200c
}
