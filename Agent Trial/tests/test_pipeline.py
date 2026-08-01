#!/usr/bin/env python3
"""Validation suite for the MTNG hydrostatic-bias pipeline.

Run standalone::

    python tests/test_pipeline.py                 # synthetic only, no data needed
    python tests/test_pipeline.py --data DIR      # + real-data sanity checks

or under pytest::

    pytest tests/test_pipeline.py -v

Design principle: **a test that needs the real data cannot be the first line of
defence**, because it only tells you something is wrong, never what.  So the
suite is built around synthetic halos whose true mass is known analytically.
If the algebra, the units and the fitter are right, the pipeline recovers that
mass exactly (Tier 1) or to a quantified tolerance (Tier 2).  Only then do the
real-data checks (Tier 3) mean anything.

Tier 2 is the scientifically interesting part: it measures how much apparent
hydrostatic bias the *fitting procedure alone* injects into a halo that is, by
construction, in perfect hydrostatic equilibrium.  Any measured b below that
floor is a fitting artefact, not physics.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtng_hse import config as C
from mtng_hse.gnfw import (
    GNFWFit, fit_gnfw, local_log_slope, log_pressure, log_slope,
)
from mtng_hse.hse import bias_from_identity, hse_at_radius, hse_mass_cgs
from mtng_hse.profiles import aperture_value, cumulative_profile, interp_profile

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    _RESULTS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}")
    if detail:
        print(textwrap.indent(detail.rstrip(), "         "))


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Synthetic halo builders
# ---------------------------------------------------------------------------
def powerlaw_halo(
    r200_mpc_h: float = 1.5,
    kT_keV: float = 5.0,
    a_rho: float = 2.0,
    p_P: float = 2.5,
):
    """A halo whose pressure and gas density are exact power laws.

    For P = B r^-p and rho = A r^-a, hydrostatic equilibrium gives, exactly,

        M(r) = p * B / (G * A) * r^(1 + a - p)

    with no approximation.  This isolates the algebra and the unit chain: the
    log-slope is recovered exactly by any sane estimator, so a failure here is
    a factor of h, a Mpc->cm, or a mu.
    """
    r = C.R_EFF.copy()
    rho0 = 1e-27                                   # g/cm^3 at r = R200c
    P_th0 = kT_keV * C.KEV_ERG * rho0 / (C.MU * C.PROTON_G)

    rho_gas = rho0 * r ** (-a_rho)
    P_th = P_th0 * r ** (-p_P)
    P_e = P_th / C.PTH_OVER_PE                     # pipeline multiplies it back

    r200_cm = r200_mpc_h * C.TNG_TO_CM
    m_true_g = (p_P * P_th0) / (C.G_CGS * rho0) * r200_cm   # at r/R200c = 1
    m_true_tng = m_true_g / C.MSUN_G / C.TNG_TO_MSUN

    return dict(
        P_e=P_e, rho_gas=rho_gas, r200=r200_mpc_h,
        m200_tng=m_true_tng, m_true_msun=m_true_g / C.MSUN_G,
        slope_true=-p_P, kT_keV=kT_keV,
    )


def gnfw_halo(
    P0: float = 3e-11,
    xc: float = 0.45,
    beta: float = 4.2,
    r200_mpc_h: float = 1.5,
    a_rho: float = 2.1,
    rho0: float = 1e-27,
    noise: float = 0.0,
    seed: int = 0,
):
    """A halo with a genuine gNFW thermal pressure profile, in exact HSE.

    Its 'true' mass at R200c follows from the same HSE relation using the
    *analytic* slope, so any discrepancy after running the pipeline is
    attributable to the fit, the interpolation, or the radial-grid convention.
    """
    rng = np.random.default_rng(seed)
    r = C.R_EFF.copy()
    ln_x = np.log(r)

    P_th = np.exp(log_pressure(ln_x, np.log(P0), np.log(xc), beta))
    if noise > 0:
        P_th = P_th * np.exp(rng.normal(0.0, noise, size=P_th.shape))
    P_e = P_th / C.PTH_OVER_PE

    rho_gas = rho0 * r ** (-a_rho)

    slope_true = float(log_slope(0.0, np.log(xc), beta))       # at r = R200c
    P_th_at_r200 = float(np.exp(log_pressure(0.0, np.log(P0), np.log(xc), beta)))
    m_true_g = hse_mass_cgs(
        r200_mpc_h * C.TNG_TO_CM, P_th_at_r200, rho0, slope_true
    )

    return dict(
        P_e=P_e, rho_gas=rho_gas, r200=r200_mpc_h,
        m200_tng=m_true_g / C.MSUN_G / C.TNG_TO_MSUN,
        m_true_msun=m_true_g / C.MSUN_G,
        slope_true=slope_true, truth=(P0, xc, beta),
    )


# ===========================================================================
# TIER 1 -- algebra, units, conventions.  Must be exact.
# ===========================================================================
def test_hse_algebra_exact():
    """Power-law halo: M_HSE must equal the analytic mass to <0.5%."""
    h = powerlaw_halo()
    res = hse_at_radius(
        h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"],
        r_over_r200=1.0, slope_source="local", p_source="data",
    )
    err = abs(res.m_hse_msun / h["m_true_msun"] - 1.0)
    ok = err < 5e-3
    record(
        "HSE algebra + unit chain (power-law halo)", ok,
        f"M_HSE/M_true - 1 = {res.m_hse_msun / h['m_true_msun'] - 1:+.3e}\n"
        f"recovered slope   = {res.slope_local:+.4f} (true {h['slope_true']:+.4f})\n"
        f"implied kT        = {res.kT_keV:.2f} keV (input {h['kT_keV']:.2f} keV)",
    )
    return ok


def test_temperature_round_trip():
    """kT recovered from P/rho must match the kT the halo was built with.

    This is the unit canary: a wrong power of h, or a missing (kpc/Mpc)^2,
    shows up here as a temperature that is orders of magnitude off, whereas in
    a fitted scaling relation it would be silently absorbed into the amplitude.
    """
    h = powerlaw_halo(kT_keV=6.5)
    res = hse_at_radius(
        h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"],
        r_over_r200=1.0, slope_source="local",
    )
    err = abs(res.kT_keV / 6.5 - 1.0)
    ok = err < 1e-6
    record("Temperature round-trip P/rho -> kT", ok,
           f"kT = {res.kT_keV:.6f} keV (expected 6.500000)")
    return ok


def test_bias_sign_convention():
    """b > 0 must mean M_HSE < M_true."""
    h = powerlaw_halo()
    res = hse_at_radius(
        h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"] * 1.25,
        r_over_r200=1.0, slope_source="local",
    )
    ok = res.bias > 0 and abs(res.bias - 0.2) < 1e-3
    record("Bias sign convention b = 1 - M_HSE/M_true", ok,
           f"M_true inflated 25% -> b = {res.bias:+.4f} (expect +0.2000)")
    return ok


def test_identity_matches_pipeline():
    """Independent re-derivation of b from (r, P, rho, slope) must agree."""
    h = gnfw_halo()
    res = hse_at_radius(h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"])
    b_id = bias_from_identity(
        h["r200"], res.P_th_cgs, res.rho_gas_cgs, res.slope_fit, res.m_true_msun
    )
    ok = abs(b_id - res.bias) < 1e-10
    record("Algebraic identity reproduces pipeline b", ok,
           f"pipeline {res.bias:+.8f}  identity {b_id:+.8f}  "
           f"diff {b_id - res.bias:+.2e}")
    return ok


def test_analytic_vs_numeric_slope():
    """Analytic gNFW log-slope vs np.gradient, as the old code computed it."""
    ln_x = np.log(C.R_EFF)
    ln_xc, beta = np.log(0.45), 4.2
    exact = log_slope(ln_x, ln_xc, beta)
    lnP = log_pressure(ln_x, -24.0, ln_xc, beta)
    numeric = np.gradient(lnP, ln_x)

    at_r200 = np.argmin(np.abs(C.R_EFF - 1.0))
    d200 = abs(numeric[at_r200] - exact[at_r200])
    ok = d200 < 5e-3
    record("Analytic slope vs np.gradient", ok,
           f"|difference| at R200c = {d200:.2e}  "
           f"(analytic {exact[at_r200]:+.4f})\n"
           f"max |difference| over grid = "
           f"{np.max(np.abs(numeric - exact)):.2e}")
    return ok


def test_purity():
    """Profile helpers must not mutate their inputs (the old `*=` bug)."""
    prof = np.abs(np.random.default_rng(1).normal(1.0, 0.1, (5, C.N_BINS)))
    r200 = np.full(5, 1.5)
    snapshot = prof.copy()
    cumulative_profile(prof, r200)
    cumulative_profile(prof, r200)
    ok = np.array_equal(prof, snapshot)
    record("cumulative_profile is pure (idempotent loading)", ok,
           "input array unchanged after two calls" if ok
           else "INPUT ARRAY WAS MUTATED")
    return ok


# ===========================================================================
# TIER 2 -- quantified systematics.  These produce numbers you must report.
# ===========================================================================
def test_gnfw_round_trip():
    """Fitter must recover known gNFW parameters from clean data."""
    truth = (3e-11, 0.45, 4.2)
    h = gnfw_halo(*truth)
    fit = fit_gnfw(C.R_EFF, h["P_e"])
    got = (fit.P0 * C.PTH_OVER_PE, fit.xc, fit.beta)
    rel = [abs(g / t - 1) for g, t in zip(got, truth)]
    ok = fit.converged and max(rel) < 1e-3
    record("gNFW fitter recovers known parameters", ok,
           f"P0  {got[0]:.4e} vs {truth[0]:.4e}   ({rel[0]:.1e})\n"
           f"xc  {got[1]:.4f}     vs {truth[1]:.4f}       ({rel[1]:.1e})\n"
           f"beta {got[2]:.4f}    vs {truth[2]:.4f}       ({rel[2]:.1e})\n"
           f"RMS ln-residual = {fit.rms_log_resid:.2e}")
    return ok


def test_fit_induced_bias_floor():
    """How much apparent bias does the *fit alone* inject?

    The halo is in perfect hydrostatic equilibrium by construction, so the true
    answer is b = 0.  Whatever comes out is the noise floor of the method, and
    no discovered relation should be interpreted below it.
    """
    rows = []
    worst = 0.0
    for noise in (0.0, 0.02, 0.05, 0.10):
        biases = []
        for seed in range(30):
            h = gnfw_halo(noise=noise, seed=seed)
            res = hse_at_radius(h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"])
            if res.ok:
                biases.append(res.bias)
        biases = np.asarray(biases)
        rows.append(
            f"ln P scatter {noise:4.2f} -> b = {np.mean(biases):+.4f} "
            f"+/- {np.std(biases):.4f}   (n={len(biases)})"
        )
        worst = max(worst, abs(np.mean(biases)))
    ok = worst < 0.02
    record("Fit-induced bias floor (halo in exact HSE)", ok,
           "\n".join(rows) + f"\nlargest |mean b| = {worst:.4f} (threshold 0.02)")
    return ok


def test_radial_grid_convention():
    """Quantify the old bin-edge-vs-centre choice."""
    h = gnfw_halo()
    fit_centre = fit_gnfw(C.R_EFF, h["P_e"])
    fit_edge = fit_gnfw(C.RBIN_OUTER, h["P_e"])

    s_c = float(fit_centre.slope_at(1.0))
    s_e = float(fit_edge.slope_at(1.0))
    dm = abs(s_e / s_c - 1.0)

    ok = np.isfinite(dm)
    record("Radial-grid convention: shell centre vs outer edge", ok,
           f"slope at R200c: centres {s_c:+.4f}   edges {s_e:+.4f}\n"
           f"=> fractional change in M_HSE = {dm * 100:.2f}%\n"
           f"   (a systematic of this size sits directly on top of b ~ 0.1-0.2)")
    return ok


def test_effective_radius_convention():
    """Volume-weighted vs geometric vs arithmetic shell centre."""
    e = C.RBIN_EDGES
    vol = C.R_EFF
    geo = np.sqrt(e[1:] * e[:-1])
    ari = 0.5 * (e[1:] + e[:-1])
    sl = slice(1, None)                      # skip bin 0 (a sphere, not a shell)
    d_geo = np.max(np.abs(vol[sl] / geo[sl] - 1))
    d_ari = np.max(np.abs(vol[sl] / ari[sl] - 1))
    ok = d_geo < 1e-3 and d_ari < 1e-3
    record("Shell-centre convention is immaterial (bins 1+)", ok,
           f"max |volume-weighted / geometric - 1| = {d_geo:.2e}\n"
           f"max |volume-weighted / arithmetic - 1| = {d_ari:.2e}\n"
           f"bin 0 spans [0, {e[1]:.3f}] R200c and is excluded everywhere")
    return ok


def test_aperture_index_offset():
    """Reproduce and quantify the hard-coded-index aperture error."""
    rho = np.tile(C.R_EFF ** -2.0, (1, 1))
    cum = cumulative_profile(rho, np.array([1.5]))
    by_radius = float(aperture_value(cum, 1.0)[0])
    by_index = float(cum[0, 101])            # the old R200_BIN_IDX
    off = by_index / by_radius - 1.0
    ok = np.isfinite(off)
    record("Aperture by radius vs hard-coded index 101", ok,
           f"RBIN_EDGES[102] = {C.RBIN_EDGES[102]:.4f} R200c, so index 101 of a\n"
           f"cumulative profile measures inside {C.RBIN_OUTER[101]:.4f} R200c.\n"
           f"=> M_gas('R200c') was overestimated by {off * 100:+.2f}%")
    return ok


def test_slope_degeneracy():
    """Show that beta and xc are degenerate but the slope is not."""
    h = gnfw_halo(noise=0.03, seed=7)
    fit = fit_gnfw(C.R_EFF, h["P_e"])
    cov = fit.cov
    sd = np.sqrt(np.diag(cov))
    corr = cov[1, 2] / (sd[1] * sd[2])
    s = float(fit.slope_at(1.0))
    ss = fit.slope_sigma_at(1.0)
    ok = abs(corr) > 0.8 and (ss / abs(s)) < abs(sd[2] / fit.beta)
    record("beta/xc degenerate; log-slope well determined", ok,
           f"corr(ln xc, beta)        = {corr:+.3f}\n"
           f"sigma(beta)/beta         = {sd[2] / fit.beta:.3f}\n"
           f"sigma(slope)/|slope|     = {ss / abs(s):.3f}   <-- use this, not beta\n"
           f"slope at R200c           = {s:+.4f} +/- {ss:.4f}")
    return ok


def test_fit_range_sensitivity():
    """How much does b move when the fit range moves?"""
    h = gnfw_halo(noise=0.05, seed=3)
    rows, vals = [], []
    for lo, hi in [(0.03, 2.50), (0.10, 2.00), (0.15, 1.80), (0.30, 2.00)]:
        res = hse_at_radius(
            h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"],
            fit_range=(lo, hi),
        )
        vals.append(res.bias)
        rows.append(f"[{lo:.2f}, {hi:.2f}] -> b = {res.bias:+.4f}, "
                    f"slope = {res.slope_fit:+.4f}")
    spread = float(np.max(vals) - np.min(vals))
    ok = np.isfinite(spread)
    record("Sensitivity of b to the gNFW fit range", ok,
           "\n".join(rows) + f"\nspread = {spread:.4f} in b "
           "-- report this as a systematic, do not model it as physics")
    return ok


def test_slope_source_disagreement():
    """gNFW slope vs model-free local slope: the model systematic."""
    diffs = []
    for seed in range(30):
        h = gnfw_halo(noise=0.05, seed=seed)
        res = hse_at_radius(h["P_e"], h["rho_gas"], h["r200"], h["m200_tng"])
        if res.ok and np.isfinite(res.slope_local):
            diffs.append(res.slope_local / res.slope_fit - 1.0)
    diffs = np.asarray(diffs)
    ok = len(diffs) > 20
    record("gNFW slope vs local power-law slope", ok,
           f"fractional difference = {np.mean(diffs):+.4f} "
           f"+/- {np.std(diffs):.4f} over {len(diffs)} halos\n"
           "this maps 1:1 onto b, so carry both estimators through the analysis")
    return ok


# ===========================================================================
# TIER 3 -- real data.  Only meaningful once Tiers 1-2 pass.
# ===========================================================================
def run_real_data_checks(data_dir: str, snapshot: str = "z=0.0") -> bool:
    section(f"TIER 3 -- real data ({data_dir}/{snapshot})")
    need = ["M200c.npy", "R200c.npy", "y_profiles.npy", "ne_profiles.npy",
            "mStar_profiles.npy"]
    base = os.path.join(data_dir, snapshot)
    missing = [f for f in need if not os.path.exists(os.path.join(base, f))]
    if missing:
        record("data files present", False, f"missing: {', '.join(missing)}")
        return False

    m200 = np.load(os.path.join(base, "M200c.npy"))
    r200 = np.load(os.path.join(base, "R200c.npy"))
    y_prof = np.load(os.path.join(base, "y_profiles.npy"))
    ne_prof = np.load(os.path.join(base, "ne_profiles.npy"))
    ms_prof = np.load(os.path.join(base, "mStar_profiles.npy"))

    shapes_ok = (
        y_prof.shape[1] == C.N_BINS
        and ne_prof.shape == y_prof.shape
        and len(m200) == len(y_prof)
    )
    record("array shapes consistent", shapes_ok,
           f"m200 {m200.shape}, y {y_prof.shape}, ne {ne_prof.shape}")
    if not shapes_ok:
        return False

    sel = np.where(m200 > C.MASS_CUT)[0]
    record("sample size above mass cut", len(sel) > 100,
           f"{len(sel)} halos with M200c > {C.MASS_CUT:.0e} (1e10 Msun/h) "
           f"= {C.MASS_CUT * 1e10:.0e} Msun/h")

    rng = np.random.default_rng(0)
    sub = rng.choice(sel, size=min(300, len(sel)), replace=False)

    P_e = y_prof * C.P_SCALING
    rho = ne_prof * C.RHO_SCALING * C.TNG_TO_G / C.TNG_TO_CM**3

    rows = [hse_at_radius(P_e[i], rho[i], r200[i], m200[i]) for i in sub]
    good = [r for r in rows if r.ok]
    frac = len(good) / len(rows)
    record("gNFW fit success rate", frac > 0.9,
           f"{len(good)}/{len(rows)} = {frac:.1%}")
    if not good:
        return False

    def band(vals, key, label):
        v = np.asarray(vals)
        lo, hi = C.SANITY[key]
        med = float(np.median(v))
        inside = float(np.mean((v > lo) & (v < hi)))
        record(f"{label} in physical range", inside > 0.9,
               f"median {med:.4g}, 16-84% "
               f"[{np.percentile(v, 16):.4g}, {np.percentile(v, 84):.4g}], "
               f"expected ({lo}, {hi}), {inside:.1%} inside")
        return inside > 0.9

    all_ok = True
    all_ok &= band([r.kT_keV for r in good], "kT_R200_keV", "kT(R200c)")
    all_ok &= band([r.bias for r in good], "bias", "hydrostatic bias b")
    all_ok &= band([r.slope_fit for r in good], "slope_R200", "dlnP/dlnr(R200c)")

    cum_gas = cumulative_profile(ne_prof[sub] * C.RHO_SCALING, r200[sub])
    f_gas = aperture_value(cum_gas, 1.0) / m200[sub]
    all_ok &= band(f_gas[np.isfinite(f_gas)], "f_gas", "f_gas(<R200c)")

    cum_star = cumulative_profile(ms_prof[sub], r200[sub])
    f_star = aperture_value(cum_star, 1.0) / m200[sub]
    all_ok &= band(f_star[np.isfinite(f_star)], "f_star", "f_star(<R200c)")

    b = np.asarray([r.bias for r in good])
    print(f"\n  Bias summary on {len(b)} halos: "
          f"median b = {np.median(b):+.3f}, "
          f"mean = {np.mean(b):+.3f}, scatter = {np.std(b):.3f}")
    print("  Literature expectation for M_HSE at R200c: b ~ 0.1-0.3.")
    return all_ok


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="path to MTNG_data/ to enable real-data checks")
    ap.add_argument("--snapshot", default="z=0.0")
    args = ap.parse_args()

    section("TIER 1 -- algebra, units, conventions (must be exact)")
    t1 = [
        test_hse_algebra_exact(),
        test_temperature_round_trip(),
        test_bias_sign_convention(),
        test_identity_matches_pipeline(),
        test_analytic_vs_numeric_slope(),
        test_purity(),
    ]

    section("TIER 2 -- quantified systematics (report these numbers)")
    t2 = [
        test_gnfw_round_trip(),
        test_fit_induced_bias_floor(),
        test_radial_grid_convention(),
        test_effective_radius_convention(),
        test_aperture_index_offset(),
        test_slope_degeneracy(),
        test_fit_range_sensitivity(),
        test_slope_source_disagreement(),
    ]

    t3 = [run_real_data_checks(args.data, args.snapshot)] if args.data else []
    if not args.data:
        section("TIER 3 -- real data")
        print("  skipped (pass --data /path/to/MTNG_data to enable)")

    section("SUMMARY")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    for name, ok, _ in _RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {len(_RESULTS) - n_fail}/{len(_RESULTS)} passed")
    if n_fail:
        print("\n  Tier 1 failures are blocking: do not run any search until they pass.")
    return 1 if n_fail else 0


# pytest entry points
def test_tier1_algebra():          assert test_hse_algebra_exact()
def test_tier1_temperature():      assert test_temperature_round_trip()
def test_tier1_sign():             assert test_bias_sign_convention()
def test_tier1_identity():         assert test_identity_matches_pipeline()
def test_tier1_slope():            assert test_analytic_vs_numeric_slope()
def test_tier1_purity():           assert test_purity()
def test_tier2_roundtrip():        assert test_gnfw_round_trip()
def test_tier2_bias_floor():       assert test_fit_induced_bias_floor()


if __name__ == "__main__":
    sys.exit(main())
