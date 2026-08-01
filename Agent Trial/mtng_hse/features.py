"""Registry of candidate predictors, with observability enforced in code.

The rule that matters
---------------------
The deliverable is a formula an observer can *apply*.  An observer measures
M_HSE and wants M_true; they do not know M_true.  So

    **M200c may not appear as a predictor.**

The original RF used ``inp[:, 0] = m200_rf`` -- the true mass -- as its first
and most important feature for predicting the bias.  That answers the physics
question "how does bias depend on halo mass", but it cannot be turned into a
mass estimator, and it inflates the apparent skill because M200c correlates
with everything.  The mass scale here is ``M_HSE``, which is observable, and
the resulting formula

    M_true = M_HSE / (1 - b(observables))

is directly usable.

Observability tiers
-------------------
``DIRECT``      routinely measured today (SZ, X-ray, optical).
``FEASIBLE``    measurable in principle with current instruments, but hard --
                typically because it needs resolved profiles out to R200c.
``PROJECTED``   the simulation gives a 3-D quantity; only a projected analogue
                is observable, so a projection must be applied before use.
``SIM_ONLY``    not observable; permitted as a *diagnostic* to interpret
                residuals, never as a term in the final equation.

``allowed_in_equation`` is what the constraint checker enforces.  Anything
SIM_ONLY is diagnostic-only by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from . import config as C

__all__ = ["Observability", "Feature", "REGISTRY", "predictors", "diagnostics"]


class Observability(str, Enum):
    DIRECT = "DIRECT"
    FEASIBLE = "FEASIBLE"
    PROJECTED = "PROJECTED"
    SIM_ONLY = "SIM_ONLY"


@dataclass
class Feature:
    name: str
    rationale: str
    observability: Observability
    how_observed: str
    fn: Callable[..., np.ndarray] | None = None

    @property
    def allowed_in_equation(self) -> bool:
        return self.observability is not Observability.SIM_ONLY


def _f(name, rationale, obs, how):
    return Feature(name=name, rationale=rationale, observability=obs, how_observed=how)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, Feature] = {f.name: f for f in [

    # -- mass scale --------------------------------------------------------
    _f("M_hse", "The observable mass scale. Replaces M200c, which an observer "
                "does not have. Also sets the amplitude of every other term.",
       Observability.DIRECT,
       "X-ray or SZ hydrostatic mass; this is the quantity being corrected."),

    # -- the two factors the bias literally decomposes into ----------------
    _f("slope_R200", "dlnP/dlnr at R200c. One of the two exact factors in "
                     "M_HSE = -(rP/G rho) dlnP/dlnr, so it is not a proxy for "
                     "anything -- it is half the answer.",
       Observability.FEASIBLE,
       "Resolved SZ (e.g. NIKA2, MUSTANG-2, ACT) or X-ray pressure profile "
       "differentiated near R200c. Hard, but done in X-COP-class datasets."),

    _f("kT_R200", "mu m_p P/rho at R200c: the gas temperature where HSE is "
                  "evaluated. The other exact factor, in the combination "
                  "kT/(G M/R) = T/T_vir.",
       Observability.FEASIBLE,
       "X-ray spectroscopic temperature in the outermost annulus, or "
       "P_SZ / n_e,X-ray."),

    _f("slope_local_R200", "Model-free power-law slope near R200c. Differs "
                           "from slope_R200 by the gNFW model systematic; the "
                           "difference is a fit-quality flag in disguise.",
       Observability.FEASIBLE, "Same data, no parametric model assumed."),

    # -- concentrations ----------------------------------------------------
    _f("c_Y_0p50", "Y(<0.5 R200c)/Y(<R200c). The pressure-concentration that "
                   "already works in the Y-M relation; expected to track "
                   "cool-core vs disturbed morphology.",
       Observability.DIRECT, "Resolved SZ maps, or X-ray derived Y_X."),

    _f("c_Y_0p15", "Y(<0.15 R200c)/Y(<R200c). Core-excised aperture, the "
                   "standard X-ray choice for suppressing cool-core scatter.",
       Observability.DIRECT, "Same, with the usual 0.15 R core excision."),

    _f("c_gas_0p50", "M_gas(<0.5 R200c)/M_gas(<R200c). Gas concentration; "
                     "traces a different physical quantity from c_Y "
                     "(mass vs pressure weighting).",
       Observability.DIRECT, "X-ray surface-brightness deprojection."),

    _f("c_gas_0p15", "M_gas(<0.15 R200c)/M_gas(<R200c).",
       Observability.DIRECT, "As above."),

    _f("Y_ratio_0p65", "Y(<0.65 R200c)/Y(<R200c): an R500c-like aperture ratio "
                       "using the stand-in radius until true R500c is available.",
       Observability.DIRECT, "Standard SZ aperture photometry."),

    # -- reparameterisations ----------------------------------------------
    _f("logit_c_Y_0p50", "ln[c/(1-c)]. Concentrations live in (0,1) with a "
                         "skewed distribution; a bounded coordinate is rarely "
                         "the one a relation is linear in. Cheap to try, and "
                         "the analogue of the reparameterisation trick that "
                         "worked in the gwBenchmarks ringdown task.",
       Observability.DIRECT, "Transformation of an observable."),

    # -- baryon content ----------------------------------------------------
    _f("Mstar_over_Mgas", "Stellar-to-gas mass ratio: a feedback / "
                          "star-formation-efficiency proxy. Dominated the "
                          "Y-M residuals, so worth carrying over.",
       Observability.DIRECT,
       "Optical/IR stellar mass + X-ray gas mass. Aperture-matched."),

    _f("f_gas_hse", "M_gas(<R200c)/M_HSE. Note the denominator: using M200c "
                    "here would smuggle the answer in. The observable version "
                    "is biased high by exactly the factor we are trying to "
                    "predict, which is itself informative.",
       Observability.DIRECT, "X-ray gas mass over hydrostatic mass."),

    # -- morphology / relaxation ------------------------------------------
    _f("gnfw_rms_resid", "RMS ln-residual of the gNFW fit. A halo whose "
                         "pressure profile refuses to be a smooth gNFW is a "
                         "disturbed halo, and disturbance is the leading "
                         "driver of non-thermal support. Free to compute.",
       Observability.FEASIBLE,
       "Goodness of fit of the same parametric model to real data -- an "
       "observer computes this too."),

    _f("ellip_projected", "Projected axis ratio of the gas distribution. "
                          "Morphology is the classic relaxation indicator and "
                          "correlates with hydrostatic bias.",
       Observability.PROJECTED,
       "X-ray/SZ isophotal ellipticity. NOTE: derive from Mtensor_Gas by "
       "projecting along a line of sight FIRST -- the raw 3-D tensor "
       "eigenvalues are not observable."),

    # -- diagnostics only --------------------------------------------------
    _f("M200c", "True mass. Diagnostic only: needed to define b, and useful "
                "for asking 'does the bias depend on mass', but it cannot "
                "appear in a formula an observer would apply.",
       Observability.SIM_ONLY, "Not observable."),

    _f("f_gas_true", "M_gas/M200c. Diagnostic only -- contains M200c.",
       Observability.SIM_ONLY, "Not observable."),

    _f("ellip_3d", "3-D gas inertia-tensor axis ratios from Mtensor_Gas. "
                   "Diagnostic only: use to check how much signal is lost by "
                   "projection, then use ellip_projected in the equation.",
       Observability.SIM_ONLY, "Only a projection is observable."),

    _f("redshift", "Diagnostic only. z=0 sample here; if z=0.5 is added later "
                   "the E(z) scaling must be handled explicitly rather than "
                   "absorbed into a fitted constant.",
       Observability.SIM_ONLY, "Observable, but excluded to keep the relation "
                               "self-similar rather than redshift-calibrated."),
]}


def predictors() -> list[str]:
    """Names usable in a candidate equation."""
    return [n for n, f in REGISTRY.items() if f.allowed_in_equation]


def diagnostics() -> list[str]:
    """Names usable only for interpreting residuals."""
    return [n for n, f in REGISTRY.items() if not f.allowed_in_equation]


def check_expression_features(used: list[str]) -> tuple[bool, list[str]]:
    """Constraint check: are all features in ``used`` allowed in an equation?

    Called by the harness on every candidate.  A prose rule in a spec file is
    not a constraint; this is.
    """
    unknown = [u for u in used if u not in REGISTRY]
    forbidden = [u for u in used if u in REGISTRY and not REGISTRY[u].allowed_in_equation]
    return (not unknown and not forbidden), unknown + forbidden


if __name__ == "__main__":
    print(f"{'feature':<22} {'observability':<12} allowed")
    print("-" * 52)
    for name, f in REGISTRY.items():
        print(f"{name:<22} {f.observability.value:<12} "
              f"{'yes' if f.allowed_in_equation else 'NO (diagnostic)'}")
    print(f"\n{len(predictors())} usable predictors, "
          f"{len(diagnostics())} diagnostics")
