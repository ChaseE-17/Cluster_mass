# manual/ — the manual Y–M pipeline

Status: LIVE · Updated: 2026-09-14 · Maintainer: both
Scope: index + map for the by-hand method · Describes: MTNG_HSE_YM.ipynb, docs/, cgas/

The by-hand version of the pipeline. One canonical notebook, a literature/decision
record in `docs/`, and a runnable symbolic-regression sub-project in `cgas/`.

## Canonical notebook

**`MTNG_HSE_YM.ipynb`** is the canonical, current notebook — the science reference for
the whole repo. Everything else here supports it.

- `archive/` — superseded notebooks (precursors + hand-made `.bak`s). Git-ignored,
  reference only; see `archive/README.md`. There is no other "live" notebook.
- The notebook is tracked with an `nbstripout` filter: your outputs stay on disk, git
  stores a stripped copy. Keep the `scatter` venv active when committing it.

## What the pipeline does (cite sections, not cell numbers)

```
load pooled MTNG snapshots
  → radial profile tools (cumulative_profile, aperture_value, concentration)
  → gNFW pressure fit (fit_gnfw)
  → HSE mass & bias   b = 1 − M_HSE / M200c
  → tiered feature framework (DIRECT / FEASIBLE / PROJECTED / SIM_ONLY)
  → feature selection: RF + SHAP + Spearman redundancy grouping
  → symbolic regression (PySR; commented out by default — hours-long)
Two targets:  (a) the HSE-bias proxy   (b) the Y–M scatter  y = M200c/(A·Y200^0.6)  (§17)
```

Physics derivations (HSE mass, the `1−b` decomposition, the Y–M target, the tier
framework) live **inline in the notebook's markdown cells**. Reference them by section.

## docs/ index

Start with the two synthesis/decision docs — they index the raw reports beside them.

| File | Status | Purpose |
|---|---|---|
| `docs/feature_selection_YM_SYNTHESIS.md` | LIVE | **Start here.** Consolidated feature-selection action plan; indexes the workstream reports |
| `docs/redshift/redshift_snapshot_decision.md` | LIVE | The z=0 vs z=0.5 vs both decision; indexes the 5 supporting `redshift/` reports |
| `docs/lit_cluster_mass_YM.md` | LIVE | Literature scan on Y–M / cluster mass (Wadekar 2023 anchored) |
| `docs/lit_rf_sr_methodology.md` | LIVE | RF → SHAP → SR methodology do's and don'ts |
| `docs/observable_catalog_YM.md` | LIVE | Observability tier per feature; indexes `docs/catalog_sources/` |
| `docs/redteam_YM_feature_selection.md` | LIVE | Correctness/leakage audit of the §17 method |
| `docs/sim_feature_selection_survey.md` | LIVE | Cross-simulation survey of feature-choice practice |
| `docs/catalog_sources/` (4 files) | LIVE | Per-observable deep detail (sz_pressure, xray_gas, optical_stellar, morphology_dynamical) |
| `docs/simfeature_sources/` (3 files) | LIVE | Per-simulation feature practice |
| `docs/feature_selection_YM.md` | **STALE** | Pre-refactor §17 method reference; superseded by `_SYNTHESIS.md`. Do not cite |

## cgas/ — runnable SR sub-project

`cgas/` extracts the notebook's §17 (constants, profile math, the Y–M scatter target)
into a standalone **propose-and-prune symbolic-regression loop** with a *frozen
evaluator the proposer cannot redefine*. It is the clean-room prototype that
`agentic_ym/` grew from. Its own `README.md`, `PLAN.md`, and `PROTOCOL.md` (the JSON
agent contract) are the best-documented component in the repo — read those to run it.
