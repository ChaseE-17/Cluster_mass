# MTNG hydrostatic mass bias — corrected forward model

Stage 0 of the plan in `PLAN.md`. Everything here is the *forward model*: it
computes the hydrostatic mass and the bias from MTNG profiles, correctly and
testably. No equation search lives here.

## Quick start

```bash
python tests/test_pipeline.py                       # synthetic only, no data
python tests/test_pipeline.py --data /path/MTNG_data # + real-data checks
jupyter lab notebooks/MTNG_HSE_v2.ipynb              # edit DATA_DIR first
```

`tests/test_pipeline.py` is the gate. Tier 1 must pass exactly before anything
else is meaningful; Tier 2 prints systematics to quote in the paper; Tier 3
checks the real catalogue against physical ranges.

## Layout

| file | role |
|---|---|
| `mtng_hse/config.py` | every constant and convention, in one place |
| `mtng_hse/profiles.py` | cumulative profiles; apertures by **radius**, not index |
| `mtng_hse/gnfw.py` | gNFW model, analytic log-slope, fit + covariance |
| `mtng_hse/hse.py` | `M_HSE`, `b = 1 − M_HSE/M_200c`, the algebraic identity |
| `mtng_hse/data.py` | pure loader + per-halo table builder |
| `mtng_hse/features.py` | feature registry with observability enforcement |

## Conventions pinned here

- `b = 1 − M_HSE / M_200c` (literature sign; positive = under-estimate)
- radii in `r/R200c`; profile values live at volume-weighted shell centres
- `M200c` is **not** a permitted predictor — the observable mass scale is `M_HSE`
