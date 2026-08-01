# Plan — agentic discovery of an analytic hydrostatic mass-bias relation

Clean slate. Nothing below is inherited from the previous attempt.

---

## 1. The reframing your answers force

You said: both deliverables matter, but **the workflow is the thing being
optimised**, and if the equation doesn't beat PySR that's acceptable.

That is not a small preference — it changes what kind of object this project is.
If the equation were the deliverable, you'd build a pipeline and run it. If the
*workflow* is the deliverable, then **the workflow is the object under study,
and it needs an experimental design**: arms, controls, ablations, and a
pre-registered outcome measure. A single agent run that produces a nice formula
demonstrates nothing about the method, because you can't tell which part of the
method did the work.

Concretely, the paper's claim will be some version of *"a validation-constrained
agentic loop with a physics-informed ansatz finds better/more interpretable
analytic relations than one-shot symbolic regression."* To support that you need:

| Arm | What it isolates |
|-----|------------------|
| **A. PySR one-shot** | The control. Same data, same splits, same metric. |
| **B. Agent loop, no ansatz** | Does agentic iteration alone help? |
| **C. Agent loop + physics ansatz (L0/L1)** | The GWAgent claim, tested rather than assumed. |
| **D. C + per-term influence feedback** | The IGSR claim. |
| **E. C + D + archive/population** | Does search structure matter, or just the feedback? |

Plus two ablations that are cheap and directly address the reviewer's first
question:

- **Evaluator locking on/off** — measure how often the unlocked agent produces a
  number the locked harness disagrees with. This is a *result*, not just hygiene:
  it quantifies, on a real scientific task, the failure mode both of your PI's
  papers describe anecdotally.
- **Split-seed replication** — rerun the winning arm on 3 different splits. With
  N ≈ 2100 this is the difference between a finding and a fluctuation.

Each arm gets the same wall-clock/token budget. That's the experiment. Everything
in §4 is infrastructure for running it.

---

## 2. The scientific target, restated

At `r = R200c`, hydrostatic equilibrium gives, exactly:

```
M_HSE = − (R200c · P_th / (G ρ_gas)) · s200 ,        s200 ≡ dlnP/dlnr|_{R200c}
1 − b ∝ (kT200 / μ m_p) · R200c/(G M200c) · (−s200)
```

So `b` is *gas temperature at R200c relative to virial* × *pressure steepness
there*. Two factors, both measurable. This gives a four-level residual ladder,
which is what the search actually operates on:

| Level | Model | Free params | Purpose |
|---|---|---|---|
| **L0** | the identity above | 0 | must be exact; it's a unit test, not a model |
| **L1** | physics ansatz: `b ≈ f_nth(r)`-type closed form, or a 1–2 parameter map from `(kT/T_vir, s200)` to `b` | 1–2 | the residual target |
| **L2** | over-complete ridge/LASSO on ~10²–10³ basis columns from registered features | many | generates the *information*, not the answer |
| **L3** | compactify L2 by influence ranking into a short closed form | ~4–6 | the deliverable |

The important structural point: **the search never regresses `b` on raw columns.**
It searches for corrections to a model that already knows the physics. That is
the single change with the best evidence behind it (GWAgent report their
unguided runs were worse; their compactification went 2955 → 75 → O(10) terms
with accuracy preserved).

### Predictors must be observable

This is the change with the largest effect on the *result*, and it comes from
your point 6. An observer measures `M_HSE` and wants `M_true`; they do not know
`M200c`. So `M200c` cannot appear in the formula — yet it was the leading RF
feature in the old notebook. The mass scale is `M_HSE`, and the deliverable is

```
M_true = M_HSE / (1 − b(observables))
```

`mtng_hse/features.py` enforces this in code, with four observability tiers
(DIRECT / FEASIBLE / PROJECTED / SIM_ONLY). On `Mtensor_Gas`: the 3-D inertia
tensor is `SIM_ONLY`, but a **projected** axis ratio along a random line of sight
is `PROJECTED` and is a real observable (X-ray/SZ isophotal ellipticity is
routinely measured). So project first, then use it. Keep the 3-D version as a
diagnostic to measure how much signal projection costs — that comparison is
itself a small publishable result.

---

## 3. Statistics for N ≈ 2100

2100 is enough, with the right protocol. It is not enough for a single fixed
train/test split.

- **Lockbox: 20% (~420 halos), opened once.** Stratified in `log M200c`.
- **Selection: repeated stratified 5-fold CV on the remaining ~1680**, 5 repeats.
  A single held-out 525 gives an MSE standard error around 6–10%; 5×5-fold CV on
  1680 cuts the selection variance substantially and uses every halo. Since the
  candidate equations have ≤6 parameters, the *fit* is not data-limited — the
  *selection* is, so spend the data there.
- **Every comparison is paired.** Bootstrap over halos on the *difference* of
  squared residuals between two candidates, not on each MSE separately. Paired CIs
  are far tighter because halo-to-halo scatter cancels.
- **Accept a candidate only if the paired ΔMSE 95% CI excludes zero.** This
  replaces "improves by >1%", which at this N is a coin flip.
- **Report the fit-induced floor.** The test suite measures how much apparent `b`
  the gNFW fit alone injects for a halo in exact HSE: at 5% ln-P scatter it is
  −0.007 ± 0.039 per halo. Structure in the residuals below that is not physics.

If this turns out to be too tight in practice, adding z=0.5 roughly doubles N —
but it also reintroduces `E(z)` bookkeeping and makes the relation
redshift-calibrated rather than self-similar. Hold it in reserve; don't spend it
up front.

---

## 4. Stages

### Stage 0 — Forward model (done; delivered here)
`mtng_hse/` + `tests/test_pipeline.py` + `notebooks/MTNG_HSE_v2.ipynb`.
Gate: Tier 1 exact, Tier 2 systematics recorded, Tier 3 real-data ranges sane.

### Stage 1 — Lock the evaluation
`harness/` read-only to the agent. Splits frozen and hashed. Lockbox labels in a
separate file not present on disk during search. The **loop** runs the scorer,
never the agent — the agent writes `candidates.py` and reads results from
`archive.jsonl` on the next iteration.

Audit checks, run every iteration, exit non-zero on violation:
1. coverage — every candidate scored on 100% of its split
2. central recomputation — agent-reported numbers ignored; disagreement logged as
   a finding, not silently overwritten
3. AST scan for stubs, constant predictions, hard-coded metrics
4. AST constraint check — parameter count, and the observability whitelist from
   `features.py` (a prose rule is not a constraint)
5. loss-series anomaly detection — suspiciously regular decrements, duplicate
   losses across "different" candidates
6. train/CV gap monitor
7. provenance — git SHA, config hash, cache hash on every row

### Stage 2 — Feature construction
Seeded registry is in `features.py`. The agent may add scalars, but each must
carry a physical rationale **written before** its correlation with `b` is
computed, an observability tier, and a stability test (recompute on a coarsened
radial grid; >5% drift means grid noise, not physics).

### Stage 3 — Ladder + search
L0 → L1 by hand (supervised). L2/L3 in the loop. Population archive keyed by
(parameter count, feature subset), which yields the complexity–accuracy Pareto
front you want as a figure anyway, directly comparable to PySR's.

### Stage 4 — Experiment
Run arms A–E, plus the two ablations, plus 3 split seeds on the winner.

### Stage 5 — Lockbox, once
Then write.

---

## 5. Supervision policy

You asked me to judge. My recommendation:

**Supervise Stages 0–2. Automate Stage 3. Supervise Stage 5.**

The reasoning is empirical rather than cautious-by-default: in both of your PI's
papers, the agent failures were overwhelmingly *setup* failures, not *search*
failures — wrong metric, partial evaluation, wrong physics in the reconstruction,
training metrics reported as validation, constraint violated. Those are all
decisions made once, early, that silently poison everything downstream. Search
itself is the part where agents did well and where iteration genuinely helps.

So: you personally sign off on the target definition, the splits, the metric, and
the feature whitelist. After that the loop can run overnight unattended, because
by then the harness — not your attention — is what's enforcing correctness. That
is also the honest version of the methods section: "human-specified: X; 
agent-discovered: Y", their Table S1.

One exception: watch the *first* 5 iterations of any new arm. Failure modes that
will recur 200 times are visible in the first few.

---

## 6. Layout

```
hse-bias/
├─ mtng_hse/           # forward model (delivered)
│   ├─ config.py       # every constant, named, with reasoning
│   ├─ profiles.py     # radius-based apertures
│   ├─ gnfw.py         # model, analytic slope, fit + covariance
│   ├─ hse.py          # M_HSE, bias, the identity
│   ├─ data.py         # pure loader, table builder
│   └─ features.py     # registry + observability enforcement
├─ tests/test_pipeline.py     # delivered; Tier 1/2/3
├─ notebooks/MTNG_HSE_v2.ipynb
├─ harness/            # READ-ONLY TO AGENT  (Stage 1)
│   ├─ splits.py  metrics.py  evaluate.py  audit.py  constraints.py
├─ ladder/             # L0/L1/L2 reference implementations
├─ arms/               # one config per experimental arm
├─ candidates.py       # agent-writable
├─ archive.jsonl       # every candidate ever scored, including rejects
└─ figs/
```

`archive.jsonl` must include rejects. GWAgent's Fig. 1b — kept vs. discarded
trials across agent steps — is compelling precisely because the discards are
shown, and you get that figure for free if the log is complete.

---

## 7. Order of work

| # | Task | Gate |
|---|------|------|
| 1 | Run `tests/test_pipeline.py --data …` on the real catalogue | Tier 3 ranges sane; record N and fit-failure rate |
| 2 | Run the v2 notebook; check §7 old-vs-new shift | you know how much the fixes moved `b` |
| 3 | Build `harness/`, freeze splits, lock it | audit catches a deliberately planted stub |
| 4 | Baselines: constant `b`, `b(M_HSE)` power law, RF ceiling, PySR one-shot | numbers with paired CIs |
| 5 | L0 identity + L1 ansatz | residual target defined |
| 6 | Arms B–E, first 5 iterations each supervised | no audit violations |
| 7 | Unattended runs to budget | archive populated |
| 8 | 3 split seeds on the winner | form is stable |
| 9 | Lockbox, once | final number |

---

## 8. Open items

1. **Real-data Tier 3 has not run** — I don't have the catalogue. Numbers 1–2
   above are the first thing to do; if the bias distribution comes out far from
   `b ~ 0.1–0.3`, something in the unit chain still needs attention and the
   `kT(R200c)` check will say which.
2. **`Mtensor_Gas.npy` didn't upload.** Needed to build `ellip_projected`.
   Also: is it available at z=0.5, and is it mass-weighted or count-weighted?
3. **`FIT_RANGE = (0.10, 2.00)` is my choice, not a measurement.** The notebook
   sweeps it; pick the value that minimises the fit-induced floor on real data
   and freeze it *before* the search starts, not after.
4. **`P_SCALING`'s `1e-6`** is inherited from the `Y_scaling` convention and I
   could not verify it without data. `test_physical_ranges` will catch it via
   `kT(R200c)` if it's wrong.
5. **L1 ansatz form** — worth one conversation. The natural candidates are a
   Nelson+14-style `f_nth(r)` closed form, or a direct 2-parameter map from
   `(kT/T_vir, s200)`. Which one you pick determines what the residual means.
