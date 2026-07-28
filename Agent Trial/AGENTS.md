# AGENTS.md — Y–M Relation Improvement (MillenniumTNG)

## Project goal

Derive an **analytic** equation that predicts halo mass `M_200c` from
SZ / gas / stellar summary quantities better than the canonical
power-law benchmark

```
M_pred = A * Y200**alpha          # benchmark to beat
```

"Better" means **lower test-set MSE of the relative residual
`M_pred / M_true - 1`**, evaluated under the protocol in
[Evaluation protocol](#evaluation-protocol). Equations may be
non-linear but should stay short (see [Complexity budget](#complexity-budget)).

The reference implementation of the data pipeline, the benchmark, and
a Random-Forest sanity check lives in `MTNG_copy.py` /
`MTNG_copy.ipynb`. **Do not edit those files** — treat them as
read-only ground truth. Put new work in the modules described in
[Deliverables](#deliverables).

## Data and conventions

Loaded inside `MTNG_copy.py` (do not duplicate this logic — import
from there or call a helper that reproduces it):

- `m200` is in units of `1e10 Msun/h` and **already absorbs `E(z)^{2/5}`**:
  the script does `m200 = m200 * Ez**(2/5)`. The training target is
  therefore `M_200c * E(z)^{2/5}` and **candidate equations must not
  depend explicitly on `z` or `E(z)`**.
- Mass cut: keep only halos with `m200 > 5e3` (= `5e13 Msun/h`).
  Use the existing `mask = (m200 > MASS_CUT)` and the suffix-`H`
  arrays (`m200H`, `Y200H`, `mGasH`, `GasConcH`, `YConcH`, `mStarH`).
- Train/test split: **always** reuse
  ```python
  rng = np.random.default_rng(0)
  maskTest = rng.choice([True, False], size=len(m200H), p=[0.5, 0.5])
  ```
  Train on `~maskTest`, evaluate on `maskTest`. Never reshuffle, never
  peek at the test set during model selection.

### Allowed features

Candidate expressions may use any of:

| symbol      | meaning                                              |
|-------------|------------------------------------------------------|
| `Y200`      | Compton-Y integrated to `R200c` (with `Y_scaling`)   |
| `YConc`     | `Y(<R_inner) / Y(<R200c)`                            |
| `mGas`      | gas mass inside `R200c`                              |
| `GasConc`   | `mGas(<R_inner) / mGas(<R200c)`                      |
| `mStar`     | stellar mass inside `R200c`                          |
| `mStar/mGas`| stellar-to-gas ratio                                 |
| profiles    | `yProf`, `neProf`, `mStarProf` — agent may construct **new scalar summaries** from these (e.g. slopes, alternative apertures, ratios at other bin indices), provided each new scalar is documented in `candidates.py` |

Disallowed: `z`, `E(z)`, `R200c` as a direct feature, `T500c` (kept out
to keep things SZ/gas/stars only), and any quantity derived from
`m200` itself.

## Complexity budget

Soft target of **~10 tokens** per equation. Token counting is
intentionally loose — count each variable, numeric constant, and
operator/function call (`+`, `-`, `*`, `/`, `**`, `exp`, `log`, ...) as
roughly one token. Going slightly over is OK if the agent justifies
why the extra term is needed (e.g. it removes a clear residual
trend); going much over is not. Prefer ~5–8-token expressions when
they suffice.

## Workflow (every refinement round)

1. **Propose** one candidate expression `M_pred = f(features; theta)`,
   with free coefficients `theta`. State briefly *why* this form was
   chosen (residual structure from the previous round, or, for round 1,
   the power-law benchmark).
2. **Fit** `theta` on the training set with
   `scipy.optimize.curve_fit`, using **linear-in-M weighting** to match
   the RF benchmark in `MTNG_copy.py`
   (`sample_weight=pow(m200H,1)`).
   Concretely, fit `M_pred` to `m200H_train` with
   ```python
   sigma = m200H_train / np.sqrt(m200H_train)   # = sqrt(m200H_train)
   curve_fit(f, X_train, m200H_train, p0=..., sigma=sigma,
             absolute_sigma=False, maxfev=20000)
   ```
   (this minimizes the M-weighted squared relative residual). If
   `curve_fit` fails to converge, fall back to
   `scipy.optimize.least_squares` on the same residual and note it.
3. **Evaluate** on the test set:
   - primary metric: unweighted MSE of `M_pred/M_true - 1` on
     `maskTest`,
   - also report: M-weighted MSE of the same residual, the relative
     scatter `std(M_pred/M_true - 1)` in the same log-mass bins as
     `MTNG_copy.py` (`temp = np.logspace(3.6, 4.7, num=7)`), and the
     bin-wise mean residual (bias).
4. **Diagnose residuals.** Plot or tabulate
   `M_pred/M_true - 1` against every allowed feature (and against
   any new profile-derived scalar in play). Identify the strongest
   remaining trend — that trend motivates the next candidate.
5. **Refine** by either retuning a coefficient, swapping a feature,
   or adding **at most one** new term/factor. Do not jump to a
   completely unrelated functional form without first explaining what
   in the residuals required it.

## Stopping criterion

Stop iterating when a new candidate fails to improve the **primary
test metric** (unweighted MSE of `M_pred/M_true - 1`) by **more than
1%** over the current best, *and* does not meaningfully reduce
binned scatter or bias. Report the final equation, its fitted
coefficients, and its margin over the benchmark.

## Evaluation protocol

A candidate is "better than the benchmark" only if **both** of the
following hold on `maskTest`:

1. its primary MSE is strictly lower than the benchmark
   `M_pred = A * Y200**alpha` refitted on the same training set with
   the same weighting (do not hard-code `A = 1.97e17`,
   `alpha = 0.605`; refit them so the comparison is apples-to-apples),
2. its binned relative scatter is no worse than the benchmark in any
   bin where the benchmark has ≥30 halos.

Always report the benchmark numbers next to every candidate.

## Deliverables

Create and maintain (do **not** modify `MTNG_copy.py` /
`MTNG_copy.ipynb`):

- `candidates.py` — each candidate equation as a pure function
  ```python
  def cand_NN(X, *theta): ...    # returns M_pred in 1e10 Msun/h
  ```
  plus, where applicable, a small helper for any new profile-derived
  scalar (with a one-line docstring explaining what it measures).
- `run_candidates.py` — driver that
  1. loads / reconstructs the `H`-suffixed arrays and `maskTest`
     exactly as in `MTNG_copy.py` (import or reproduce verbatim),
  2. refits the power-law benchmark on the training set,
  3. fits every candidate in `candidates.py`,
  4. prints a comparison table (rows = candidates, columns =
     fitted params, train MSE, test MSE, M-weighted test MSE,
     max binned scatter, max binned bias, % improvement vs.
     benchmark),
  5. saves residual-vs-feature diagnostic plots for the current best
     candidate into a `figs/` folder.
- `CANDIDATES.md` — append-only log, one section per round:
  - the proposed expression (LaTeX or plain),
  - motivation (what residual structure prompted it),
  - fitted coefficients,
  - test metrics vs. benchmark,
  - what the residuals now suggest for the next round.

## Hard rules

- Never edit `MTNG_copy.py` or `MTNG_copy.ipynb`.
- Never use the test set (`maskTest == True`) to choose features,
  functional forms, or coefficients.
- Never include `z`, `E(z)`, or any function of `m200` itself as a
  feature in a candidate.
- Always weight training by `m200H` (linear) to match the RF
  benchmark.
- Always refit the benchmark alongside the candidate; never compare
  against stale hard-coded benchmark coefficients.
- Keep equations short — if a round's candidate exceeds ~10 tokens,
  the agent must explicitly justify each extra token.
