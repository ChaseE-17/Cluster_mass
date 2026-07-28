# CANDIDATES.md — Y–M relation refinement log

Append-only round log per `AGENTS.md`. Every round states the proposed
expression, what residual structure motivated it, the fitted
coefficients, the test metrics next to a freshly-refitted power-law
benchmark, and a one-sentence diagnosis of what the residuals say about
the next round.

Common conventions (constant across all rounds):

* Dataset: `MTNG_copy.py` z=0.5 + z=0 catalogues, mass cut
  `m200 > 5e3` (= `5e13 Msun/h`). 10,546 halos pass.
* Train/test split: `np.random.default_rng(0)` 50/50 →
  5,278 train, 5,268 test.
* Training weight: `sample_weight = m200H` (linear, matches the
  RF benchmark).
* Primary metric: **unweighted MSE of `M_pred / M_true − 1`** on the
  test set.
* Bin grid for binned scatter / bias: `np.logspace(3.6, 4.7, num=7)`.

---

## Round 0 — Power-law benchmark (refitted)

**Expression** (4 tokens):
`M = A * Y200**alpha`

**Motivation.** Re-fit the canonical Y–M power law on the same
training set, with the same weighting, so every later candidate can be
compared apples-to-apples. (`MTNG_copy.py` hard-coded `A = 1.97e17`,
`alpha = 0.605`; we refit instead.)

**Fitted coefficients.**

| param | value      |
| ----- | ---------- |
| `A`   | 1.636e+07 (in 1e10 Msun/h units; ≈ 1.636e17 Msun/h) |
| `alpha` | 0.5895   |

**Test metrics.**

|                       | benchmark |
| --------------------- | --------- |
| MSE (unweighted)      | 4.3186e-3 |
| MSE (M-weighted)      | 4.7334e-3 |
| max bin scatter       | 0.0690    |
| max abs bin bias      | 0.0279    |

Bin counts (M~5e13 → 4e14): 1267, 1925, 1115, 554, 270, 90 — last bin has
≥30 halos so it counts under the binned-scatter criterion.

**Residual diagnosis (training set).** Strongest remaining
correlations: `mStar/mGas` (Pearson −0.39), `YConc` (Pearson ≈ −0.05
linear, but a clear ~7% peak-to-peak swing in binned residuals as in
the published MTNG plot). Two corrections are obvious next steps;
`YConc` is the one already documented in the reference notebook, so
take it first.

---

## Round 1 — Add a linear YConc correction

**Expression** (8 tokens):
`M = A * Y200**alpha * (1 - b*YConc)`

**Motivation.** Reproduce the paper's YConc correction as the first
physically-motivated extension of the power law.

**Fitted coefficients.**

| param  | value    |
| ------ | -------- |
| `A`    | 2.448e+07 |
| `alpha` | 0.6033  |
| `b`    | 0.3761   |

**Test metrics vs. benchmark.**

|                       | benchmark | round 1   | Δ vs. bench |
| --------------------- | --------- | --------- | ----------- |
| MSE (unweighted)      | 4.3186e-3 | 3.3981e-3 | **−21.31 %** |
| MSE (M-weighted)      | 4.7334e-3 | 3.4235e-3 | −27.7 %     |
| max bin scatter       | 0.0690    | 0.0622    | better      |
| max abs bin bias      | 0.0279    | 0.0220    | better      |

Both AGENTS.md "better than benchmark" criteria pass: lower MSE *and*
no bin (≥30 halos) gets worse on scatter.

**Residual diagnosis (training set).** Re-running the residual–vs–
feature analysis on round 1's residuals:

| feature       | Pearson | Spearman | binned peak-to-peak |
| ------------- | ------: | -------: | ------------------: |
| Y200          |  +0.009 |   +0.125 | 0.013               |
| YConc         |  −0.029 |   +0.024 | 0.010               |
| GasConc       |  +0.041 |   +0.067 | 0.017               |
| mGas          |  +0.014 |   +0.119 | 0.037               |
| mStar         |  −0.079 |   −0.089 | 0.017               |
| **mStar/mGas** | **−0.393** | **−0.403** | **0.071**       |

`mStar/mGas` is, by a factor of ~3 over any other feature, the
strongest remaining trend, monotonic, and physically meaningful (proxy
for star-formation efficiency / feedback strength).

---

## Round 2 — Add an `mStar/mGas` correction *(final selection)*

**Expression** (11 tokens, slightly over the soft 10-token budget;
justified by the +1 token bringing in the strongest residual
correlator):

`M = A * Y200**alpha * (1 - b*YConc + c*(mStar/mGas))`

**Motivation.** Round-1 residuals showed `mStar/mGas` was *the*
dominant remaining trend. Pack the new correction into the same
bracket as the YConc term to keep the expression compact.

**Fitted coefficients.**

| param  | value    |
| ------ | -------- |
| `A`    | 2.537e+07 |
| `alpha` | 0.6188  |
| `b`    | 0.3893   |
| `c`    | 1.275    |

The sign of `c` is positive (and large), as expected: high `mStar/mGas`
halos under-predict M with the round-1 form, so the correction must
*raise* M_pred there.

**Test metrics vs. benchmark.**

|                       | benchmark | round 2   | Δ vs. bench | Δ vs. round 1 |
| --------------------- | --------- | --------- | ----------- | ------------- |
| MSE (unweighted)      | 4.3186e-3 | **2.6729e-3** | **−38.11 %** | −21.34 %       |
| MSE (M-weighted)      | 4.7334e-3 | 2.7345e-3 | −42.2 %     | −20.1 %       |
| max bin scatter       | 0.0690    | 0.0553    | better      | better        |
| max abs bin bias      | 0.0279    | 0.0185    | better      | better        |

All AGENTS.md acceptance criteria pass.

**Residual diagnosis (training set).**

| feature       | Pearson | Spearman | binned peak-to-peak |
| ------------- | ------: | -------: | ------------------: |
| Y200          |  +0.009 |   +0.060 | 0.025               |
| YConc         |  −0.069 |   −0.016 | 0.009               |
| GasConc       |  −0.024 |   −0.000 | 0.017               |
| mGas          |  −0.010 |   +0.038 | 0.017               |
| mStar         |  −0.014 |   +0.046 | 0.022               |
| mStar/mGas    |  +0.032 |   −0.005 | 0.014               |

The dominant `mStar/mGas` correlation (−0.39 → +0.03) is killed.
**Every** allowed feature now has |Pearson| < 0.07 and binned-residual
peak-to-peak ≤ 0.025 (≈ noise floor at fixed mass, given the typical
relative scatter is ~0.05). The non-monotonic Y200 bin pattern
(−0.017, +0.005, +0.008, +0.006, −0.001, −0.002) hints at faint
log-curvature in the Y power law, but the magnitude is small.

---

## Round 3 — Swap the linear bracket for `exp(...)` *(rejected)*

**Expression** (11 tokens):
`M = A * Y200**alpha * exp(-b*YConc + c*(mStar/mGas))`

**Motivation.** Round-2 residuals showed faint non-linear curvature in
both `YConc` and `mStar/mGas`. Replacing `(1 + x)` with `exp(x)` adds
Taylor-2 (and higher) corrections at *zero parameter cost* and the
same token count.

**Fitted coefficients.**

| param  | value    |
| ------ | -------- |
| `A`    | 2.530e+07 |
| `alpha` | 0.6177  |
| `b`    | 0.4199   |
| `c`    | 1.302    |

**Test metrics.**

|                       | round 2   | round 3   | Δ vs. round 2 |
| --------------------- | --------- | --------- | ------------- |
| MSE (unweighted)      | 2.6729e-3 | 2.7040e-3 | **+1.16 %** (worse) |
| max bin scatter       | 0.0553    | 0.0558    | slightly worse |

**Verdict.** Rejected. The first-order Taylor expansion of `exp` is
already what the round-2 linear form captures; the higher-order terms
introduce ~1% extra test MSE in the regime probed.

---

## Round 4 — Add a `GasConc` correction *(rejected, triggers stopping criterion)*

**Expression** (13 tokens):
`M = A * Y200**alpha * (1 - b*YConc + c*(mStar/mGas) - d*GasConc)`

**Motivation.** GasConc had a non-trivial bin pattern in round-2
residuals despite small `Pearson`, and traces a different physical
quantity (gas-mass concentration) from `YConc` (pressure
concentration).

**Fitted coefficients.**

| param  | value    |
| ------ | -------- |
| `A`    | 3.019e+07 |
| `alpha` | 0.6293  |
| `b`    | 0.0405   |
| `c`    | 1.129    |
| `d`    | 0.5073   |

Note that `b` collapsed from 0.39 to 0.04 — the fit shifted weight
from `YConc` to `GasConc`, indicating the two are largely *redundant*
once `Y` and `mStar/mGas` are in the model.

**Test metrics.**

|                       | round 2   | round 4   | Δ vs. round 2 |
| --------------------- | --------- | --------- | ------------- |
| MSE (unweighted)      | 2.6729e-3 | 2.6614e-3 | **+0.43 %** (only) |
| MSE (M-weighted)      | 2.7345e-3 | 2.6069e-3 | better        |
| max bin scatter       | 0.0553    | 0.0562    | **slightly worse** |
| max abs bin bias      | 0.0185    | 0.0169    | better        |

**Verdict.** Rejected by the stopping criterion: improvement is below
the 1% threshold *and* the max binned scatter does not improve. The
extra parameter is buying mostly redundancy.

---

## Final answer

After 4 refinement rounds, AGENTS.md's stopping criterion fires at
Round 4. The selected analytic mass estimator is

```
M_pred = A * Y200**alpha * (1 - b * YConc + c * (mStar / mGas))
```

with

| `A`         | `alpha` | `b`     | `c`    |
| ----------- | ------- | ------- | ------ |
| 2.537e+07   | 0.6188  | 0.3893  | 1.275  |

(`A` in units of 1e10 Msun/h; the SI-friendly equivalent is
A ≈ 2.54×10¹⁷ in Msun/h units, with Y in the scaled units of
`MTNG_copy.py`.)

**Margin over the refitted power-law benchmark on the test set:**

| metric              | benchmark | final     | improvement |
| ------------------- | --------- | --------- | ----------- |
| MSE of `M/M_true−1` | 4.3186e-3 | 2.6729e-3 | **−38.1 %** |
| M-weighted MSE      | 4.7334e-3 | 2.7345e-3 | **−42.2 %** |
| max bin scatter     | 0.0690    | 0.0553    | −19.9 %     |
| max abs bin bias    | 0.0279    | 0.0185    | −33.7 %     |

All four diagnostics improve. The final test scatter floor of ≈ 5.2 %
(`sqrt(MSE)`) is consistent with the analytic-model ceiling visible in
the reference RF benchmarks.
