"""Driver: fit & evaluate every candidate in candidates.py against the
freshly-refitted power-law benchmark, then save residual diagnostic
plots for the current best.

Usage (from WSL `base` env):
    python run_candidates.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

from data_loader import load_dataset
from candidates import CANDIDATES, BENCHMARK, Candidate, pack_features
import metrics as M

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def fit_candidate(cand: Candidate, ds, mask_train: np.ndarray):
    """Fit one candidate on the training set, return (popt, fit_status)."""
    X_all = pack_features(ds, cand.feature_names)
    X_train = X_all[:, mask_train]
    y_train = ds.m200H[mask_train]
    sigma = M.fit_sigma(y_train)

    try:
        popt, _ = curve_fit(
            cand.func,
            X_train,
            y_train,
            p0=list(cand.p0),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        status = "curve_fit"
    except Exception as e:
        # Fallback per AGENTS.md: scipy.optimize.least_squares on the
        # M-weighted relative residual.
        def residual(theta):
            pred = cand.func(X_train, *theta)
            return np.sqrt(y_train) * (pred / y_train - 1.0)

        sol = least_squares(residual, x0=list(cand.p0), max_nfev=20000)
        popt = sol.x
        status = f"least_squares ({type(e).__name__})"
    return popt, status


def evaluate_candidate(cand: Candidate, popt, ds, mask_test: np.ndarray, mask_train: np.ndarray):
    X_all = pack_features(ds, cand.feature_names)
    pred_train = cand.func(X_all[:, mask_train], *popt)
    pred_test = cand.func(X_all[:, mask_test], *popt)
    train_metrics = M.evaluate(pred_train, ds.m200H[mask_train])
    test_metrics = M.evaluate(pred_test, ds.m200H[mask_test])
    return pred_train, pred_test, train_metrics, test_metrics


def fmt_params(names, popt):
    return ", ".join(f"{n}={v:.4g}" for n, v in zip(names, popt))


def print_table(rows: list[dict]) -> None:
    """Pretty-print a comparison table of all candidates."""
    cols = [
        ("name", 24),
        ("params", 56),
        ("train_MSE", 12),
        ("test_MSE", 12),
        ("test_MSE_w", 12),
        ("max_scat", 10),
        ("max_|bias|", 11),
        ("vs bench", 10),
    ]
    header = " | ".join(f"{c:<{w}}" for c, w in cols)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(f"{str(r[c]):<{w}}" for c, w in cols)
        print(line)
    print("=" * len(header))


def plot_residual_diagnostics(name: str, pred_test, ds, mask_test):
    """Six-panel residual-vs-feature plot for the current best candidate."""
    res = pred_test / ds.m200H[mask_test] - 1.0
    feats = {
        r"$M_{200c}\,(10^{10}\,M_\odot/h)$": ds.m200H[mask_test],
        r"$Y_{200}$":                         ds.Y200H[mask_test],
        r"$c_Y$":                              ds.YConcH[mask_test],
        r"$c_{\rm gas}$":                      ds.GasConcH[mask_test],
        r"$M_\star/M_{\rm gas}$":              (ds.mStarH / ds.mGasH)[mask_test],
        r"$M_{\rm gas}$":                      ds.mGasH[mask_test],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (label, x) in zip(axes.flat, feats.items()):
        ax.scatter(x, res, s=4, alpha=0.4, color="C1")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$M_{\rm pred}/M_{\rm true} - 1$")
        if any(label.startswith(s) for s in (r"$M_", r"$Y_")):
            ax.set_xscale("log")
        ax.set_ylim(-0.4, 0.4)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Residuals: {name}")
    fig.tight_layout()
    out = FIGS_DIR / f"residuals_{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_binned_scatter(rows: list[dict], bench_metrics: M.Metrics):
    """Compare binned relative scatter, candidate / benchmark, vs. mass."""
    fig, ax = plt.subplots(figsize=(6, 4))
    centers = bench_metrics.bin_centers * 1e10
    ax.semilogx(centers, np.ones_like(centers), "k--", lw=0.8, label="benchmark")
    for r in rows:
        if r["name"] == "cand_benchmark":
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = r["test_metrics"].bin_scatter / bench_metrics.bin_scatter
        ax.semilogx(centers, ratio, "o-", label=r["name"], lw=1.0)
    ax.set_xlabel(r"$M_{200c}\times E(z)^{2/5}\,[h^{-1}M_\odot]$")
    ax.set_ylabel(r"scatter / benchmark scatter")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "binned_scatter_ratio.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("Loading dataset...")
    ds = load_dataset()
    mask_test = ds.mask_test
    mask_train = ~mask_test
    print(f"  N_total={ds.m200H.size}  N_train={mask_train.sum()}  N_test={mask_test.sum()}")

    rows = []
    bench_test_metrics = None
    fits = {}
    for cand in CANDIDATES:
        popt, status = fit_candidate(cand, ds, mask_train)
        pred_train, pred_test, mtr_tr, mtr_te = evaluate_candidate(
            cand, popt, ds, mask_test, mask_train
        )
        fits[cand.name] = dict(popt=popt, pred_train=pred_train, pred_test=pred_test)
        if cand.name == "cand_benchmark":
            bench_test_metrics = mtr_te
            improvement_str = "--"
        else:
            improv = 100.0 * (bench_test_metrics.mse - mtr_te.mse) / bench_test_metrics.mse
            improvement_str = f"{improv:+.2f}%"
        rows.append(
            dict(
                name=cand.name,
                params=fmt_params(cand.param_names, popt),
                train_MSE=f"{mtr_tr.mse:.4e}",
                test_MSE=f"{mtr_te.mse:.4e}",
                test_MSE_w=f"{mtr_te.mse_weighted:.4e}",
                max_scat=f"{mtr_te.max_bin_scatter:.4f}",
                **{"max_|bias|": f"{mtr_te.max_abs_bin_bias:.4f}"},
                **{"vs bench": improvement_str},
                test_metrics=mtr_te,
                fit_status=status,
            )
        )
        print(f"  fitted {cand.name:<22} via {status}: {fmt_params(cand.param_names, popt)}")

    print()
    print_table(rows)

    print("\nFit-status notes:")
    for r in rows:
        print(f"  {r['name']}: {r['fit_status']}")

    print("\nBenchmark binned scatter & bias:")
    for c, n, s, b in zip(
        bench_test_metrics.bin_centers,
        bench_test_metrics.bin_counts,
        bench_test_metrics.bin_scatter,
        bench_test_metrics.bin_bias,
    ):
        print(f"  M~{c*1e10:.2e}  N={n:5d}  scatter={s:.4f}  bias={b:+.4f}")

    # ---- Stopping-criterion logic (AGENTS.md):
    # Walk through rounds in order; advance to round N+1 only if it improves
    # primary test MSE by more than 1% AND its max binned scatter is no worse.
    non_bench = [r for r in rows if r["name"] != "cand_benchmark"]
    selected = None
    print("\nApplying AGENTS.md stopping criterion (>=1% MSE improvement"
          " AND no worse max binned scatter):")
    for r in non_bench:
        if selected is None:
            selected = r
            print(f"  -> accept {r['name']} (first non-benchmark candidate)")
            continue
        prev = selected
        improv = (
            (prev["test_metrics"].mse - r["test_metrics"].mse)
            / prev["test_metrics"].mse * 100.0
        )
        scat_ok = r["test_metrics"].max_bin_scatter <= prev["test_metrics"].max_bin_scatter + 1e-6
        if improv > 1.0 and scat_ok:
            print(f"  -> accept {r['name']}: improv={improv:+.2f}% over "
                  f"{prev['name']}, scatter ok")
            selected = r
        else:
            print(f"  -> reject {r['name']}: improv={improv:+.2f}% over "
                  f"{prev['name']}, scatter_ok={scat_ok}")

    if non_bench:
        raw_best = min(non_bench, key=lambda r: r["test_metrics"].mse)
        print(f"\nRaw lowest-test-MSE candidate: {raw_best['name']} "
              f"(test MSE = {raw_best['test_metrics'].mse:.4e})")
        print(f"AGENTS.md-selected final candidate: {selected['name']} "
              f"(test MSE = {selected['test_metrics'].mse:.4e})")

        # Plot residuals for the AGENTS.md-selected final candidate.
        out = plot_residual_diagnostics(
            selected["name"], fits[selected["name"]]["pred_test"], ds, mask_test
        )
        print(f"  residual plot -> {out}")
        out = plot_binned_scatter(rows, bench_test_metrics)
        print(f"  binned-scatter plot -> {out}")
    else:
        print("No non-benchmark candidates yet.")


if __name__ == "__main__":
    main()
