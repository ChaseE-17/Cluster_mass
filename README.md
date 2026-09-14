# Cluster Mass — the Y–M relation pipeline

Building an **analytic proxy for the Y–M relation** (integrated SZ signal ↔ cluster
mass) in MTNG cluster halos, via an ML pipeline (random forest → SHAP → symbolic
regression). The work is done two ways in parallel: **by hand** (a notebook) and by
an **agentic loop**.

## Start here

| You want to… | Go to |
|---|---|
| Understand the science / the manual method | [`manual/`](manual/) → `MTNG_HSE_YM.ipynb` (canonical) |
| Run or extend the agentic loop | [`agentic_ym/`](agentic_ym/) → `AGENTS.md` |
| Know how the repo is laid out | the **Directory map** below |
| Have an agent work here | it auto-loads [`CLAUDE.md`](CLAUDE.md) — read that too |

## Directory map

| Path | Status | What it is |
|---|---|---|
| `manual/` | **LIVE** | Manual method: notebook + `docs/` (lit & decisions) + `cgas/` (runnable SR sub-project) |
| `agentic_ym/` | **LIVE** | Agentic discovery loop. **Its own git repo** (nested), not part of this one |
| `ym_arm_obs/`, `ym_arm_obs_theory/` | **WORKTREE** | Git worktrees of `agentic_ym`, one per experiment arm. Derived — do not edit as source |
| `MTNG_data/` | **DATA** | MTNG group catalogs & profiles (tracked, large) |
| `scatter/` | **VENV** | Python 3.11 virtualenv (git-ignored). The project's real interpreter |
| `figs/` | output | Generated figures |
| `agentic/`, `agentic_old/`, `agentic_old_2/`, `Old Loops/`, `_old_ym_projct/` | **LEGACY** | Prior iterations. Reference only — **not** the current method |

## The two methods (and how they relate)

```
manual/MTNG_HSE_YM.ipynb   →   manual/cgas/        →   agentic_ym/
  (science reference:            (clean-room            (full agentic
   HSE mass, Y–M target,          prototype of the       proposer/harness
   feature selection, SR)         §17 SR loop)           loop, 3 arms)
```

- **Manual** — one canonical notebook, `manual/MTNG_HSE_YM.ipynb`. Data → radial
  profiles → gNFW pressure fit → HSE mass & bias `b = 1 − M_HSE/M200c` → tiered
  feature selection (RF + SHAP) → symbolic regression → two targets (HSE bias, and
  the Y–M scatter `y = M200c/(A·Y200^0.6)`, §17). Literature and design decisions
  live in `manual/docs/`.
- **Agentic** — a bounded "agent proposes an equation / frozen harness scores it"
  loop with anti-cheat (split lockbox, AST constraints). See `agentic_ym/AGENTS.md`.

## Setup

```bash
source scatter/bin/activate          # Python 3.11 venv (numpy/scipy/pandas/pysr…)
```

- **Two nested git repos:** this repo (notebooks + data) and `agentic_ym/` (the loop)
  each have their own `.git`. The `ym_arm_obs*` dirs are worktrees of `agentic_ym` and
  must be created **from inside `agentic_ym`** (see `agentic_ym/BRINGUP.md`).
- **Notebooks** are tracked with an `nbstripout` git filter: outputs stay in your
  working copy but are stripped from what git stores. Agents committing notebooks
  need `nbstripout` installed in the venv (`pip install nbstripout`).

## Documentation conventions

Docs are kept consistent by a light contract (details in [`CLAUDE.md`](CLAUDE.md)):
every live doc carries a **status header** (`Status · Updated · Maintainer · Scope`),
references code by **stable anchor** (`§17`, a function name) not by volatile cell
number, and is listed in its directory's index. A warn-only pre-commit check
(`tools/check_docs.py`) flags docs missing a header or marked `STALE`. Enable it once
per clone with:

```bash
git config core.hooksPath tools/hooks
```
