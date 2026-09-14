# CLAUDE.md — manual/ local rules

Auto-loaded when you work under `manual/`. Read `manual/README.md` for the map; the
root `CLAUDE.md` has the repo-wide environment and doc contract. Local rules:

- **`MTNG_HSE_YM.ipynb` is the only live notebook.** Everything in `archive/` is dead
  and git-ignored — never edit it or treat it as source. If you need old behavior, read
  it, don't revive it.
- **Notebooks strip on commit.** The `nbstripout` filter removes cell outputs from what
  git stores; keep the `scatter` venv active so the filter runs. Don't try to commit
  outputs — they're intentionally excluded.
- **Cite the notebook by section (`§17`), not by cell number.** Cell indices drift as
  the notebook grows and silently rot doc references.
- **Docs carry status headers** (see root `CLAUDE.md`). When you change the notebook or
  `cgas/` code, update the doc that `Describes:` it and bump its `Updated:` date. When
  you add/remove a `docs/` file, update the index table in `manual/README.md`.
- **`docs/feature_selection_YM.md` is STALE** — use `docs/feature_selection_YM_SYNTHESIS.md`
  instead. Don't propagate the stale one.
- **`cgas/` has a frozen evaluator.** Don't redefine the scoring/target inside a
  proposal; follow `cgas/PROTOCOL.md`.
