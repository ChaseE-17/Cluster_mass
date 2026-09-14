# CLAUDE.md — operating manual for agents in this repo

This file is auto-loaded into your context. It tells you how to navigate, what not to
touch, and how to keep the docs consistent. The human-readable map is in
[`README.md`](README.md) — read it for the directory layout; it is not duplicated here.

## Environment (do this first)

- **Activate the venv before running anything:** `source scatter/bin/activate`.
  System `python3` is 3.9 with no numpy; bare `python`/harness calls fail without the
  venv. The real interpreter is `scatter/` (Python 3.11).
- **Two nested git repos:** the top-level repo (notebooks + data) and `agentic_ym/`
  each have their own `.git`. A commit in one does not touch the other.
- **Worktrees:** `ym_arm_obs/` and `ym_arm_obs_theory/` are git worktrees of
  `agentic_ym`, one per arm. Create them **from inside `agentic_ym`**
  (`git worktree add -b run/<arm> ../ym_arm_obs`), never from the top-level repo, or
  you get a worktree full of notebook content. See `agentic_ym/BRINGUP.md`.
- **Notebooks** use an `nbstripout` git filter (outputs stripped on commit, kept in the
  working tree). It runs via the venv python; keep the venv active when committing `.ipynb`.

## Where NOT to go

`agentic/`, `agentic_old/`, `agentic_old_2/`, `Old Loops/`, `_old_ym_projct/` are
**LEGACY** — prior iterations kept for reference. Do not read them to understand the
current method, and do not edit them. The live methods are `manual/` and `agentic_ym/`.

## Reading order

- Working in **`manual/`** → `manual/README.md`, then the notebook by section, then the
  relevant `manual/docs/` file. (`manual/CLAUDE.md` has the local rules.)
- Working in **`agentic_ym/`** → `AGENTS.md` → `CONCEPTS.md` → the arm's `prompts/` file.

## Documentation contract

Follow these whenever you change code or docs, so docs stay consistent across many
agents and the human editing in parallel.

1. **Update the doc when you change the thing it describes.** If you change code, update
   the doc that references it (each doc names what it `Describes:` in its header). If you
   add or remove a doc, update that directory's README index table.
2. **Status header on every live doc.** First lines of any `.md` under a live method:
   ```
   Status: LIVE | STALE | ARCHIVE   ·   Updated: YYYY-MM-DD   ·   Maintainer: human | agent | both
   Scope: <one line>   ·   Describes: <files/sections this doc tracks>
   ```
   Bump `Updated:` when you edit. Mark `STALE` (don't silently leave wrong) if you can't
   fix it now. The checker is lenient: `**bold**` markers are fine and `Date:` counts as
   `Updated:` (many existing docs use that older form) — but new/edited docs should use
   the full form above, including `Maintainer:`.
3. **Respect maintainer ownership.** Do **not** edit docs marked `Maintainer: human`
   during a run (e.g. `agentic_ym/CONCEPTS.md`). Append-only logs marked
   `Maintainer: agent` are yours; the human won't hand-edit those mid-run.
4. **Reference by stable anchor, not volatile index.** Cite notebook `§17` or a function
   name — never "cell 87" or a raw cell count (those rot as the notebook grows).
5. **Keep the map in one place.** The directory map lives in `README.md`; link to it,
   don't recopy it.

A warn-only check (`tools/check_docs.py`, run via the `core.hooksPath tools/hooks`
pre-commit hook) reports docs missing a header or marked `STALE`. It never blocks a
commit — it's a punch-list, for you and the human both.
