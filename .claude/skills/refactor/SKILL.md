---
name: refactor
description: Restructure PRISMA code without changing results (rename, move, split or merge functions, unify duplicated logic, move logic out of app.py). Use when the user asks to clean up, reorganise or deduplicate code.
argument-hint: "<what to restructure>"
---

Refactor following the CLAUDE.md "Refactor" workflow. Outputs must stay identical.

1. **Map callers** in `app.py` (callbacks, component ids) and the modules. Use the
   `ecc:code-explorer` agent when the target is used across several packages.
2. **Design**: if the refactor changes module boundaries use the `ecc:architect` agent first.
   Never merge two similar functions without diffing their numerics.
3. **Baseline**: capture outputs of the affected functions on a representative grid (scratchpad
   script), plus run `python -m pytest tests/`.
4. Change in small steps; after each, re-run the baseline and tests — results must match.
   No physics/numerical changes in the same step.
5. **Review**: use the `ecc:python-reviewer` agent on the final diff.
6. Update `docs/` / `README.md` if names or locations changed. Component ids and
   callback outputs must not change unless the refactor is about them.

The refactor: $ARGUMENTS
