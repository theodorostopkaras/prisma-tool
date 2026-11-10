---
name: new-analysis
description: Add or change a scientific calculation, metric, threshold, default or plot in PRISMA (new ratio, χ² term, probe score, attenuation metric, regime classifier, new tab panel). Use whenever a formula or numerical procedure is added or modified.
argument-hint: "<what to compute / change>"
---

Implement this following the CLAUDE.md "New analysis" workflow and analysis priors.

1. **Science first.** State the formula, units and reference; verify it on arXiv / NASA ADS.
   If the expected trend or regime is unclear, ask the science agents:
   `ism-physicist` (PDR physics, CR attenuation, probe ranking), `astrochemist` (rates/networks),
   `radio-astronomer` (observables, units, map fitting, errors).
2. **Placement.** Search for existing helpers first. If the change needs a new module or touches more
   than ~3 public functions, use the `ecc:architect` agent for the placement before writing code.
3. **Implement** with a docstring giving equation, units, validity range and approximations.
   Keep computation out of `app.py` callbacks: put it in the module, call it from the callback.
4. **Validate**: limiting cases and the expected direction of change when one parameter
   varies; add one small test in `tests/`.
5. **Review**: use the `ecc:python-reviewer` agent on the diff, then `ecc:pr-test-analyzer`
   if the test covers only the happy path of new branching logic.
6. If it adds or changes UI, use `ui-designer` for layout and figure styling.
7. Update `README.md` / `docs/` in the same change.

The task: $ARGUMENTS
