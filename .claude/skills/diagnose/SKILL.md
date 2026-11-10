---
name: diagnose
description: Investigate a discrepancy between PRISMA model grids/fits and observations, or a result that looks physically wrong (fitted ζ, n, χ maps; line ratios; attenuation verdicts; probe ranking). Use before changing any code for such a discrepancy.
argument-hint: "<description of the discrepancy>"
---

Investigate this discrepancy. Do not jump to a code fix.

1. Classify: is it in the comparison, the model grid, the PRISMA code, or a combination?
2. Use the `radio-astronomer` agent to check the comparison is fair (units, beam/filling,
   calibration scale, error maps, which transitions).
3. Use the `ism-physicist` agent for the expected value/trend from first principles.
4. Use the `astrochemist` agent if a species abundance or chemistry pathway is involved.
5. If the PRISMA code is suspected, trace it yourself (or with `ecc:code-explorer` for
   paths across app.py and modules), then hand over to the `fix-bug` skill.
6. Synthesize: cause, evidence, recommended next step (no change / comparison fix / code fix).

The discrepancy: $ARGUMENTS
