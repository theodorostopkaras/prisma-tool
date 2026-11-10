---
name: fix-bug
description: Fix a bug in PRISMA (wrong grid/ratio/χ²/ζ values, crash, NaNs, callback error, odd plot). Use whenever the user reports an error, traceback, or suspicious output from app.py or a PRISMA module.
argument-hint: "<error, traceback, or description of the wrong output>"
---

Fix this bug following the CLAUDE.md "Bug investigation" workflow and analysis priors.

1. **Classify** the bug: software (index, axis order, mask, dtype, key), numerical
   (interpolation, log/linear, NaN propagation, percentiles on few cells), or
   physics/convention (formula, unit, definition).
2. **Locate the root cause.** If the faulty code path spans several modules (e.g. app.py
   callbacks plus a helper module) and you cannot find all callers with one or two greps, use the
   `ecc:code-explorer` agent to trace the flow and list every caller.
3. **If the output might be physically wrong rather than just broken** (unexpected trend,
   sign, magnitude), ask `ism-physicist` (physics/attenuation/thermal) and/or `astrochemist`
   (rates, pathways) what the expected behaviour is before changing code.
4. **Fix once** in the shared helper all callers route through. Add or extend a test in `tests/`
   that fails without the fix; run it with `python -m pytest tests/test_<name>.py`.
5. **If the fix touched error handling, fallbacks, NaN masking or `try/except`**, use the
   `ecc:silent-failure-hunter` agent on the changed files.
6. **Review**: use the `ecc:python-reviewer` agent on the diff (skip for one-line fixes).
7. Update `docs/` if behaviour changed. If the user corrected you, append to `tasks/lessons.md`.
   If the bug is visual (layout, theme, figure styling), hand the fix to `ui-designer`.

Report: classification, root cause (`file:line`), fix, test result.

The bug: $ARGUMENTS
