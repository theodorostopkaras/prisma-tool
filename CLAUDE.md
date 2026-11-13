# PRISMA

**PDR Results Interactive Simulation Mapping & Analysis** — a Dash/Plotly app for
photon-dominated-region model grids (**KOSMA-τ** and **Meudon** HDF5, one file per model,
n_H × FUV × ζ, plus optional mass / metallicity / attenuation).

- Entry point: `app.py` (layout + ~115 callbacks; very large — grep, never read whole)
- Science modules at the repo root (see map below); styling in `assets/theme.css`, `plot_style.py`
- Procedure docs: `docs/` (attenuation workflow, probe margin, good values)
- Several modules are vendored from KoSens (`~/Desktop/Projects/Kosens_toolkit`); a bug
  found here may exist there too — mention it
- Lessons from past corrections: `tasks/lessons.md` — check it, and append to it
- Install / update deps: `python -m pip install --upgrade -r requirements.txt`
- Tests: `python -m pytest tests/`
- Full tab list, filename encoding and usage: `README.md`

## Coding standards

- PEP8, 4 spaces, max line length 99; `snake_case` / `CamelCase` / `UPPER_CASE`
- Docstrings on public functions; type hints where useful
- Computation lives in the modules; `app.py` callbacks stay thin glue
- New dependencies go in `requirements.txt` in the same change

## Change discipline

- SIMPLICITY FIRST — smallest change that solves the problem
- No drive-by refactors, unrelated cleanups, or new docs the user did not ask for
- Document scientific calculations when adding or changing them
- On bug reports: fix from logs/errors/tests; do not ask for hand-holding
- If stuck: stop and re-plan; do not thrash
- After the user corrects your approach, append to `tasks/lessons.md`
  (date — mistake — rule that prevents it)
- **Keep `README.md` and `docs/` in sync** with any change to an algorithm, default,
  threshold, unit, tab or control

## Analysis priors (apply to every change and every result)

1. **Physical plausibility is the test.** A map that renders cleanly can still be wrong. Ask
   "does this make sense for a PDR at this n, χ, ζ?" — C⁺ emission rises with χ, CO-traced
   quantities sit deeper, CR attenuation acts at N ≳ 1e20 cm⁻².
2. **Respect the existing code and its history.** Check `tasks/lessons.md`, `docs/` and
   `git log -p` for *why* something odd exists before changing it.
3. **Model outputs are smooth across the grid.** Jumps, stripes or checkerboards between
   neighbouring n/χ/ζ cells usually mean an interpolation, axis-order, mask or NaN bug.
   A real sharp transition (H/H₂, C⁺/C/CO) must be justified explicitly.
4. **Small values are real.** No floors, clips, threshold gates or species cuts without
   physical justification and user approval.
5. **Units and naming are part of correctness.** `k` (coefficient) vs `R` (volumetric rate)
   vs τ (timescale); ζ_H vs ζ_H₂; K km/s vs erg s⁻¹ cm⁻² sr⁻¹; χ in Draine units; log vs
   linear axes. KOSMA-τ vs Meudon conventions differ — check which file type you're in.
6. **Every scientific claim is checked against the literature** (arXiv / NASA ADS).

## Who does what (subagents)

Science specialists in `.claude/agents/` are **read-only consultants**:

| Question | Agent |
|---|---|
| Rates, reaction pathways, chemical networks, Chemistry tab | `astrochemist` |
| PDR physics, thermal balance, expected trends, CR attenuation, probe ranking | `ism-physicist` |
| Observed maps/cubes/spectra, units, beams, calibration, map fitting | `radio-astronomer` |
| "Is this abundance/ratio wrong?" | `astrochemist` + `ism-physicist` |
| "Model vs observation disagrees" | `radio-astronomer` → `ism-physicist` → `astrochemist` |

Design specialist (may edit layout / CSS / figure styling only):

| Question | Agent |
|---|---|
| Tab layout, controls, theme, dark mode, Plotly styling, colour scales, accessibility | `ui-designer` |

Engineering roles use installed agents:

| Task | Agent |
|---|---|
| Trace a flow across `app.py` callbacks and modules | `ecc:code-explorer` (or `Explore`) |
| New module / module boundaries | `ecc:architect` |
| Review any Python change | `ecc:python-reviewer` |
| Swallowed errors, silent NaN / fallback paths | `ecc:silent-failure-hunter` |
| Test coverage of a change | `ecc:pr-test-analyzer` |

Spawn agents only when the task warrants it (multi-module, scientific doubt, visual redesign,
or the user asks); small edits are done inline.

**Project skills** (`.claude/skills/`) run the workflows below and call the agents where
needed — invoke the matching one for every change request:
`fix-bug`, `new-analysis`, `refactor`, `diagnose` (model–observation discrepancies),
`ui-change` (look and interaction).

## Workflows

### Bug investigation
1. Classify: software (index, axis order, mask, dtype, key, callback id), numerical
   (interpolation, log/linear, NaN propagation), or physics/convention (formula, unit).
2. Reproduce on the smallest grid/slice; find the root cause in the shared helper and grep
   every caller (including `app.py` callbacks) before editing.
3. Fix once; add/extend a test in `tests/` that fails without the fix.
4. Run tests; check the tab output is physically sensible.

### New analysis / scientific calculation
1. Reference and formula first (arXiv/ADS); consult a science agent if the trend is unclear.
2. Reuse existing helpers (`grid_naming`, `grid_interp`, `line_catalog`, `model_config`, …).
3. Docstring: equation, units, validity, approximations.
4. Validate: limiting cases and direction of change; one test in `tests/`.
5. UI for it via `ui-designer`; update `README.md` / `docs/`.

### UI change
**Always invoke the `impeccable` and `design-taste-frontend` skills**
(`.claude/skills/impeccable`, `.claude/skills/taste-skill`) for any request about how the app
looks or for design ideas/critique — before proposing or editing anything.
1. Find the component id in `app.py` and its rules in `assets/theme.css`.
2. Use theme tokens and `plot_style.py`; keep ids and callback signatures stable.
3. Check both light and dark themes; never hide or clip data for looks.

### Refactor
1. Map callers (modules + `app.py` callbacks/ids).
2. Capture current outputs on a representative grid first.
3. Small steps; outputs identical afterwards. Never mix refactor and physics change.

### Pre-commit checklist
```
[ ] Root cause fixed in the shared place; all callers checked
[ ] python -m pytest tests/ passes
[ ] Output physically plausible and smooth across the grid
[ ] No new floors / cutoffs / species gates without approval
[ ] Units and naming consistent (k / R / τ, ζ, intensity units, KOSMA-τ vs Meudon)
[ ] UI checked in light and dark theme (if touched)
[ ] README.md / docs/ / requirements.txt updated if needed
[ ] tasks/lessons.md appended if the user corrected the approach
[ ] No unrelated changes; PEP8, ≤ 99 columns
```

## Context / token hygiene

- `app.py` (~19k lines), `chem_network.py`, `map_fit.py`, `atten_compare.py` are large —
  grep before reading, read targeted slices
- FITS / HDF5 / PNG files: reference by path, never load into context
- Scratch scripts go in the session scratchpad, never the repo root

## Module map

| Module | Role |
|---|---|
| `app.py` | Dash layout, tabs, callbacks |
| `grid_naming.py` | Model filename decoding → grid axes |
| `grid_interp.py` | Grid interpolation / resampling (KoSens3D-compatible) |
| `grid_fit.py` | 3-D SIMLINE intensity cubes, FITS-map fitting |
| `model_config.py` | PDR config JSON (Model setup tab) |
| `chem_network.py` | Reaction ranking (Chemistry tab) |
| `diag_chem.py` | Diagnose chemistry-only + overlay directory pairing |
| `transition_fronts.py`, `rgb_phase.py` | H/H₂, C⁺/C/CO fronts, RGB phase maps |
| `line_catalog.py` | Molecular line identification in a frequency window |
| `smli_labels.py`, `simline_spectra.py` | SIMLINE transition labels, PV FITS spectra |
| `cr_attenuation.py`, `atten_compare.py` | CR attenuation, attenuation impact |
| `probe_ranking.py` | CRIR probe scores, CR vs FUV dominance |
| `map_fit.py`, `map_fit_extras.py` | Fit observed FITS maps to the grid |
| `obs_spectrum_fits.py`, `cube_viewer.py` | Observed spectra, cube viewer |
| `plot_style.py`, `assets/theme.css` | Figure fonts, design system |
