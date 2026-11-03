# Attenuation impact — prisma-tool notes and good values

The other two documents on this page (`attenuation_workflow.md`,
`probe_margin_explained.md`) are the KoSens documentation, kept verbatim so
they can be re-synced. This file says where their names live in the tab, what
the 3-D mode adds, and which values to look for.

## Where the KoSens names live in this tab

| KoSens name (in the docs) | In prisma-tool |
|---|---|
| `HDF5_DIR_NO_ATTEN` / `HDF5_DIR_ATTEN` | main grid / **overlay grid** on the Load tab (SIMLINE overlay for line intensities) |
| `grid_slice_filters` (fixed n) | **Mode** → single n_H slice; mass, metallicity and the other axes follow the sliders |
| `target_shape` | **Resample to n per axis**: ζ × χ for a single slice (native densities), ζ × χ × n_H in 3-D (default 24³) |
| `plot_triple_grid_ratio` (stage 3) | Grids / Intensities tab with the overlay loaded |
| `plot_attenuation_shift_summary` | **1 · Breadth** |
| `plot_attenuation_fuv_response` | **2 · Magnitude & probe** |
| `reconcile_attenuation_species` | **3 · Verdicts** |
| `x_shift_scan_direction`, `x_shift_match_rtol`, `obs_limit` | Setup panel — these change the cells, so they need **Compute** |
| every other threshold | **Thresholds** panel — applied instantly to the computed cells |
| `fuv_swing_dex` / `g0_swing_dex` / `n_swing_dex` | *Swing* / *χ swing* / *n_H swing*; in single-density mode the first two are identical |
| `attenuation_zeta_regime_verdicts` | **3 · Verdicts** → ζ regimes |
| `attenuation_regime_verdicts_3d` | 3-D → verdicts per environment (n_H / χ / ζ bins in the setup panel); `best_regimes` is the table under the matrix and sets its row order; `cube` is the pooled verdict table |
| `attenuation_verdict_frame_3d` | 3-D → the one-row-per-species table under the environment matrix |
| `plot_attenuation_impact_matrix_3d` | 3-D → attenuation impact per environment (+ its ranking table) |
| `plot_probe_scope_3d`, `plot_probe_boxes_3d`, `plot_response_slices_3d` | 3-D → scope of each probe, and the two per-species figures under it |
| G₀, `G0_level` | χ, `chi_level` |

Settings that must agree between the steps (exclude ice / isotopologues, max
per constituent) are entered once and passed to every step, so they cannot
drift apart.

Species selection follows KoSens: *exclude ice* and *exclude isotopologues* in
every mode, and in 3-D mode `require_catalog` (**3-D: only species in CDMS/JPL**,
on by default) — only species, and ratios whose both sides, have laboratory
spectroscopy are graded; the dropped keys are listed in the notes above the
figures. `exclude_species`, `top_n`, `metric` and `min_f_att` of the impact
matrix are in the 3-D setup panel. `plot_triple_grid_ratio(...,
overplot_delta_median=True)` is the cyan line on the x-shift panel of the Grids
/ Intensities tab (row median |Δ| on the top axis, over the same cells as the
rankings; click its legend entry to hide it).

## 3-D mode

Every ζ-row of a 3-D grid sits at a fixed **(n_H, χ)**: it is one cosmic-ray
experiment in one environment. The shift Δ, the slope S and the response R are
computed on those full rows first; everything else is a reduction of that cell
table, so the same three steps can be run on any subset of cells. Every cell
with a model counts: unmatched cells and cells below the detection limit enter
at Δ = 0 and R = 0, and no species is excluded.

- **Pooled** — all densities together. The swing of the per-row median R
  (max − min, nothing trimmed) is split by axis: `g0_swing_dex` (*χ swing*:
  across χ at fixed n_H) and `n_swing_dex` (*n_H swing*: across n_H at fixed χ);
  `fuv_swing_dex` (*Swing*) is the total. The CR-led test uses **only the χ
  part**: χ is a confounder, while the density dependence is largely the
  attenuation itself (more column, more shielded mass, `f_att` rising with n_H),
  so it is reported, not penalised.
- **Verdict per environment** — each axis is cut into equal-log-width bins
  between its own min and max (default 3 → low / mid / high, 27 boxes; or give
  edge values). The three steps are re-run on the cells of each box, so the
  verdict is local: *"at low n_H, high χ and high ζ, HCO⁺ is a clean probe and
  C⁺ is a G0-modulated one"*. Pick an environment to get its own *Best CR
  probes* and *Lines that mislead most* lists.
- **Attenuation impact per environment** — median |Δ| per species and box,
  columns grouped by ζ. ▲ / ▼ mark R > 0 / R < 0 (brighter / fainter when
  attenuated). Colour strips under this matrix and under the verdict matrix give
  each column's n_H, χ and ζ level (light → dark = low → high) instead of the
  environment names; hover a cell for the box and its ranges. Rows are the species with the largest median over the boxes; the
  ranking table lists every species. With the attenuated grid loaded as **HDF5
  overlay**, each column carries the median CR-attenuated mass fraction
  `f_att = ∫4πr² n_H [N_eff ≥ 10²⁰ cm⁻²] dr / ∫4πr² n_H dr`, with
  `N_eff = N(H₂) + N(H)/2` from the clump surface. A column with
  `f_att ≤ min f_att` never switched attenuation on: it is blanked and left out
  of the ranking and of the probe scope. `f_att` is also a grid quantity on the
  Grids tab and appears in each CR attenuation profile's legend.
- **Scope of each probe** — is the response the same in every box? Only boxes
  with |R| ≥ response floor enter. `within_ratio` = median box χ swing / median
  box |R| sets the class; `between_ratio` = spread of the signed R between boxes
  / median |R|. *Universal probe*: both < 1. *Conditional probe*: clean inside
  each box but different between boxes — *Depends on* names the axis that must
  be known. *Not a probe*: χ moves R more than attenuation does even inside a
  box. `within_n_ratio` (density) is drawn in the second panel and does not
  enter the class; `corr(R, f_att)` says how far the density dependence *is* the
  attenuation signal.
- **Per-species figures** — every box of one species in the probe plane
  (colour = n_H, symbol = χ, size = ζ, ★ = whole cube, hollow = below the
  response floor), and the signed R per (n_H, χ) box for each ζ level (green
  outline = robust attenuation probe box, ✕ = no attenuation).

Shifts are matched on the full row before any split, and a no-match cell is
Δ = 0, so cutting ζ into environments cannot move a row end.

## Good values

Units of the ζ error: 0.1 dex = ×1.26, 0.3 dex = ×2, 0.5 dex = ×3, 1 dex = ×10.

### Step 1 · Breadth

| Quantity | Look for | Why |
|---|---|---|
| median \|Δ\| | small = safe to fit without attenuation; large = correct first | the typical ζ error over every cell with a model (no \|S\| gate; unmatched and unobservable cells as 0). The sort key |
| stacked bar vs the "all models" line | — | the gap above a bar is the models that did **not** move |
| `std_abs_shift_dex` | small | the shift is nearly the same everywhere, so one number describes it |
| `frac_observable` | large | reported, not a cut: the other cells entered at 0, so a faint line ranks low |
| regime `n_models` | many | a handful of cells is a noisy regime median — do not follow it up on its own |
| winner map | — | "under these conditions, follow up this species"; ranks \|Δ\| only, so a G0-modulated species can win |

### Step 2 · Magnitude and probe quality

Error panel (left figure) — read in this order:

| Quantity | Look for | Why |
|---|---|---|
| `error_p90` (band right end) | **≤ your tolerance** | decides safety: a line is safe to use without attenuation only if its worst decile is |
| `error_dex` (bar) | small = safe; large = bad over at least half the grid | decides rank, not safety; equals the step-1 median \|Δ\| |
| band width p10–p90 | **narrow** | one number describes the species; wide = regime-dependent → go to the step-1 heatmaps |
| bar colour (`frac_observable`) | **dark** (≈ 1) | pale = most cells are below the detection limit and entered at 0 |

Probe plane (right figure):

| Quantity | Look for | Why |
|---|---|---|
| \|R\| `abs_response_dex` | **large**, ≥ response floor (0.1 dex ≈ 26 %) | does attenuation change the line at all? |
| χ swing `g0_swing_dex` | **small** | how much that response changes with the FUV field (max − min of the per-row median R, at fixed n_H in 3-D) |
| `cr_dominated` = χ swing < \|R\| (kept as two numbers, no combined score) | **True, with \|R\| large** (below the diagonal) | True: if this line changes, blame cosmic rays; False: you cannot tell why |
| n_H swing `n_swing_dex` (3-D) | — | reported, not penalised: the density dependence is largely the attenuation itself |
| `fuv_trend` | ≈ 0 | dex of R per dex of χ; the sign gives the direction |
| `den_contrib` (ratios) | **not ≈ 0** | ≈ 0: the denominator is inert and the ratio restates its numerator |
| `n_fuv_rows` | ≥ 3 | swing needs 2 rows, trend 3 |
| S `slope_dex` | **large** | a steep row resolves ζ; the ideal probe has a large R on a steep row and therefore a *small* \|Δ\| |

### Step 3 · Verdicts

| Verdict | Meaning | Use it for |
|---|---|---|
| **robust attenuation probe** | responds, CR-led, and the ζ error matters | the best lines to measure ζ — and you must correct them for attenuation |
| **robust, small bias** | clean probe, ζ error below the threshold | good ζ probe that attenuation barely biases |
| **G0-modulated** | χ swing beats the response | do not use it to measure ζ; its ζ error can still be large (it may lead *Lines that mislead most*) |
| **no response** | \|R\| below the floor | insensitive to attenuation — or undetectable over most of the cells (check *Observable*) |
| **no data** | missing from one of the two tables, or no usable row | check the overlay |

The flags c / m are the second and third questions (CR-led, matters); the
verdict is the first question that fails, so a reader who disagrees with a
threshold can see which species would flip. The thresholds are labels on
numbers computed identically for every species, not cuts. The two lists answer
different questions and are allowed to disagree: **Best CR probes** is for
choosing a line to observe, **Lines that mislead most** is for choosing what to
correct. A whole-grid median under-ranks a species sensitive in one ζ band only:
read its grade in the ζ regimes or the 3-D boxes.
