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
| `target_shape` | **Resample ζ × χ** (n_H keeps its native slices) |
| `plot_triple_grid_ratio` (stage 3) | Grids / Intensities tab with the overlay loaded |
| `plot_attenuation_shift_summary` | **1 · Breadth** |
| `plot_attenuation_fuv_response` | **2 · Magnitude & probe** |
| `reconcile_attenuation_species` | **3 · Verdicts** |
| `x_shift_scan_direction`, `x_shift_match_rtol`, `obs_limit` | Setup panel — these change the cells, so they need **Compute** |
| every other threshold | **Thresholds** panel — applied instantly to the computed cells |
| `fuv_swing_dex` | `swing_dex` (the margin subtracts it); in single-density mode the two are identical |

Parameters that the workflow says must agree between the steps (min |S|,
exclude ice / isotopologues, max unmatched, max censored, min observable, max
per constituent) are entered once and passed to every step, so they cannot
drift apart.

## 3-D mode

Every ζ-row of a 3-D grid sits at a fixed **(n_H, χ)**: it is one cosmic-ray
experiment in one environment. The shift Δ, the slope S, the response R and
the censoring bound are computed on those full rows first; everything else is a
reduction of that cell table, so the same three steps can be run on any subset
of cells.

- **Pooled** — all densities together. The swing is taken over all
  (n_H, χ) rows, so `probe_margin > 0` means the attenuation response beats its
  own spread over *both* density and FUV. Two attribution columns say which axis
  spoils a species: `fuv_swing_dex` (median over densities of the χ-swing at
  fixed n_H) and `n_swing_dex` (spread across densities of the per-density
  median R). Large `n_swing_dex` = the response depends on density.
- **Verdict by density** — the whole procedure re-run on each density slice,
  next to the pooled verdict. A row that keeps its colour is a verdict you can
  quote at any density; a row that changes is density-specific.
- **Verdict per environment** — the n_H, χ and ζ axes are split at the edges
  you give (`auto` = once at the log midpoint → low/high, 8 environments). The
  three steps are re-run on the cells of each environment, and the swing is
  measured only across the environment's own (n_H, χ) rows, so the verdict is
  local: *"at low n_H, high χ and high ζ, HCO⁺ is a clean probe and C⁺ is an
  FUV tracer"*. An environment with fewer than *min cells* cells, or fewer than
  two rows, for a species grades it **too few models** rather than guessing.
  Pick an environment to get its own *Best CR probes* and *Lines that mislead
  most* lists.

Splitting ζ into environments cuts every row: cells near the upper ζ edge of a
low-ζ environment have little rightward room, so their censored fraction rises.
The bound is still the full row's end, never the environment's.

## Good values

Units of the ζ error: 0.1 dex = ×1.26, 0.3 dex = ×2, 0.5 dex = ×3, 1 dex = ×10.

### Step 1 · Breadth

| Quantity | Look for | Why |
|---|---|---|
| `frac_shifted` | **large** (≥ 0.5 = "broad") | how much of the matched map moved; small = the effect sits in one corner → *CR-led, narrow* |
| `frac_unmatched` | **small** (≤ 0.5 to be ranked) | an unmatched cell counts as movement here but is a cell with no measurement; above the gate the species is *unmeasurable* |
| `frac_trusted` | **large** | share of matched cells on rows steep enough (\|S\| ≥ min) to back the median |
| median \|Δ\| | small = safe to fit without attenuation; large = correct first | the typical ζ error over trusted matched cells |
| `std_abs_shift_dex` | small | the shift is nearly the same everywhere, so one number describes it |
| `frac_observable` | ≥ min observable fraction | detectable over most of the grid (line intensities with a detection limit) |
| regime `n_models` | many | a handful of cells is a noisy regime median — do not follow it up on its own |
| winner map | — | "under these conditions, follow up this species"; ranks \|Δ\| only, so an FUV tracer can win |

### Step 2 · Magnitude and probe quality

Error panel (left figure) — read in this order:

| Quantity | Look for | Why |
|---|---|---|
| bar colour / ▶ (`frac_censored`) | **pale, no ▶** (0) | > 0: every number of that species is a lower limit; > 0.5: not ranked |
| `error_p90` (band right end) | **≤ your tolerance** | decides safety: a line is safe to use without attenuation only if its worst decile is |
| `error_dex` (bar) | small = safe; large = bad over at least half the grid | decides rank, not safety |
| band width p10–p90 | **narrow** | one number describes the species; wide = regime-dependent → go to the step-1 heatmaps |
| `bound_gap` | **≈ 0** | how far imputed bounds moved the median; large = leaning on imputation |

Probe plane (right figure):

| Quantity | Look for | Why |
|---|---|---|
| \|R\| `abs_response_dex` | **large**, ≥ response floor (0.1 dex ≈ 26 %) | does attenuation change the line at all? |
| `swing_dex` | **small** | how much that response changes with the environment (G₀; n_H too in 3-D) |
| `probe_margin` = \|R\| − swing | **positive and large** (below the diagonal) | > 0: if this line changes, blame cosmic rays; < 0: you cannot tell why |
| `fuv_trend` | ≈ 0 | dex of R per dex of G₀; the sign gives the direction |
| `den_contrib` (ratios) | **not ≈ 0** | ≈ 0: the denominator is inert and the ratio restates its numerator |
| `n_fuv_rows` | ≥ 3 | swing needs 2 rows, trend 3 |
| S `slope_dex` | **large** | a steep row resolves ζ; the ideal probe has a large R on a steep row and therefore a *small* \|Δ\| |

### Step 3 · Verdicts

| Verdict | Meaning | Use it for |
|---|---|---|
| **CRIR probe** | broad, honest, CR-led, and the ζ error matters | the best lines to measure ζ — and you must correct them for attenuation |
| **CR-led, small bias** | clean probe, ζ error below the threshold | good ζ probe that attenuation barely biases |
| **CR-led, narrow** | CR-led but only a small part of the grid moves | usable only in that regime — see the winner map / heatmaps |
| **FUV tracer** | swing beats the response | do not use it to measure ζ; its ζ error can still be large (it may lead *Lines that mislead most*) |
| **unmeasurable** | mostly unmatched or censored | no number to trust; not a low grade, an absent measurement |
| **no response** | \|R\| below the floor | insensitive to attenuation |
| **unobservable** | below the detection limit over most of the grid | nobody can detect it |
| **too few models** | (environments) too few cells or rows | widen the environment edges |
| **no data** | missing from one of the two tables | check the overlay |

The flags b / h / c / m are the four questions (broad, honest, CR-led, matters);
the verdict is the first one that fails, so a reader who disagrees with a
threshold can see which species would flip. The two lists answer different
questions and are allowed to disagree: **Best CR probes** is for choosing a
line to observe, **Lines that mislead most** is for choosing what to correct.
