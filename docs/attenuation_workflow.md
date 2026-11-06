# Grid attenuation comparison — full workflow

The method behind the Attenuation impact tab. Function names refer to the
steps of the workflow; their implementation is in `atten_compare.py`.

---

## 0. The question

Two PDR model trees over the same `(ζ_H, G_0)` plane, identical in every respect
except cosmic-ray attenuation:

| case | tree | meaning |
|---|---|---|
| reference | `HDF5_DIR_NO_ATTEN` | constant CR field, α = 0 |
| attenuated | `HDF5_DIR_ATTEN` | attenuated CR field, α = 0.4 |

**What does an observer lose by fitting a real, attenuated source against a constant-CR
model grid?**

Two sub-questions that the notebook keeps deliberately separate:

- *Which lines mislead me most if I ignore attenuation?* → a **correction priority** list.
- *Which lines are the cleanest cosmic-ray probes?* → a **line selection** list.

They are not the same list, and the workflow exists to stop them being confused.

> **Hard requirement.** The two runs must differ **only** in attenuation. Any other
> difference is mathematically indistinguishable from an attenuation shift in every
> figure below.

---

## 1. Stage 1 — configuration (notebook §2)

The only cell that is edited. Same knobs as `grid.ipynb`, plus the two directory roots.

```python
grid_type_definition = "rel_abund"      # or 'intensity', 'column_density', 'level_abund'
type_of_grid         = "pdrgrid"        # 'pdrgrid' or 'simlinegrid'; dustgrid is not dual-wired
target_shape         = (60, 60)         # interpolation target for both cases
species_transitions  = 'all'            # sweep everything — the point is to find the movers

grid_slice_filters = {'mass': 100, 'fuv': None, 'crir': None, 'n': 1e3}

HDF5_DIR_NO_ATTEN = ".../new_teo_grid/pdrgrid_hdf5"
HDF5_DIR_ATTEN    = ".../new_teo_grid_atten/pdrgrid_hdf5"
```

`grid_slice_filters` fixes the non-axis dimensions so that what remains is a genuine 2D
`(ζ_H, G_0)` plane: here a constant-mass (100 M☉), constant-density (10³ cm⁻³) slice.
`axes_names_determination` then reduces the six candidate axis names to the two that
actually vary.

`dustgrid` raises `NotImplementedError` — the dual path is only wired for `pdrgrid` and
`simlinegrid`.

---

## 2. Stage 2 — run the pipeline twice (notebook §3)

`_run_one_case(...)` is the whole `grid.ipynb` chain applied to one model tree:

```
process_grids(...)          → final_pdr_grid_dataframe, all_grids, error grids/stats,
                              interpolated_grids, original_grids
create_final_grids(...)     → my_grids  (dict: name → {'grid', y_axis, x_axis})
create_transition_ratios()  → ratio grids, merged into my_grids
check_max_values()          → sanity print
```

It is called twice:

```python
final_df_a0, my_grids_a0, *_ = _run_one_case(directory_prefix=directory_prefix_a0)  # reference
final_df_a1, my_grids_a1, *_ = _run_one_case(directory_prefix=directory_prefix_a1)  # attenuated
```

This is the expensive cell — the full processing chain, run twice.

A third dictionary is then built by cellwise division over the **shared keys only**:

```python
my_grids_ratio[name]['grid'] = my_grids_a0[name]['grid'] / my_grids_a1[name]['grid']
```

guarded by `(g0 > 0) & isfinite(g0) & isfinite(g1)`, NaN elsewhere. The printed line
`Keys in ref. / atten. / ratio dict` is the check that both trees produced the same
species set — a mismatch here means the two runs are not comparable and everything
downstream is meaningless.

---

## 3. The one table underneath everything

`collect_attenuation_x_shift_table(my_grids_ref, my_grids_atten, ...)` walks both grids
cell by cell and returns a long frame, **one row per `(species, iy, ix)`**:

| column | meaning |
|---|---|
| `species`, `iy`, `ix` | identity and grid indices |
| `x`, `y`, `x_name`, `y_name` | physical coordinates (ζ_H, G_0) |
| `value_ref`, `value_atten` | the two intensities in that cell |
| `shift_dex`, `abs_shift_dex` | Δ, the horizontal match shift (0 where no match, NaN only where there is no model) |
| `slope_dex` | S, the local CR sensitivity of the **reference** grid (diagnostic only, never used as a filter) |

**Every figure in the workflow is a reduction of this one table.** That is why the two
plot functions can be joined on `species` in stage 7 with no risk of inconsistent inputs.

### 3.1 The horizontal shift, Δ

`horizontal_intensity_matched_x_shift_dex` takes the reference intensity as a target and
scans the attenuated row at the same `y` for the `x` that reproduces it, within
`x_shift_match_rtol` (default 0.02 = 2%), interpolating linearly in `log10(x)` between
samples:

```
Δ = log10( x_match / x_nom )
```

Under the default **rightward-only** scan this has a direct observational meaning.
Fitting an attenuated source against the constant-CR grid returns the `ζ_inferred` that
solves `I_const(ζ_inferred) = I_atten(ζ_true)` — exactly the match performed here. So

```
Δ = log10( ζ_true / ζ_inferred )     ← the dex error in the inferred ionization rate
```

Attenuation weakens the CR field, the line looks like a lower-ζ model, and ζ comes out
too low.

### 3.2 The local CR sensitivity, S

`_log_intensity_slope` differentiates the reference grid along each row by central
differences (one-sided at the edges), taking spacing from `log10(x)` itself because rows
need not share an x sampling:

```
S = d log10(I_ref) / d log10(x)
```

How many dex of intensity the species gains per dex of CRIR. Measured on the **reference
grid only**: the question it answers is whether the grid you would be fitting against can
resolve ζ at all in that cell.

### 3.3 The intensity response, R

Computed inside `plot_attenuation_fuv_response` from the same two intensity columns, at
the same cell, **with no matching involved**:

```
R(x, y) = log10( I_atten(x, y) / I_const(x, y) )
```

The intensity lost at fixed ζ when attenuation is switched on. Because it needs no match,
no tolerance band and no scan direction, R is defined in every cell where both
intensities are positive — **including the cells where the shift search fails**.

### 3.4 How the three relate — the key relation

R is a *vertical* distance between the two curves at fixed x. Δ is a *horizontal*
distance at fixed intensity. S is the slope that converts one into the other:

```
|Δ|  ≈  R / S
```

This is the single most important relation in the workflow.

Where the reference curve flattens (S → 0), a small vertical drop R demands an enormous
horizontal displacement — or none reproduces it at all and the match fails. **The shift
blows up not because cosmic rays matter more there, but because the species has stopped
responding to them.**

**No gate on S is applied.** Every cell with a model enters every median, small shifts and
flat-row cells included; cells below the detection limit enter at 0 (§3.5).
The price is a bias in the 2D whole-grid numbers: a species that shifts a little over
most of the map outranks one that is strongly sensitive in a single CRIR band, because
the median of the latter is pulled down by the inert part of its rows. The per-ζ regime
verdicts (§7.1) and the 3D per-box verdicts grade each CRIR band separately and remove
that bias, so no filtering is needed to protect them.

The relation is a local linearisation and is **never used to compute anything**: Δ always
comes from the real match search, so it stays correct where rows are curved.

### 3.5 What a no-match cell means

The subtlest point in the method. `_x_match_intensity_to_x` returns no match for
genuinely different reasons:

1. the target lies outside the attenuated row's entire intensity envelope;
2. the row's shape changed so that no sample matches within tolerance;
3. under the rightward-only scan, a match exists only on the left half and is discarded.

Only (1) is fairly described as *"the required ζ lies past the end of the grid"*. Cases
(2) and (3) mean that **no horizontal displacement reproduces the attenuated intensity at
all**, because the two curves differ by something this construction does not model — a
vertical offset, or a displacement along `G_0`. Attributing that to a cosmic-ray
inference error would be wrong.

The cases cannot be separated from the shift alone. **The choice made here: a no-match
cell is read as "no shift" (Δ = 0)**, neither as missing data nor as a lower bound on a
larger shift. `collect_attenuation_x_shift_table` writes 0 into every cell that has a
model in both grids but no match; only cells with no model at all (non-finite intensity,
non-positive x) stay NaN. A no-match cell therefore pulls every median towards zero
rather than up, and a species whose attenuation is not reproducible by any horizontal
displacement is graded as barely shifted, not as badly biased.

Where case (1) holds, the true shift is larger than 0, so every median built this way is a
**conservative** (low) estimate of the ζ error, never an imputed one. This bites most in
the high-ζ part of the grid, where there is little room left to shift into.

**Unobservable cells follow the same rule.** A cell where a line behind the key is below
`obs_limit` (100 mK, intensity grids only) shows no change anyone could measure, so it
enters at **Δ = 0 and R = 0** (`_zero_unobservable_shifts`, applied inside both ranking
functions; the `pixels` table keeps the raw shift and the `observable` flag).

Consequences that run through the rest of this document:

- **Every species is summarised over the same cells with the same rule, and no species is
  excluded.** There are no sample cuts on unmatched, censored, shifted or observable
  fractions. A species detectable over a small part of the grid ranks low instead of
  ranking on that part alone; `frac_observable` is reported next to it.
- The only thresholds left are labels on the continuous numbers (`response_floor_dex`,
  `min_error_dex`, the `swing < |R|` diagonal, §7) and display choices (`max_species`,
  `max_per_constituent`, `min_abs_response_dex = 0`).

---

## 4. Stage 3 — triple-panel comparison (notebook §4)

```python
GRf.plot_triple_grid_ratio(
    my_grids_a0, my_grids_a1,
    custom_grid_unit=intensity_unit_choice,
    set_x_name=r"$\zeta_H$ [s$^{-1}$]", set_y_name=r"$G_0$ [Draine]",
    grid_type=grid_type_definition,
    plot_contours=True, flux_scale="log", color_map="magma", contour_color="white",
    x_shift_scan_direction="rightward", x_shift_match_rtol=0.02,
)
```

Three panels **per species**: constant-CR grid, attenuated grid, and the CRIR shift in
dex. The first argument must be the constant (α = 0) reference — the sign of the third
panel depends on it.

This is the *per-species, spatially resolved* view: it shows **where** on the plane a
given species shifts. Everything after this reduces that spatial detail down to rankings.

---

## 5. Stage 4 — *which* species move, and *where* (notebook §5)

```python
shift_overview = GRf.plot_attenuation_shift_summary(
    my_grids_a0, my_grids_a1,
    x_shift_scan_direction="rightward", x_shift_match_rtol=0.02,
    shift_abs_edges=(0.1, 0.3, 0.5, 1.0), obs_limit=GRf.OBS_INTENSITY_LIMIT,
    n_x_bins=4, n_y_bins=4, n_top_models=8,
    n_top_summary_species=12, summary_only=True,
)
```

Every statistic is taken over **every cell with a model**, the same for every species:
counts, medians, heatmaps and the top-model lists all use that set, with unmatched and
unobservable cells entering as Δ = 0 (§3.5). Species are ranked by `median_abs_shift_dex`
— one number, computed identically for all of them. No species is dropped.

### Figures

| # | content |
|---|---|
| 1 | left: stacked bar of models per \|Δ\| dex bin (lowest bin omitted; unmatched and unobservable cells sit in it at 0); every species has the same model count, marked by the dotted "all models" line, so the gap above a bar is the models that did **not** move; right: median \|Δ\| per species over all cells |
| 2 | heatmaps of median \|Δ\| over log-spaced x and y bins — every cell contributes, unmatched and unobservable ones as 0. This is the number to compare with the triple plot's shift panel, read with blank (no-match) and hatched (unobservable) pixels as 0 |
| 3 | **winner map** — in each `(x, y)` regime, the species with the largest median \|Δ\|. Read as *"under these initial conditions, follow up this species."* Return here at the end |
| 4–5 | figures 1 and 2 repeated for the top `n_top_summary_species` only |

### Returned dict

`pixels`, `species_summary`, `top_models`, `top_overall`, `regime_median`,
`regime_winners`.

`species_summary` carries `n_cells`, `n_valid` (cells with a model), `frac_observable`
(reported, not a cut) and the mean / median / std / max of \|Δ\|, sorted by
`median_abs_shift_dex`. Every species appears in every figure and table.

---

## 6. Stage 5 — magnitude and attribution (notebook §6)

```python
divergence = GRf.plot_attenuation_fuv_response(shift_overview["pixels"])
```

Reads the **same pixels table** — nothing is recomputed, so the two figures are directly
comparable and joinable. Two panels, ranked **independently**.

### Left panel — how wrong is ζ?

Per species, every cell with a model contributes its `|Δ|`, unmatched and unobservable
cells as 0 (§3.5). No slope gate is applied, so `error_dex` is the median over the same
cells as the summary's `median_abs_shift_dex` — the two are the same number.

- bar = median of that array
- band = p10–p90 across the grid — a *spread over cells*, not an uncertainty on
  the median (drawn capless for that reason: a capped error bar reads as "this
  median is poorly known", and it is not)
- bar colour = `frac_observable`, the fraction of cells above the detection limit (the
  rest entered at 0)

### Right panel — is it cosmic rays, or is it FUV?

Both axes are reductions of the R field (unobservable cells at R = 0), and the reduction
is **two-stage, along ζ first**. Each grid row is one CR experiment at fixed `G_0`, so each row collapses to its
median R. That gives one robust number per FUV value, and the FUV question becomes
whether that number stays flat as `G_0` moves.

| axis | quantity | meaning |
|---|---|---|
| x | `abs_response_dex` | \|median over rows of the per-row median R\| — how large the attenuation effect typically is |
| y | `fuv_swing_dex` | spread of the per-row medians, max − min over every row (nothing trimmed: the grid is a complete enumeration). In 3D the verdict uses its G₀ part `g0_swing_dex`; in 2D the two are equal — how much that effect depends on where in the radiation field you sit |

Marker size = the inference error `|Δ|`; colour = species identity (legend).

Two dividers, neither arbitrary:

- **Diagonal `y = x`** is `swing = |R|`. *Below it* the attenuation response exceeds its
  own spread across `G_0`: a **CR-dominated response**. *Above it* `G_0` moves the response
  more than attenuation does: an **G0-modulated response**. This describes the response
  to attenuation — it does not make the species an FUV tracer.
- **Vertical line at `response_floor_dex`** asks the prior question: is there any response
  at all? Left of it the species barely moves, so which side of the diagonal it falls on
  is noise.

`|R|` and the swing are kept as **two separate numbers** — no combined score. A single
`|R| − swing` value (the former `probe_margin`) hid which of the two drove it. The
returned table carries `cr_dominated = fuv_swing_dex < abs_response_dex` for the side of
the diagonal.

A third figure plots the per-row median R against `G_0` for the same species and colours:
the profile the swing summarises (swing = its extent, `fuv_trend` = its slope).

### Why the two panels rank independently

Since `|Δ| ≈ R / S`, the ideal CRIR probe — a large response on a **steep** row — has a
**small** inference error. Ranking the right panel by `error_dex` would sort exactly the
best probes to the bottom and truncate them away before they could be drawn.

So: left panel takes the largest `error_dex`, the probe and profile panels the largest `|R|`,
from the same pool of every species. The two sets overlap but are not the same; the
returned table flags each with `ranked` and `probe_ranked` (display selection only).

Both axes are logarithmic and zoomed to the data, with equal limits so the diagonal stays
at 45°. A zero swing is pinned to the bottom edge.

---

## 7. Stage 6 — reconciliation, one verdict per species (notebook §7)

```python
views = GRf.reconcile_attenuation_species(
    shift_overview, divergence,
    response_floor_dex=0.1, min_error_dex=0.3,
)
```

The two functions read the same table with the same cell rule, so their medians agree;
the response function adds whether the response is CR- or G0-modulated.

### The three-question cascade (applied in order)

| step | flag | test | question |
|---|---|---|---|
| 1 | — | `abs_response_dex >= response_floor_dex` | does attenuation change the line at all? |
| 2 | `cr_led` | `fuv_swing_dex < abs_response_dex` | is the response CR- or G0-modulated? |
| 3 | `matters` | `error_dex >= min_error_dex` | is the ζ error worth caring about? (0.3 dex = ×2) |

These are **labels on numbers computed identically for every species, not cuts on the
sample**: every species stays in every table. The **first failing test names the
verdict**, so a grade can always be traced back to the step that decided it. Verdicts, in
cascade order:

`no response` → `G0-modulated` → `robust attenuation probe` → `robust, small bias`, plus `no data`
for a species missing from one of the two tables.

The flags come back alongside the verdict, so a reader who disagrees with a threshold can
see which species would flip.

### Three views of the same rows

| key | question | sorted by | restricted to |
|---|---|---|---|
| `table` | the full graded record | verdict tier, then `error_dex` | everything |
| `problematic` | **which lines mislead me most?** | `error_dex` ↓ | everything with an `error_dex` |
| `tracers` | **which lines are the best CR probes?** | `abs_response_dex` ↓ | CR-led verdicts |

They are allowed to disagree, and the disagreement is the point:

- `problematic` is a **correction priority** list. An `G0-modulated` species can lead it — the ζ error
  is real, its cause simply is not cosmic rays, and the `verdict` column says so.
- `tracers` is a **line selection** list. It deliberately drops the `matters` cut: a clean
  probe whose ζ error falls just under `min_error_dex` is still a clean probe, it simply
  biases you less. This is why a species can rank higher here than in `table`, where the
  verdict tier comes first.

### 7.1 Per-ζ regime verdicts

```python
zeta_views = GRf.attenuation_zeta_regime_verdicts(
    shift_overview["pixels"], n_bins=3,
    response_floor_dex=0.1, min_error_dex=0.3,
)
```

Cuts the CRIR axis into `n_bins` equal-log-width regimes (`low` / `mid` / `high`) and runs
the unchanged stages 4–6 on each regime's cells; the FUV axis is not binned, so the swing
still spans the full `G_0` range. This is what corrects the 2D bias of §3.4: a line inert
at low ζ and responsive at high ζ has a whole-grid median near zero, but grades on its
own merits in the `high` regime. Shifts were matched on the full row before the cut.
Returns `whole_grid`, `regimes` (one graded row per species and regime) and
`best_regimes`. The 3D workflow (`grid_3d.attenuation_regime_verdicts_3d`) does the same
per n·G₀·ζ box.

---

## 8. Stage 7 — where to go next

```python
probes = views["table"].query("verdict == 'robust attenuation probe'")["species"]
shift_overview["regime_winners"].query("species in @probes")
```

Take the `robust attenuation probe` rows back to `regime_winners` from stage 4 to see **where** in
`(ζ_H, G_0)` to follow each one up, and to `top_models` for the individual models
behind the largest shifts.

A species you would act on is:

> responding (step 1) · **below the diagonal** (step 2) · carrying a large
> **`error_dex`** (step 3).

But if you are choosing a line to **observe** rather than a bias to **correct**, read the
`tracers` table, not the `problematic` one.

---

## 9. Parameters that must agree across the workflow

| parameter | default | role |
|---|---|---|
| `exclude_ice` | True | drop J-prefixed grain-surface species and their ratios |
| `exclude_isotopologues` | False | drop 13C/18O/17O/15N/34S/33S isotopologues and their ratios — **pass the same value to both plot functions**; run once with each value to separate the two investigations (headers, figures and files are tagged, files end in `_no_iso` / `_with_iso`) |
| `x_shift_scan_direction` | `'rightward'` | only this gives Δ its ζ-error meaning |
| `x_shift_match_rtol` | 0.02 | intensity match band for the shift search |
| `obs_limit` | `OBS_INTENSITY_LIMIT` (intensity) / None (abundance) | detection limit; cells below it enter at Δ = 0, R = 0 |
| `response_floor_dex` | 0.1 | the vertical divider — pass the same value to `reconcile_attenuation_species` |

Both inputs to `reconcile_attenuation_species` must come from the **same** `pixels`
table, and both functions must have run with the same `exclude_ice` and
`exclude_isotopologues`,
or the columns are not commensurable. The defaults already match.

---

## 10. Pitfalls

1. **Two trees differing in anything but attenuation** — undetectable, and invalidates
   every figure.
2. **Reading the winner map as a tracer ranking.** It ranks by \|Δ\| alone; a G0-modulated species
   with a large ζ error wins there and is still the wrong line to observe. That is exactly
   what the reconciliation exists to catch.
3. **Trusting a large \|Δ\| where S is small.** No slope gate is applied, so a cell on a
   flat stretch of the reference row can carry a large \|Δ\| that means loss of CR
   sensitivity rather than a stronger CR effect. Check `slope_dex` in `top_models` before
   following up a single cell.
4. **Reading a whole-grid 2D median as the species' best case.** Unmatched and
   unobservable cells count as 0 and inert cells are not filtered, so a species sensitive
   (or detectable) in one CRIR band only is under-ranked. Read its grade in the per-ζ
   regimes (§7.1) or the 3D boxes.
5. **Comparing the shift panel with the medians by eye** without reading blank (no-match)
   and hatched (unobservable) pixels as 0 — the medians do exactly that.
6. **Reading a median as the true error.** It is conservative: unmatched cells may hide
   shifts larger than the grid can represent.
7. **A species missing from one of the two tables** returns `no data`, not a low grade.

---

## 11. Column guide — how to read every returned table

One line per column. A name that appears in two tables means the same thing in both.

### `pixels` — one row per grid cell (`collect_attenuation_x_shift_table`)

| column | meaning | how to read |
|---|---|---|
| `species` | Line key, or ratio key 'Num/Den'. | A ratio restates its numerator when den_contrib is near 0. |
| `label` | Short display name of the species. | Cosmetic; index by species. |
| `iy` | Row index of the cell on the interpolated mesh. | One row = one G0 value, i.e. one CR experiment at fixed FUV. |
| `ix` | Column index of the cell on the interpolated mesh. | Increases with x (ζ). |
| `x` | x-axis value of the cell (ζ_H). | The ζ_nom a match starts from. |
| `y` | y-axis value of the cell (G0). | — |
| `x_name` | Grid key the x values come from. | — |
| `y_name` | Grid key the y values come from. | — |
| `value_ref` | Grid value in the constant-CR model, I_const. | The grid an observer would fit a source with. |
| `value_atten` | Grid value in the attenuated model, I_atten. | Plays the role of the real (attenuated) source. |
| `shift_dex` | Signed shift Δ = log10(ζ_match / ζ_nom): how far along ζ the constant-CR grid must move to reproduce I_atten. 0 = both grids have a model but no match (read as no shift); NaN = no model in that cell. Unobservable cells keep their raw value here; the ranking functions read them as 0. | Rightward scan: positive Δ = ζ comes out too low if attenuation is ignored. The triple plot draws the raw match (no-match pixels blank); every table reads them as 0. |
| `abs_shift_dex` | \|shift_dex\|. | The magnitude every median and bin uses. |
| `slope_dex` | Local CR sensitivity S = dlog10(I_const) / dlog10(ζ). | \|Δ\| ~ R / S, so where \|S\| is small a large shift means the line stopped responding to ζ, not a strong CR effect. Diagnostic only: no statistic filters on it. |
| `observable` | True where every line behind the key is >= obs_limit in both grids. | Ratios are judged on their two lines, never on the ratio value, so a bright/faint ratio is unobservable wherever the faint line is. All True when obs_limit is None. False cells enter every statistic at 0. |

### `species_summary` and regime tables (`plot_attenuation_shift_summary`)

| column | meaning | how to read |
|---|---|---|
| `n_cells` | Cells of the species (every grid cell). | Cells are interpolated mesh points (target_shape), not independent models, so neighbouring cells are not independent evidence. |
| `n_valid` | Cells with a model (finite shift; unmatched and unobservable as 0). | Equals n_cells unless part of the grid has no model. |
| `mean_abs_shift_dex` | Mean \|Δ\| over all cells (unmatched and unobservable as 0). | Pulled by outliers; prefer the median. |
| `median_abs_shift_dex` | Median \|Δ\| over all cells (unmatched and unobservable as 0). The sort key. | No slope gate: small and zero shifts pull it down. Equals error_dex. |
| `std_abs_shift_dex` | Standard deviation of \|Δ\| over all cells. | Small = the shift is nearly uniform over the grid. |
| `max_abs_shift_dex` | Largest \|Δ\|. | Worst single cell; top_models says where it sits (check its slope_dex). |
| shift-bin columns | Cells whose \|Δ\| falls in this bin (one column per \|Δ\| range, e.g. \|Δ\| < 0.1). | The lowest bin is left out of the bar figure: it asks how many moved. |
| `frac_observable` | Fraction of the grid above obs_limit. | Reported, not a cut: the other cells entered at 0. 1.0 for every species usually means obs_limit was None. |
| `x_bin` | Log-spaced x (ζ) bin of the regime tables. | — |
| `y_bin` | Log-spaced y (G0) bin of the regime tables. | — |
| `n_models` | Cells in that (x_bin, y_bin) regime. | A handful of cells = a noisy regime median; do not follow it up on its own. |

### Response table (`plot_attenuation_fuv_response`)

| column | meaning | how to read |
|---|---|---|
| `error_dex` | Median \|Δ\| over all cells (unmatched and unobservable as 0). The dex error in the inferred ζ if attenuation is ignored (0.3 dex = factor 2). | A median rewards breadth: a moderate shift everywhere outranks a large one in a corner (§3.4). Conservative where matches fall off the grid. |
| `error_p10` | 10th percentile of the cells behind error_dex. | Spread over the grid, not an uncertainty on the median. |
| `error_p90` | 90th percentile of the cells behind error_dex. | The worst-decile error. A wide p10-p90 band = the error depends on where in the grid the source sits. |
| `n_error_cells` | Cells behind error_dex. | — |
| `response_dex` | Signed median of R = log10(I_atten / I_const) over the per-G0-row medians; unobservable cells enter at R = 0. | Needs no match. Negative = attenuation dims the line. |
| `abs_response_dex` | \|response_dex\|. | Below response_floor_dex (0.1 dex ~ 26%) the line barely responds: verdict 'no response'. |
| `fuv_swing_dex` | Spread of the per-G0-row median R: max - min over every row. In 3D split into `g0_swing_dex` (across G0 at fixed n; the verdict's swing) and `n_swing_dex` (across n at fixed G0; reported). | How much the attenuation response changes with the FUV field. Small = the CR effect does not care about G0. |
| `fuv_trend` | Slope of R against log10(G0), in dex of R per dex of G0. | 0 = FUV-independent; the sign gives the direction. |
| `n_fuv_rows` | G0 rows with a finite R. | swing needs >= 2 rows, trend >= 3. |
| `cr_dominated` | fuv_swing_dex < abs_response_dex. | True: the response exceeds its own G0 spread (below the probe-panel diagonal). False: G0-modulated response. |
| `den_contrib` | \|R_den\| / (\|R_num\| + \|R_den\|) for a ratio whose two lines both have rows. | Near 0 = the denominator is inert and the ratio restates its numerator. NaN for single lines or a missing constituent. |
| `frac_observable` | Fraction of the grid above obs_limit. | Reported, not a cut; colours the error bars. |
| `ranked` | Drawn in the left (error) panel. | Display only: max_species, max_per_constituent, min_abs_response_dex. `max_species=None` (the PRISMA tab) draws every species, split over figures. |
| `probe_ranked` | Drawn in the right (probe) panel. | Display only, as ranked. |

### Reconciliation (`reconcile_attenuation_species`)

| column | meaning | how to read |
|---|---|---|
| `cr_led` | fuv_swing_dex < abs_response_dex. | False -> verdict 'G0-modulated'. |
| `matters` | error_dex >= min_error_dex. | Is the ζ error worth correcting? Not applied to the tracers view. |
| `verdict` | First step the species fails, in cascade order. | no data: missing from one table \| no response: \|R\| < response_floor_dex \| G0-modulated: swing >= \|R\| \| robust attenuation probe: CR-led and the error matters \| robust, small bias: CR-led, error below min_error_dex. |
| `problematic` view | View: every species with an error_dex, sorted by it. | Which lines mislead you most if attenuation is ignored. A species can lead it and still be G0-modulated: the error is real, its cause is not cosmic rays. |
| `tracers` view | View: CR-led verdicts sorted by abs_response_dex. | The best CRIR probes. The matters cut is not applied: a clean probe with a small ζ error is still a clean probe, it simply biases you less. |

---
