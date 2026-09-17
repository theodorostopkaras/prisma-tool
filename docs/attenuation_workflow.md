# Grid attenuation comparison — full workflow

Companion to `Scripts/GRID/grid_attenuation_compare.ipynb`.
Every function named here lives in `src/kosens/grid/grid_functions.py`.

The rendered method PDF (with figures on a synthetic grid) is
`Scripts/function_documentation/kosens_grid_attenuation_workflow.pdf`, built by
`python Scripts/GRID/make_attenuation_method_doc.py`.

---

## 0. The question

Two KOSMA-τ model trees over the same `(ζ_H, G_0)` plane, identical in every respect
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
| `shift_dex`, `abs_shift_dex` | Δ, the horizontal match shift (NaN where no match) |
| `slope_dex` | S, the local CR sensitivity of the **reference** grid |

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

Hence every median over shifts is gated on `|S| >= min_slope_dex` (default 0.1, i.e.
intensity moving by less than ~26% per decade of CRIR is discarded).

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

The cases cannot be separated from the shift alone. That is why stage 5 gates on *how
many* of a species' cells are affected rather than trusting any single cell.

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

## 5. Stage 4 — Step 1, breadth: *which* species move, and *where* (notebook §5)

```python
shift_overview = GRf.plot_attenuation_shift_summary(
    my_grids_a0, my_grids_a1,
    x_shift_scan_direction="rightward", x_shift_match_rtol=0.02,
    min_abs_shift_dex=0.1, shift_abs_edges=(0.1, 0.3, 0.5, 1.0),
    max_frac_unmatched=0.5,
    n_x_bins=4, n_y_bins=4, n_top_models=8,
    n_top_summary_species=12, summary_only=True,
)
```

Ranks species by `n_affected = n_shifted + n_unmatched` — **an unmatched cell counts as
evidence that the species moved**. That is defensible for this question: *"did it move"*
is well posed everywhere, even where the *size* of the move is not. Counts deliberately
ignore the `min_slope_dex` gate for the same reason; medians and the top-model lists
apply it.

> **Consequence that drives the whole reconciliation:** this ranking is about **breadth,
> not magnitude**, and it *rewards* unmatched cells. The next stage treats those same
> cells as evidence of ignorance.

### The ceiling on that reward: `max_frac_unmatched`

Rewarding unmatched cells is defensible while they are a **minority**. A species whose
grid is mostly unmatched has moved almost everywhere and been measured almost nowhere:
its median rests on the matched leftovers, and — for the same reason as §3.5 — a no-match
cell can mean the two curves differ by something *no horizontal x-shift reproduces* (a
vertical offset, a displacement along `y`) rather than a large CR shift. Topping this
ranking on such cells is therefore not evidence that cosmic rays moved the species.

Species with `frac_unmatched > max_frac_unmatched` (default 0.5) are excluded from the
rankings, all five figures and the winner map, and returned with `ranked = False`. This
is the deliberate twin of `max_frac_censored` in stage 5: **the two functions now rank
the same pool of species**, so a name at the top of one is a name the other has also
seen fit to measure.

Set `max_frac_unmatched=1.0` to restore the ungated behaviour.

### Figures

| # | content |
|---|---|
| 1 | left: stacked bar of models per \|Δ\| dex bin (lowest bin omitted, grey = unmatched); every species has the same model count, marked by the dotted "all models" line, so the gap above a bar is the models that did **not** move; right: median \|Δ\| per species among matched cells |
| 2 | heatmaps of median \|Δ\| over log-spaced x and y bins — only slope-gated cells contribute, so a bin can be blank where the reference grid is flat in x |
| 3 | **winner map** — in each `(x, y)` regime, the species with the largest median \|Δ\|. Read as *"under these initial conditions, follow up this species."* Return here at the end |
| 4–5 | figures 1 and 2 repeated for the top `n_top_summary_species` only |

### Returned dict

`pixels`, `species_summary`, `top_models`, `top_overall`, `regime_median`,
`regime_winners`.

`species_summary` carries `n_valid` (matched cells), `n_unmatched`, `n_trusted`,
`frac_shifted`, `frac_unmatched`, `median_abs_shift_dex`, and `ranked` (False for species
held back by `max_frac_unmatched`). Gated species stay in this table on purpose —
`reconcile_attenuation_species` needs the row to grade them `unmeasurable` rather than
`no data` — but appear in no figure, in `top_models`, or in `regime_median` /
`regime_winners`.

---

## 6. Stage 5 — Steps 2–4: magnitude, honesty, attribution (notebook §6)

```python
divergence = GRf.plot_attenuation_fuv_response(shift_overview["pixels"])
```

Reads the **same pixels table** — nothing is recomputed, so the two figures are directly
comparable and joinable. Two panels, ranked **independently**.

### Left panel — Step 4: how wrong is ζ?

Per species, each cell contributes its `|Δ|` if it passes the slope gate and a match was
found. A trusted cell with **no** match is not dropped: it enters at its lower bound
`log10(x_row_max / x_nom)`, because dropping it would delete precisely the
worst-determined models from a ranking of worst-determined models.

- bar = median of that array
- band = p10–p90 across the grid — a *spread over cells*, not an uncertainty on
  the median (drawn capless for that reason: a capped error bar reads as "this
  median is poorly known", and it is not)
- bar colour = fraction of contributing cells that are lower limits
- `>` in the printout marks a median that is itself a bound; on the figure
  that is the ▶ at the band's end (no numbers are drawn — the bar and band
  ends already carry them)
- ▶ at the band's right end marks the same for p90: censored cells enter below
  their true value, so every percentile of that species is a lower limit

### Step 2 — is the number measured or imputed?

Because a no-match cell is ambiguous (§3.5), imputing one is only defensible while such
cells are a **minority**. Past `frac_censored = 0.5` the median itself falls on an imputed
bound rather than a measured shift, and the species is no longer being measured at all.

Species above `max_frac_censored` are excluded from the ranking and both panels, returned
with `ranked = False`.

> Without this gate the ranking **inverts**: the less measurable a species is, the more
> censored cells it has, the higher its imputed bound, and the higher it ranks.

### Right panel — Step 3: is it cosmic rays, or is it FUV?

Both axes are reductions of the R field, and the reduction is **two-stage, along ζ
first**. Each grid row is one CR experiment at fixed `G_0`, so each row collapses to its
median R. That gives one robust number per FUV value, and the FUV question becomes
whether that number stays flat as `G_0` moves.

| axis | quantity | meaning |
|---|---|---|
| x | `abs_response_dex` | \|median over rows of the per-row median R\| — how large the attenuation effect typically is |
| y | `fuv_swing_dex` | p90 − p10 of the per-row medians — how much that effect depends on where in the radiation field you sit |

Marker size = the inference error `|Δ|`; colour = species identity (legend).

Two dividers, neither arbitrary:

- **Diagonal `y = x`** is `probe_margin = 0`, the very quantity the panel is ranked by.
  *Below it* the attenuation response exceeds the response's own spread across `G_0`, so
  the line is **reporting cosmic rays**. *Above it* the species moves more with the
  radiation field than attenuation ever moves it — an **FUV tracer**, not a CR one.
- **Vertical line at `response_floor_dex`** asks the prior question: is there any response
  at all? Left of it the species barely moves, so which side of the diagonal it falls on
  is noise.

```
probe_margin = |R| − fuv_swing_dex
```

Deliberately a **difference, not a ratio**: both terms are already in dex, so no epsilon
is needed and nothing blows up as the swing approaches zero. It reads as *the response
that survives the worst of the species' own FUV variability*.

### Why the two panels rank independently

Since `|Δ| ≈ R / S`, the ideal CRIR probe — a large response on a **steep** row — has a
**small** inference error. Ranking the right panel by `error_dex` would sort exactly the
best probes to the bottom and truncate them away before they could be drawn.

So: left panel takes the largest `error_dex`, right panel the largest `probe_margin`,
from the same gated pool. The two sets overlap but are not the same; the returned table
flags each with `ranked` and `probe_ranked`.

Axes share fixed limits (`probe_axis_max`, raised automatically if any species would fall
outside, so nothing is ever clipped) so two runs of this figure can be compared by eye.

---

## 7. Stage 6 — Step 5: reconciliation, one verdict per species (notebook §7)

```python
views = GRf.reconcile_attenuation_species(
    shift_overview, divergence,
    min_frac_shifted=0.5, max_frac_unmatched=0.5, max_frac_censored=0.5,
    response_floor_dex=0.1, min_error_dex=0.3,
)
```

The two functions read the same table and rank differently, so the difference between
them is **informative rather than a conflict to average away**. The summary counts an
unmatched cell as movement; the response function counts it as ignorance. A species can
top the summary on unmatched cells and be dropped outright by the response function —
that combination is itself the finding: *wide effect, unmeasurable magnitude*.

### The four-question cascade (applied in order)

| step | flag | test | question |
|---|---|---|---|
| 1 | `broad` | `frac_shifted >= min_frac_shifted` | does it touch a useful part of the grid? |
| 2 | `honest` | `frac_unmatched <= max_frac_unmatched` **and** `frac_censored <= max_frac_censored` | is the magnitude measured, or imputed? |
| 3 | `cr_led` | `probe_margin > 0` | is the response CR, or `G_0`? |
| 4 | `matters` | `error_dex >= min_error_dex` | is the ζ error worth caring about? (0.3 dex = ×2) |

Step 2 is now also applied *upstream*, inside each plot function — `max_frac_unmatched`
in stage 4, `max_frac_censored` in stage 5 — so a species failing it is already absent
from both sets of figures by the time it gets here. It still reaches the cascade, because
both functions keep it in their returned table, and `unmeasurable` is a finding worth
naming rather than a row worth deleting.

The **first failing test names the verdict**, so a grade can always be traced back to the
step that decided it. Verdicts, in cascade order:

`no response` → `unmeasurable` → `FUV tracer` → `CR-led, narrow` → `CRIR probe` →
`CR-led, small bias`, plus `no data` for a species missing from one of the two tables.

All four flags come back alongside the verdict, so a reader who disagrees with a
threshold can see which species would flip.

### Two columns worth reading directly

- **`bound_gap` = `error_dex` − `median_abs_shift_dex`.** The summary's median uses
  matched cells only; the response function's uses matched **plus** imputed bounds. A
  large gap means that species' ranking leans on imputation rather than measurement.
- **`frac_trusted_shift` vs `frac_trusted_fuv`.** Same name, **different denominators**
  (trusted among *matched* cells, versus among cells where R is defined, which includes
  unmatched ones). They will not agree numerically and neither is wrong.
  **Never average them.**

### Three views of the same rows

| key | question | sorted by | restricted to |
|---|---|---|---|
| `table` | the full graded record | verdict tier, then `error_dex` | everything |
| `problematic` | **which lines mislead me most?** | `error_dex` ↓ | `honest` species only |
| `tracers` | **which lines are the best CR probes?** | `probe_margin` ↓ | CR-led verdicts |

They are allowed to disagree, and the disagreement is the point:

- `problematic` is a **correction priority** list. An FUV tracer can lead it — the ζ error
  is real, its cause simply is not cosmic rays, and the `verdict` column says so.
- `tracers` is a **line selection** list. It deliberately drops the `matters` cut: a clean
  probe whose ζ error falls just under `min_error_dex` is still a clean probe, it simply
  biases you less. This is why a species can rank higher here than in `table`, where the
  verdict tier comes first.

---

## 8. Stage 7 — where to go next

```python
probes = views["table"].query("verdict == 'CRIR probe'")["species"]
shift_overview["regime_winners"].query("species in @probes")
```

Take the `CRIR probe` rows back to `regime_winners` from stage 4 to see **where** in
`(ζ_H, G_0)` to follow each one up, and to `top_models` for the individual KOSMA-τ models
behind the largest shifts.

A species you would act on is:

> **broad** in step 1 · **honest** in step 2 · **below the diagonal** in step 3 ·
> carrying a large **`error_dex`** in step 4.

But if you are choosing a line to **observe** rather than a bias to **correct**, read the
`tracers` table, not the `problematic` one.

---

## 9. Parameters that must agree across the workflow

| parameter | default | role |
|---|---|---|
| `min_slope_dex` | 0.1 | CR-sensitivity gate — **set identically in both plot functions** |
| `exclude_ice` | True | drop J-prefixed grain-surface species and their ratios |
| `exclude_isotopologues` | False | drop 13C/18O/17O/15N/34S/33S isotopologues and their ratios — **pass the same value to both plot functions**; run once with each value to separate the two investigations (headers, figures and files are tagged, files end in `_no_iso` / `_with_iso`) |
| `x_shift_scan_direction` | `'rightward'` | only this gives Δ its ζ-error meaning |
| `x_shift_match_rtol` | 0.02 | intensity match band for the shift search |
| `max_frac_unmatched` | 0.5 | step-2 gate inside `plot_attenuation_shift_summary` — **pass the same value to `reconcile_attenuation_species`** |
| `max_frac_censored` | 0.5 | step-2 gate inside `plot_attenuation_fuv_response` |
| `response_floor_dex` | 0.1 | the vertical divider — pass the same value to `reconcile_attenuation_species` |

Both inputs to `reconcile_attenuation_species` must come from the **same** `pixels`
table, and both functions must have run with the same `min_slope_dex`, `exclude_ice` and
`exclude_isotopologues`,
or the columns are not commensurable. The defaults already match.

---

## 10. Pitfalls

1. **Two trees differing in anything but attenuation** — undetectable, and invalidates
   every figure.
2. **Reading the winner map as a tracer ranking.** It ranks by \|Δ\| alone; an FUV tracer
   with a large ζ error wins there and is still the wrong line to observe. That is exactly
   what steps 2–4 exist to catch.
3. **Trusting a large \|Δ\| at high `G_0`.** Since S varies with `G_0`, that can mean loss
   of CR sensitivity rather than a stronger CR effect — hence the slope gate.
4. **Averaging `frac_trusted_shift` with `frac_trusted_fuv`.** Different denominators.
5. **Dropping no-match cells** from the magnitude ranking. They are censored observations,
   not missing data — imputed as lower bounds, and gated at 50%.
   The mirror-image mistake is **keeping a species whose grid is mostly no-match**: past
   50% neither ranking is measuring it any more, which is what `max_frac_unmatched` and
   `max_frac_censored` cut off at either end.
6. **A species missing from one of the two tables** returns `no data`, not a low grade.

---

## 11. Column guide — how to read every returned table

One line per column. A name that appears in two tables means the same thing in both,
except `frac_trusted`, `ranked` and `observable`, whose rows state both meanings.

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
| `shift_dex` | Signed shift Δ = log10(ζ_match / ζ_nom): how far along ζ the constant-CR grid must move to reproduce I_atten. NaN = no match. | Rightward scan: positive Δ = ζ comes out too low if attenuation is ignored. NaN cells are never filled in any map; only error_dex imputes a bound for some of them (see frac_censored). |
| `abs_shift_dex` | \|shift_dex\|. | The magnitude every median and bin uses. |
| `slope_dex` | Local CR sensitivity S = dlog10(I_const) / dlog10(ζ). | \|Δ\| ~ R / S, so where \|S\| < min_slope_dex the shift blows up because the line stopped responding to ζ, not because the CR effect is strong. Such cells are not trusted: medians, heatmaps and top-model lists skip them. |
| `observable` | pixels: True where every line behind the key is >= obs_limit in both grids. reconcile table: frac_observable >= min_frac_observable. | Ratios are judged on their two lines, never on the ratio value, so a bright/faint ratio is unobservable wherever the faint line is. All True when obs_limit is None. |

### `species_summary` and regime tables (`plot_attenuation_shift_summary`)

| column | meaning | how to read |
|---|---|---|
| `n_cells` | Observable cells of the species. | Cells are interpolated mesh points (target_shape), not independent KOSMA-τ models, so neighbouring cells are not independent evidence. |
| `n_valid` | Cells with a match (finite shift). | — |
| `n_unmatched` | Cells with no match (n_cells - n_valid), for any reason, no slope gate. | Includes flat-row and row-end cells, which frac_censored leaves out. |
| `n_trusted` | Matched cells passing the slope gate \|S\| >= min_slope_dex. | These cells back every median of the summary. |
| `n_shifted` | Matched cells with \|Δ\| >= min_abs_shift_dex (no slope gate). | 'Did it move' is well posed even on flat rows, so this count is ungated. |
| `n_affected` | n_shifted + n_unmatched. | How many models moved or could not be matched; first sort key of species_summary. |
| `frac_shifted` | n_shifted / n_valid. | Breadth: how much of the matched map moved. Small = the effect sits in a corner (verdict 'CR-led, narrow'). |
| `frac_unmatched` | n_unmatched / n_cells. | How much of the shift map is empty. Not frac_censored: it counts every no-match cell over the whole observable map. Above max_frac_unmatched the species is not ranked. |
| `frac_trusted` | Share of cells passing the slope gate. species_summary: among matched cells. Response table: among cells where R is defined (unmatched included). | Same name, different denominators: the two will not agree and neither is wrong. Never average them. |
| `mean_abs_shift_dex` | Mean \|Δ\| over trusted matched cells. | Pulled by outliers; prefer the median. |
| `median_abs_shift_dex` | Median \|Δ\| over trusted matched cells only. | Pure measurement, nothing imputed. Compare with error_dex through bound_gap. |
| `std_abs_shift_dex` | Standard deviation of \|Δ\| over trusted matched cells. | Small = the shift is nearly uniform over the grid. |
| `max_abs_shift_dex` | Largest trusted \|Δ\|. | Worst single cell; top_models says where it sits. |
| `no match` | Unmatched cells (grey stack in the summary bar figure). | Equals n_unmatched. |
| shift-bin columns | Matched cells whose \|Δ\| falls in this bin (one column per \|Δ\| range, e.g. \|Δ\| < 0.1). | The lowest bin is left out of the bar figure: it asks how many moved. |
| `frac_observable` | Fraction of the grid above obs_limit. | Every other count and median is over these cells only. Below min_frac_observable the species is dropped. 1.0 for every species usually means obs_limit was None. |
| `ranked` | species_summary: passed max_frac_unmatched and min_frac_observable, so it is drawn. Response table: drawn in the left (error) panel. | False rows stay for reference. In the response table False also covers max_frac_censored, min_abs_response_dex, max_species and max_per_constituent. |
| `x_bin` | Log-spaced x (ζ) bin of the regime tables. | — |
| `y_bin` | Log-spaced y (G0) bin of the regime tables. | — |
| `n_models` | Trusted matched cells in that (x_bin, y_bin) regime. | A handful of cells = a noisy regime median; do not follow it up on its own. |

### Response table (`plot_attenuation_fuv_response`)

| column | meaning | how to read |
|---|---|---|
| `error_dex` | Median \|Δ\| over trusted cells: measured shifts plus the lower bound log10(ζ_row_max / ζ_nom) for censored cells. The dex error in the inferred ζ if attenuation is ignored (0.3 dex = factor 2). | A median rewards breadth: a moderate shift everywhere outranks a large one in a corner. A lower limit whenever frac_censored > 0; trust it only with a low frac_censored (see bound_gap). |
| `error_p10` | 10th percentile of the cells behind error_dex. | Spread over the grid, not an uncertainty on the median. |
| `error_p90` | 90th percentile of the cells behind error_dex. | The worst-decile error. A wide p10-p90 band = the error depends on where in the grid the source sits. |
| `frac_censored` | censored / (measured trusted + censored). Censored = trusted, no match, ζ_nom not at the row end; such cells enter error_dex at their lower bound. | How much of error_dex is imputed rather than measured. Not frac_unmatched: flat-row and row-end no-match cells are left out, and the denominator is only the error cells. Above 0.5 the median is itself an imputed bound, set by where the ζ grid ends; above max_frac_censored the species is not ranked. |
| `n_error_cells` | Cells behind error_dex (measured + censored). | — |
| `response_dex` | Signed median of R = log10(I_atten / I_const) over the per-G0-row medians. | Needs no match, so it is defined where the shift is NaN. Negative = attenuation dims the line. |
| `abs_response_dex` | \|response_dex\|. | Below response_floor_dex (0.1 dex ~ 26%) the line barely responds: verdict 'no response'. |
| `fuv_swing_dex` | p90 - p10 of the per-G0-row median R. | How much the attenuation response changes with the FUV field. Small = the CR effect does not care about G0. |
| `fuv_trend` | Slope of R against log10(G0), in dex of R per dex of G0. | 0 = FUV-independent; the sign gives the direction. |
| `n_fuv_rows` | G0 rows with a finite R. | swing needs >= 2 rows, trend >= 3. |
| `probe_margin` | abs_response_dex - fuv_swing_dex. | > 0: the response exceeds its own G0 spread (CR-led, below the probe-panel diagonal). < 0: FUV tracer. Ranks the right panel. |
| `den_contrib` | \|R_den\| / (\|R_num\| + \|R_den\|) for a ratio whose two lines both have rows. | Near 0 = the denominator is inert and the ratio restates its numerator. NaN for single lines or a missing constituent. |
| `probe_ranked` | Drawn in the right (probe) panel. | — |

### Reconciliation (`reconcile_attenuation_species`)

| column | meaning | how to read |
|---|---|---|
| `broad` | frac_shifted >= min_frac_shifted. | False -> verdict 'CR-led, narrow'. |
| `honest` | frac_unmatched <= max_frac_unmatched and frac_censored <= max_frac_censored. | Both are needed: the first guards the shift map, the second guards error_dex. False -> verdict 'unmeasurable'. |
| `cr_led` | probe_margin > 0. | False -> verdict 'FUV tracer'. |
| `matters` | error_dex >= min_error_dex. | Is the ζ error worth correcting? Not applied to the tracers view. |
| `verdict` | First step the species fails, in cascade order. | no data: missing from one table \| unobservable: frac_observable too low \| no response: \|R\| < response_floor_dex \| unmeasurable: not honest \| FUV tracer: probe_margin <= 0 \| CR-led, narrow: not broad \| CRIR probe: passes every step and the error matters \| CR-led, small bias: passes every step, error below min_error_dex. |
| `bound_gap` | error_dex - median_abs_shift_dex: how far the imputed bounds moved the median. | The effect of imputation, not how many cells were imputed (that is frac_censored). Below 50% censored the gap depends on the censored count and the spread of the measured shifts, not on the bound size: a near-uniform species keeps a small gap however many cells are censored. Negative when the bounds sit below the measured shifts (ζ_nom close to the row end). High frac_censored and a large gap: treat error_dex as a lower limit. |
| `problematic` view | View: honest species sorted by error_dex. | Which lines mislead you most if attenuation is ignored. A species can lead it and still be an FUV tracer: the error is real, its cause is not cosmic rays. |
| `tracers` view | View: CR-led verdicts sorted by probe_margin. | The best CRIR probes. The matters cut is not applied: a clean probe with a small ζ error is still a clean probe, it simply biases you less. |

---
