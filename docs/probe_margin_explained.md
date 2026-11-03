# `probe_margin` and the inference error Δ

> **Superseded.** `probe_margin` is no longer computed. `plot_attenuation_fuv_response`
> keeps `|R|` (`abs_response_dex`) and the swing (`fuv_swing_dex`) as separate numbers,
> plots the per-`G₀` median R as a profile, and labels the probe-panel regions
> *CR-dominated* / *G0-modulated response*; the verdict `FUV tracer` is now
> `G0-modulated`. See `attenuation_compare_workflow.md`. The Δ material below still holds.

> **Definition in one line.** `probe_margin` is the magnitude of a line's
> response to CR attenuation minus the spread (swing) of that response
> across the FUV axis: the part of the attenuation signal that is not
> degenerate with a change in `G₀`.

Figures: `python Scripts/GRID/make_probe_margin_explainer.py`.
The numerical values in Figures 1–5 are from the synthetic grid of the method
document; they are identical to what `plot_attenuation_fuv_response` prints for
that grid.

**Notation.** All logarithms are base 10. `x ≡ ζ_H` (CR ionization rate, grid
x-axis), `y ≡ G₀` (FUV field, grid y-axis). A grid cell is indexed by
`(i, j)` = (row in `G₀`, column in `ζ_H`). `I_const` and `I_atten` are the
line intensities of the constant-CR and attenuated-CR model grids at the same
nominal `(ζ_H, G₀)`. "dex" denotes a difference in log10.

---

## 1. The response field R

For every cell the response is defined as

```
R(i, j)  =  log10( I_atten(i, j) / I_const(i, j) )
```

i.e. the logarithmic change in line intensity caused by attenuation at fixed
nominal parameters. `R < 0` means attenuation lowers the intensity. The
quantity is evaluated cell-wise and involves no interpolation or matching;
cells in which either intensity is non-positive or non-finite, or which fall
below the observability limit, are masked (NaN).

![R over the grid](Figures/probe_margin/1_r_field.png)

The two synthetic species illustrate the two limiting cases:

- **CO 1-0** — `R ≈ −1.40` in every cell, so `∂R/∂log G₀ = 0`. The attenuation
  response does not depend on the FUV field.
- **[OI] 63µm** — `R > 0` at low `G₀` and `R < 0` at high `G₀`, so
  `∂R/∂log G₀ < 0`. The sign of the response itself depends on the FUV field.

The distinction matters for inference. For CO 1-0 an observed intensity change
relative to the constant-CR prediction can only be attributed to attenuation.
For [OI] 63µm the same change could equally be produced by a different `G₀`, so
the attenuation signal and the FUV field are degenerate. `probe_margin`
measures how strongly that degeneracy affects each species.

---

## 2. Reducing each row to a single value

A row of the grid is a sequence of models at fixed `G₀` and varying `ζ_H`, so
it is a one-parameter CR experiment. The ζ_H-dependence is removed by taking
the row median:

```python
r_row[i] = np.nanmedian(response[i, :])      # median over ζ_H at fixed G₀
```

For a grid with `N_y` rows this yields a vector `r_row ∈ ℝ^{N_y}` that depends
on `G₀` only (here `N_y = 6`). The median is chosen over the mean so that
individual cells (e.g. near a chemical transition) cannot dominate the row
value.

![row medians](Figures/probe_margin/2_row_medians.png)

- **CO 1-0**: `r_row = −1.40` for all rows (zero dispersion).
- **[OI] 63µm**: `r_row` decreases monotonically from `+0.60` to `−0.90`.

Both summary statistics below are functionals of this vector.

---

## 3. The two statistics

![number line](Figures/probe_margin/3_number_line.png)

### `abs_response_dex` — location

```
|R|  =  | median_i ( r_row[i] ) |
```

The median over rows of the row medians; this is the typical magnitude of the
attenuation response. (`response_dex` stores the signed value.)

### `fuv_swing_dex` — dispersion along G₀

```
swing  =  max(r_row) − min(r_row)
```

It requires at least two valid rows; otherwise it is NaN. It is the **full
range**: the grid is a complete, deterministic enumeration of environments, not
a noisy sample, so no row is trimmed — a claim that the response does not
depend on `G₀` must hold for every row (`grid_functions._swing`). An earlier
version used the inter-decile range P90 − P10; it was dropped because it
ignored real extreme environments. In 3D the swing of a box is split into a
`G₀` part (at fixed density, the one the verdict tests) and a density part
(`grid_functions._split_swing`).

For **[OI] 63µm**:

```
sorted r_row:   -0.90   -0.60   -0.30    0.00   +0.30   +0.60

|R|   = |median| = |(-0.30 + 0.00)/2| = 0.15
swing = +0.60 − (−0.90)              = 1.50
```

For **CO 1-0** all elements are equal, so the swing is 0:

```
|R|   = 1.40
swing = 0.00
```

---

## 4. The margin

```
probe_margin  =  |R| − swing
```

![the subtraction](Figures/probe_margin/4_subtraction.png)

| species | \|R\| | swing | margin | interpretation |
|---|---|---|---|---|
| CO 1-0 | 1.40 | 0.00 | **+1.40** | large response, independent of `G₀` → attenuation-dominated |
| C+ 158µm | 0.25 | 0.88 | **−0.63** | FUV-induced dispersion is 3.5× the response |
| [OI] 63µm | 0.15 | 1.20 | **−1.05** | response is negligible compared with its `G₀` dependence |

`probe_margin > 0` means the typical attenuation response is larger than the
change in that response across the central 80 % of the FUV range. So a
response of that size cannot come from a change in `G₀` alone.

**Why a difference and not a ratio.** Both terms are in dex, so their
difference is also in dex and can be read directly as a log-intensity. A ratio
`|R|/swing` diverges as `swing → 0` and is undefined at `swing = 0`, which is
exactly the ideal case, so it cannot rank the species of most interest.

---

# The inference error Δ

`probe_margin` addresses whether an intensity change can be attributed to
cosmic rays. The left panel of `plot_attenuation_fuv_response` addresses a
separate quantity: the bias in the inferred `ζ_H` when an attenuated source is
fitted with the constant-CR grid. This bias is `Δ`. It is obtained by matching
intensities along the `ζ_H` axis, not from `R`.

---

## 5. Definition of Δ

Consider one row (fixed `G₀`) and the two intensity profiles `I_const(ζ)` and
`I_atten(ζ)` along it.

```
     ^  log I
     |
     |                                                  ----- I_const
     |                                       -----------
     |                           ------------
     |               ------------                       ----- I_atten
     |    ----o------........................----o--  <- equal intensity
     |  --                       ------------
     +--------|----------------------------------|-----------------> log ζ_H
           ζ_nom                              x_match
              |<----------  Δ  (dex)  ---------->|
```

`R` is the vertical separation of the two profiles at fixed `ζ`. `Δ` is their
horizontal separation at fixed intensity.

For a cell at `ζ_nom`, `x_match` is the `ζ` on the same row at which the
attenuated model reproduces the constant-CR intensity:

```
I_atten(x_match)  =  I_const(ζ_nom)      (relative tolerance match_rtol = 0.02)

Δ  =  log10( x_match / ζ_nom )
```

(`horizontal_intensity_matched_x_shift_dex` in `grid_functions.py`; stored as
`shift_dex` / `abs_shift_dex` in the pixel table.)

---

## 6. Δ as the bias of a constant-CR fit

Fitting an observed (attenuated) intensity with the constant-CR grid means
solving

```
I_const(ζ_inferred)  =  I_atten(ζ_true)
```

for `ζ_inferred`. This is the matching equation of §5 with
`ζ_inferred = ζ_nom` and `ζ_true = x_match`. Hence

```
Δ  =  log10( ζ_true / ζ_inferred )
```

`Δ` is the logarithmic bias of the inferred CR ionization rate, with no further
assumptions. `|Δ| = 0.3, 0.5, 1.0` dex correspond to factors of 2.0, 3.2 and 10
in `ζ_H`.

**Rightward-only scan.** In the attenuated model the ionization rate decreases
with column density, so the column-averaged rate is lower than the nominal
(surface) value. The attenuated model at `ζ_true` is therefore closest to a
constant-CR model at some `ζ_eff < ζ_true`. A constant-CR fit recovers
`ζ_inferred ≈ ζ_eff`, so `Δ > 0` and `x_match ≥ ζ_nom`. This argument does not
depend on the sign of `∂I/∂ζ`; it follows from attenuation reducing the
effective rate. `scan_direction="rightward"` encodes this expectation. A match
only at `x < ζ_nom` would contradict it.

---

## 7. Relation between R, Δ and the CR sensitivity S

Define the local CR sensitivity of the constant-CR model

```
S  =  ∂ log10 I_const / ∂ log10 ζ          (column `slope_dex`, dex/dex)
```

Write `log I_atten(x) = log I_const(x) + R(x)` and expand
`log I_const(x_match)` to first order about `ζ_nom`:

```
log I_atten(x_match)  =  log I_const(x_match) + R(x_match)
                      ≈  log I_const(ζ_nom) + S·Δ + R(ζ_nom)
```

Setting this equal to `log I_const(ζ_nom)` (the matching condition) gives

```
S·Δ + R  ≈  0      ⇒      Δ  ≈  −R / S,      |Δ| ≈ |R| / |S|
```

The approximation assumes that `S` and `R` are approximately constant over
`[ζ_nom, x_match]`. That is, it is valid when `|Δ|` is small compared with the
scale over which `log I_const` departs from linearity in `log ζ`. For
large `|Δ|` the matched value from §5 is the exact quantity and the ratio is
only indicative.

| case | \|R\| | S (dex/dex) | \|Δ\| ≈ \|R\|/\|S\| |
|---|---|---|---|
| steep row | 1.40 | 0.70 | 2.0 dex |
| same R, shallow row | 1.40 | 0.20 | 7.0 dex |

Two consequences for the ranking:

1. **|Δ| is not a measure of the attenuation effect alone.** It scales as
   `1/|S|`, so it diverges as the line becomes insensitive to `ζ`, even though
   in that limit the line carries no CR information.
2. **A good CR probe has a small |Δ|.** A large `|R|` together with a large
   `|S|` gives a small bias. Sorting by `|Δ|` would place exactly these species
   at the bottom, which is why the right panel is ranked by `probe_margin`
   instead.

---

## 8. Cell-level failure modes

### 8.1 Low sensitivity: no slope gate

Because `|Δ| ∝ 1/|S|`, cells with small `|S|` can produce large,
ill-conditioned values of Δ. **No gate on `|S|` is applied**: every cell with
a model is retained, flat-row cells and small shifts included (cells below the
detection limit enter at 0, §8.2). `slope_dex` is kept in the table as a
diagnostic only. Δ and R are taken over the same cells as in
`plot_attenuation_shift_summary`, so both functions operate on an identical
cell set.

The reason is that a gate on `|S|` removes the cells where the line does not
respond to ζ — and with them the small and zero shifts. The medians then
describe only the ζ-sensitive part of each row and overstate the bias the map
shows as a whole. Keeping every observable cell has the opposite, known bias
on the whole 2D grid: a species that shifts a little over most of the map
outranks one strongly sensitive in a single CRIR band, whose median is pulled
down by the inert part of its rows. The per-ζ regime verdicts
(`attenuation_zeta_regime_verdicts`) and the 3D per-box verdicts grade each
CRIR band separately and remove that bias.

A large `|Δ|` in a single cell with small `|S|` still means loss of CR
sensitivity rather than a strong CR effect; check `slope_dex` in `top_models`
before following one up.

### 8.2 No match and undetectable cells: read as no shift

`collect_attenuation_x_shift_table` writes `Δ = 0` into every cell that has a
model in both grids but no `x_match`: a no-match cell is read as *"attenuation
moved nothing measurable here"*. Only cells with no model at all stay NaN.
Cells below the detection limit (`obs_limit`, intensity grids only) are treated
the same way inside the ranking functions (`_zero_unobservable_shifts`):
`Δ = 0` and `R = 0`, because a change nobody can detect is no change.

Nothing is imputed. A missing match has two readings that the shift alone
cannot separate:

- **Past the grid edge.** Where the attenuated profile is monotonic, the
  required `ζ_true` lies beyond the upper end of the row, so the true `Δ` is
  *larger* than anything the row can represent.
- **Model mismatch.** No horizontal displacement reproduces the attenuated
  intensity at all: the two profiles differ by an offset that no shift along
  `ζ_H` can absorb, or the degeneracy runs along `G₀`, which this construction
  does not model.

Counting both as 0 means every median and percentile is a **conservative**
(low) estimate of the ζ error wherever the first reading holds — most often in
the high-`ζ_H` part of the grid, where little room is left to shift into. It
never inflates the error with a value that was not measured.

The same rule applies to every species, and no species is excluded on the
fraction of its cells that are unmatched or undetectable. Each species is
therefore summarised over the same cells, and a species detectable over only
a small part of the grid ranks low instead of ranking on that part alone.
`frac_observable` is reported next to it.

---

## 9. The left panel

All quantities are computed from the per-cell `|Δ|` of one species over all
its cells (no-match and undetectable cells as 0, §8.2):

| element | quantity |
|---|---|
| bar length | median of `|Δ|` — `error_dex` |
| band | P10–P90 of `|Δ|` — `error_p10`, `error_p90` |
| bar colour | `frac_observable` (colour bar 0 → 1): fraction of the grid above the detection limit; the rest entered at 0 |
| numeric labels | omitted; the only free position is beside P90, where a label would be misread as annotating P90. Values are in the printout and the table |
| ordering | decreasing `error_dex` |

> **The band is a dispersion, not an uncertainty.** The statistics are computed
> over the full set of grid cells. That set is a complete enumeration of the
> model grid, not a random sample, so the median carries no sampling
> uncertainty. The band is the P10–P90 range of `|Δ|` across parameter space,
> i.e. how strongly the bias varies with `(ζ_H, G₀)`. It is drawn without caps
> to avoid confusion with an error bar. In the real run it spans
> ≈ 0.1–2.4 dex. The axes along which this variation occurs are not shown in
> this panel (see §10).
>
> **Zeros pull the whole band down.** Every unmatched or undetectable cell
> enters at 0, so every percentile is conservative (§8.2). A pale bar (low
> `frac_observable`) means much of that pull comes from undetectable cells.

### 9.1 Interpretation of median and dispersion

![how to read one bar](Figures/probe_margin/6_reading_order.png)

*(Schematic archetypes; they are not taken from the synthetic grid of Figures
1–5.)*

By definition at least half of the cells satisfy `|Δ| ≥ error_dex`,
so a large median already establishes that the bias affects most of the grid.
The band adds information about **homogeneity** across parameter space.

| | narrow band | wide band |
|---|---|---|
| **large median** | homogeneous large bias; a single value describes the grid | large bias with strong parameter dependence; a single value is not representative |
| **small median** | bias negligible throughout | bias negligible in most of the grid but large in a sub-region |

The bottom-right case is under-represented by the ranking, because the ranking
sorts on the median: a species with a small median and a large P90 appears low
in the list despite a severe bias in part of the grid.

### 9.2 Decision procedure

```
                       select a species
                                |
                                v
        1.  error_p90 <= tolerance?
                                |
               yes -------------+------------- no
                |                               |
                v                               v
    attenuation can be neglected         attenuation cannot be
    for this line (stop)                 neglected
                                                 |
                                                 v
        2.  error_dex > tolerance?
                                                 |
                            yes -----------------+----------------- no
                             |                                       |
                             v                                       v
                  bias exceeds tolerance                   bias exceeds tolerance
                  in >= 50 % of cells                      only in a sub-region
                             |                                       |
                             +-------------------+-------------------+
                                                 v
        3.  regime heatmaps: along which parameter, in which region?
            (`atten_shift_regime_heatmaps_*.pdf`)
```

P90 is tested before the median because the criterion for neglecting
attenuation is an upper bound on the bias over (nearly) the whole grid.
`error_p90` decides admissibility, while `error_dex` only determines rank.
The zeros of §8.2 pull every percentile down, so passing step 1 is
conservative only where the matches stayed inside the grid; read a small
median together with P90 (§9.1, bottom-right case) and the bar colour.

---

## 10. Localising the parameter dependence

**Magnitude of the dependence.** The band width `error_p90 − error_p10`
measures how much `|Δ|` varies across the cells. A narrow band means
the median is representative. A wide band means the median averages over
regions with substantially different bias, and it should not be quoted without
the percentiles.

**Direction of the dependence.** The band is a marginal distribution and
therefore contains no information about *where* in parameter space the large
values occur. That information is in the regime heatmaps of
`plot_attenuation_shift_summary` (`atten_shift_regime_heatmaps_*.pdf`). They
show the median `|Δ|` per species and per parameter bin, marginalised along
`ζ_H` in one panel and along `G₀` in the other. A gradient in the ζ_H panel
indicates that the bias depends on the CR ionization rate, and a gradient in
the G₀ panel indicates that it depends on the FUV field. Both figures use the
same cells (no-match and undetectable cells as 0, no slope gate), so a heatmap bin
and a bar are computed from the same cell set.

Two table columns describe the `G₀` dependence of `R` (not of Δ):

- `fuv_swing_dex` — spread of the row medians (full range by default, §3). The row median
  removes the ζ_H dependence, so this isolates the variation with `G₀`.
- `fuv_trend` — ordinary least-squares slope of `r_row` against
  `log10 G₀` (dex of R per dex of G₀; at least 3 rows required). It gives the
  sign of the dependence, which a range cannot.

If a species has a wide Δ band but a small `fuv_swing_dex`, the variation of Δ
is not driven by `G₀`. It must then arise within rows: from the ζ_H dependence
of `R`, or, since `|Δ| ≈ |R|/|S|`, from the ζ_H dependence of the sensitivity
`S`.

On the right panel, marker size encodes `|Δ|`, and each legend entry lists its
`|Δ|` in brackets, so both quantities can be read together.

---

## 11. Worked example — real grid

The procedure applied to `Figures/atten_fuv_response.pdf`, using the eight
species selected by the pipeline. This run **predates the constituent cap**
(equivalent to `max_per_constituent = max_species` today). It is kept because
it illustrates the problem the cap addresses (§12).

It also **predates the current cell convention**: it used the `|S| ≥ 0.1`
slope gate, imputed no-match cells as lower bounds (the `>` marks and the
censoring paragraph below), and dropped undetectable cells. Rerun today,
no-match and undetectable cells enter as 0, flat cells are kept and no species
is excluded, so no median would carry `>` and the medians would be lower. The
steps illustrate the reading procedure, not current numbers.

Medians as printed; percentiles read from the axis:

| species | median \|Δ\| | P10 | P90 | colour |
|---|---|---|---|---|
| HNC/H2O+ | >0.97 | ≈0.1 | ≈2.4 | orange |
| HNC/OH+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/13CO+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/18OH+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/H218O+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/NH2+ | >0.90 | ≈0.1 | ≈2.4 | orange |
| HNC/CO+ | >0.89 | ≈0.1 | ≈2.4 | orange |
| N2H+/H+ | >0.88 | ≈0.1 | ≈3.4 | orange |

**(Historical) censoring.** In that run close to half of each sample consisted
of imputed bounds, so every value in the table was a lower limit. This step no
longer exists (§8.2).

**Step 1 — P90.** ≈2.4 dex for the HNC ratios (a factor ≈250 in ζ_H) and
≈3.4 dex for N2H+/H+ (a factor ≈2.5 × 10³), both lower limits. This exceeds
any plausible tolerance, so none of the eight lines admits neglecting
attenuation.

**Step 2 — median.** ≈0.9 dex, i.e. a bias of about a factor 8 in at least
half of the cells. The bias is therefore widespread, not confined to a
sub-region.

**Step 3 — dispersion.** P10 ≈ 0.1 and P90 ≈ 2.4 dex: the bias varies by more
than two orders of magnitude in ζ_H across the grid. These species fall in the
"large median, wide band" class of §9.1: the bias is both widespread and
strongly parameter-dependent, which requires
`atten_shift_regime_heatmaps_rightward_top_median.pdf` to locate.

**Constituent degeneracy.** Seven of the eight entries share the numerator HNC.
As shown in §12, these are not independent results. The panel effectively
contains two findings, one for HNC and one for N2H+/H+, and the first is
repeated seven times because the ranking operates on keys, not on the
underlying species.

---

## 12. Constituent degeneracy in ratio rankings

If one species' grid contains an artefact (a chemical discontinuity, an
abundance floor, an interpolation defect), it appears in every ratio that
contains the species. The ranking then shows several mutually consistent
entries that look like independent confirmation.

### 12.1 Decomposition over a ratio

Both `R` and `S` are logarithmic quantities and decompose exactly:

```
R_ratio = R_num − R_den            S_ratio = S_num − S_den
```

and, to the order of §7,

```
|Δ_ratio|  ≈  |R_num − R_den| / |S_num − S_den|
```

If the denominator is insensitive to both CRs and attenuation
(`|R_den| ≪ |R_num|`, `|S_den| ≪ |S_num|`), then
`Δ_ratio = Δ_num · [1 + O(R_den/R_num, S_den/S_num)]`. All such ratios share
the numerator's value up to first-order corrections. The seven HNC medians lie
within 0.08 dex of each other (0.89–0.97), which is consistent with this limit
and not what independent observables would be expected to give.

### 12.2 Implementation

**`max_per_constituent` (default 2).** This is the maximum number of entries in
a top-N selection that may share a constituent species. It applies to both
panels of `plot_attenuation_fuv_response` and to the summary bars and heatmaps
of `plot_attenuation_shift_summary` (`_cap_per_constituent`,
`grid_functions.py`). The default of 2 allows one consistency check (the same
numerator against a second, independent denominator) without letting one
species fill the selection.

The cap affects only which entries are drawn. All species keep their rows in
the returned tables, the paginated per-species figures are unaffected, and the
suppressed entries are printed:

```
  [4 higher-ranked entries suppressed by max_per_constituent=2:
   HNC/D3, HNC/D4, HNC/D5, HNC — the same species restated against
   other denominators, one finding not many]
```

`max_per_constituent = max_species` restores the uncapped top-N.

**`den_contrib`.** This column of the `plot_attenuation_fuv_response` table
(also printed as `den=`) is

```
den_contrib  =  |R_den| / ( |R_num| + |R_den| )
```

evaluated from the constituents' own `response_dex` rows. It is the fractional
contribution of the denominator to the total absolute response.
`den_contrib → 0` means the ratio's response is carried by the numerator
alone. It is NaN if a constituent has no row of its own (not in `species_list`)
or if the key is not a two-component ratio.

The two work together: the cap limits redundancy in the selection, and
`den_contrib` identifies which constituent is responsible for a ratio's
response.

---

### 12.3 Disjointness of the two panels

| left panel (largest \|Δ\|) | right panel (largest probe_margin) |
|---|---|
| HNC/H2O+, HNC/OH+, HNC/13CO+, HNC/18OH+, HNC/H218O+, HNC/NH2+, HNC/CO+, N2H+/H+ | SO2/CH3OH+, SO2/SO2+, HCO+/13CO2, 13CO/N2+, CO/N2+, H13CO+/13CO2, N2+, C+/N2+ |

The two sets have no species in common. This is expected: the left panel ranks
by `|Δ|` ∝ `|R|/|S|`, and the right by `|R| − swing`. A species can be extreme
in one metric and unremarkable in the other.

A large `probe_margin` does not imply a small bias. `SO2/CH3OH+` has the
largest margin in the run (`|R| ≈ 1.02`, swing ≈ 0.31, margin ≈ +0.71), yet
its legend entry reads `(>0.57)`. A constant-CR fit of this line
underestimates ζ_H by at least 0.57 dex (a factor ≥ 3.7). The remaining seven
have `|R| ≈ 0.5–0.6`, swing ≈ 0.1–0.2 (margins ≈ +0.35 to +0.45), and
`|Δ| ≈ 0.52–0.60` dex, all lower limits.

> The panels address two distinct questions: whether an intensity change is
> attributable to CRs rather than to `G₀` (right), and the bias of ζ_H under a
> constant-CR model (left). Satisfying the first does not bound the second.
> In this grid, every species with a positive margin still carries a bias of
> at least ≈0.5 dex, so attenuation cannot be neglected for any of the
> candidate probes.

---

## 13. The right panel

![the probe panel](Figures/probe_margin/5_probe_panel.png)

The right panel of `plot_attenuation_fuv_response` plots `fuv_swing_dex`
against `abs_response_dex`, so the dashed line `y = x` is the locus
`probe_margin = 0`:

- **below the diagonal** — `probe_margin > 0`: the response exceeds its FUV
  dispersion; the intensity change is attributable to attenuation.
- **above the diagonal** — `probe_margin < 0`: the response is degenerate with
  `G₀`; the line does not constrain ζ_H independently.
- **left of the dotted line** — `|R| < response_floor_dex` (default 0.1 dex,
  a 26 % intensity change). The response is below the floor, and the sign of
  the margin is not interpreted.

The sign criterion is the `cr_led` flag (`probe_margin > 0`) of
`reconcile_attenuation_species`. Consequently the `tracers` category contains
no species with negative margin, while `problematic` may.

---

## Summary of quantities

| quantity | definition | preferred |
|---|---|---|
| `R` | `log10(I_atten / I_const)` per cell | — |
| `S` (`slope_dex`) | `∂log I_const / ∂log ζ_H`; diagnostic, not a filter | large \|S\| |
| `Δ` (`error_dex`) | median `log10(ζ_true / ζ_inferred)`; ≈ \|R\|/\|S\| to first order | small |
| `frac_observable` | fraction of the grid above the detection limit; the rest enters Δ and R at 0 (reported, not a cut) | — |
| `abs_response_dex` | \|median over rows of the row-median R\| | large |
| `fuv_swing_dex` | max − min of the row-median R (3D: split into `g0_swing_dex`, tested, and `n_swing_dex`) | small |
| `fuv_trend` | OLS slope of row-median R vs log G₀ | ≈ 0 |
| `probe_margin` | `abs_response_dex − fuv_swing_dex` | positive and large |
| `den_contrib` | `|R_den| / (|R_num| + |R_den|)` | not ≈ 0 (≈ 0: numerator-dominated) |

**`probe_margin > 0`:** the attenuation response exceeds its FUV dependence, so
an intensity change is attributable to cosmic rays.
**`probe_margin < 0`:** the response is degenerate with `G₀`, so an intensity
change cannot be attributed uniquely.
