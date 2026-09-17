# `probe_margin`, in pictures

> **The whole idea in one line:** how much of a species' attenuation signal is
> left over once you subtract how much that signal *wobbles* as you move across
> the FUV axis.

Figures: `python Scripts/GRID/make_probe_margin_explainer.py`.
Numbers are the synthetic ones from the method doc — they reproduce what
`plot_attenuation_fuv_response` actually prints on that grid.

---

## The raw material: R

For every cell of the grid you have one number:


```
R  =  log10( I_atten / I_const )
```

*How much intensity this line loses when attenuation is switched on, at this
exact (ζ_H, G₀).* Negative = fainter. That's it — no matching, no tolerance.

![R over the grid](Figures/probe_margin/1_r_field.png)

Two species, two completely different situations:

- **CO 1-0** — the same colour everywhere. Attenuation costs it 1.4 dex no
  matter where you are.
- **[OI] 63µm** — orange at the bottom, blue at the top. Attenuation makes it
  *brighter* at low `G₀` and *fainter* at high `G₀`.

**This is the entire distinction `probe_margin` is trying to capture.** If you
observed a line and saw its intensity change, CO's change could only be
attenuation. [OI]'s change might just mean you were looking at a different part
of the radiation field.

---

## Step 1 — squash each row to one number

Each horizontal row of the grid sits at **one fixed `G₀`**, so it is one
cosmic-ray experiment. Take the median of R along that row:

```python
r_row[i] = np.nanmedian(response[i, :])      # median over ζ
```

The 6-row grid gives you a **vector of 6 numbers**, one per FUV value:

![row medians](Figures/probe_margin/2_row_medians.png)

- **CO 1-0 is a vertical line** — every row gives −1.40.
- **[OI] 63µm is tilted** — the row median slides from +0.60 down to −0.90.

Everything from here on is just two summary numbers describing that vector.

---

## Step 2 — the two numbers, read off the same six dots

![number line](Figures/probe_margin/3_number_line.png)

### `abs_response_dex` = |R| — *how big is the effect*

The median of the six, then absolute value.

### `fuv_swing_dex` = "swing" — *how much does it depend on `G₀`*

Sort the six, find the value 10% of the way up (`p10`) and 90% of the way up
(`p90`), subtract.

For **[OI] 63µm** the six numbers are:

```
sorted:   -0.90   -0.60   -0.30    0.00   +0.30   +0.60
                    ↑                             ↑
                p10 = -0.75                   p90 = +0.45

|R|   = |median| = |-0.15|          = 0.15   ← tiny effect
swing = p90 - p10 = 0.45 - (-0.75)  = 1.20   ← huge G_0 dependence
```

(`p10` lands between the 1st and 2nd sorted values, so numpy interpolates:
halfway between −0.90 and −0.60 gives −0.75.)

For **CO 1-0** all six numbers are −1.40, so `p10 = p90` and:

```
|R|   = 1.40      ← big effect
swing = 0.00      ← identical at every G_0
```

> **Why `p90 − p10` and not `max − min`?** So one freak row can't set the whole
> number. It spans 80% of the FUV range and lets the extreme 10% at each end
> fall away.

---

## Step 3 — subtract

```
probe_margin = |R| − swing
```

![the subtraction](Figures/probe_margin/4_subtraction.png)

Read the green bar as the signal and the blue bar as the amount of that signal
you have to give away because it isn't stable across `G₀`. The diamond is what
survives.

| species | \|R\| | swing | margin | verdict |
|---|---|---|---|---|
| CO 1-0 | 1.40 | 0.00 | **+1.40** | huge effect, perfectly stable → great probe |
| C+ 158µm | 0.25 | 0.88 | **−0.63** | small effect, wobbles 3.5× more than its own size |
| [OI] 63µm | 0.15 | 1.20 | **−1.05** | barely responds, but swings wildly with `G₀` |

A **subtraction** rather than a ratio because both terms are already in dex —
and because the *best* case is `swing → 0`, which a ratio would send to
infinity.

---

# The other half of the figure: Δ, the dex error

`probe_margin` answers *"is this line telling me about cosmic rays?"*. The
**left** panel of `plot_attenuation_fuv_response` answers a different question:
*"if I ignore attenuation altogether, how wrong is the ζ_H I fit?"* That number
is `Δ`, and it is built from a horizontal match, not from R.

---

## Δ in one picture

Stay on **one row** of the grid (one fixed `G₀`). You have two curves along ζ:
the constant-CR grid `I_const(ζ)` and the attenuated grid `I_atten(ζ)`.

```
     ^  log I
     |
     |                                                  ----- I_const
     |                                       -----------
     |                           ------------
     |               ------------                       ----- I_atten
     |    ----o------........................----o--  <- same intensity
     |  --                       ------------
     +--------|----------------------------------|-----------------> log ζ_H
           ζ_nom                              x_match
              |<----------  Δ  (dex)  ---------->|
```

**R is the vertical gap between the two curves; Δ is the horizontal one.**
Same physics, measured along two different axes.

Take the intensity the **constant** grid predicts at `ζ_nom`. Walk along the
**attenuated** row until you find the ζ where the attenuated model produces that
same intensity (within `match_rtol = 2%`). That is `x_match`, and

```
Δ  =  log10( x_match / ζ_nom )
```

(`horizontal_intensity_matched_x_shift_dex`, `grid_functions.py:7809`; stored as
`shift_dex` / `abs_shift_dex` in the pixel table.)

---

## Why that horizontal distance *is* the inference error

Fitting a real source — which is attenuated — against your constant-CR grid
means solving

```
I_const(ζ_inferred)  =  I_atten(ζ_true)
```

for `ζ_inferred`. That is the *same equation* the match above solves, with
`ζ_true = x_match` and `ζ_inferred = ζ_nom`. So

```
Δ  =  log10( ζ_true / ζ_inferred )   ←  the dex you get wrong
```

Nothing more is needed to turn a grid diagnostic into an observational error
bar: 0.3 dex means a factor 2 wrong in ζ_H, 0.5 dex a factor 3, 1.0 dex a
factor 10.

**Why the scan is rightward-only.** Attenuation removes CR flux, so the line
comes out fainter, so it looks like a *lower*-ζ constant-CR model. The inferred
ζ is therefore too low and `ζ_true > ζ_inferred`: Δ is positive and the match
lives at `x ≥ ζ_nom`. `scan_direction="rightward"` encodes exactly that
expectation — a leftward match would be a sign the physics is not what we think.

---

## R and Δ are the same effect, divided by the slope

Let `S = dlog10(I_const)/dlog10(ζ)` — the local CR sensitivity, the `slope_dex`
column. Expand the match equation to first order around `ζ_nom`:

```
log10 I_atten(x_match)  ≈  log10 I_const(x_match) + R
                        ≈  log10 I_const(ζ_nom) + S·Δ + R
```

and that must equal `log10 I_const(ζ_nom)`, so

```
S · Δ  +  R  =  0        →        |Δ|  ≈  |R| / |S|
```

**R is the numerator of the error; the slope is its denominator.**

| species | \|R\| | S (dex/dex) | \|Δ\| ≈ \|R\|/\|S\| |
|---|---|---|---|
| steep row | 1.40 | 0.70 | 2.0 dex |
| same R, flat row | 1.40 | 0.20 | 7.0 dex |

Two consequences, both of which the function is built around:

1. **A big \|Δ\| is not automatically a big attenuation effect.** It can just be
   a flat row: `S → 0` sends Δ to infinity while the species has *stopped*
   responding to cosmic rays at all.
2. **The best CRIR probe has a small error.** A large R on a *steep* row — the
   ideal probe — divides down to a tiny Δ. This is why the right panel is ranked
   by `probe_margin` and not by Δ: an error-sorted list buries exactly the
   species you were looking for.

---

## The two things that can go wrong with a cell

### 1. Flat row → the slope gate

Where `|S|` is small, Δ is numerically enormous and physically meaningless. Only
cells with

```
|S|  ≥  min_slope_dex          (default 0.1 dex/dex ≈ 26% per decade of ζ_H)
```

are *trusted*; `frac_trusted` in the returned table is the fraction of the
species' grid that survives. Δ and R are gated on the identical `slope_dex`
numbers used by `plot_attenuation_shift_summary`, so the two functions never
disagree about which cells count.

### 2. No match on the row → censoring

Sometimes no `x_match` exists inside the row. That is **not** missing data, and
dropping it would delete precisely the worst-determined models from the ranking.
The usual reason is that the required `ζ_true` lies past the end of the grid
row, so the cell is **right-censored**: the error is *larger* than anything
measurable there. It enters at its lower bound

```
Δ_bound  =  log10( ζ_row_max / ζ_nom )
```

```
        ζ_nom                        end of grid row
          |                                 |
          |<───── Δ_bound ─────────────────>|   ← what we record
          |<───── Δ_true ──────────────────────────>?  ← unmeasurable
```

Note what `ζ_row_max` actually is: the largest `x_match` that is finite *and*
positive on that row, not the nominal end of the ζ_H axis. If the last few
columns of the row never produced a match, the bound is measured to the last
**matchable** column, which makes `Δ_bound` conservative — the true displacement
is at least that large, and possibly larger than even the axis would allow.

Two guards on that imputation:

- `Δ_bound ≤ 0` (i.e. `ζ_nom` already sits at the row end) means there is no
  rightward room at all, so the cell carries no information and is dropped.
  Note that this drops the cell from the **denominator** as well: `frac_censored`
  is `n_censored / n_err`, where `n_err` counts only the cells that actually
  contribute to the median (matched *plus* imputed). It is therefore the share of
  the *reported* sample that is a bound, not the share of the trusted grid —
  a species can have a small `frac_censored` and still be built on very few
  cells, which is why `n_error_cells` is reported next to it.
- A no-match cell can *also* mean that **no horizontal displacement reproduces
  the attenuated intensity** — the curves differ by a vertical offset, or by a
  displacement along `G₀`, which this whole construction does not model. The two
  cases are not separable from the shift alone, so the imputation is only
  defensible while censored cells are a minority. Past
  `max_frac_censored = 0.5` the median itself falls on an imputed bound rather
  than on a measurement, and the species is dropped from the ranking and both
  figures (`ranked = False`, `probe_ranked = False`) with a printed note.

The 0.5 threshold is an order-statistic argument, not a taste call. The median
is the middle value of the pooled sample of measured `|Δ|` and imputed
`Δ_bound`. Once more than half the sample is imputed, the middle value is an
imputed entry **by construction**, whatever the numbers are — so the headline
error of that species would be a statement about where the grid ends rather than
about cosmic rays, and ranking on it would push the least measurable species to
the top for the wrong reason. Below the threshold the majority of the sample is
real measurement (though an individual median can still land on a bound if the
bounds happen to fall mid-distribution — the guard bounds the failure mode, it
does not eliminate it). Excluded species are **not deleted**: they stay in the
returned table with `ranked = False` / `probe_ranked = False` and their
`frac_censored` printed, so the exclusion is auditable rather than silent.

Whenever `frac_censored > 0` the median is a **lower limit**, and both the
printout carries a `>` and the figure a limit arrow. The direction is
one-sided: every
imputed cell enters *below* its true value, never above, so the pooled median can
only be pulled down relative to the truth. `bound_gap` in the probe-ranking table
(`error_dex - median_abs_shift_dex`) quantifies exactly this — how much of a
species' reported error rests on imputed bounds rather than on matched shifts.

---

## Reading the left panel

Everything on it comes from the per-cell `|Δ|` of one species, over trusted
cells only:

| what you see | what it is |
|---|---|
| bar length | **median** \|Δ\| over trusted cells — `error_dex` |
| band | **p10–p90** of \|Δ\| across the grid — `error_p10`, `error_p90` |
| bar colour | `frac_censored`, the fraction of cells that are lower limits (colour bar 0→1) |
| ▶ at the band's right end | at least one censored cell → every percentile of this species, p90 included, is a lower limit |
| no numbers | on purpose — the bar length *is* the median and the band ends *are* p10/p90, and a label only fitted beside the band's right edge, where it read as if it annotated p90. Exact values: the printout, or `error_dex` / `error_p10` / `error_p90` |
| ordering | largest median error at the top |

> **The band is not an error bar.** It used to be drawn as a capped
> `errorbar`, which reads as *uncertainty on the median* — and it is not that.
> The median of hundreds of grid cells is very well determined; the band is
> the **spread of \|Δ\| across those cells**, i.e. how differently the same
> species behaves at different (ζ_H, G₀). The two are numerically miles apart:
> the uncertainty on these medians is negligible, while the drawn span in the
> real run is 0.1 → 2.4 dex. It is now drawn capless, as a band, for exactly
> that reason — and note that the variation it encodes runs along axes that do
> not appear anywhere on this panel, which is what the regime heatmaps are for.
>
> **Censoring applies to the whole band, not just the median.** An unmatched
> cell enters at `Δ_bound = log10(ζ_row_max / ζ_nom)`, which lies *below* its
> true value, so every percentile of a censored species is a lower limit and
> the whole band is pushed leftward. The ▶ sits at p90 because that is the end
> a safety call is made on: a worst case that looks tolerable may be an imputed
> bound, and the bar colour alone does not say so.

> Read it as: **"ignore attenuation and this is how many dex of ζ_H you get
> wrong — and how much that depends on where in the grid you sit."** A short bar
> with a long band is a species whose error is small in most of the grid and
> bad in one corner; a dark bar is a species whose error is mostly *bounds*, not
> measurements.

### The four bar shapes

![how to read one bar](Figures/probe_margin/6_reading_order.png)

*(These four are schematic archetypes, not the synthetic pixels used in figures
1–5 — no single grid shows all four corners at once.)*

The bar is a **median**, so a large bar already says the problem is widespread:
by construction at least half the trusted cells have `|Δ|` ≥ the bar length. You
do not need a narrow band to establish *prevalence* — the median has done
that. What the band adds is **uniformity**.

| | narrow band | wide band |
|---|---|---|
| **large bar** | badly biased *uniformly* — one number transfers to any source in the grid | badly biased at the median, tolerable at the p10 end, far worse at the p90 end — still a real problem, just a regime-dependent one |
| **small bar** | genuinely safe to ignore attenuation | **the trap** — fine over most of the grid, catastrophic in one corner |

So "large bar + narrow band" is the cleanest *quotable* case, not the only
dangerous one. A large bar with a wide band is not less of a problem; it is a
problem you cannot summarise in one number, which is exactly what sends you to
the heatmaps below. And the bottom-right cell is the one the ranking itself
under-sells: it sorts on the median, so a species that is catastrophic in one
corner sits low in the list.

### The procedure, in order

```
                       pick a species' bar
                                |
                                v
        1.  bar colour / limit arrow  —  dark, or arrow shown?
                                |
               yes -------------+------------- no
                |                               |
                v                               v
    every number below is a              numbers are measured
    LOWER LIMIT: "at least"                    |
                |                               |
                +---------------+---------------+
                                v
        2.  p90  (right end of the band)  —  p90 <= your tolerance?
                                |
               yes -------------+------------- no
                |                               |
                v                               v
    safe to ignore attenuation          you have a problem
    for this line; stop                          |
                                                 v
        3.  bar (median)  —  is the median above tolerance too?
                                                 |
                            yes -----------------+----------------- no
                             |                                       |
                             v                                       v
                  bad EVERYWHERE                            bad SOMEWHERE
                  (>= half the grid);                       (wide band);
                  one number describes it                   one number does not
                             |                                       |
                             +-------------------+-------------------+
                                                 v
        4.  regime heatmaps  —  along which axis, and in which corner?
            (`atten_shift_regime_heatmaps_*.pdf`)
```

Step 2 before step 3 is deliberate: `error_p90` is the number that decides
safety, `error_dex` only the number that decides *rank*. A species is safe to
ignore only if its **worst** case is tolerable; it is dangerous as soon as its
worst case is not, whatever its median says.

### Answering the second half — *how much*, then *where*

The bar answers the first half of that sentence. The second half is answered in
two steps, by two different objects.

**How much** — the band. `error_p10`–`error_p90` is the spread of `|Δ|` over
the species' trusted cells, so the band *width* is the size of the grid
dependence. Narrow: one number describes the species everywhere, and the median
is a fair summary. Wide: the median is an average over cells that disagree with
each other, and quoting it alone is misleading — the same species is a good CR
probe in part of the grid and a bad one elsewhere.

**Where** — not on this panel at all. The band is a spread, and a spread has
thrown the coordinates away: it tells you the error varies by so many dex across
the grid, never along which axis or in which corner. That is what the **regime
heatmaps** of `plot_attenuation_shift_summary` are for
(`atten_shift_regime_heatmaps_*.pdf`). They are two panels of species × parameter
bin, coloured by **median `|Δ|` per bin** — one panel binned along `x` (ζ_H), one
along `y` (`G₀`) — so a wide band resolves into a direction: a gradient across
the x-panel means the error grows with CR rate, a gradient across the y-panel
means it is `G₀` that decides. Both figures gate cells on the identical
`slope_dex` numbers (see above), so a heatmap bin and a bar are the same Δ,
sliced differently: read the bar for the ranking, the heatmap for whether that
rank holds where *your* source sits.

Two related columns answer a narrower version of the same question, but about
`R` rather than `Δ`, and only along `G₀`: `fuv_swing_dex` is p90 − p10 of the
**per-row median** response, and since each row is a fixed-`G₀` experiment,
taking the median within the row removes the ζ_H variation and isolates the `G₀`
dependence; `fuv_trend` is the regression slope of those row medians against
`log₁₀ G₀`, which gives that dependence a *sign* — something a spread cannot.
Because they are built on `R`, a wide Δ band beside a small `fuv_swing_dex`
is informative in itself: the grid dependence is not along `G₀`, so it lives
within the rows — either `R` varies along ζ_H, or, just as likely given
Δ ≈ |R|/|S|, the CR sensitivity `S` does.

Δ also appears on the right panel twice, so the two halves can be read together:
marker **size** is `|Δ|`, and each legend entry carries its `|Δ|` in brackets.

---

## Worked example — the real run

Everything above is machinery. This is the procedure run on the actual output in
`Figures/atten_fuv_response.pdf`, on the eight species the pipeline itself
selected — **run before the constituent cap existed**, i.e. what you get today
with `max_per_constituent` set as high as `max_species`. It is kept in that form
on purpose: it is the clearest picture of the failure the cap was added for.

The left panel's eight, with the medians it printed and the band ends read
off the axis:

| species | median \|Δ\| | p10 | p90 | colour |
|---|---|---|---|---|
| HNC/H2O+ | >0.97 | ≈0.1 | ≈2.4 | orange |
| HNC/OH+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/13CO+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/18OH+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/H218O+ | >0.93 | ≈0.1 | ≈2.4 | orange |
| HNC/NH2+ | >0.90 | ≈0.1 | ≈2.4 | orange |
| HNC/CO+ | >0.89 | ≈0.1 | ≈2.4 | orange |
| N2H+/H+ | >0.88 | ≈0.1 | ≈3.4 | orange |

**Step 1 — colour and the limit arrow.** Every bar is orange, not pale
yellow, and every printed median carries `>`. On the 0→1 colour bar that is roughly 0.4–0.5: *half of the
cells behind each of these bars are imputed bounds, not measurements.* They sit
directly under the `max_frac_censored = 0.5` guard — species worse than that were
already dropped and printed as excluded. So before reading a single value: these
are **lower limits**, and the true errors are larger than what is drawn. Nothing
below this step can be quoted as an equality.

**Step 2 — p90.** ≈2.4 dex for the seven HNC ratios, ≈3.4 dex for N2H+/H+ —
and every one of them carries the ▶, so these worst cases are themselves lower
bounds. Three dex is a factor of ~2500 in ζ_H. Whatever your tolerance is, it is not that. All
eight fail at step 2, so all eight are lines you cannot use while ignoring
attenuation.

**Step 3 — the median.** ≈0.9 dex, so at least half the grid is wrong by nearly
an order of magnitude. That is the "bad **everywhere**" cell of the table — this
is not one pathological corner dragging up an otherwise fine species.

**Step 4 — but the band is enormous.** p10 ≈ 0.1 against p90 ≈ 2.4: the same
species is nearly harmless in the bottom decile of cells and catastrophic in the
top. So it is *both* — bad on the typical cell **and** strongly regime-dependent.
One number cannot describe it, which is precisely the trigger to go to
`atten_shift_regime_heatmaps_rightward_top_median.pdf` and ask which axis and
which corner. This is the case the 2×2 table calls "large bar, wide band".

**A caution the ranking used to be unable to give you.** Seven of the eight
share the same numerator, HNC. This is not eight independent verdicts; it is
essentially one statement about HNC seen against seven different denominators,
plus one about N2H+/H+. Read that way, the left panel says *HNC ratios are the
worst thing in this grid to interpret without attenuation* — a single finding,
listed eight times because the ranking sorts rows, not physical causes. That is
now capped rather than left to the reader; see the next section.

## One species cannot take the whole list

The run above is the failure mode in its pure form: eight slots, one physical
cause. It matters beyond tidiness — if HNC's grid has a pathology anywhere (a
chemistry cliff, an abundance floor, a bad interpolation), that single defect
appears as eight agreeing bars and reads as an overwhelming, reproducible
result. The whole follow-up then chases one species' grid.

### Why ratio families are not independent

Both quantities are logarithmic, so they decompose **exactly** across a ratio:

```
R_ratio = R_num - R_den            S_ratio = S_num - S_den
|Δ_ratio| ≈ |R_ratio| / |S_ratio|
```

If a denominator is CR-inert (`R_den → 0`, `S_den → 0`) then
`Δ_ratio → Δ_num` identically. Seven denominators like that give seven copies
of HNC's number — which is exactly what the medians show: 0.89, 0.90, 0.93,
0.93, 0.93, 0.97. **A spread of 0.08 dex across seven supposedly different
observables is not a coincidence, it is the fingerprint.** Genuinely
independent observables do not agree that closely.

### What the code does about it

**1. `max_per_constituent` (default 2)** caps how many entries in any top-N
selection may share a constituent species. It applies to both panels of
`plot_attenuation_fuv_response` and to the summary bars/heatmaps of
`plot_attenuation_shift_summary` (`_cap_per_constituent`,
`grid_functions.py`). Two, not one: a second denominator for the same numerator
is a genuine robustness check, a seventh is a monoculture.

Nothing is deleted. The cap decides only what is *drawn*; every species keeps
its row and its numbers in the returned tables, the paginated per-species
figures still show everything, and the suppressed entries are printed:

```
  [4 higher-ranked entries suppressed by max_per_constituent=2:
   HNC/D3, HNC/D4, HNC/D5, HNC — the same species restated against
   other denominators, one finding not many]
```

The crowding *is* a result — "these are all the same species" — so it moves
into one printed line instead of occupying seven of the eight bars. Set
`max_per_constituent=max_species` to get the old plain top-N back.

**2. `den_contrib`**, a column on the `plot_attenuation_fuv_response` table and
a `den=` field in its printout, is

```
den_contrib = |R_den| / (|R_num| + |R_den|)
```

read off the *same table* — the constituents' own rows. Near 0 means the
denominator contributes nothing to the ratio's response, i.e. the row is a
numerator statement wearing a ratio's name. It is NaN when a constituent has no
row of its own (not in `species_list`, or the key is a single species).

Use them together: the cap frees the slots, `den_contrib` names the cause.

---

### The two panels do not share a single species

| left panel (worst error) | right panel (best probe) |
|---|---|
| HNC/H2O+, HNC/OH+, HNC/13CO+, HNC/18OH+, HNC/H218O+, HNC/NH2+, HNC/CO+, N2H+/H+ | SO2/CH3OH+, SO2/SO2+, HCO+/13CO2, 13CO/N2+, CO/N2+, H13CO+/13CO2, N2+, C+/N2+ |

Zero overlap, and that is the point of having two panels. The left ranks on `Δ`
— *how wrong you get ζ_H* — and the right on `margin = |R| − swing` — *whether
the change you see can be blamed on cosmic rays at all*. A species can be
enormous on one and absent from the other.

Note also what the right panel does **not** absolve. `SO2/CH3OH+` is the
cleanest probe in the run — `|R| ≈ 1.02`, swing `≈ 0.31`, so `margin ≈ +0.71`,
far below the diagonal — and its legend entry still reads `(>0.57)`. The best CR
probe in this grid still misplaces ζ_H by **at least half a dex** if you fit it
against a constant-CR grid. The remaining seven cluster at `|R| ≈ 0.5–0.6` with
swing `≈ 0.1–0.2` (margins ≈ +0.35 to +0.45) and carry `|Δ|` of 0.52–0.60, all
with `>`.

> The two panels answer, in order: **"can I trust this line to be about cosmic
> rays?"** (right) and **"and how far off will I be anyway?"** (left). Passing
> the first does not excuse you from the second. In this run *every* species
> that passes the probe test still carries ≥0.5 dex of error — which is the
> quantitative case for not ignoring attenuation in this grid at all.

---

## Where it lands on the real plot

![the probe panel](Figures/probe_margin/5_probe_panel.png)

The right-hand panel of `plot_attenuation_fuv_response` plots exactly these two
numbers against each other, so the dashed diagonal `y = x` **is**
`probe_margin = 0`:

- **below the diagonal** → margin positive → the effect beats its own wobble →
  *this line reports cosmic rays*
- **above the diagonal** → margin negative → *FUV tracer, don't use it to
  measure ζ*
- **left of the dotted line** → `|R| < 0.1`, no response worth discussing, so
  which side it's on is noise

That sign test is literally the `cr_led` flag in the reconciliation cascade
(`grid_functions.py:9944`), which is why `tracers` can never contain a negative
margin and `problematic` freely can.

---

## Cheat sheet

| number | plain English | good value |
|---|---|---|
| `R` | dex the line moves when attenuation is switched on | — |
| `S` (`slope_dex`) | dex the line moves per dex of ζ_H — CR sensitivity | **large** |
| `Δ` (`error_dex`) | ≈ \|R\|/\|S\|: dex of ζ_H you get wrong ignoring attenuation | **small** |
| `frac_censored` | share of cells where Δ is only a lower bound | **small** |
| `abs_response_dex` | how much attenuation moves this line | **large** |
| `fuv_swing_dex` | how much that answer changes with `G₀` | **small** |
| `probe_margin` | the difference — signal you can actually trust | **positive and large** |
| `den_contrib` | share of a ratio's response owned by its denominator | **not near 0** — near 0 restates the numerator |

**`margin > 0`: if this line changes, blame cosmic rays.**
**`margin < 0`: if this line changes, you can't tell why.**
