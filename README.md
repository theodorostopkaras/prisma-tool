# PRISMA

**PDR Results Interactive Simulation Mapping & Analysis** — an interactive
toolkit for photodissociation-region (PDR) model grids.

- **Input:** a directory of one HDF5 file per model
- **Free axes:** density × FUV × ζ (3-D), plus mass / metallicity / attenuation if they vary
- **Data access:** the HDF5 `Metadata/Metadata` table + `species involved`

## Application tabs

After loading a main grid on **Load grids**, the other tabs become active:

| Tab | Purpose |
|---|---|
| **Load grids** | Main PDR grid, optional overlay / chemistry / line-intensity directories |
| **Model setup** | PDR config JSON summary (shared vs varying parameters, species network) |
| **Abundance profiles** | Depth profiles vs A_V or n_H |
| **Heating & cooling** | Thermal balance and rate-component breakdowns |
| **Grid slices** | 2-D contour maps over n_H, FUV, and ζ |
| **Chemistry** | Top formation / destruction reactions with contribution metrics |
| **Intensities** | Line-intensity slice maps and spectrum |
| **Spectra** | Model PV FITS velocity spectra / PV diagrams, plus an observational cube viewer |
| **Interpolation error** | Native vs resampled grids with decimation error maps (abundance + line intensities) |
| **CR attenuation** | ζ<sub>H₂</sub> vs N<sub>H</sub> profiles with 𝓛/𝓗/𝓤 reference bands |
| **Probe ranking** | Regime-aware CRIR probe scores; **CR vs FUV dominance** maps D = \|S_ζ\|/(\|S_ζ\|+\|S_χ\|) over (χ₀, ζ) per density. Method after Albertsson, Kauffmann & Menten (2018, ApJ, 868, 40; [doi:10.3847/1538-4357/aae775](https://doi.org/10.3847/1538-4357/aae775)) |
| **Attenuation impact** | Attenuation workflow: ζ error from ignoring attenuation, CR-probe quality, one verdict per tracer (single n_H or full 3-D, per environment) |
| **Map fit** | Fit observed FITS maps to the 3-D model line-intensity grid |

## How the grid is discovered

Each model file is named:

```
Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5
            |  |  |  |  |  |
            |  |  |  |  |  +-- AA  attenuation tag (optional; omitted => no attenuation, AA=0)
            |  |  |  |  +----- CC  cosmic-ray ionisation rate   ζ   = 10^(-CC)        s^-1
            |  |  |  +-------- ZZ  metallicity                  Z   = 10^((ZZ-10)/10) Z_sun
            |  |  +----------- FF  FUV field (Draine)           χ   = 10^(FF/10)
            |  +-------------- MM  clump mass                   M   = 10^(MM/10)      M_sun
            +----------------- DD  gas density                  n_H = 10^(DD/10)      cm^-3
```

The tool scans the chosen directory, decodes every filename, and creates **one
slider per parameter that actually varies** on disk. If you later add models at
different masses (or metallicities, attenuations), the corresponding slider
simply shows up — no code changes needed.

Non-attenuated model sets may omit the final ``_AA`` token entirely
(e.g. ``Model100_50_20_00_10_15.hdf5`` instead of ``…_15_00.hdf5``). Those
files are indexed with ``atten = 0``.

## Abundance profiles & heating / cooling

For the selected model, depth profiles are read from the HDF5 file (via the
metadata table and the `Relative densities` matrix) and shown against
either **A_V** or **n_H** (log/linear):

**Abundance profiles** tab:

1. Gas & dust temperature
2. H / H₂ relative abundance (with the H→H₂ transition marked)
3. C⁺ / C / CO relative abundance
4. A custom set of species chosen from the dropdown

**Heating & cooling** tab:

5. **Heating / cooling balance** — total heating Γ, total cooling Λ, and the
   **cosmic-ray heating** Γ_CR (dashed, highlighted).
6. **Heating-rate components (all 7)** — H₂ de-excitation, H₂ photodissociation,
   H₂ formation, cosmic rays (drawn thicker), photo-electric, C ionization,
   chemical reactions.
7. **Cooling-rate components (all 16)** — [OI] 63/44/146 µm, CO, [CII], [CI]
   610/230/370 µm, [SiII], ¹³CO, Lyα, H₂O, gas-grain, OH, [OI] 6300 Å, H₂
   kinetic dissociation.

The heating/cooling rates come from
`Local quantities/Auxiliary/Thermal balance/{Heating,Cooling} rates`
(`erg cm⁻³ s⁻¹`); labels are taken from the metadata table. In the breakdown
panels the line dash pattern is cycled alongside the colour so all components
stay distinguishable.

**Grid-integrated rates** integrates every rate as ∫4πr² Γ dr (the rates are
already per volume) and plots it against the **x axis** parameter, one panel per
selected value of the **panel** parameter (the rate-regime map uses the panel
parameter as its y axis). Both lists hold only the parameters the loaded grid
varies (single-valued ones are listed last); the defaults are ζ on x and n_H in
the panels, falling back to the grid's varying axes when ζ or n_H is fixed. The
same rule sets the three axes of **Grids → 3D fronts** and the axes pooled by
**Grids → Abundance PDF**, which also has an **x axis** choice (local ζ_H/n_H by
default, or A_V, n_H, N_H, N(H₂)) and a **Split by** choice (one panel per value
of a varying grid parameter, ζ by default, or none). The absolute
figure takes a rate floor (default 0) and **Top N channels** (a channel is drawn
if it is among the N largest in at least one plotted model). The **share bars**
below give each channel's % of the total heating / cooling, with the attenuated
overlay grid hatched next to the main grid at every x value.

## Model setup (JSON configs)

The **Model setup** tab scans PDR configuration files next to the grid:

```
<grid_dir>/Models/Model…/config_files/pdr_config_*.json
```

(alternate layouts such as `<grid_dir>/Models/` or a sibling `Models/` folder are
also tried automatically).

For the loaded grid it shows:

- **Current model** — grid parameters for the slider selection
- **Chemical species network** — full species list from the configs
- **Grid parameters (vary)** / **Other varying** — parameters that differ between model folders
- **Shared simulation setup** — collapsible sections for settings identical in every config

Each parameter table has columns **Parameter | Value | Comment**. Comments come
from the JSON `*_comment` fields (PDRNEW.INP names); a few fields without JSON
comments use descriptive fallbacks (element abundances, file paths, etc.).

## 2-D parameter-slice contour grids

The **Grid slices** tab shows three contour panels side by side (identical square
figsize when no overlay is loaded):

| Panel | X axis | Y axis | Fixed by slider |
|-------|--------|--------|-----------------|
| n_H vs FUV | density | FUV | cosmic-ray rate ζ |
| ζ vs n_H | ζ | density | FUV field |
| ζ vs FUV | ζ | FUV | density |

Each panel has its **own slider** for the third parameter (independent of the
profile sliders). Cosmic-ray rate ζ is always plotted on the horizontal axis.

Pick the **contoured quantity** from the dropdown (T_gas, T_dust, n_H at the
cloud edge, or **X(species)** — clump-integrated relative abundance). Axes use log₁₀ values on linear scales. Use **Contour Z
scale** for log or linear coloring. Mass, metallicity, and attenuation are held
at their grid mid-points when building each slice.

### Interpolation

Before plotting, native grid points are resampled to a finer mesh (default
60 × 60):

- **Values** interpolated in log₁₀ space (intensities, abundances, etc.)
- **Axes** resampled in log₁₀ parameter space (n_H, χ, ζ)
- **Missing / zero** cells imputed along rows and columns
  so sparse line-intensity or incomplete grids do not produce blocky holes

Controls on **Grid slices** and **Intensities**:

| Control | Meaning |
|---|---|
| Interpolated grid size (ny × nx) | Output mesh resolution |
| X / Y log₁₀ limit (upper) | Optional axis cutoffs |
| Interpolation method | linear, cubic, nearest, or spline |
| Clip to original bounds | Prevent overshoot outside native min/max |

Implementation: `grid_interp.py`.

### Overlay triple panels (x-shift)

When an **overlay grid** is loaded (attenuated models), each slice plane expands
to **three** sub-plots in a row:

1. Main grid (reference)
2. Overlay grid (attenuated)
3. **Horizontal x-shift** — log₁₀(x_match / x_nom) in dex, from intensity matching
   along the x-axis

Use **X-shift scan direction** and **Intensity match tolerance (rtol)** to
control the matching. On the shift panel, **positive** offsets are **red** and
**negative** offsets are **blue** (`RdBu_r`).

## Line intensities

Line intensities are read directly from the model HDF5 files when present.
Alternatively, load a **line-intensity directory** on the Load tab (files named
`jtemp_Model…_<species>.smli`, `jerg_…`, or `tau_…` optical-depth tables; ``.smlc``
is accepted for ``tau`` as well).

The **Intensities** tab provides:

- Three **2-D slice maps** (same planes as Grid slices) for one chosen transition
- **Line spectrum** — all transitions for the current slider selection
- Quantities: **K km/s** (`jtemp`), **erg s⁻¹ cm⁻² Hz⁻¹** (`jerg`), or **τ** (`tau`)
- Same interpolation and x-shift controls as Grid slices

An optional **overlay line-intensity** directory (attenuated `.smli` files) enables the
same triple-panel x-shift layout as for HDF5 grid slices.

## Model spectra (PV FITS)

Position–velocity cubes written as
``Model<tag>_DD_MM_FF_ZZ_CC_AA_<species>.<transition>.fits`` (brightness
temperature) and ``…<transition>-tau.fits`` (optical depth) in the same
directory as the ``.smli`` files. The **Spectra** tab reads these cubes
via ``simline_spectra.py``:

- **Quantity** — **T<sub>mb</sub> [K]** (default PV cubes) or **τ** optical depth
  (``*-tau.fits`` cubes, matching ``tau_Model…`` ``.smli`` tables)
- **1-D spectrum** — spatial mean over all position columns (default), or
  comma-separated position offsets in arcsec to overplot individual columns.
  Select **multiple transitions** to compare lines on the same velocity axis.
- **PV diagram** — 2-D heatmap for the **first** selected transition only
  and linear / log brightness scale

Species and transition lists come from the indexed PV FITS files at the current
grid slider selection.

### Observational overlay and Gaussian fitting

Point **FITS file path** at a CLASS/GILDAS MATRIX table (``hdu_index=1``,
``SPECTRUM`` column, velocity from ``VELO-LSR`` / ``DELTAV``) or a spectral
image cube (``hdu_index=0``). Click **Browse** to pick the file in a system
window, or paste a path. Controls:

- **Spectrum selection** — spatial mean, peak-region average (bright half of
  moment-0), or a single row/pixel index
- **Line core limits** — velocities *outside* this range define the continuum
  (polynomial degree 1); inside is the line core for the Gaussian fit
- **N Gaussians** — multi-component line model (Astropy LevMar)
- **Optional Peak / v₀ / FWHM** per component — starting values (blank = auto).
  **Lock filled values** holds those parameters instead of fitting them.
  Width is FWHM, not σ.
- **Overlay** — plot the observed spectrum on the model 1-D panel
- **Fit Gaussians** — continuum + sum of Gaussians; shows per-component
  amplitudes, centres, widths, integrated intensities, and total
  ``∫I dv`` with propagated uncertainty

Implementation: ``obs_spectrum_fits.py``.

### Cube viewer

The **Cube viewer** sub-tab under **Spectra** is a custom reader for the IRAM
30 m CLASS/GILDAS ``MATRIX`` tables (and ordinary NAXIS=3 image cubes). Those
CLASS files are *not* 3-D FITS images — each row is one 15″ OTF cell with a
full spectrum — so a standard cube viewer has nothing to slice until they are
gridded here:

- **Map** (left) — moment-0 (``∫T dv``) or peak temperature, collapsed over the
  line-core velocity window. Missing OTF cells stay blank (no interpolation).
  RA offset increases to the left (sky convention).
- **Spectrum** (right) — **Pixel / region** uses a map click or a box/lasso
  average; **Mean (whole map)** is the nan-mean of every filled OTF cell
  (same spatial-mean option as the model observational overlay).
  **Spectrum X axis** switches between LSR velocity and observed frequency
  (GHz; radio definition).
- **Possible lines** — a separate panel under the map/spectrum lists catalog
  rest frequencies that fall in the cube’s frequency coverage (typically
  ~50 MHz for a CLASS OTF cube, so you see the target line plus hyperfine
  structure such as N₂H⁺ 1–0). The list is filled as soon as a cube is
  loaded (it does not depend on the selected pixel). Default is a local
  3 mm list; **CDMS**, **JPL** or **CDMS + JPL** query those catalogues
  directly (``astroquery.linelists``) for rest frequencies that fall in the
  cube’s frequency window (the same window whether the spectrum is plotted in
  velocity or frequency). E_u is computed from the catalogue’s lower-state
  energy, and up to 80 lines below the E_u limit are listed, lowest E_u first.
  If one service is down, its error is shown above the rows of the other. **Source VLSR** Doppler-shifts catalog
  lines onto the observed frame. Optional purple markers overlay the spectrum.
- **Measurements** — peak T, v(peak), ν(peak), centroid, half-max FWHM, integrated
  intensity, RMS, S/N, and sky position. Optional multi-Gaussian fit adds
  per-component peak, v₀, FWHM, σ, and ``∫I dv``. You can type Peak / v₀ /
  FWHM starting values for each component (or lock filled values).
- Species labels come from the **filename + RESTFREQ**, not the CLASS ``LINE``
  keyword (often a leftover backend name such as ``HCN_LSB``).
- **Regions (masks)** — the map tool bar has box, circle/ellipse and
  polygon drawing tools plus an eraser. Shapes stay editable (drag / resize); the
  spectrum, measurements and Gaussian fit use the mean over every pixel whose
  centre lies inside any region. **Add circle** places a circle by numbers:
  centre ΔRA/ΔDec in arcsec (empty = map centre), radius in arcsec or in pc with
  a distance (empty = half the beam BMAJ). **Clear regions** goes back to the
  brightest pixel.

Point **FITS file or directory** at a single cube or a folder of CLASS exports
(e.g. the DR21 IRAM ``grid15`` set), or a CASA image cube
(``*.image.fits`` / ``*.pbcor.fits``, including 4-D cubes with a dummy Stokes
axis). Click **Browse** and pick any FITS file in that folder (the folder path
is filled in). Implementation: ``cube_viewer.py``.

## Interpolation error check

The **Interpolation error** tab compares native and resampled grids:

1. **Original** — native model grid (one point per grid folder)
2. **Interpolated** — log-space resample to the chosen ny × nx mesh
3. **Error** — checkerboard decimation → re-interpolation on the native mesh;
   relative (%) or absolute error in linear flux / abundance units

When both data sources are loaded, each slice plane shows:

- **Abundance / diagnostic** row (from the main HDF5 grid quantity dropdown)
- **Line intensity** row (species + transition)

## CR attenuation profiles

The **CR attenuation** tab shows:

- **ζ<sub>H₂</sub> vs N<sub>H</sub> = N(H) + 2N(H₂)** (or A<sub>V</sub>) from the model
  structure output (`cosray` × 20/13, `cd_prof_h`, `cd_prof_h2`)
- Reference bands 𝓛, 𝓗, 𝓤 with ± factor-of-2
  uncertainty shading, placed at N<sub>H</sub> = 2 N(H₂)
- **Add current model** — snapshot the slider selection; **Add all overlay matches**
  — add every attenuated model at the current density / mass / FUV / metallicity
- Optional log-log extrapolation to 10²⁵ cm⁻² and attenuation-threshold vertical
  line (stopping rate, default 10²⁰ cm⁻²)

No observational data points are plotted. Implementation: ``cr_attenuation.py``.

Controls match the slice tabs (interpolation method, grid size, axis limits) plus
**error decimation factor**, **error metric**, **relative threshold**,
**flux scale**, and **colormap** (default Magma). Error panel uses `RdYlGn_r`
(green = low error, red = high).

## Formation / destruction reactions (chemistry grid)

The per-model `pdrgrid` HDF5 files do **not** contain reaction rates — those live
in a separate **chemistry grid** (files named `chem_Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5`,
e.g. in `chemistrygrid/`). Load that directory on the **Load grids** tab to
enable the **Chemistry** tab:

- **`<species>`: formation reactions**
- **`<species>`: destruction reactions**

Pick the species, set **Top reactions / depth point** (N), and choose a
**Ranking metric**:

| Metric | Description |
|---|---|
| **Fractional contribution** (default) | ∫ 4π r² R dr / ∫ 4π r² R_tot dr: share of the clump-integrated total rate |
| **Rate per X particle** | ∫ 4π r² R dr / ∫ 4π r² n(X) dr = (dN/dt) / N_X (s⁻¹); summed over all reactions it gives 1/τ |

R is the volumetric reaction rate (cm⁻³ s⁻¹) from the `frate_*`/`drate_*` columns. It is
integrated without an extra n(X) weight, because `drate_X` already contains n(X).

The **reaction pathway network** follows the selected model (its top formation /
destruction reactions). A copy of the grid sliders sits above the network, synced
with the main ones, so the model can be changed without scrolling up;
**Upstream / Downstream depth**, **Max nodes** and **Branches / species** set the
extent of the graph.

### Reaction selection (same union as `top_reactions_plot`)

1. At **every depth point**, the N reactions with the largest metric are found.
2. A reaction is plotted if it was in the top-N at **any** depth point.
3. Legend / table ordering follows the selected ranking metric.

Reaction rates are read from
`Local quantities/Chemistry/<species>/{Formation,Destruction} rates <species>`
with labels from the chem-grid metadata table, plotted against A_V
(`Local quantities/Positions`).

## Observational map fit

The **Map fit** tab builds 3-D line-intensity cubes (density × FUV × ζ) and
runs ``fit_fits_maps_to_grids_3d`` against user-supplied observed FITS
maps. Requires:

- Main grid + line intensities loaded

Provide JSON mappings of spectroscopic line keys to FITS paths and optional
per-line errors. Output includes fitted n_H, χ, and ζ maps and reduced χ² maps
(written to the chosen output directory).

With an attenuated line-intensity overlay loaded and **ε_att** ticked, the maps are
refitted with the attenuated grids (``fitted_atten_*``) and
ε_att (Eq. 16) = log₁₀(ζ_atten / ζ_const) is shown and written to
``fitted_<z>_atten_shift_dex.fits``. **Distance / circle radii / scale bar** draw
circles of physical radius (pc) at the map centre and a scale bar on every
fitted map.

Implementation: `grid_fit.py`, `map_fit.py`, `map_fit_extras.py`.

## Overlaying a second (attenuated) grid

An optional **overlay grid** can be loaded from a second directory (e.g. your
attenuated grid). On profile plots it appears as **dashed, translucent lines**
(legend entries suffixed with `· att`). The overlay model is matched to the
current slider selection on **every axis except attenuation**, so the
attenuated counterpart of each model (different `AA` tag) lines up automatically.

The same matching applies to chemistry grids and line-intensity directories. On slice
tabs, overlays additionally enable the triple-panel x-shift view described above.

> The overlay assumes the same chemical network / species ordering as
> the main grid (true for an attenuated version of the same grid).

## Attenuation impact

With the attenuated grid loaded as the overlay (line-intensity overlay for line
intensities), this tab runs the procedure
`plot_attenuation_shift_summary` → `plot_attenuation_fuv_response` →
`reconcile_attenuation_species` and keeps its two questions apart: *which lines
mislead most if attenuation is ignored* (correction priority) and *which lines
are the cleanest CR probes* (line selection).

1. **Breadth** — how many models move per |Δ| bin, median |Δ| per ζ / χ
   (/ n_H) bin, and the winner map (who moves most where).
2. **Magnitude & probe** — ζ error bars (median, p10–p90 spread, observable
   fraction) and the probe plane |R| vs its χ swing.
3. **Verdicts** — responds → CR-led → matters; *Best CR probes* and
   *Lines that mislead most*, verdicts per ζ regime, plus, in 3-D mode, verdicts
   per (n_H, χ, ζ) box, the attenuation impact matrix (with f_att and the clump
   column log N_H per box), the scope of each probe (universal / conditional) and
   the **observer error table** (Δ, Δ_lin, S_ζ, S_χ, r, D per box, CSV / LaTeX
   download). Verdicts: *robust attenuation probe*, *robust, small bias*,
   *G0-modulated*, *no response*.

Every cell with a model counts for every species: unmatched cells and cells
below the detection limit enter at Δ = 0 and R = 0, and no species is excluded.

**Compute** builds both cubes and runs the per-cell shift search (the slow
part); every threshold then applies instantly to the cached cells. Each step
has "How to read" panels, and the **Guide** sub-tab renders
`docs/attenuation_workflow.md`, `docs/probe_margin_explained.md` and `docs/attenuation_good_values.md` (name mapping, 3-D mode, and
the good value of every quantity).

Implementation: `atten_compare.py` (numerics checked by
`tests/test_atten_compare.py`, plus Plotly figures).

## Install & run

```bash
git clone https://github.com/theodorostopkaras/prisma-tool.git
cd prisma-tool
python -m pip install --upgrade -r requirements.txt
python app.py --install      # Linux: add PRISMA, with its icon, to the applications menu
```

Then start PRISMA from the applications menu (search "PRISMA"; right-click the
running app -> *Add to Favorites* to pin it to the dock), or from the terminal:

```bash
python app.py                          # or: python app.py --dir /path/to/pdrgrid_hdf5
```

PRISMA opens in its own app window (via `pywebview`); closing the window quits
PRISMA completely. `--install` points the menu entry at this folder and this
Python, so re-run it if you move the folder or switch Python environments. To
remove it: `rm ~/.local/share/applications/prisma.desktop`.

Use `--browser` to serve PRISMA to a browser tab at <http://127.0.0.1:8050>
instead. That is also the fallback if `pywebview` is missing or the window
crashes, and always the case with `--debug`. Started from the menu (no
terminal to press Ctrl+C in), a fallback PRISMA quits by itself about 3 minutes
after its browser tab is closed.

On the **Load** tab, click **Browse** to
open a system file window and pick any model file in the grid folder (the
folder path is filled in automatically), or paste a path and click
**Load grid** (tick *recursive* to walk sub-folders).

To update, run `git pull` and re-run the `pip install --upgrade` line, so new
or newer packages are picked up (see [Updates](#updates)).

**Required packages** (from `requirements.txt`):

| Package | Used for |
|---|---|
| `dash`, `plotly` | Interactive UI and figures |
| `numpy`, `scipy` | Arrays, interpolation, numerics |
| `pandas` | Per-cell tables of the **Attenuation impact** workflow |
| `h5py` | Model / chemistry HDF5 grids |
| `matplotlib` | Contour paths for map-fit / cube overlays (headless) |
| `astropy` | FITS cubes, WCS, Gaussian line fitting |
| `astroquery` | CDMS / JPL line ID on **Spectra → Cube viewer** |
| `tqdm` | Progress bar during map fitting |
| `Pillow` | Image export |
| `pywebview` | Native app window (Qt backend on Linux) |

The cube viewer’s **local 3 mm catalog** works with the core stack only.
**CDMS / JPL** need `astroquery` and a network connection. To install that
piece on its own:

```bash
python -m pip install 'astroquery>=0.4.8'
```

**Optional: Graphviz** (a system program, not a pip package) lays out the
**Chemistry** reaction pathway network with curved, non-overlapping arrows.
Without it, or for very large networks (≈ > 40 species), the built-in layout
is used.

```bash
conda install graphviz        # or: sudo apt install graphviz / brew install graphviz
```

### Updates

When a newer PRISMA release is published on GitHub, an **Update** badge appears
in the header with the commands to update (`git pull`, then the `pip install`
line above) and a link to the release notes. PRISMA asks GitHub once at start-up;
offline or without a newer release, nothing is shown. Only published releases
count, not individual commits.

**Publishing a release (maintainer):** set `__version__` in `update_check.py`
(e.g. `'1.1.0'`), commit, then `git tag v1.1.0 && git push origin v1.1.0`, and on
GitHub open *Releases -> Draft a new release*, pick the tag, write what changed
and publish. Drafts and pre-releases do not trigger the badge.

### CLI options

| flag | default | meaning |
|---|---|---|
| `--dir` | bundled path | directory of per-model `.hdf5` files |
| `--recursive` | off | search sub-directories too |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8050` | port |
| `--debug` | off | Dash debug / hot-reload |
| `--browser` | off | browser tab instead of the app window |
| `--install` | off | add PRISMA to the Linux applications menu, then exit |

## Code layout

| Module | Role |
|---|---|
| `app.py` | Dash UI, callbacks, plotting |
| `desktop_window.py` | Native app window (pywebview), menu entry (`--install`) |
| `update_check.py` | Version number, GitHub release check |
| `grid_interp.py` | 2-D / 3-D grid resampling |
| `grid_fit.py` | 3-D intensity cubes and FITS map fit |
| `map_fit.py` | Core FITS-to-grid fitting (`fit_fits_maps_to_grids_3d`) |
| `map_fit_extras.py` | Ratio grids, chi² analysis helpers for map fitting |
| `simline_spectra.py` | Model PV FITS spectra and PV diagrams |
| `cr_attenuation.py` | CR attenuation profiles and reference bands |
| `atten_compare.py` | Reference vs attenuated grid comparison (attenuation workflow, x-shift match, Plotly figures) |
| `obs_spectrum_fits.py` | Observational FITS spectra extraction and Gaussian fitting |
| `cube_viewer.py` | CLASS MATRIX / spectral-cube map + spectrum |
| `line_catalog.py` | Local 3 mm line list and optional CDMS / JPL queries |
| `smli_labels.py` | Line transition label formatting |
| `model_config.py` | JSON config scan and Model setup panel |

## Export

- **Download model (ASCII)** dumps the selected model's depth profiles
  (A_V, n_H, T_gas, T_dust and the relative abundance of every species).
- The camera icon on each plot saves a high-resolution PNG.

## Notes

- Profiles are read using fixed HDF5 dataset paths
  (`Local quantities/Positions`, `Local quantities/Gas state`,
  `Local quantities/Densities/Relative densities`) and the
  `Additional output/species involved` list, which aligns 1:1 with the density
  columns. Field column indices are resolved from `Metadata/Metadata` for
  robustness.
- State is held in a single server-side global (last loaded directory wins),
  matching the single-user, local-use design of the original tool. For a
  multi-user deployment, move `_grid` / caches into `flask_caching` keyed by
  session.
