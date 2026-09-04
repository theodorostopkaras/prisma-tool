# KOSMA-τ Grid Explorer (`Kosma-online-tool`)

Interactive browser for **KoSens3D** photodissociation-region (PDR) model grids,
inspired by the 3D-PDR Grid Explorer but adapted to the KOSMA-τ output format.

The key differences from the original 3D-PDR tool:

| | 3D-PDR explorer | KOSMA-τ explorer (this tool) |
|---|---|---|
| Input | one combined `grid.h5` (uploaded) | a **directory of one HDF5 file per model** |
| Free axes | FUV × ζ (2-D) | density × FUV × ζ (3-D), plus mass / metallicity / attenuation if they vary |
| Data access | fixed column indices | KOSMA-τ `Metadata/Metadata` table + `species involved` |

## Application tabs

After loading a main grid on **Load grids**, the other tabs become active:

| Tab | Purpose |
|---|---|
| **Load grids** | Main PDR grid, optional overlay / chemistry / SIMLINE directories |
| **Model setup** | PDR config JSON summary (shared vs varying parameters, species network) |
| **Abundance profiles** | Depth profiles vs A_V or n_H |
| **Heating & cooling** | Thermal balance and rate-component breakdowns |
| **Grid slices** | 2-D contour maps over n_H, FUV, and ζ |
| **Chemistry** | Top formation / destruction reactions with contribution metrics |
| **Intensities** | SIMLINE line-intensity slice maps and spectrum |
| **Spectra** | SimLine PV FITS velocity spectra / PV diagrams, plus a CARTA-style observational cube viewer |
| **Interpolation error** | Native vs resampled grids with decimation error maps (abundance + SIMLINE) |
| **CR attenuation** | ζ<sub>H₂</sub> vs N<sub>H₂</sub> profiles with Padovani 𝓛/𝓗/𝓤 reference bands |
| **Map fit** | Fit observed FITS maps to the 3-D SIMLINE model grid (KoSens3D) |

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
KOSMA-τ metadata table and the `Relative densities` matrix) and shown against
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
cloud edge, or **X(species)** — clump-integrated relative abundance matching
KoSens grid plots). Axes use log₁₀ values on linear scales. Use **Contour Z
scale** for log or linear coloring. Mass, metallicity, and attenuation are held
at their grid mid-points when building each slice.

### KoSens-style interpolation

Before plotting, native grid points are resampled to a finer mesh (default
60 × 60) using the same approach as KoSens3D `resampled_grid_data`:

- **Values** interpolated in log₁₀ space (intensities, abundances, etc.)
- **Axes** resampled in log₁₀ parameter space (n_H, χ, ζ)
- **Missing / zero** cells imputed along rows and columns (KoSens flux imputation)
  so sparse SIMLINE or incomplete grids do not produce blocky holes

Controls on **Grid slices** and **Intensities**:

| Control | Meaning |
|---|---|
| Interpolated grid size (ny × nx) | Output mesh resolution |
| X / Y log₁₀ limit (upper) | Optional axis cutoffs |
| Interpolation method | linear, cubic, nearest, or spline |
| Clip to original bounds | Prevent overshoot outside native min/max |

Implementation: `grid_interp.py` (`resample_grid_2d_kosens`, `resample_grid_3d_kosens`).

### Overlay triple panels (x-shift)

When an **overlay grid** is loaded (attenuated models), each slice plane expands
to **three** sub-plots in a row:

1. Main grid (reference)
2. Overlay grid (attenuated)
3. **Horizontal x-shift** — log₁₀(x_match / x_nom) in dex, from intensity matching
   along the x-axis (KoSens `grid_attenuation_compare`)

Use **X-shift scan direction** and **Intensity match tolerance (rtol)** to
control the matching. On the shift panel, **positive** offsets are **red** and
**negative** offsets are **blue** (`RdBu_r`).

## SIMLINE intensities

Load a **SIMLINE directory** on the Load tab (files named
`jtemp_Model…_<species>.smli`, `jerg_…`, or `tau_…` optical-depth tables; ``.smlc``
is accepted for ``tau`` as well).

The **Intensities** tab provides:

- Three **2-D slice maps** (same planes as Grid slices) for one chosen transition
- **Line spectrum** — all transitions for the current slider selection
- Quantities: **K km/s** (`jtemp`), **erg s⁻¹ cm⁻² Hz⁻¹** (`jerg`), or **τ** (`tau`)
- Same interpolation and x-shift controls as Grid slices

An optional **overlay SIMLINE** directory (attenuated `.smli` files) enables the
same triple-panel x-shift layout as for HDF5 grid slices.

## SIMLINE spectra (PV FITS)

When SimLine is run with **`-fits`**, position–velocity cubes are written as
``Model<tag>_DD_MM_FF_ZZ_CC_AA_<species>.<transition>.fits`` (brightness
temperature) and ``…<transition>-tau.fits`` (optical depth) in the same
directory as the ``.smli`` files. The **Spectra** tab reads these cubes
locally via ``simline_spectra.py`` (no KoSens dependency):

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
- **N Gaussians** — multi-component line model (Astropy LevMar, same as KoSens)
- **Optional Peak / v₀ / FWHM** per component — starting values (blank = auto).
  **Lock filled values** holds those parameters instead of fitting them.
  Width is FWHM, not σ.
- **Overlay** — plot the observed spectrum on the SimLine 1-D panel
- **Fit Gaussians** — continuum + sum of Gaussians; shows per-component
  amplitudes, centres, widths, integrated intensities, and total
  ``∫I dv`` with propagated uncertainty

Implementation: ``obs_spectrum_fits.py`` (vendored from KoSens ``spectrum_fits``;
no KoSens import).

### Cube viewer (CARTA-like)

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
  (same spatial-mean option as the SimLine observational overlay).
  **Spectrum X axis** switches between LSR velocity and observed frequency
  (GHz; radio definition).
- **Possible lines** — a separate panel under the map/spectrum lists catalog
  rest frequencies that fall in the cube’s frequency coverage (typically
  ~50 MHz for a CLASS OTF cube, so you see the target line plus hyperfine
  structure such as N₂H⁺ 1–0). The list is filled as soon as a cube is
  loaded (it does not depend on the selected pixel). Default is a local
  3 mm list; **Splatalogue (CDMS/JPL)** queries rest frequencies that fall
  in the cube’s frequency window (the same window whether the spectrum is
  plotted in velocity or frequency). ``astroquery`` ≥ 0.4.8 returns those
  IDs as ``orderedfreq`` in MHz. **Source VLSR** Doppler-shifts catalog
  lines onto the observed frame. Optional purple markers overlay the spectrum.
- **Measurements** — peak T, v(peak), ν(peak), centroid, half-max FWHM, integrated
  intensity, RMS, S/N, and sky position. Optional multi-Gaussian fit adds
  per-component peak, v₀, FWHM, σ, and ``∫I dv``. You can type Peak / v₀ /
  FWHM starting values for each component (or lock filled values).
- Species labels come from the **filename + RESTFREQ**, not the CLASS ``LINE``
  keyword (often a leftover backend name such as ``HCN_LSB``).

Point **FITS file or directory** at a single cube or a folder of CLASS exports
(e.g. the DR21 IRAM ``grid15`` set), or a CASA image cube
(``*.image.fits`` / ``*.pbcor.fits``, including 4-D cubes with a dummy Stokes
axis). Click **Browse** and pick any FITS file in that folder (the folder path
is filled in). Implementation: ``cube_viewer.py``.

## Interpolation error check

The **Interpolation error** tab reproduces KoSens3D
``plot_interpolation_comparison`` / ``resampled_grid_data`` with
``calculate_error=True``:

1. **Original** — native model grid (one point per grid folder)
2. **Interpolated** — KoSens-style resample to the chosen ny × nx mesh
3. **Error** — checkerboard decimation → re-interpolation on the native mesh;
   relative (%) or absolute error in linear flux / abundance units

When both data sources are loaded, each slice plane shows:

- **Abundance / diagnostic** row (from the main HDF5 grid quantity dropdown)
- **SIMLINE intensity** row (species + transition)

## CR attenuation profiles

The **CR attenuation** tab follows the KoSens ``CR_atten_plot`` notebook:

- **ζ<sub>H₂</sub> vs N<sub>H₂</sub>** (or A<sub>V</sub>) from KOSMA structure output
  (`cosray` × 20/13, `cd_prof_h2`)
- **Padovani et al. (2018/2024)** reference bands 𝓛, 𝓗, 𝓤 with ± factor-of-2
  uncertainty shading (on the N<sub>H₂</sub> axis)
- **Add current model** — snapshot the slider selection; **Add all overlay matches**
  — add every attenuated model at the current density / mass / FUV / metallicity
- Optional log-log extrapolation to 10²⁵ cm⁻² and attenuation-threshold vertical
  line (stopping rate, default 10²⁰ cm⁻²)

No observational data points are plotted. Implementation: ``cr_attenuation.py``
(vendored Padovani polynomials from KoSens ``functions_for_cratten``).

Controls match the slice tabs (interpolation method, grid size, axis limits) plus
**error decimation factor**, **error metric**, **relative threshold**,
**flux scale**, and **colormap** (default Magma). Error panel uses `RdYlGn_r`
(green = low error, red = high), as in KoSens.

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
| **Fractional contribution** (default) | Volume-weighted ∫ 4π r² k n dr as % of total (matches `KOSMA_tau_READ_chem.ipynb`) |
| **Mass-weighted rate** | Integrated rate weighted by species density |

### Reaction selection (same union as `top_reactions_plot`)

1. At **every depth point**, the N reactions with the largest metric are found.
2. A reaction is plotted if it was in the top-N at **any** depth point.
3. Legend / table ordering follows the selected ranking metric.

Reaction rates are read from
`Local quantities/Chemistry/<species>/{Formation,Destruction} rates <species>`
with labels from the chem-grid metadata table, plotted against A_V
(`Local quantities/Positions`).

## Observational map fit

The **Map fit** tab builds 3-D SIMLINE intensity cubes (density × FUV × ζ) and
runs ``fit_fits_maps_to_grids_3d`` against user-supplied observed FITS
maps. Requires:

- Main grid + SIMLINE directory loaded

Provide JSON mappings of spectroscopic line keys to FITS paths and optional
per-line errors. Output includes fitted n_H, χ, and ζ maps and reduced χ² maps
(written to the chosen output directory).

Implementation: `grid_fit.py`, `map_fit.py`, `map_fit_extras.py`.

## Overlaying a second (attenuated) grid

An optional **overlay grid** can be loaded from a second directory (e.g. your
attenuated grid). On profile plots it appears as **dashed, translucent lines**
(legend entries suffixed with `· att`). The overlay model is matched to the
current slider selection on **every axis except attenuation**, so the
attenuated counterpart of each model (different `AA` tag) lines up automatically.

The same matching applies to chemistry grids and SIMLINE directories. On slice
tabs, overlays additionally enable the triple-panel x-shift view described above.

> The overlay assumes the same KOSMA-τ chemical network / species ordering as
> the main grid (true for an attenuated version of the same grid).

## Install & run

From the project directory:

```bash
python -m pip install -r requirements.txt
python app.py --dir /path/to/pdrgrid_hdf5
```

Then open <http://127.0.0.1:8050>. On the **Load** tab, click **Browse** to
open a system file window and pick any model file in the grid folder (the
folder path is filled in automatically), or paste a path and click
**Load grid** (tick *recursive* to walk sub-folders).

If you already had the tool installed, re-run the `pip install` line after
pulling updates so new packages (in particular **astroquery**) are picked up.

**Required packages** (from `requirements.txt`):

| Package | Used for |
|---|---|
| `dash`, `plotly` | Interactive UI and figures |
| `numpy`, `scipy` | Arrays, interpolation, numerics |
| `h5py` | KOSMA-τ / chemistry HDF5 grids |
| `astropy` | FITS cubes, WCS, Gaussian line fitting |
| `astroquery` | Splatalogue (CDMS/JPL) line ID on **Spectra → Cube viewer** |
| `tqdm` | Progress bar during map fitting |
| `Pillow` | Image export |

The cube viewer’s **local 3 mm catalog** works with the core stack only.
**Splatalogue** needs `astroquery` and a network connection. To install that
piece on its own:

```bash
python -m pip install 'astroquery>=0.4.7'
```

### CLI options

| flag | default | meaning |
|---|---|---|
| `--dir` | bundled path | directory of per-model `.hdf5` files |
| `--recursive` | off | search sub-directories too |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8050` | port |
| `--debug` | off | Dash debug / hot-reload |

## Code layout

| Module | Role |
|---|---|
| `app.py` | Dash UI, callbacks, plotting |
| `grid_interp.py` | KoSens-aligned 2-D / 3-D grid resampling |
| `grid_fit.py` | 3-D intensity cubes and FITS map fit |
| `map_fit.py` | Core FITS-to-grid fitting (`fit_fits_maps_to_grids_3d`) |
| `map_fit_extras.py` | Ratio grids, chi² analysis helpers for map fitting |
| `simline_spectra.py` | SimLine PV FITS spectra and PV diagrams (local, no KoSens) |
| `cr_attenuation.py` | CR attenuation profiles and Padovani reference bands (local, no KoSens) |
| `obs_spectrum_fits.py` | Observational FITS spectra extraction and Gaussian fitting |
| `cube_viewer.py` | CLASS MATRIX / spectral-cube map + spectrum (CARTA-like) |
| `line_catalog.py` | Local 3 mm line list and optional Splatalogue queries |
| `smli_labels.py` | SIMLINE transition label formatting |
| `model_config.py` | JSON config scan and Model setup panel |

## Export

- **Download model (ASCII)** dumps the selected model's depth profiles
  (A_V, n_H, T_gas, T_dust and the relative abundance of every species).
- The camera icon on each plot saves a high-resolution PNG.

## Notes

- Profiles are read using fixed KOSMA-τ dataset paths
  (`Local quantities/Positions`, `Local quantities/Gas state`,
  `Local quantities/Densities/Relative densities`) and the
  `Additional output/species involved` list, which aligns 1:1 with the density
  columns. Field column indices are resolved from `Metadata/Metadata` for
  robustness.
- State is held in a single server-side global (last loaded directory wins),
  matching the single-user, local-use design of the original tool. For a
  multi-user deployment, move `_grid` / caches into `flask_caching` keyed by
  session.
