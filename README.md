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
| **Interpolation error** | Native vs resampled grids with decimation error maps (abundance + SIMLINE) |
| **Map fit** | Fit observed FITS maps to the 3-D SIMLINE model grid (KoSens3D) |

## How the grid is discovered

Each model file is named:

```
Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5
            |  |  |  |  |  |
            |  |  |  |  |  +-- AA  attenuation tag
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
`jtemp_Model<tag>_DD_MM_FF_ZZ_CC_AA_<species>.smli` or `jerg_…`).

The **Intensities** tab provides:

- Three **2-D slice maps** (same planes as Grid slices) for one chosen transition
- **Line spectrum** — all transitions for the current slider selection
- Units: **K km/s** (`jtemp`) or **erg s⁻¹ cm⁻² Hz⁻¹** (`jerg`)
- Same interpolation and x-shift controls as Grid slices

An optional **overlay SIMLINE** directory (attenuated `.smli` files) enables the
same triple-panel x-shift layout as for HDF5 grid slices.

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

```bash
cd Kosma-online-tool
pip install -r requirements.txt

# uses the built-in default directory; or pass your own:
python app.py --dir /path/to/pdrgrid_hdf5
```

Then open <http://127.0.0.1:8050>. You can also paste a directory path into the
box at the top of the page and click **Load grid** (tick *recursive* to walk
sub-folders).

**Dependencies:** `dash`, `plotly`, `numpy`, `h5py`, `scipy`, `astropy`.
Optional: `tqdm` (progress bar during map fitting).

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
