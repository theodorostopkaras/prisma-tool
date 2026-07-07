# KOSMA-τ Grid Explorer (`Kosma-online-tool`)

Interactive browser for **KoSens3D** photodissociation-region (PDR) model grids,
inspired by the 3D-PDR Grid Explorer but adapted to the KOSMA-τ output format.

The key differences from the original 3D-PDR tool:

| | 3D-PDR explorer | KOSMA-τ explorer (this tool) |
|---|---|---|
| Input | one combined `grid.h5` (uploaded) | a **directory of one HDF5 file per model** |
| Free axes | FUV × ζ (2-D) | density × FUV × ζ (3-D), plus mass / metallicity / attenuation if they vary |
| Data access | fixed column indices | KOSMA-τ `Metadata/Metadata` table + `species involved` |

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
slider per parameter that actually varies** on disk. The current grid varies
density × FUV × ζ (5 × 6 × 5 = 150 models) at fixed mass and metallicity, so
three sliders appear. If you later add models at different masses (or
metallicities, attenuations), the corresponding slider simply shows up — no code
changes needed.

## What is plotted

For the selected model, the depth profiles are read straight from the HDF5 file
(via the KOSMA-τ metadata table and the `Relative densities` matrix) and shown
as four panels, against either **A_V** or **n_H** (log/linear):

1. Gas & dust temperature
2. H / H₂ relative abundance (with the H→H₂ transition marked)
3. C⁺ / C / CO relative abundance
4. A custom set of species chosen from the dropdown

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

### CLI options

| flag | default | meaning |
|---|---|---|
| `--dir` | bundled path | directory of per-model `.hdf5` files |
| `--recursive` | off | search sub-directories too |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8050` | port |
| `--debug` | off | Dash debug / hot-reload |

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
