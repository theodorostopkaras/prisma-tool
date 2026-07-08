"""
FITS map fitting to 2-D / 3-D PDR model grids (vendored from KoSens3D).
"""

from __future__ import annotations

import numbers
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

from map_fit_extras import (
    BLUE,
    GREEN,
    NC,
    ORANGE,
    RED,
    _chi2_contribution,
    _chi2_plot_surface,
    _infer_axis_display_label,
    chi2_analysis_3d,
    create_transition_ratios,
    create_transition_ratios_3d,
)

# Delta chi^2 thresholds for 2 free parameters (2D confidence regions).
_CHI2_DELTA_2D = (2.30, 6.17, 11.62)  # ~68%, 95%, 99.7%
_CHI2_CONF_LABELS = (r'68\%', r'95\%', r'99.7\%')
#########################################################################################################################################
#                                                       FITS map fitting to grid models
#                                                       Helper functions
#########################################################################################################################################
def convert_intensity_units(data, from_unit, to_unit, species_name=None):
    """
    Convert intensity values between different units.
    
    Supported units:
    - 'K km/s' or 'Kkm/s' or 'K*km/s'
    - 'erg/s/cm^2' or 'erg s-1 cm-2' or 'erg/s/cm2'
    - 'W m-2 sr-1' or 'W m^-2 sr^-1' or 'W/m^2/sr'
    - 'Jy sr-1' or 'Jy/sr' or 'Jy*sr^-1'
    
    Parameters
    ----------
    data : numpy.ndarray or float
        Intensity values to convert (can be scalar or array).
    from_unit : str
        Source unit (will be normalized/cleaned). Can also be read from FITS BUNIT.
    to_unit : str
        Target unit (will be normalized/cleaned).
    species_name : str or tuple(str, int), optional
        Species name for conversion to/from K km/s (requires molecular data).
        Examples: 'CO', 'CI', 'HCO+', 'CO(2-1)', or ('CO', 2).
    
    Returns
    -------
    numpy.ndarray or float
        Converted intensity values (same type as input).
    
    Notes
    -----
    Conversions involving K km/s require knowing the species and upper level.
    The species/transition can be provided as a tuple (name, upper_level) or
    embedded in the string (e.g. 'CO(2-1)'). The function uses the
    intensity_unit_change function which reads molecular data from INP files.
    
    For large arrays, the function optimizes by computing conversions for unique
    values and mapping back to the full array.
    """

    def extract_upper_level_transition(raw_name):
        """
        Extract species name (without transition) and the upper level number.
        
        Parameters
        ----------
        raw_name : str
            Species name in format like 'CO(1-0)', 'C(2-1)', or 'C+(3/2-1/2)'
            
        Returns
        -------
        tuple
            (clean_species_name, upper_level). Returns (raw_name, None)
            if no transition can be extracted. For C+ the upper level defaults to 1.
        """
        try:
            if raw_name is None:
                return None, None
            if isinstance(raw_name, (list, tuple)) and len(raw_name) >= 2:
                return raw_name[0], int(raw_name[1])

            # Extract the transition part from parentheses
            transition_part = raw_name.split('(')[1].split(')')[0]
            upper_level_str = transition_part.split('-')[0]

            # Special case for C+ which only has one upper level transition (3/2)
            if raw_name.startswith('C+'):
                return raw_name.split('(')[0], 1

            # Handle fractional transitions like '3/2'
            if '/' in upper_level_str:
                upper_level_str = upper_level_str.split('/')[0]

            clean_name = raw_name.split('(')[0]
            return clean_name, int(upper_level_str)
        except (IndexError, ValueError):
            print(f"{ORANGE}Warning: Could not extract transition from {raw_name}, using provided value{NC}")
            clean_name = raw_name.split('(')[0] if raw_name else None
            return clean_name, None

    # Handle scalar input
    is_scalar = not isinstance(data, np.ndarray)
    if is_scalar:
        data = np.array([data])
    
    data = np.asarray(data, dtype=float)
    # Normalize unit strings (remove spaces, handle variations)
    def normalize_unit(unit_str):
        if unit_str is None:
            return None
        unit_str = str(unit_str).strip().lower()
        # Remove spaces and handle common variations
        unit_str = unit_str.replace(' ', '').replace('*', '').replace('^', '')
        unit_str = unit_str.replace('-', '').replace('_', '')
        # Handle common unit abbreviations
        unit_str = unit_str.replace('km/s', 'kms').replace('kms', 'kms')
        unit_str = unit_str.replace('km/s', 'kms')
        # Handle erg variations
        if 'erg' in unit_str and 's' in unit_str and 'cm' in unit_str:
            return 'ergscm2'
        # Handle W m^-2 sr^-1 variations
        if 'w' in unit_str and 'm' in unit_str and ('sr' in unit_str or 'ster' in unit_str):
            return 'wm2sr1'
        # Handle Jy sr^-1 variations
        if 'jy' in unit_str and ('sr' in unit_str or 'ster' in unit_str):
            return 'jysr1'
        # Handle K km/s variations
        if 'k' in unit_str and 'km' in unit_str and 's' in unit_str:
            return 'kkms'
        return unit_str
    
    parsed_species_name, parsed_upper_level = extract_upper_level_transition(species_name)

    norm_from_unit = normalize_unit(from_unit)
    norm_to_unit = normalize_unit(to_unit)
    
    # If units are the same, return data unchanged
    if norm_from_unit == norm_to_unit:
        return data
    
    # Convert to common intermediate unit (erg/s/cm^2)
    # First, convert from source unit to erg/s/cm^2
    if norm_from_unit in ['kkm/s', 'kkms', 'kkm/s', 'kkms']:
        # K km/s to erg/s/cm^2 requires species info
        if parsed_species_name is None or parsed_upper_level is None:
            raise ValueError("Conversion from K km/s requires species name and upper level")
        # This is complex - would need to reverse the intensity_unit_change function
        # For now, we'll handle this case separately
        raise NotImplementedError("Conversion from K km/s to other units requires reverse calculation")
    
    elif norm_from_unit in ['erg/s/cm2', 'ergs1cm2', 'ergscm2']:
        # Already in erg/s/cm^2
        data_erg = data
    elif norm_from_unit in ['wm2sr1', 'wm2sr', 'wmsr', 'wm-2sr-1']:
        # W m^-2 sr^-1 to erg/s/cm^2 (integrated over 4π sr)
        # 1 W = 10^7 erg/s, 1 m^2 = 10^4 cm^2 -> 1 W m^-2 = 10^3 erg/s/cm^2
        # Multiply by 4π to remove steradian denominator.
        data_erg = data * 1e3 * 4 * np.pi
    elif norm_from_unit in ['jysr1', 'jysr', 'jysr']:
        # Jy sr^-1 to erg/s/cm^2
        # 1 Jy = 10^-23 erg/s/cm^2/Hz
        # For line intensities, we need frequency information
        # This is complex and requires frequency - for now, raise error
        raise NotImplementedError("Conversion from Jy sr^-1 requires frequency information")
    else:
        raise ValueError(f"Unsupported source unit: {from_unit}. "
                         f"Supported units: K km/s, erg/s/cm^2, W m^-2 sr^-1, Jy sr^-1")
    
    # Now convert from erg/s/cm^2 to target unit
    if norm_to_unit in ['kkm/s', 'kkms', 'kkm/s', 'kkms']:
        # erg/s/cm^2 to K km/s requires species info
        if parsed_species_name is None or parsed_upper_level is None:
            raise ValueError("Conversion to K km/s requires species name and upper level")
        
        raise NotImplementedError(
            'Conversion to K km/s from other units requires molecular line data '
            'not bundled in Kosma-online-tool. Match FITS BUNIT to the SIMLINE idef.'
        )
    
    elif norm_to_unit in ['erg/s/cm2', 'ergs1cm2', 'ergscm2']:
        # Already in erg/s/cm^2
        return data_erg
    
    elif norm_to_unit in ['wm2sr1', 'wm2sr', 'wmsr']:
        # erg/s/cm^2 to W m^-2 sr^-1 (assume emission over 4π sr)
        return data_erg / (1e3 * 4 * np.pi)
    
    elif norm_to_unit in ['jysr1', 'jysr', 'jysr']:
        # erg/s/cm^2 to Jy sr^-1 requires frequency
        raise NotImplementedError("Conversion to Jy sr^-1 requires frequency information")
    
    else:
        raise ValueError(f"Unsupported target unit: {to_unit}. "
                         f"Supported units: K km/s, erg/s/cm^2, W m^-2 sr^-1, Jy sr^-1")
#-----------------------------------------------------------------------------------------------------------
def _find_axis_mesh(grid_dict, axis_name, fallback_names):
    """Find axis mesh in grid dictionary."""
    if axis_name in grid_dict:
        print(f"{GREEN}Found axis name {axis_name} in grid dictionary{NC}")
        return grid_dict[axis_name]
    for name in fallback_names:
        if name in grid_dict:
            print(f"{ORANGE}Provided name {axis_name} not found, using fallback name {name}{NC}")
            return grid_dict[name]
    print(f"{RED}Could not find {axis_name} in grid dictionary{NC}")
    return None
#-----------------------------------------------------------------------------------------------------------
def _normalize_unit(unit_str):
    """Normalize unit string for comparison."""
    if not unit_str:
        return ''
    return unit_str.strip().lower().replace(' ', '').replace('*', '').replace('^', '')
#-----------------------------------------------------------------------------------------------------------
def _get_species_value(line_name, species_info):
    """Get species value from line name or species_info."""
    if species_info and line_name in species_info:
        mapped = species_info[line_name]
        return mapped if isinstance(mapped, (list, tuple)) else mapped
    return line_name
#-----------------------------------------------------------------------------------------------------------
def _read_fits_2d(fits_path):
    """Read 2D data from a FITS file.

    Some workflows store already-integrated maps with extra singleton dimensions
    (e.g. shape ``(1, ny, nx)``). In that case we squeeze the singleton axes
    and return a true 2D array. If the data still has more than 2 dimensions
    after squeezing, we raise to avoid silently using an arbitrary slice of a
    real cube.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    if data is None:
        raise ValueError(f"FITS file at {fits_path!r} has no data in primary HDU.")

    data = np.asarray(data)
    if data.ndim != 2:
        squeezed = np.squeeze(data)
        if squeezed.ndim != 2:
            raise ValueError(
                f"FITS file at {fits_path!r} is not 2D integrated data: "
                f"got shape {data.shape}. After squeezing singleton axes, got "
                f"shape {squeezed.shape}."
            )
        data = squeezed

    return data, header
#-----------------------------------------------------------------------------------------------------------
def _convert_units_if_needed(data, from_unit, to_unit, line_name, species_info):
    """Convert intensity units if needed.

    Returns
    -------
    tuple
        (converted_data, did_convert) where `did_convert` is True only if
        conversion was attempted (units differ) and succeeded (no exception).
    """
    if not from_unit or not to_unit:
        print(f"{ORANGE}Warning: No units provided for {line_name}. Using original units.{NC}")
        return data, False
    if _normalize_unit(from_unit) == _normalize_unit(to_unit):
        print(f"{GREEN}Units are the same for {line_name}. Using original units.{NC}")
        return data, False
    try:
        species_value = _get_species_value(line_name, species_info)
        converted = convert_intensity_units(
            data, from_unit, to_unit, species_name=species_value
        )
        return converted, True
    except Exception as e:
        print(f"{ORANGE}Warning: Could not convert units for {line_name}.\n" 
        f"Check from_unit and to_unit inputs of convert_intensity_units function: {e}\n"
              f"Using original units.{NC}")
        return data, False
#-----------------------------------------------------------------------------------------------------------
def _construct_axis_from_header(header, shape, axis_name, axis_type):
    """Try to construct axis data from FITS header keywords."""
    ny, nx = shape
    axis_keywords = [
        f'{axis_name.upper()}_AXIS', f'{axis_name.upper()}_VALUES',
        f'AXIS_{axis_type.upper()}', f'{axis_type.upper()}_AXIS'
    ]
    for keyword in axis_keywords:
        if keyword in header:
            try:
                axis_data = header[keyword]
                if isinstance(axis_data, (list, np.ndarray)):
                    axis_data = np.array(axis_data)
                    if axis_data.ndim == 1:
                        if axis_type == 'x':
                            axis_data, _ = np.meshgrid(axis_data, np.arange(ny), indexing='ij')
                        else:
                            _, axis_data = np.meshgrid(np.arange(nx), axis_data, indexing='ij')
                return axis_data
            except Exception:
                continue
    return None
#-----------------------------------------------------------------------------------------------------------
def _ensure_2d_meshgrids(x_data, y_data, grid_shape):
    """Ensure axis data are 2D meshgrids matching grid shape."""
    # ny_grid, nx_grid = grid_shape
    
    if x_data.ndim == 1:
        if y_data.ndim == 1:
            Y_mesh, X_mesh = np.meshgrid(y_data, x_data, indexing='ij')
        else:
            Y_mesh, X_mesh = np.meshgrid(np.unique(y_data), x_data, indexing='ij')
        return X_mesh, Y_mesh
    elif y_data.ndim == 1:
        Y_mesh, X_mesh = np.meshgrid(y_data, np.unique(x_data), indexing='ij')
        return X_mesh, Y_mesh
    else:
        if x_data.shape == grid_shape:
            return x_data, y_data
        elif x_data.shape == grid_shape[::-1]:
            return x_data.T, y_data.T
        else:
            return x_data, y_data
#-----------------------------------------------------------------------------------------------------------
def _extract_spatial_coordinates(header, shape):
    """Extract spatial coordinates from FITS header using WCS."""
    try:
        wcs = WCS(header)
        if wcs.naxis > 2:
            wcs_2d = wcs.celestial
        else:
            wcs_2d = wcs
        
        ny, nx = shape
        y_pix, x_pix = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        
        try:
            if wcs_2d.naxis == 2:
                world_coords = wcs_2d.pixel_to_world_values(x_pix, y_pix)
                if isinstance(world_coords, tuple) and len(world_coords) >= 2:
                    x_coords = world_coords[0]
                    y_coords = world_coords[1]
                else:
                    x_coords = x_pix
                    y_coords = y_pix
            else:
                x_coords = x_pix
                y_coords = y_pix
        except Exception:
            x_coords = x_pix
            y_coords = y_pix
    except Exception:
        ny, nx = shape
        y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    return x_coords, y_coords
#-----------------------------------------------------------------------------------------------------------
def _create_output_header(base_header, param_name, param_units, param_map):
    """Create output FITS header with parameter info."""
    header = base_header.copy()
    if 'CTYPE1' not in header or 'CTYPE2' not in header:
        print(f"{ORANGE}Warning: CTYPE1/CTYPE2 not found. WCS may not be preserved.{NC}")
    header['BUNIT'] = param_units if param_units else 'dimensionless'
    header['TITLE'] = f'Fitted {param_name} map'
    header['DATAMIN'] = np.nanmin(param_map)
    header['DATAMAX'] = np.nanmax(param_map)
    header['COMMENT'] = f'Fitted parameter: {param_name}'
    return header
#-----------------------------------------------------------------------------------------------------------
def _load_grid_from_fits(fits_path, x_axis_name, y_axis_name):
    """Load grid data from FITS file."""
    with fits.open(fits_path) as grid_fits:
        grid_array, grid_header = _read_fits_2d(fits_path)
        ny_grid, nx_grid = grid_array.shape
        
        x_axis_data = y_axis_data = None
        if len(grid_fits) > 1:
            for ext_idx in range(1, len(grid_fits)):
                ext_name = grid_fits[ext_idx].name.upper()
                ext_data = grid_fits[ext_idx].data
                if ext_data is None:
                    continue
                if ext_data.ndim > 2:
                    ext_data = ext_data[0] if ext_data.ndim == 3 else ext_data[0, 0]
                
                if ext_data.shape == (nx_grid, ny_grid):
                    ext_data = ext_data.T
                elif ext_data.shape != (ny_grid, nx_grid) and ext_data.ndim != 1:
                    print(f"{ORANGE}Warning: Extension {ext_name} shape mismatch. Skipping.{NC}")
                    continue
                
                if x_axis_name.upper() in ext_name or 'XAXIS' in ext_name:
                    x_axis_data = ext_data
                elif y_axis_name.upper() in ext_name or 'YAXIS' in ext_name:
                    y_axis_data = ext_data
        
        if x_axis_data is None:
            x_axis_data = _construct_axis_from_header(grid_header, (ny_grid, nx_grid), x_axis_name, 'x')
        if y_axis_data is None:
            y_axis_data = _construct_axis_from_header(grid_header, (ny_grid, nx_grid), y_axis_name, 'y')
        
        if x_axis_data is None or y_axis_data is None:
            print(f"{ORANGE}Warning: Creating default meshgrids{NC}")
            Y_mesh, X_mesh = np.meshgrid(np.arange(ny_grid), np.arange(nx_grid), indexing='ij')
            x_axis_data = X_mesh if x_axis_data is None else x_axis_data
            y_axis_data = Y_mesh if y_axis_data is None else y_axis_data
        
        x_axis_data, y_axis_data = _ensure_2d_meshgrids(x_axis_data, y_axis_data, (ny_grid, nx_grid))
        
        if x_axis_data.shape != (ny_grid, nx_grid) or y_axis_data.shape != (ny_grid, nx_grid):
            raise ValueError(f"Axis shape mismatch: x={x_axis_data.shape}, y={y_axis_data.shape}, grid=({ny_grid}, {nx_grid})")
        
        return {
            'grid': grid_array,
            x_axis_name: x_axis_data,
            y_axis_name: y_axis_data
        }
#-----------------------------------------------------------------------------------------------------------
def _is_scalar_error(error_source):
    """True if *error_source* is a single constant error (not a map)."""
    if isinstance(error_source, numbers.Number):
        return True
    if isinstance(error_source, np.ndarray) and error_source.ndim == 0:
        return True
    return False
#-----------------------------------------------------------------------------------------------------------
def _normalize_observed_error_map(error_source,
                                  data,
                                  ny,
                                  nx,
                                  error_fraction,
                                  line_name,
                                  target_units=None,
                                  species_info=None):
    """
    Build a 2D error map from a FITS path, array, scalar, or default fraction.

    Scalars (int/float) are broadcast to the full map shape. When no error is
    supplied, ``error_fraction * |data|`` is used (default 20%).
    """
    data = np.asarray(data, dtype=float)

    if error_source is None:
        err = np.abs(data) * error_fraction
        return np.maximum(err, np.nanmax(data) * 1e-6)

    if isinstance(error_source, str):
        err_data, err_header = _read_fits_2d(error_source)
        if err_data.shape != (ny, nx):
            raise ValueError(
                f"Error FITS for {line_name} has shape {err_data.shape}, "
                f"expected ({ny}, {nx})"
            )
        err_unit = err_header.get('BUNIT', '')
        if err_unit and target_units:
            err_data, _ = _convert_units_if_needed(
                err_data, err_unit, target_units, line_name, species_info
            )
        return np.asarray(err_data, dtype=float)

    if _is_scalar_error(error_source):
        err_val = float(error_source)
        if not np.isfinite(err_val) or err_val <= 0:
            raise ValueError(
                f"Scalar error for {line_name} must be a positive finite number, "
                f"got {error_source!r}"
            )
        return np.full((ny, nx), err_val, dtype=float)

    err_arr = np.asarray(error_source, dtype=float)
    if err_arr.shape != (ny, nx):
        raise ValueError(
            f"Error array for {line_name} has shape {err_arr.shape}, "
            f"expected ({ny}, {nx}) or a scalar"
        )
    return err_arr
#-----------------------------------------------------------------------------------------------------------
def _read_observed_data(observed_fits_files, target_units, species_info, error_fraction, obs_errors):
    """Read and process all observed FITS files."""
    observed_data = {}
    observed_errors = {}
    
    first_line = list(observed_fits_files.keys())[0]
    first_data, first_header = _read_fits_2d(observed_fits_files[first_line])
    if first_data.ndim > 2:
        print(f"Warning: FITS file has {first_data.ndim} dimensions. Using first 2D slice.")
    ny, nx = first_data.shape
    
    print(f'{NC}-'*30)
    print(f"{BLUE}Reading observed FITS files{NC}")
    print('-'*30)
    
    for line_name, fits_path in observed_fits_files.items():
        data, header = _read_fits_2d(fits_path)
        if data.shape != (ny, nx):
            raise ValueError(f"FITS file for {line_name} has shape {data.shape}, expected ({ny}, {nx})")
        
        fits_unit = header.get('BUNIT', '')
        if fits_unit and target_units:
            if _normalize_unit(fits_unit) != _normalize_unit(target_units):
                print(f"{GREEN}_Converting {line_name} from {fits_unit} to {target_units}{NC}")
            data, did_convert = _convert_units_if_needed(
                data, fits_unit, target_units, line_name, species_info
            )
            if did_convert:
                print(f"{GREEN}Successfully converted {line_name} to {target_units}{NC}")
        else:
            print(f"{ORANGE}Warning: BUNIT not found in the fits header for {line_name}. Units are not converted.")
        
        observed_data[line_name] = data

        error_source = obs_errors.get(line_name) if obs_errors is not None else None
        observed_errors[line_name] = _normalize_observed_error_map(
            error_source,
            data,
            ny,
            nx,
            error_fraction,
            line_name,
            target_units=target_units,
            species_info=species_info,
        )
    
    return observed_data, observed_errors, first_header, ny, nx
#-----------------------------------------------------------------------------------------------------------
def _find_matching_axis_key(desired_name, available_keys, fallback_names):
    """Find matching axis key from available keys."""
    if desired_name in available_keys:
        return desired_name
    for fallback in fallback_names:
        if fallback in available_keys:
            return fallback
    return available_keys[0] if available_keys else desired_name
#-----------------------------------------------------------------------------------------------------------
def _create_ratios_from_data(line_names_only, observed_data, observed_errors, first_header, 
                              ny, nx, grid_dicts, x_axis_name, y_axis_name, error_fraction):
    """Create transition ratios from observed data."""
    x_spatial, y_spatial = _extract_spatial_coordinates(first_header, (ny, nx))
    
    if line_names_only and line_names_only[0] in grid_dicts:
        first_grid = grid_dicts[line_names_only[0]]
        axis_keys = [key for key in first_grid.keys() if key != 'grid']
        x_axis_key = _find_matching_axis_key(
            x_axis_name, axis_keys,
            ['densities', 'crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine']
        )
        y_axis_key = _find_matching_axis_key(
            y_axis_name, axis_keys,
            ['crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine', 'densities']
        )
    else:
        x_axis_key = x_axis_name
        y_axis_key = y_axis_name
    
    observed_data_dict = {}
    for line_name in line_names_only:
        observed_data_dict[line_name] = {
            'grid': observed_data[line_name],
            x_axis_key: x_spatial,
            y_axis_key: y_spatial
        }
    
    print(f"{GREEN}Created observed data dictionary with spatial coordinates (shape: {ny}, {nx}){NC}")
    print(f"{GREEN}Using axis keys: {x_axis_key} and {y_axis_key} to match grid structure{NC}")

    ratio_dicts = create_transition_ratios(observed_data_dict, line_names_only)
    ratio_data = {}
    ratio_errors = {}
    
    if ratio_dicts:
        print(f"{GREEN}Created {len(ratio_dicts)} ratio(s) from observed data:{NC}")
        for ratio_name, ratio_dict in ratio_dicts.items():
            ratio_data[ratio_name] = ratio_dict['grid']
            
            if '/' in ratio_name:
                num_name, den_name = ratio_name.split('/', 1)
                if num_name in observed_data and den_name in observed_data:
                    num_data = observed_data[num_name]
                    den_data = observed_data[den_name]
                    num_err = observed_errors[num_name]
                    den_err = observed_errors[den_name]
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rel_err_num = np.divide(num_err, np.abs(num_data), 
                                                out=np.full_like(num_data, np.inf), 
                                                where=(num_data != 0))
                        rel_err_den = np.divide(den_err, np.abs(den_data), 
                                                out=np.full_like(den_data, np.inf), 
                                                where=(den_data != 0))
                        ratio_err = np.abs(ratio_data[ratio_name]) * np.sqrt(rel_err_num**2 + rel_err_den**2)
                        ratio_err = np.where(np.isfinite(ratio_err), ratio_err, 
                                            np.abs(ratio_data[ratio_name]) * error_fraction)
                        ratio_errors[ratio_name] = np.maximum(ratio_err, np.nanmax(ratio_data[ratio_name]) * 1e-6)
                else:
                    err = np.abs(ratio_data[ratio_name]) * error_fraction
                    ratio_errors[ratio_name] = np.maximum(err, np.nanmax(ratio_data[ratio_name]) * 1e-6)
            else:
                err = np.abs(ratio_data[ratio_name]) * error_fraction
                ratio_errors[ratio_name] = np.maximum(err, np.nanmax(ratio_data[ratio_name]) * 1e-6)
    else:
        print(f"{ORANGE}No ratios could be created from the provided lines.{NC}")
    
    return ratio_data, ratio_errors
#-----------------------------------------------------------------------------------------------------------
def _compute_chi2_grid(obs_values, obs_errs, grid_arrays):
    """Build the combined chi-squared grid over model parameter space."""
    if not obs_values:
        return None

    first_name = next(iter(obs_values.keys()))
    if first_name not in grid_arrays:
        return None

    chi2_grid = np.zeros_like(grid_arrays[first_name], dtype=float)
    for data_name, obs_val in obs_values.items():
        if data_name not in grid_arrays:
            continue
        model_grid = grid_arrays[data_name]
        obs_err = obs_errs[data_name]
        chi2_grid += _chi2_contribution(model_grid, obs_val, obs_err)

    return chi2_grid
#-----------------------------------------------------------------------------------------------------------
def _median_pixel_observations(all_observed_data,
                               all_observed_errors,
                               ref_mask,
                               data_names,
                               grid_arrays):
    """Median observed values/errors over the reference footprint (for summary plots)."""
    obs_values = {}
    obs_errs = {}
    for name in data_names:
        if name not in grid_arrays:
            continue
        if name not in all_observed_data or name not in all_observed_errors:
            continue

        data = np.asarray(all_observed_data[name], dtype=float)[ref_mask]
        err = np.asarray(all_observed_errors[name], dtype=float)[ref_mask]
        valid = np.isfinite(data) & np.isfinite(err) & (err > 0)
        if np.count_nonzero(valid) < 1:
            continue

        obs_values[name] = float(np.median(data[valid]))
        obs_errs[name] = float(np.median(err[valid]))

    return obs_values, obs_errs
#-----------------------------------------------------------------------------------------------------------
def plot_chi2_corner(obs_values,
                     obs_errs,
                     grid_arrays,
                     X_mesh,
                     Y_mesh,
                     x_axis_name,
                     y_axis_name,
                     x_axis_units='',
                     y_axis_units='',
                     *,
                     fig_dir_PATH=False,
                     or_PATH=False,
                     output_prefix='fitted',
                     plot_format='pdf',
                     font_size=19,
                     fig_size=None,
                     cmap='viridis_r',
                     show_confidence_contours=True,
                     subtitle=''):
    """
    Corner-style plot of the chi-squared grid in parameter space.

    Layout matches the standard KoSens3D chi-squared figures: a central 2D
    :math:`\\chi^2` surface in log-parameter space with marginalized 1D
    profiles on the top and right, confidence contours, and a summary of the
    observed constraints used in the fit.

    Parameters
    ----------
    obs_values, obs_errs : dict
        Per-line observed intensities and errors at one sky position (or a
        representative spectrum, e.g. median over the map).
    grid_arrays : dict
        Model intensity grids on the parameter mesh.
    X_mesh, Y_mesh : numpy.ndarray
        2D meshgrids of the fitted parameters.
    subtitle : str, optional
        Extra text in the figure title (e.g. pixel index or ``median footprint``).

    Returns
    -------
    str or None
        Path to the saved figure, or None if not saved.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    chi2_grid = _compute_chi2_grid(obs_values, obs_errs, grid_arrays)
    if chi2_grid is None:
        print(f"{ORANGE}Warning: could not build chi^2 grid; skipping corner plot.{NC}")
        return None

    chi2_min = float(np.nanmin(chi2_grid))
    min_y_idx, min_x_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    best_x = float(X_mesh[min_y_idx, min_x_idx])
    best_y = float(Y_mesh[min_y_idx, min_x_idx])

    x_axis = np.asarray(X_mesh[0, :], dtype=float)
    y_axis = np.asarray(Y_mesh[:, 0], dtype=float)
    if np.any(x_axis <= 0) or np.any(y_axis <= 0):
        print(
            f"{ORANGE}Warning: chi^2 corner plot requires positive axis values; "
            f"skipping.{NC}"
        )
        return None

    chi2_vs_x = np.nanmin(chi2_grid, axis=0)
    chi2_vs_y = np.nanmin(chi2_grid, axis=1)

    log_x = np.log10(x_axis)
    log_y = np.log10(np.asarray(y_axis, dtype=float))
    log_x_mesh = np.log10(np.asarray(X_mesh, dtype=float))
    log_y_mesh = np.log10(np.asarray(Y_mesh, dtype=float))

    x_label = _infer_axis_display_label(x_axis_name, axis_units=x_axis_units)
    y_label = _infer_axis_display_label(y_axis_name, axis_units=y_axis_units)

    if fig_size is None:
        fig_size = (14, 12)

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(
        2, 2, figure=fig, width_ratios=[4, 1.15], height_ratios=[1.15, 4],
        hspace=0.06, wspace=0.06,
    )
    ax_x = fig.add_subplot(gs[0, 0])
    ax_2d = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_2d)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_axis_off()

    contourf, delta_chi2, _ = _chi2_plot_surface(
        ax_2d, log_x_mesh, log_y_mesh, chi2_grid, chi2_min, cmap, cap=25.0,
    )

    legend_elements = []
    if show_confidence_contours:
        colors_conf = ['white', 'cyan', 'yellow']
        linestyles = ['-', '--', ':']
        for i, (delta, label_pct) in enumerate(zip(_CHI2_DELTA_2D, _CHI2_CONF_LABELS)):
            ax_2d.contour(
                log_x_mesh,
                log_y_mesh,
                delta_chi2,
                levels=[delta],
                colors=[colors_conf[i]],
                linewidths=2.0,
                linestyles=linestyles[i],
            )
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=colors_conf[i],
                    linewidth=2.0,
                    linestyle=linestyles[i],
                    label=rf'{label_pct} ($\Delta\chi^2$={delta:.1f})',
                )
            )

    ax_2d.scatter(
        np.log10(best_x),
        np.log10(best_y),
        marker='*',
        s=350,
        c='red',
        edgecolors='white',
        linewidths=2.0,
        zorder=10,
    )
    legend_elements.append(
        Line2D(
            [0], [0],
            marker='*',
            color='w',
            markerfacecolor='red',
            markersize=14,
            label='Best fit',
        )
    )
    ax_2d.legend(handles=legend_elements, loc='upper right', fontsize=font_size - 4, frameon=True)

    rel_x = chi2_vs_x - chi2_min
    rel_y = chi2_vs_y - chi2_min
    ax_x.plot(log_x, rel_x, color='#2E86AB', lw=2.0)
    ax_x.axvline(np.log10(best_x), color='red', linestyle='--', linewidth=1.5)
    ax_x.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, label=r'$\Delta\chi^2=1$')
    ax_x.scatter(np.log10(best_x), 0.0, marker='*', s=120, c='red', edgecolors='white', zorder=5)

    ax_y.plot(rel_y, log_y, color='#C73E1D', lw=2.0)
    ax_y.axhline(np.log10(best_y), color='red', linestyle='--', linewidth=1.5)
    ax_y.axvline(1.0, color='gray', linestyle=':', linewidth=1.0)
    ax_y.scatter(0.0, np.log10(best_y), marker='*', s=120, c='red', edgecolors='white', zorder=5)

    profile_ymax = min(
        15.0,
        max(float(np.nanmax(rel_x)), float(np.nanmax(rel_y))) + 2.0,
    )
    profile_ylim = (-0.5, profile_ymax)
    ax_x.set_ylim(profile_ylim)
    ax_y.set_xlim(profile_ylim)

    ax_x.set_ylabel(r'$\Delta\chi^2$', fontsize=font_size)
    ax_x.tick_params(axis='x', labelbottom=False)
    ax_y.set_xlabel(r'$\Delta\chi^2$', fontsize=font_size)
    ax_y.tick_params(axis='y', labelleft=False)
    ax_2d.set_xlabel(rf'$\log_{{10}}$({x_label})', fontsize=font_size)
    ax_2d.set_ylabel(rf'$\log_{{10}}$({y_label})', fontsize=font_size)

    cbar = fig.colorbar(
        contourf,
        ax=ax_2d,
        location='right',
        pad=0.10,
        fraction=0.05,
        shrink=0.85,
    )
    cbar.set_label(r'$\Delta\chi^2$', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size - 2)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    title = (
        rf'$\chi^2$ corner: {x_axis_name} vs {y_axis_name}'
        rf'\nBest fit: {x_axis_name}={best_x:.2e}, {y_axis_name}={best_y:.2e}, '
        rf'$\chi^2_{{\min}}$={chi2_min:.2f}'
    )
    if subtitle:
        title += rf' ({subtitle})'
    ax_2d.set_title(title, fontsize=font_size, weight='bold')

    obs_lines = []
    for name, obs_val in obs_values.items():
        if name not in obs_errs:
            continue
        obs_err = obs_errs[name]
        if name in grid_arrays:
            model_at_best = float(grid_arrays[name][min_y_idx, min_x_idx])
            residual = (model_at_best - obs_val) / obs_err
            obs_lines.append(
                f'{name}: obs={obs_val:.3e} +/- {obs_err:.3e}\n'
                f'    model={model_at_best:.3e}, '
                f'residual={residual:+.2f} sigma'
            )
        else:
            obs_lines.append(f'{name}: obs={obs_val:.3e} ± {obs_err:.3e}')

    info_text = (
        rf'$\chi^2_{{\min}}$ = {chi2_min:.3f}' + '\n'
        rf'{x_axis_name} = {best_x:.3e}' + '\n'
        rf'{y_axis_name} = {best_y:.3e}' + '\n'
        + '-' * 22 + '\n'
        + '\n'.join(obs_lines)
    )
    ax_info.text(
        0.02, 0.98, info_text,
        transform=ax_info.transAxes,
        va='top', ha='left',
        fontsize=font_size - 5,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95),
    )

    for ax in (ax_x, ax_2d, ax_y):
        ax.tick_params(axis='both', which='major', labelsize=font_size - 1, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.90)

    plot_path = None
    if fig_dir_PATH and or_PATH:
        os.chdir(fig_dir_PATH)
        plot_path = f'{output_prefix}_chi2_corner.{plot_format}'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        os.chdir(or_PATH)
        print(
            f"{GREEN}Saved chi^2 corner plot: "
            f"{os.path.join(fig_dir_PATH, plot_path)}{NC}"
        )
    else:
        plt.show()
    return plot_path
#-----------------------------------------------------------------------------------------------------------
def _fit_pixel(obs_values, obs_errs, grid_arrays, X_mesh, Y_mesh):
    """Fit a single pixel to find best parameters."""
    chi2_grid = _compute_chi2_grid(obs_values, obs_errs, grid_arrays)
    if chi2_grid is None:
        return np.nan, np.nan

    chi2_min = np.nanmin(chi2_grid)
    chi2_tolerance_exact = max(chi2_min * 1e-12, np.finfo(float).eps * 10)
    exact_min_indices = np.where(np.abs(chi2_grid - chi2_min) <= chi2_tolerance_exact)

    if len(exact_min_indices[0]) > 0:
        x_vals_exact = X_mesh[exact_min_indices]
        y_vals_exact = Y_mesh[exact_min_indices]

        y_vals_unique = np.unique(y_vals_exact)
        is_flat_in_y = len(y_vals_unique) == 1 or (np.nanmax(y_vals_exact) - np.nanmin(y_vals_exact)) < np.finfo(float).eps * 100

        if is_flat_in_y and len(exact_min_indices[0]) > 1:
            chi2_tolerance_good = chi2_min * 0.01
            good_indices = np.where(chi2_grid <= chi2_min + chi2_tolerance_good)

            if len(good_indices[0]) > len(exact_min_indices[0]):
                x_vals_good = X_mesh[good_indices]
                y_vals_good = Y_mesh[good_indices]
                weights = 1.0 / (chi2_grid[good_indices] + 1e-10)
                weights = weights / np.sum(weights)
                return np.sum(weights * x_vals_good), np.sum(weights * y_vals_good)
            else:
                weights = 1.0 / (chi2_grid[exact_min_indices] + 1e-10)
                weights = weights / np.sum(weights)
                return np.sum(weights * x_vals_exact), np.sum(weights * y_vals_exact)
        elif len(exact_min_indices[0]) > 1:
            weights = 1.0 / (chi2_grid[exact_min_indices] + 1e-10)
            weights = weights / np.sum(weights)
            return np.sum(weights * x_vals_exact), np.sum(weights * y_vals_exact)
        else:
            min_y_idx = exact_min_indices[0][0]
            min_x_idx = exact_min_indices[1][0]
            return X_mesh[min_y_idx, min_x_idx], Y_mesh[min_y_idx, min_x_idx]
    else:
        min_y_idx, min_x_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        return X_mesh[min_y_idx, min_x_idx], Y_mesh[min_y_idx, min_x_idx]
#-----------------------------------------------------------------------------------------------------------
def _plot_fitted_parameter_kde(
        param_map,
        param_name,
        param_units='',
        *,
        use_log_scale=False,
        fig_dir_PATH=False,
        or_PATH=False,
        output_prefix='fitted',
        plot_format='pdf',
        font_size=19,
        fig_size=None,
        kde_color='#2E86AB',
        param_label=None):
    """Plot a 1D KDE of finite fitted parameter values from a parameter map."""
    display_name = param_label if param_label is not None else param_name
    vals = np.asarray(param_map, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if use_log_scale:
        vals = vals[vals > 0]
        if vals.size < 2:
            print(
                f"{ORANGE}Warning: not enough positive finite values for KDE of "
                f"{param_name}; skipping KDE plot.{NC}"
            )
            return None
        plot_vals = np.log10(vals)
        inner = display_name.strip('$') if display_name.startswith('$') and display_name.endswith('$') else display_name
        x_label = rf'$\log_{{10}}({inner})$'
        if param_units:
            x_label += rf' [{param_units}]'
    else:
        if vals.size < 2:
            print(
                f"{ORANGE}Warning: not enough finite values for KDE of "
                f"{param_name}; skipping KDE plot.{NC}"
            )
            return None
        plot_vals = vals
        if not display_name.startswith('$'):
            display_name = _infer_axis_display_label(param_name, axis_units=param_units)
        x_label = display_name

    kde = gaussian_kde(plot_vals)
    x_grid = np.linspace(plot_vals.min(), plot_vals.max(), 256)
    density = kde(x_grid)

    if fig_size is None:
        fig_size = (8, 5)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.fill_between(x_grid, density, alpha=0.35, color=kde_color)
    ax.plot(x_grid, density, color=kde_color, lw=2.2)
    ax.axvline(
        np.median(plot_vals),
        color='black',
        ls='--',
        lw=1.8,
        label=rf"median = {np.median(plot_vals):.3g}",
    )
    ax.axvline(
        np.mean(plot_vals),
        color='0.45',
        ls=':',
        lw=1.8,
        label=rf"mean = {np.mean(plot_vals):.3g}",
    )
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel('Probability density', fontsize=font_size)
    ax.set_title(
        rf"KDE: {display_name} ($N={plot_vals.size}$ pixels)",
        fontsize=font_size + 1,
        weight='bold',
    )
    ax.legend(loc='best', fontsize=font_size - 2, frameon=True)
    ax.tick_params(axis='both', which='major', labelsize=font_size, pad=8, width=2.5, length=8)
    for spine_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine_name].set_linewidth(2.5)
        ax.spines[spine_name].set_color('black')
    plt.tight_layout()

    kde_plot_path = None
    if fig_dir_PATH and or_PATH:
        os.chdir(fig_dir_PATH)
        kde_plot_path = f'{output_prefix}_{param_name}_kde.{plot_format}'
        plt.savefig(kde_plot_path, dpi=300, bbox_inches='tight')
        os.chdir(or_PATH)
        print(
            f"{GREEN}Saved KDE plot: {os.path.join(fig_dir_PATH, kde_plot_path)}{NC}"
        )
    else:
        plt.show()
    return kde_plot_path
#-----------------------------------------------------------------------------------------------------------
def plot_fitted_parameter_maps(x_param_map,
                               y_param_map,
                               x_axis_name,
                               y_axis_name,
                               x_axis_units='',
                               y_axis_units='',
                               z_param_map=None,
                               z_axis_name=None,
                               z_axis_units='',
                               z_axis_label=None,
                               x_axis_label=None,
                               y_axis_label=None,
                               set_x_name=None,
                               set_y_name=None,
                               set_z_name=None,
                               set_mass_name='M',
                               fits_header=None,
                               fig_dir_PATH=False,
                               or_PATH=False,
                               output_prefix='fitted',
                               plot_format='pdf',
                               font_size=False,
                               fig_size=None,
                               cmap='plasma',
                               use_log_colorbar=False,
                               plot_contours=False,
                               plot_specific_contours=False,
                               plot_kde=True):
    """
    Create colormap plots of fitted parameter maps with WCS coordinates (RA/DEC).

    When ``plot_kde`` is True, also creates one KDE plot per parameter map showing
    the distribution of fitted values over valid pixels.
    
    Parameters
    ----------
    x_param_map : numpy.ndarray
        2D array of fitted x-axis parameter values.
    y_param_map : numpy.ndarray
        2D array of fitted y-axis parameter values.
    x_axis_name : str
        Name of the x-axis parameter.
    y_axis_name : str
        Name of the y-axis parameter.
    x_axis_units : str, optional
        Units for x-axis parameter.
    y_axis_units : str, optional
        Units for y-axis parameter.
    z_param_map : numpy.ndarray, optional
        2D array of fitted third parameter (e.g. clump mass).
    z_axis_name : str, optional
        Name of the third fitted parameter.
    z_axis_units : str, optional
        Units for the third parameter.
    fits_header : astropy.io.fits.Header, optional
        FITS header with WCS information for coordinate display.
    fig_dir_PATH : str, optional
        Directory path to save plots. If False, plots are displayed but not saved.
    or_PATH : str, optional
        Original working directory path to return to after saving plots.
    output_prefix : str, optional
        Prefix for output plot filenames.
    plot_format : str, optional
        Format for saved plots ('pdf', 'png', etc.).
    font_size : int, optional
        Font size for plot labels.
    fig_size : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap name for plots.
    use_log_colorbar : bool, optional
        Whether to use logarithmic scale for colorbar.
    plot_contours : bool, optional
        Whether to overlay contour lines on the parameter maps.
    plot_specific_contours : array-like or bool, optional
        Specific contour levels to draw. If False, uses automatic levels.
    plot_kde : bool, optional
        If True, add KDE distribution plots for x and y fitted parameters.
        Default is True.
    
    Returns
    -------
    tuple
        (x_plot_path, y_plot_path, x_kde_plot_path, y_kde_plot_path, z_plot_path, z_kde_plot_path)
        Paths to saved plot files (or None if not saved). z paths are None when no z map.
    """
    # Store original directory
    original_dir = os.getcwd()
    z_plot_path = None
    z_kde_plot_path = None

    x_display = x_axis_label or _infer_axis_display_label(
        x_axis_name, set_x_name, axis_units=x_axis_units, set_mass_name=set_mass_name,
    )
    y_display = y_axis_label or _infer_axis_display_label(
        y_axis_name, set_y_name, axis_units=y_axis_units, set_mass_name=set_mass_name,
    )
    z_display = z_axis_label
    if z_axis_name and z_display is None:
        z_display = _infer_axis_display_label(
            z_axis_name, set_z_name, axis_units=z_axis_units, set_mass_name=set_mass_name,
        )

    # Try to create WCS from header if available
    wcs = None
    if fits_header is not None:
        try:
            wcs_full = WCS(fits_header)
            # Slice WCS to 2D if needed (for spatial coordinates only)
            if wcs_full.naxis > 2:
                wcs = wcs_full.celestial  # Get 2D celestial WCS
            else:
                wcs = wcs_full
            # Verify WCS has spatial coordinates (RA/DEC)
            if wcs.naxis != 2:
                wcs = None
                print(f"{ORANGE}Warning: WCS does not have 2D spatial coordinates. "
                      f"Using pixel coordinates.{NC}")
        except Exception as e:
            print(f"{ORANGE}Warning: Could not create WCS from header: {e}. "
                  f"Using pixel coordinates.{NC}")
            wcs = None
    
    # Create figure size
    if fig_size is None:
        fig_size = (8, 10)
    
    def _get_contour_levels(data, user_levels):
        """
        Determine contour levels from user input or automatically.
        """
        if user_levels not in (False, None):
            return user_levels
        if np.ma.isMaskedArray(data):
            finite_vals = data.compressed()
        else:
            finite_vals = np.asarray(data, dtype=float)
            finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size == 0:
            return None
        data_min = np.nanmin(finite_vals)
        data_max = np.nanmax(finite_vals)
        if data_min == data_max:
            return None
        return np.linspace(data_min, data_max, 10)
    
    def _power_formatter():
        """Return a formatter for tick labels in 10^n style."""
        from matplotlib.ticker import FuncFormatter
        def _fmt(val, _pos):
            if not np.isfinite(val) or val <= 0:
                return ""
            exp = int(np.round(np.log10(val)))
            return r"$10^{%d}$" % exp
        return FuncFormatter(_fmt)

    def _apply_power_ticks(axis_obj):
        """Apply 10^n formatting to both axes when using pixel coordinates."""
        formatter = _power_formatter()
        axis_obj.xaxis.set_major_formatter(formatter)
        axis_obj.yaxis.set_major_formatter(formatter)

    # Parameters that span orders of magnitude -> auto-enable a log colorbar.
    # Includes density, CRIR and FUV/G0 so all three maps share a log scale by default.
    axes_keywords = ['density', 'densities', 'n_', 'n ', 'crir_values', 'crir',
                     'fuv', 'fuv_values', 'fuv_draine', 'fuv_habing', 'g_0', 'g0', 'draine', 'habing']
    is_x_axis_log = any(keyword in x_axis_name.lower() for keyword in axes_keywords)
    is_y_axis_log = any(keyword in y_axis_name.lower() for keyword in axes_keywords)
    
    # Plot x-axis parameter map
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': wcs} if wcs else {})
    
    # Prepare data for plotting
    plot_data_x = x_param_map.copy()
    use_log_x = use_log_colorbar or is_x_axis_log
    if use_log_x:
        # Mask zeros and negative values for log scale
        plot_data_x = np.ma.masked_where(plot_data_x <= 0, plot_data_x)
        norm = LogNorm(vmin=np.nanmin(plot_data_x), vmax=np.nanmax(plot_data_x))
    else:
        norm = None
    
    # Create colormap plot
    im = ax.imshow(plot_data_x, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    
    # Add contour overlays if requested
    contour_levels_x = _get_contour_levels(plot_data_x, plot_specific_contours)
    specific_levels_x = plot_specific_contours if plot_specific_contours not in (False, None) else None
    if plot_contours and contour_levels_x is not None:
        contour = ax.contour(plot_data_x, levels=contour_levels_x, colors='black',
                             linewidths=1.0)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
    if specific_levels_x is not None and contour_levels_x is not None:
        ax.contour(plot_data_x, levels=specific_levels_x, colors='red',
                   linewidths=1.5, linestyles='--')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=font_size if font_size else 19)
    
    # Set colorbar label
    cbar.set_label(x_display, fontsize=font_size if font_size else 19)
    
    # Format colorbar
    if use_log_x:
        from matplotlib.ticker import LogFormatterMathtext
        cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        cbar.update_ticks()
    
    # Set axis labels
    if wcs is not None:
        # Use WCS coordinates (RA/DEC)
        ax.set_xlabel('RA', fontsize=font_size if font_size else 19)
        ax.set_ylabel('DEC', fontsize=font_size if font_size else 19)
        # Format coordinate labels
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    else:
        # Use pixel coordinates
        ax.set_xlabel('X (pixels)', fontsize=font_size if font_size else 19)
        ax.set_ylabel('Y (pixels)', fontsize=font_size if font_size else 19)
        _apply_power_ticks(ax)
    
    # Set title
    ax.set_title(f'{x_display} Map', fontsize=font_size if font_size else 19 + 2, weight='bold')
    
    # Format axes
    # WCS axes have restrictions on minor tick parameters
    if wcs is not None:
        # For WCS axes, only set major tick parameters and minor tick length
        ax.tick_params(axis='both', which='major', labelsize=font_size if font_size else 19, pad=10, width=2.5, length=10)
        ax.tick_params(axis='both', which='minor', length=5)
        # Format coordinate axes
        for coord in ax.coords:
            coord.set_ticklabel(size=font_size if font_size else 19 )
    else:
        # For regular axes, set all parameters
        ax.tick_params(axis='both', which='major', labelsize=font_size if font_size else 19, pad=10, width=2.5, length=10)
        ax.tick_params(axis='both', which='minor', labelsize=font_size if font_size else 19, width=1.5, length=5)
    
    for axis_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis_name].set_linewidth(2.5)
        ax.spines[axis_name].set_color('black')
    
    # Save or show plot
    x_plot_path = None
    if fig_dir_PATH and or_PATH:
        os.chdir(fig_dir_PATH)
        x_plot_path = f'{output_prefix}_{x_axis_name}_map.{plot_format}'
        plt.savefig(x_plot_path, dpi=300, bbox_inches='tight')
        os.chdir(or_PATH)
        print(f"{GREEN}Saved x-axis parameter plot: {os.path.join(fig_dir_PATH, x_plot_path)}{NC}")
    elif fig_dir_PATH and not or_PATH:
        print(f"{ORANGE}Warning: fig_dir_PATH specified but or_PATH not provided. Plot not saved.{NC}")
        plt.show()
    else:
        plt.show()

    # Plot y-axis parameter map
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': wcs} if wcs else {})
    
    # Prepare data for plotting
    plot_data_y = y_param_map.copy()
    use_log_y = use_log_colorbar or is_y_axis_log
    if use_log_y:
        # Mask zeros and negative values for log scale
        plot_data_y = np.ma.masked_where(plot_data_y <= 0, plot_data_y)
        norm = LogNorm(vmin=np.nanmin(plot_data_y), vmax=np.nanmax(plot_data_y))
    else:
        norm = None
    
    # Create colormap plot
    im = ax.imshow(plot_data_y, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    
    # Add contour overlays if requested
    contour_levels_y = _get_contour_levels(plot_data_y, plot_specific_contours)
    specific_levels_y = plot_specific_contours if plot_specific_contours not in (False, None) else None
    if plot_contours and contour_levels_y is not None:
        contour = ax.contour(plot_data_y, levels=contour_levels_y, colors='black',
                             linewidths=1.0)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
    if specific_levels_y is not None and contour_levels_y is not None:
        ax.contour(plot_data_y, levels=specific_levels_y, colors='red',
                   linewidths=1.5, linestyles='--')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=font_size if font_size else 19)
    
    # Set colorbar label
    cbar.set_label(y_display, fontsize=font_size if font_size else 19)
    
    # Format colorbar
    if use_log_y:
        from matplotlib.ticker import LogFormatterMathtext
        cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        cbar.update_ticks()
    
    # Set axis labels
    if wcs is not None:
        # Use WCS coordinates (RA/DEC)
        ax.set_xlabel('RA', fontsize=font_size if font_size else 19)
        ax.set_ylabel('DEC', fontsize=font_size if font_size else 19)
        # Format coordinate labels
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    else:
        # Use pixel coordinates
        ax.set_xlabel('X (pixels)', fontsize=font_size if font_size else 19)
        ax.set_ylabel('Y (pixels)', fontsize=font_size if font_size else 19)
        _apply_power_ticks(ax)
    
    # Set title
    ax.set_title(f'{y_display} Map', fontsize=font_size if font_size else 19 + 2, weight='bold')
    
    # Format axes
    # WCS axes have restrictions on minor tick parameters
    if wcs is not None:
        # For WCS axes, only set major tick parameters and minor tick length
        ax.tick_params(axis='both', which='major', labelsize=font_size if font_size else 19, pad=10, width=2.5, length=10)
        ax.tick_params(axis='both', which='minor', length=5)
        # Format coordinate axes
        for coord in ax.coords:
            coord.set_ticklabel(size=font_size if font_size else 19)
    else:
        # For regular axes, set all parameters
        ax.tick_params(axis='both', which='major', labelsize=font_size if font_size else 19, pad=10, width=2.5, length=10)
        ax.tick_params(axis='both', which='minor', labelsize=font_size if font_size else 19, width=1.5, length=5)
    
    for axis_name in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis_name].set_linewidth(2.5)
        ax.spines[axis_name].set_color('black')
    
    # Save or show plot
    y_plot_path = None
    if fig_dir_PATH and or_PATH:
        os.chdir(fig_dir_PATH)
        y_plot_path = f'{output_prefix}_{y_axis_name}_map.{plot_format}'
        plt.savefig(y_plot_path, dpi=300, bbox_inches='tight')
        os.chdir(or_PATH)
        print(f"{GREEN}Saved y-axis parameter plot: {os.path.join(fig_dir_PATH, y_plot_path)}{NC}")
    elif fig_dir_PATH and not or_PATH:
        print(f"{ORANGE}Warning: fig_dir_PATH specified but or_PATH not provided. Plot not saved.{NC}")
        plt.show()
    else:
        plt.show()

    if z_param_map is not None and z_axis_name:
        is_z_axis_log = any(
            keyword in z_axis_name.lower()
            for keyword in ['mass', 'm_', ' m', 'density', 'densities', 'n_', 'crir',
                            'fuv', 'g_0', 'g0', 'draine', 'habing']
        )
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': wcs} if wcs else {})
        plot_data_z = z_param_map.copy()
        use_log_z = use_log_colorbar or is_z_axis_log
        if use_log_z:
            plot_data_z = np.ma.masked_where(plot_data_z <= 0, plot_data_z)
            norm = LogNorm(vmin=np.nanmin(plot_data_z), vmax=np.nanmax(plot_data_z))
        else:
            norm = None
        im = ax.imshow(plot_data_z, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        contour_levels_z = _get_contour_levels(plot_data_z, plot_specific_contours)
        if plot_contours and contour_levels_z is not None:
            contour = ax.contour(plot_data_z, levels=contour_levels_z, colors='black', linewidths=1.0)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=font_size if font_size else 19)
        cbar.set_label(z_display, fontsize=font_size if font_size else 19)
        if use_log_z:
            from matplotlib.ticker import LogFormatterMathtext
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
            cbar.update_ticks()
        if wcs is not None:
            ax.set_xlabel('RA', fontsize=font_size if font_size else 19)
            ax.set_ylabel('DEC', fontsize=font_size if font_size else 19)
            ax.coords[0].set_major_formatter('hh:mm:ss')
            ax.coords[1].set_major_formatter('dd:mm:ss')
        else:
            ax.set_xlabel('X (pixels)', fontsize=font_size if font_size else 19)
            ax.set_ylabel('Y (pixels)', fontsize=font_size if font_size else 19)
            _apply_power_ticks(ax)
        ax.set_title(f'{z_display} Map', fontsize=font_size if font_size else 21, weight='bold')
        if fig_dir_PATH and or_PATH:
            os.chdir(fig_dir_PATH)
            z_plot_path = f'{output_prefix}_{z_axis_name}_map.{plot_format}'
            plt.savefig(z_plot_path, dpi=300, bbox_inches='tight')
            os.chdir(or_PATH)
            print(f"{GREEN}Saved z-axis parameter plot: {os.path.join(fig_dir_PATH, z_plot_path)}{NC}")
        else:
            plt.show()

    x_kde_plot_path = None
    y_kde_plot_path = None
    if plot_kde:
        label_fs = font_size if font_size else 19
        x_kde_plot_path = _plot_fitted_parameter_kde(
            x_param_map,
            x_axis_name,
            x_axis_units,
            use_log_scale=use_log_x,
            fig_dir_PATH=fig_dir_PATH,
            or_PATH=or_PATH,
            output_prefix=output_prefix,
            plot_format=plot_format,
            font_size=label_fs,
            param_label=x_display,
        )
        y_kde_plot_path = _plot_fitted_parameter_kde(
            y_param_map,
            y_axis_name,
            y_axis_units,
            use_log_scale=use_log_y,
            fig_dir_PATH=fig_dir_PATH,
            or_PATH=or_PATH,
            output_prefix=output_prefix,
            plot_format=plot_format,
            font_size=label_fs,
            param_label=y_display,
        )
        if z_param_map is not None and z_axis_name:
            is_z_axis_log = any(
                keyword in z_axis_name.lower()
                for keyword in ['mass', 'm_', ' m', 'density', 'densities', 'n_', 'crir',
                                'fuv', 'g_0', 'g0', 'draine', 'habing']
            )
            z_kde_plot_path = _plot_fitted_parameter_kde(
                z_param_map,
                z_axis_name,
                z_axis_units,
                use_log_scale=is_z_axis_log,
                fig_dir_PATH=fig_dir_PATH,
                or_PATH=or_PATH,
                output_prefix=output_prefix,
                plot_format=plot_format,
                font_size=label_fs,
                param_label=z_display,
            )

    # Restore original directory
    os.chdir(original_dir)

    return x_plot_path, y_plot_path, x_kde_plot_path, y_kde_plot_path, z_plot_path, z_kde_plot_path
#########################################################################################################################################
#                                                         Fit fits function
#########################################################################################################################################
def fit_fits_maps_to_grids(observed_fits_files,
                            grid_dicts,
                            x_axis_name,
                            y_axis_name,
                            x_axis_units            = '',
                            y_axis_units            = '',
                            obs_errors              = None,
                            output_dir              = None,
                            output_prefix           = 'fitted',
                            error_fraction          = 0.2,
                            create_plots            = True,
                            fig_dir_PATH            = False,
                            or_PATH                 = False,
                            plot_format             = 'pdf',
                            font_size               = False,
                            fig_size                = None,
                            cmap                    = 'plasma',
                            use_log_colorbar        = False,
                            plot_contours           = False,
                            plot_specific_contours  = False,
                            plot_kde                = True,
                            create_chi2_corner      = True,
                            chi2_corner_pixel       = None,
                            chi2_corner_cmap        = 'viridis_r',
                            target_units            = 'K km/s',
                            species_info            = None,
                            create_ratios           = False):
    """
    Fit observed FITS intensity maps to grid models and create parameter maps.
    
    This function reads observed intensity FITS files for one or more molecular lines,
    compares them with corresponding grid models, and performs chi-squared fitting
    to determine the best-fit x and y parameter values for each pixel. The results
    are saved as two FITS files: one for the x-axis parameter map and one for the
    y-axis parameter map.
    
    Parameters
    ----------
    observed_fits_files : dict
        Dictionary mapping species/line names to FITS file paths.
        Example: {'CO(1-0)': '/path/to/CO10.fits', 'CI(1-0)': '/path/to/CI10.fits'}
    grid_dicts : dict
        Dictionary mapping species/line names to grid dictionaries or FITS file paths.
        Each grid dictionary should have structure:
        {
            'grid': 2D numpy array of model intensity values,
            'densities' or x_axis_name: 2D meshgrid of x-axis values,
            'crir_values' or y_axis_name: 2D meshgrid of y-axis values
        }
        Alternatively, values can be FITS file paths (strings) containing the grid data.
        The function will read the grid from the primary HDU and look for axis data
        in extensions or header keywords. If axis data is not found, default meshgrids
        will be created.
        Example: {'CO(1-0)': {'grid': ..., 'densities': ..., 'crir_values': ...}}
        or: {'CO(1-0)': '/path/to/grid.fits'}
    x_axis_name : str
        Name of the x-axis parameter (e.g., 'densities', 'n', 'crir_values').
        This should match the key in grid_dicts. In case it does not, a match function
        will be used to find the keys from a selection of pre-defined keys.
    y_axis_name : str
        Name of the y-axis parameter (e.g., 'crir_values', 'fuv_values').
        This should match the key in grid_dicts. In case it does not, a match function
        will be used to find the keys from a selection of pre-defined keys.
    x_axis_units : str, optional
        Units for x-axis parameter (e.g., 'cm^-3', 's^-1'). Default: ''.
    y_axis_units : str, optional
        Units for y-axis parameter (e.g., 'Habing', 's^-1'). Default: ''.
    obs_errors : dict, optional
        Per-line errors. Each value may be:
        - path to a 2D error FITS map,
        - a 2D numpy array with shape ``(ny, nx)``,
        - a scalar (int/float) giving a uniform error over the whole map.
        If None or a line is missing, errors default to
        ``error_fraction * |observed|`` (20% by default).
        Example: ``{'CO(1-0)': '/path/to/CO10_err.fits', 'CI(1-0)': 0.15}``
    output_dir : str, optional
        Directory to save output FITS files. If None, uses directory of first FITS file.
    output_prefix : str, optional
        Prefix for output FITS filenames. Default: 'fitted'.
    error_fraction : float, optional
        Fraction of observed value to use as error if obs_errors not provided. Default: 0.2.
    create_plots : bool, optional
        Whether to create colormap plots of the fitted parameter maps. Default: True.
    fig_dir_PATH : str, optional
        Directory path to save plots. If False, plots are not saved.
    or_PATH : str, optional
        Original working directory path to return to after saving plots.
    plot_format : str, optional
        Format for saved plots ('pdf', 'png', etc.). Default: 'pdf'.
    font_size : int, optional
        Font size for plot labels. Default: 19.
    fig_size : tuple, optional
        Figure size (width, height) in inches. Default: (11, 8).
    cmap : str, optional
        Colormap name for plots. Default: 'plasma'.
    use_log_colorbar : bool, optional
        Whether to use logarithmic scale for colorbar. Default: False.
    plot_contours : bool, optional
        Whether to overlay contour lines on the parameter maps. Default: False.
    plot_specific_contours : array-like or bool, optional
        Specific contour levels to draw. If False, uses automatic levels.
    plot_kde : bool, optional
        If True, also plot KDE distributions of fitted x and y parameters.
        Default is True.
    create_chi2_corner : bool, optional
        If True, save a corner plot of the chi^2 grid in parameter space.
        Default is True.
    chi2_corner_pixel : tuple of int, optional
        Sky pixel ``(row, col)`` used for the corner plot. If None, uses the
        median spectrum over the reference footprint.
    chi2_corner_cmap : str, optional
        Colormap for the corner plot Delta-chi^2 panels. Default is 'viridis_r'.
    target_units : str, optional
        Target units for intensity conversion. Default: 'K km/s'.
        Supported: 'K km/s', 'erg/s/cm^2', 'W m^-2 sr^-1', 'Jy sr^-1'
    species_info : dict, optional
        Dictionary mapping line names to (species_name, upper_level_transition) tuples.
        Optional: if not provided, the function will try to infer the species
        and transition from the line name (e.g., 'CO(2-1)').
    create_ratios : bool, optional
        Whether to automatically create transition ratios from the observed data.
        If True, ratios will be created using create_transition_ratios function
        from unit-converted observed data and included in the fitting analysis.
        Default: False.
    
    Returns
    -------
    tuple
        (x_param_map, y_param_map, x_fits_path, y_fits_path)
        - x_param_map: 2D numpy array of fitted x-axis parameter values
        - y_param_map: 2D numpy array of fitted y-axis parameter values
        - x_fits_path: Path to saved x-axis parameter FITS file
        - y_fits_path: Path to saved y-axis parameter FITS file
    
    Notes
    -----
    The function performs chi-squared minimization for each pixel using exact values:
    chi^2 = sum((observed - model)^2 / error^2) for all lines
    
    The best-fit parameters are found by locating the minimum chi-squared value
    in the parameter space grid. When multiple grid points have chi-squared values
    at the exact minimum, a weighted average is used (weighted by inverse chi-squared)
    to avoid clustering artifacts. If the chi-squared surface is flat in one direction
    (degenerate solution), the function uses nearby "good" points (within 1% of minimum)
    to provide smoother parameter estimates.
    
    WCS information from the first observed FITS file is preserved in the output files.
    
    Unit Conversion:
    The function reads the BUNIT keyword from each FITS file header and converts
    intensities to target_units. Supported units include:
    - 'K km/s' or 'Kkm/s'
    - 'erg/s/cm^2' or 'erg s-1 cm-2'
    - 'W m^-2 sr^-1' or 'W m-2 sr-1'
    - 'Jy sr^-1' or 'Jy/sr'
    
    For K km/s conversions, the species/transition is inferred from the line
    name if possible (e.g., 'CO(2-1)'). You may also provide species_info
    mapping line names to (species_name, upper_level) tuples.
    """
    # Validate inputs
    if not isinstance(observed_fits_files, dict) or not isinstance(grid_dicts, dict):
        raise ValueError("observed_fits_files and grid_dicts must be dictionaries")
    
    missing_grids = [line for line in observed_fits_files.keys() if line not in grid_dicts]
    if missing_grids:
        raise ValueError(f"Missing grids for lines: {missing_grids}")
    
    # Load grids from FITS files or use provided dictionaries
    processed_grid_dicts = {}
    for line_name, grid_data in grid_dicts.items():
        if isinstance(grid_data, str):
            print(f"{GREEN}Reading grid FITS file for {line_name}: {grid_data}{NC}")
            try:
                processed_grid_dicts[line_name] = _load_grid_from_fits(grid_data, x_axis_name, y_axis_name)
                print(f"{GREEN}Successfully loaded grid for {line_name} from FITS file{NC}")
            except Exception as e:
                raise ValueError(f"Error reading grid FITS file for {line_name}: {e}")
        else:
            processed_grid_dicts[line_name] = grid_data
    
    grid_dicts = processed_grid_dicts
    
    # Check number of lines
    num_lines = len(list(observed_fits_files.keys()))
    print(f"{GREEN}_Available lines from the provided fits data: {list(observed_fits_files.keys())}{NC}")
    if num_lines <= 2:
        print(f"{ORANGE}The number of available lines is: {num_lines} and this could lead to a degenerate solution.{NC}")
        print(f"{ORANGE}The user should provide more than two lines in the observed lines input.{NC}")
        print(f"{ORANGE}If the user wants to create ratios, set the create_ratios parameter to True.{NC}")
    else:
        print(f"{GREEN}The number of available lines is: {num_lines} and this is sufficient for fitting.{NC}")
    
    # Read observed data
    observed_data, observed_errors, first_header, ny, nx = _read_observed_data(
        observed_fits_files, target_units, species_info, error_fraction, obs_errors
    )
    
    # Separate line names from ratio names
    line_names_only = [name for name in observed_fits_files.keys() if '/' not in name]
    ratio_names_provided = [name for name in observed_fits_files.keys() if '/' in name]
    
    # Handle provided ratios
    ratio_data = {}
    ratio_errors = {}
    if ratio_names_provided:
        print('-'*30)
        print(f"{BLUE}Checking for ratios and creating if requested{NC}")
        print('-'*30)
        print(f"{GREEN}Found {len(ratio_names_provided)} ratio(s) provided in input: {ratio_names_provided}{NC}")
        for ratio_name in ratio_names_provided:
            fits_path = observed_fits_files[ratio_name]
            data, header = _read_fits_2d(fits_path)
            if data.shape != (ny, nx):
                raise ValueError(f"Ratio FITS file for {ratio_name} has shape {data.shape}, expected ({ny}, {nx})")
            
            fits_unit = header.get('BUNIT', '')
            if fits_unit and target_units:
                data = _convert_units_if_needed(data, fits_unit, target_units, ratio_name, species_info)
            
            ratio_data[ratio_name] = data
            
            error_source = obs_errors.get(ratio_name) if obs_errors is not None else None
            ratio_errors[ratio_name] = _normalize_observed_error_map(
                error_source,
                data,
                ny,
                nx,
                error_fraction,
                ratio_name,
                target_units=target_units,
                species_info=species_info,
            )
    
    # Create ratios if requested
    if create_ratios:
        print('-'*30)
        print(f"{BLUE}Checking for ratios and creating if requested{NC}")
        print('-'*30)
        if ratio_names_provided:
            print(f"{ORANGE}Warning: Ratios already provided in input. Skipping automatic ratio creation.{NC}")
            print(f"{GREEN}Using provided ratios: {ratio_names_provided}{NC}")
        else:
            print(f"{GREEN}create_ratios=True: Creating ratios from unit-converted observed data.{NC}")
            created_ratios, created_errors = _create_ratios_from_data(
                line_names_only, observed_data, observed_errors, first_header,
                ny, nx, grid_dicts, x_axis_name, y_axis_name, error_fraction
            )
            ratio_data.update(created_ratios)
            ratio_errors.update(created_errors)
    elif not ratio_names_provided:
        print('-'*30)
        print(f"{BLUE}Checking for ratios and creating if requested{NC}")
        print('-'*30)
        print(f"{GREEN}create_ratios=False: Skipping automatic ratio creation.{NC}")
    
    # Extract ratio grids from model grids
    ratio_grids = {}
    ratio_names_in_grids = [name for name in grid_dicts.keys() if '/' in name]
    if ratio_names_in_grids:
        print(f"{GREEN}Found {len(ratio_names_in_grids)} ratio(s) already provided in grid_dicts: {ratio_names_in_grids}{NC}")
        for ratio_name in ratio_data.keys():
            if ratio_name in grid_dicts:
                ratio_grids[ratio_name] = grid_dicts[ratio_name]
                print(f"  {GREEN}Using provided model grid for ratio: {ratio_name}{NC}")
            else:
                print(f"  {ORANGE}Warning: Ratio {ratio_name} not found in grid_dicts. It will be skipped in fitting.{NC}")
    
    # Combine lines and ratios
    print('-'*30)
    print(f"{BLUE}Preparing grid data and data checks{NC}")
    print('-'*30)
    all_data_names = line_names_only.copy()
    all_observed_data = observed_data.copy()
    all_observed_errors = observed_errors.copy()
    
    if ratio_data:
        all_data_names.extend(list(ratio_data.keys()))
        all_observed_data.update(ratio_data)
        all_observed_errors.update(ratio_errors)
        print(f"{GREEN}Combined data for fitting: {all_data_names}{NC}")
    else:
        print(f"{GREEN}Data for fitting: {all_data_names}{NC}")
    
    # Get axis meshes
    first_grid = grid_dicts[line_names_only[0]]
    x_mesh = _find_axis_mesh(first_grid, x_axis_name, 
                             ['densities', 'crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine'])
    y_mesh = _find_axis_mesh(first_grid, y_axis_name,
                             ['crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine', 'densities'])
    
    if x_mesh is None or y_mesh is None:
        raise ValueError(f"Could not find axes in grid. Looking for x='{x_axis_name}', "
                        f"y='{y_axis_name}'. Available keys: {list(first_grid.keys())}")
    
    # Extract grid arrays
    grid_arrays = {}
    for data_name in all_data_names:
        if data_name in line_names_only:
            if 'grid' not in grid_dicts[data_name]:
                raise ValueError(f"Grid for {data_name} missing 'grid' key")
            grid_arrays[data_name] = grid_dicts[data_name]['grid']
        elif data_name in ratio_grids:
            grid_arrays[data_name] = ratio_grids[data_name]['grid']
        else:
            print(f"{ORANGE}Warning: No grid found for {data_name}, skipping.{NC}")
            continue
        
        if grid_arrays[data_name].shape != x_mesh.shape:
            raise ValueError(f"Grid shape mismatch for {data_name}: "
                           f"{grid_arrays[data_name].shape} != {x_mesh.shape}")
    
    # Ensure 2D meshgrids
    if x_mesh.ndim == 1:
        if y_mesh.ndim == 1:
            Y_mesh, X_mesh = np.meshgrid(y_mesh, x_mesh, indexing='ij')
        else:
            Y_mesh, X_mesh = np.meshgrid(np.unique(y_mesh), x_mesh, indexing='ij')
    elif y_mesh.ndim == 1:
        Y_mesh, X_mesh = np.meshgrid(y_mesh, np.unique(x_mesh), indexing='ij')
    else:
        X_mesh, Y_mesh = x_mesh, y_mesh
    
    ny_grid, nx_grid = X_mesh.shape
    if (ny_grid, nx_grid) != grid_arrays[all_data_names[0]].shape:
        raise ValueError(f"Meshgrid shape ({ny_grid}, {nx_grid}) != grid shape {grid_arrays[all_data_names[0]].shape}")
    
    # -------------------------------------------------------------------------
    # Reference footprint logic
    # -------------------------------------------------------------------------
    # We want the output parameter maps to have the same *visual footprint*
    # as the smallest input map (in terms of number of valid pixels).
    # Previously, any NaN in any required map would invalidate the pixel,
    # which shrinks the footprint to the intersection across all maps.
    #
    # Here we:
    # 1) pick a reference mask from the input map with the fewest valid pixels
    # 2) only compute fits inside that reference mask
    # 3) inside the reference mask, skip invalid lines per pixel instead of
    #    invalidating the entire pixel.
    def _valid_pixel_mask(data_map, err_map):
        """Pixel is valid if both value and error are finite and error > 0."""
        return np.isfinite(data_map) & np.isfinite(err_map) & (err_map > 0)

    candidate_ref_names = [
        name
        for name in all_data_names
        if name in grid_arrays and name in all_observed_data and name in all_observed_errors
    ]
    if not candidate_ref_names:
        raise ValueError("No valid reference candidates found for determining output footprint.")

    ref_name = None
    ref_mask = None
    min_valid_count = None
    for name in candidate_ref_names:
        mask = _valid_pixel_mask(all_observed_data[name], all_observed_errors[name])
        valid_count = int(np.count_nonzero(mask))
        if min_valid_count is None or valid_count < min_valid_count:
            min_valid_count = valid_count
            ref_name = name
            ref_mask = mask

    ref_mask = ref_mask.astype(bool)
    n_ref_pixels = int(np.count_nonzero(ref_mask))
    print(f"{ORANGE}Using reference footprint from {ref_name} ({n_ref_pixels} valid pixels).{NC}")

    # Perform fitting
    print('-'*30)
    print(f"{BLUE}Fitting {n_ref_pixels} pixels (reference footprint){NC}")
    print('-'*30)
    x_param_map = np.full((ny, nx), np.nan)
    y_param_map = np.full((ny, nx), np.nan)

    for i in tqdm(range(ny), desc="Processing rows"):
        for j in range(nx):
            if not ref_mask[i, j]:
                continue

            obs_values = {}
            obs_errs = {}

            # Skip invalid lines per pixel; only require at least 1 valid
            # constraint inside the reference footprint.
            valid_constraints = 0
            for data_name in all_data_names:
                if data_name not in grid_arrays:
                    continue
                if data_name not in all_observed_data or data_name not in all_observed_errors:
                    continue

                obs_val = all_observed_data[data_name][i, j]
                obs_err = all_observed_errors[data_name][i, j]
                if (not np.isfinite(obs_val)) or (not np.isfinite(obs_err)) or obs_err <= 0:
                    continue

                obs_values[data_name] = obs_val
                obs_errs[data_name] = obs_err
                valid_constraints += 1

            if valid_constraints < 1:
                continue
            
            x_param, y_param = _fit_pixel(obs_values, obs_errs, grid_arrays, X_mesh, Y_mesh)
            x_param_map[i, j] = x_param
            y_param_map[i, j] = y_param

    # Save output files
    first_line = list(observed_fits_files.keys())[0]
    fits_output_dir = (fig_dir_PATH if fig_dir_PATH else 
                      (output_dir if output_dir else os.path.dirname(os.path.abspath(observed_fits_files[first_line]))))
    os.makedirs(fits_output_dir, exist_ok=True)
    
    x_fits_name = f"{output_prefix}_{x_axis_name}.fits"
    y_fits_name = f"{output_prefix}_{y_axis_name}.fits"
    x_fits_path = os.path.join(fits_output_dir, x_fits_name)
    y_fits_path = os.path.join(fits_output_dir, y_fits_name)

    x_header = _create_output_header(first_header, x_axis_name, x_axis_units, x_param_map)
    y_header = _create_output_header(first_header, y_axis_name, y_axis_units, y_param_map)

    print('-'*30)
    print(f"{BLUE}Saving output FITS files{NC}")
    print('-'*30)
    original_dir = os.getcwd()
    try:
        if fig_dir_PATH and or_PATH:
            os.chdir(fits_output_dir)
        fits.PrimaryHDU(data=x_param_map, header=x_header).writeto(x_fits_name, overwrite=True)
        fits.PrimaryHDU(data=y_param_map, header=y_header).writeto(y_fits_name, overwrite=True)
    finally:
        os.chdir(or_PATH if (fig_dir_PATH and or_PATH) else original_dir)

    print(f"{GREEN}Saved x-axis parameter map: {x_fits_path}{NC}")
    print(f"{GREEN}Saved y-axis parameter map: {y_fits_path}{NC}")
    
    # Create plots if requested
    if create_plots:
        plot_fitted_parameter_maps(
            x_param_map, y_param_map,
            x_axis_name, y_axis_name,
            x_axis_units, y_axis_units,
            fits_header=first_header,
            fig_dir_PATH=fig_dir_PATH,
            or_PATH=or_PATH,
            output_prefix=output_prefix,
            plot_format=plot_format,
            font_size=font_size if font_size else 19,
            fig_size=fig_size,
            cmap=cmap,
            use_log_colorbar=use_log_colorbar,
            plot_contours=plot_contours,
            plot_specific_contours=plot_specific_contours,
            plot_kde=plot_kde,
        )

        if create_chi2_corner:
            label_fs = font_size if font_size else 19
            corner_obs, corner_err, corner_subtitle = _pixel_observations_for_chi2_analysis(
                all_observed_data,
                all_observed_errors,
                ref_mask,
                all_data_names,
                grid_arrays,
                pixel=chi2_corner_pixel,
            )

            if corner_obs:
                print('-'*30)
                print(f"{BLUE}Plotting chi^2 corner ({corner_subtitle}){NC}")
                print('-'*30)
                plot_chi2_corner(
                    corner_obs,
                    corner_err,
                    grid_arrays,
                    X_mesh,
                    Y_mesh,
                    x_axis_name,
                    y_axis_name,
                    x_axis_units,
                    y_axis_units,
                    fig_dir_PATH=fig_dir_PATH,
                    or_PATH=or_PATH,
                    output_prefix=output_prefix,
                    plot_format=plot_format,
                    font_size=label_fs,
                    cmap=chi2_corner_cmap,
                    subtitle=corner_subtitle,
                )
            else:
                print(
                    f"{ORANGE}Warning: no valid data for chi^2 corner plot; skipping.{NC}"
                )

    return x_param_map, y_param_map, x_fits_path, y_fits_path
#########################################################################################################################################
#                                              3D grid map fitting helpers
#########################################################################################################################################
def _read_fits_3d_model(fits_path):
    """Read a 3D model grid from FITS with shape (n_z, ny, nx)."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    if data is None:
        raise ValueError(f"FITS file at {fits_path!r} has no data in primary HDU.")

    data = np.asarray(data)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(
            f"FITS file at {fits_path!r} is not a 3D model grid: got shape {data.shape} "
            f"(expected 3 dimensions after squeezing singleton axes)."
        )
    return data, header
#-----------------------------------------------------------------------------------------------------------
def _ensure_3d_meshgrids(x_data, y_data, z_data, grid_shape):
    """Ensure axis data are 3D meshgrids matching grid shape (n_z, ny, nx)."""
    n_z, ny, nx = grid_shape

    if x_data.ndim == 2 and x_data.shape == (ny, nx):
        x_data = np.broadcast_to(x_data[None, :, :], grid_shape)
    if y_data.ndim == 2 and y_data.shape == (ny, nx):
        y_data = np.broadcast_to(y_data[None, :, :], grid_shape)
    if z_data.ndim == 2 and z_data.shape == (ny, nx):
        z_data = np.broadcast_to(z_data[None, :, :], grid_shape)

    def _expand_1d(axis_1d, axis_index):
        if axis_index == 0:
            full = np.broadcast_to(axis_1d[:, None, None], grid_shape)
        elif axis_index == 1:
            full = np.broadcast_to(axis_1d[None, :, None], grid_shape)
        else:
            full = np.broadcast_to(axis_1d[None, None, :], grid_shape)
        return full

    if x_data.ndim == 1:
        x_data = _expand_1d(x_data, 2)
    if y_data.ndim == 1:
        y_data = _expand_1d(y_data, 1)
    if z_data.ndim == 1:
        z_data = _expand_1d(z_data, 0)

    if x_data.shape == grid_shape:
        return x_data, y_data, z_data
    if x_data.shape == grid_shape[::-1]:
        return x_data.transpose(2, 1, 0), y_data.transpose(2, 1, 0), z_data.transpose(2, 1, 0)
    return x_data, y_data, z_data
#-----------------------------------------------------------------------------------------------------------
def _load_grid_from_fits_3d(fits_path, x_axis_name, y_axis_name, z_axis_name):
    """Load 3D grid data and axis meshgrids from a FITS file."""
    with fits.open(fits_path) as grid_fits:
        grid_array, grid_header = _read_fits_3d_model(fits_path)
        n_z, ny_grid, nx_grid = grid_array.shape

        x_axis_data = y_axis_data = z_axis_data = None
        if len(grid_fits) > 1:
            for ext_idx in range(1, len(grid_fits)):
                ext_name = grid_fits[ext_idx].name.upper()
                ext_data = grid_fits[ext_idx].data
                if ext_data is None:
                    continue
                ext_data = np.squeeze(np.asarray(ext_data))
                if ext_data.ndim > 3:
                    ext_data = ext_data.reshape(ext_data.shape[-3:])

                if ext_data.ndim == 1:
                    if x_axis_name.upper() in ext_name or 'XAXIS' in ext_name:
                        x_axis_data = ext_data
                    elif y_axis_name.upper() in ext_name or 'YAXIS' in ext_name:
                        y_axis_data = ext_data
                    elif z_axis_name.upper() in ext_name or 'ZAXIS' in ext_name or 'MASS' in ext_name:
                        z_axis_data = ext_data
                    continue

                if ext_data.shape != (n_z, ny_grid, nx_grid):
                    print(f"{ORANGE}Warning: Extension {ext_name} shape mismatch. Skipping.{NC}")
                    continue

                if x_axis_name.upper() in ext_name or 'XAXIS' in ext_name:
                    x_axis_data = ext_data
                elif y_axis_name.upper() in ext_name or 'YAXIS' in ext_name:
                    y_axis_data = ext_data
                elif z_axis_name.upper() in ext_name or 'ZAXIS' in ext_name or 'MASS' in ext_name:
                    z_axis_data = ext_data

        if x_axis_data is None:
            x_axis_data = _construct_axis_from_header(grid_header, (ny_grid, nx_grid), x_axis_name, 'x')
        if y_axis_data is None:
            y_axis_data = _construct_axis_from_header(grid_header, (ny_grid, nx_grid), y_axis_name, 'y')
        if z_axis_data is None:
            z_axis_data = _construct_axis_from_header(grid_header, (ny_grid, nx_grid), z_axis_name, 'z')

        if x_axis_data is None or y_axis_data is None or z_axis_data is None:
            print(f"{ORANGE}Warning: Creating default 3D meshgrids for {fits_path}{NC}")
            z_lin = np.arange(n_z)
            y_lin = np.arange(ny_grid)
            x_lin = np.arange(nx_grid)
            Z_mesh, Y_mesh, X_mesh = np.meshgrid(z_lin, y_lin, x_lin, indexing='ij')
            if z_axis_data is None:
                z_axis_data = Z_mesh
            if y_axis_data is None:
                y_axis_data = Y_mesh
            if x_axis_data is None:
                x_axis_data = X_mesh

        x_axis_data, y_axis_data, z_axis_data = _ensure_3d_meshgrids(
            x_axis_data, y_axis_data, z_axis_data, (n_z, ny_grid, nx_grid)
        )

        if (x_axis_data.shape != (n_z, ny_grid, nx_grid)
                or y_axis_data.shape != (n_z, ny_grid, nx_grid)
                or z_axis_data.shape != (n_z, ny_grid, nx_grid)):
            raise ValueError(
                f"3D axis shape mismatch for {fits_path}: "
                f"x={x_axis_data.shape}, y={y_axis_data.shape}, z={z_axis_data.shape}, "
                f"grid=({n_z}, {ny_grid}, {nx_grid})"
            )

        return {
            'grid': grid_array,
            x_axis_name: x_axis_data,
            y_axis_name: y_axis_data,
            z_axis_name: z_axis_data,
        }
#-----------------------------------------------------------------------------------------------------------
def _best_fit_from_chi2(chi2_grid, X_mesh, Y_mesh, Z_mesh):
    """Return best-fit (x, y, z) parameters from a 3D chi^2 grid.

    When several grid cells tie for the minimum (common when the fit is
    degenerate or has zero degrees of freedom), the result is the chi^2-weighted
    average over the tied cells so the estimate is not an arbitrary grid corner.
    """
    chi2_min = np.nanmin(chi2_grid)
    chi2_tolerance_exact = max(chi2_min * 1e-12, np.finfo(float).eps * 10)
    exact_min_indices = np.where(np.abs(chi2_grid - chi2_min) <= chi2_tolerance_exact)

    if len(exact_min_indices[0]) > 0:
        x_vals_exact = X_mesh[exact_min_indices]
        y_vals_exact = Y_mesh[exact_min_indices]
        z_vals_exact = Z_mesh[exact_min_indices]

        z_vals_unique = np.unique(z_vals_exact)
        is_flat_in_z = (
            len(z_vals_unique) == 1
            or (np.nanmax(z_vals_exact) - np.nanmin(z_vals_exact)) < np.finfo(float).eps * 100
        )

        if is_flat_in_z and len(exact_min_indices[0]) > 1:
            chi2_tolerance_good = chi2_min * 0.01
            good_indices = np.where(chi2_grid <= chi2_min + chi2_tolerance_good)
            if len(good_indices[0]) > len(exact_min_indices[0]):
                weights = 1.0 / (chi2_grid[good_indices] + 1e-10)
                weights = weights / np.sum(weights)
                return (
                    np.sum(weights * X_mesh[good_indices]),
                    np.sum(weights * Y_mesh[good_indices]),
                    np.sum(weights * Z_mesh[good_indices]),
                )
            weights = 1.0 / (chi2_grid[exact_min_indices] + 1e-10)
            weights = weights / np.sum(weights)
            return (
                np.sum(weights * x_vals_exact),
                np.sum(weights * y_vals_exact),
                np.sum(weights * z_vals_exact),
            )
        if len(exact_min_indices[0]) > 1:
            weights = 1.0 / (chi2_grid[exact_min_indices] + 1e-10)
            weights = weights / np.sum(weights)
            return (
                np.sum(weights * x_vals_exact),
                np.sum(weights * y_vals_exact),
                np.sum(weights * z_vals_exact),
            )
        min_z_idx = exact_min_indices[0][0]
        min_y_idx = exact_min_indices[1][0]
        min_x_idx = exact_min_indices[2][0]
        return (
            X_mesh[min_z_idx, min_y_idx, min_x_idx],
            Y_mesh[min_z_idx, min_y_idx, min_x_idx],
            Z_mesh[min_z_idx, min_y_idx, min_x_idx],
        )

    min_z_idx, min_y_idx, min_x_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    return (
        X_mesh[min_z_idx, min_y_idx, min_x_idx],
        Y_mesh[min_z_idx, min_y_idx, min_x_idx],
        Z_mesh[min_z_idx, min_y_idx, min_x_idx],
    )
#-----------------------------------------------------------------------------------------------------------
def _fit_pixel_3d(obs_values, obs_errs, grid_arrays, X_mesh, Y_mesh, Z_mesh):
    """Fit a single sky pixel to best (x, y, z) model parameters."""
    chi2_grid = _compute_chi2_grid(obs_values, obs_errs, grid_arrays)
    if chi2_grid is None:
        return np.nan, np.nan, np.nan
    return _best_fit_from_chi2(chi2_grid, X_mesh, Y_mesh, Z_mesh)
#-----------------------------------------------------------------------------------------------------------
def _parameter_confidence_width_dex(chi2_grid, values_1d, keep_axis, chi2_min, delta_chi2):
    """Profiled 1-sigma half-width (in dex) for a single parameter axis.

    The chi^2 grid is profiled (minimised) over the two other axes, giving a 1D
    chi^2 curve for this parameter. The confidence interval is the set of values
    with profiled chi^2 <= chi2_min + delta_chi2 (delta_chi2 = 1.0 corresponds to
    the 68% interval of a single parameter). The returned value is half the
    log10 span of that interval, i.e. an approximate +/- sigma expressed in dex.

    Returns NaN if no value satisfies the threshold (should not happen since the
    minimum always does) or if the parameter values are not strictly positive.
    """
    other_axes      = tuple(a for a in range(chi2_grid.ndim) if a != keep_axis)
    finite_grid     = np.where(np.isfinite(chi2_grid), chi2_grid, np.inf)
    profile         = np.min(finite_grid, axis=other_axes)
    within          = profile <= (chi2_min + delta_chi2)
    selected        = np.asarray(values_1d, dtype=float)[within]
    selected        = selected[np.isfinite(selected) & (selected > 0)]
    if selected.size == 0:
        return np.nan
    log_sel = np.log10(selected)
    return 0.5 * float(log_sel.max() - log_sel.min())
#-----------------------------------------------------------------------------------------------------------
def _fit_pixel_3d_with_uncertainty(obs_values, obs_errs, grid_arrays,
                                   X_mesh, Y_mesh, Z_mesh,
                                   x_values, y_values, z_values,
                                   delta_chi2):
    """Best-fit (x, y, z) plus per-parameter uncertainty and goodness of fit.

    The per-parameter uncertainties are profiled 1-sigma half-widths in dex
    (see :func:`_parameter_confidence_width_dex`). ``chi2_reduced`` uses the
    per-pixel degrees of freedom ``max(N_obs - 3, 1)``.

    Returns
    -------
    tuple
        ``(best_x, best_y, best_z, sigma_x_dex, sigma_y_dex, sigma_z_dex,
        chi2_min, chi2_reduced)``. All entries are NaN when the pixel cannot be
        fit.
    """
    nan = np.nan
    chi2_grid = _compute_chi2_grid(obs_values, obs_errs, grid_arrays)
    if chi2_grid is None or not np.any(np.isfinite(chi2_grid)):
        return nan, nan, nan, nan, nan, nan, nan, nan

    best_x, best_y, best_z = _best_fit_from_chi2(chi2_grid, X_mesh, Y_mesh, Z_mesh)
    chi2_min = float(np.nanmin(chi2_grid))

    n_obs = len(obs_values)
    dof = max(n_obs - 3, 1)
    chi2_reduced = chi2_min / dof

    sigma_x = _parameter_confidence_width_dex(chi2_grid, x_values, 2, chi2_min, delta_chi2)
    sigma_y = _parameter_confidence_width_dex(chi2_grid, y_values, 1, chi2_min, delta_chi2)
    sigma_z = _parameter_confidence_width_dex(chi2_grid, z_values, 0, chi2_min, delta_chi2)

    return best_x, best_y, best_z, sigma_x, sigma_y, sigma_z, chi2_min, chi2_reduced
#-----------------------------------------------------------------------------------------------------------
def _plot_2d_diagnostic_map(data_map, title, cbar_label, output_name,
                            fits_header=None, fig_dir_PATH=False, or_PATH=False,
                            plot_format='pdf', font_size=19, fig_size=None,
                            cmap='viridis'):
    """Plot and optionally save a single 2D diagnostic map (uncertainty or chi^2).

    Uses the FITS header WCS for sky coordinates when available, otherwise pixel
    coordinates. Saves to ``fig_dir_PATH`` when both ``fig_dir_PATH`` and
    ``or_PATH`` are provided. Returns the saved path (or None).
    """
    finite = np.asarray(data_map, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        print(f"{ORANGE}Warning: diagnostic map '{output_name}' has no finite values; skipping plot.{NC}")
        return None

    wcs = None
    if fits_header is not None:
        try:
            wcs_full = WCS(fits_header)
            wcs = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full
            if wcs.naxis != 2:
                wcs = None
        except Exception:
            wcs = None

    if fig_size is None:
        fig_size = (8, 7)
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={'projection': wcs} if wcs else {})
    im = ax.imshow(data_map, origin='lower', cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size - 2)
    if wcs:
        ax.set_xlabel('RA', fontsize=font_size)
        ax.set_ylabel('Dec', fontsize=font_size)
    else:
        ax.set_xlabel('x pixel', fontsize=font_size)
        ax.set_ylabel('y pixel', fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size - 2)
    plt.tight_layout()

    plot_path = None
    if fig_dir_PATH and or_PATH:
        os.chdir(fig_dir_PATH)
        plot_path = f'{output_name}.{plot_format}'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        os.chdir(or_PATH)
    plt.show()
    return plot_path
#-----------------------------------------------------------------------------------------------------------
def _create_model_ratios_3d(line_names_only, grid_dicts_3d):
    """Create 3D model ratio grids from line grids when not already provided."""
    existing = {name for name in grid_dicts_3d if '/' in name}
    ratio_dicts = create_transition_ratios_3d(line_names_only, grid_dicts_3d)
    new_ratios = {}
    for ratio_name, ratio_dict in ratio_dicts.items():
        if ratio_name not in existing:
            new_ratios[ratio_name] = ratio_dict
    return new_ratios
#-----------------------------------------------------------------------------------------------------------
def _pixel_observations_for_chi2_analysis(all_observed_data,
                                          all_observed_errors,
                                          ref_mask,
                                          data_names,
                                          grid_arrays,
                                          pixel=None):
    """Scalar observed values/errors for chi2_analysis_3d (one pixel or median footprint)."""
    obs_values = {}
    obs_errs = {}
    subtitle = 'median footprint'

    if pixel is not None:
        i_pix, j_pix = pixel
        subtitle = f'pixel ({i_pix}, {j_pix})'
        for name in data_names:
            if name not in grid_arrays or name not in all_observed_data:
                continue
            obs_val = all_observed_data[name][i_pix, j_pix]
            obs_err = all_observed_errors[name][i_pix, j_pix]
            if (not np.isfinite(obs_val)) or (not np.isfinite(obs_err)) or obs_err <= 0:
                continue
            obs_values[name] = float(obs_val)
            obs_errs[name] = float(obs_err)
        return obs_values, obs_errs, subtitle

    obs_values, obs_errs = _median_pixel_observations(
        all_observed_data, all_observed_errors, ref_mask, data_names, grid_arrays
    )
    return obs_values, obs_errs, subtitle
#-----------------------------------------------------------------------------------------------------------
def interactive_pixel_map(source,
                          line_name=None,
                          *,
                          use_log=True,
                          colorscale='Viridis',
                          title=None,
                          width=800,
                          height=900,
                          target_units='K km/s',
                          species_info=None):
    """Show an interactive Plotly heatmap to pick a pixel (i, j) for chi2 analysis.

    Hover over any pixel to read its array indices ``i`` (row) and ``j`` (column)
    together with the intensity value. The reported pair matches the indexing
    used by the fitter, so you can pass it directly as
    ``fit_fits_maps_to_grids_3d(..., chi2_analysis_pixel=(i, j))``.

    Parameters
    ----------
    source : dict, str, or numpy.ndarray
        One of:
        - the ``observed_fits_files`` dict (line name -> FITS path); the map for
          ``line_name`` is shown (defaults to the first entry),
        - a path to a 2D FITS file,
        - a 2D numpy array.
    line_name : str, optional
        Which line to display when ``source`` is a dict, or a label otherwise.
    use_log : bool, optional
        Show log10 of the intensity for the colour scale (positive values only).
        Hover always reports the original (linear) intensity. Default True.
    colorscale : str, optional
        Plotly colour scale name. Default 'Viridis'.
    title : str, optional
        Figure title. A sensible default is used when None.
    width, height : int, optional
        Figure size in pixels.
    target_units, species_info : optional
        Passed through to unit conversion if the FITS BUNIT differs from
        ``target_units`` (mirrors the fitter's behaviour). Conversion is skipped
        for raw numpy array input.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure (also shown immediately).

    Notes
    -----
    Row 0 is drawn at the bottom to match the ``origin='lower'`` convention used
    by the fitted-parameter maps.
    """
    import plotly.graph_objects as go
    header = None
    if isinstance(source, dict):
        if not source:
            raise ValueError("source dict is empty.")
        if line_name is None:
            line_name = next(iter(source))
        if line_name not in source:
            raise ValueError(f"line_name {line_name!r} not in source keys: {list(source)}")
        data, header = _read_fits_2d(source[line_name])
    elif isinstance(source, str):
        data, header = _read_fits_2d(source)
        line_name = line_name or os.path.basename(source)
    else:
        data = np.asarray(source, dtype=float)
        line_name = line_name or 'map'

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {data.shape}.")

    # Optional unit conversion to match the fitter (FITS sources only).
    if header is not None and target_units:
        fits_unit = header.get('BUNIT', '')
        if fits_unit:
            data, _ = _convert_units_if_needed(
                data, fits_unit, target_units, line_name, species_info
            )

    z_color = data
    colorbar_title = 'Intensity'
    if use_log:
        with np.errstate(invalid='ignore', divide='ignore'):
            z_color = np.where(data > 0, np.log10(data), np.nan)
        colorbar_title = 'log10(Intensity)'

    fig = go.Figure(data=go.Heatmap(
        z=z_color,
        customdata=data,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        hovertemplate=(
            'i (row) = %{y}<br>'
            'j (col) = %{x}<br>'
            'intensity = %{customdata:.3e}<extra></extra>'
        ),
    ))
    fig.update_layout(
        title=title or f'Pixel picker: {line_name} — hover to read (i, j)',
        xaxis_title='j  (column index)',
        yaxis_title='i  (row index)',
        width=width,
        height=height,
    )
    # Keep row 0 at the bottom (matches origin='lower' parameter maps).
    fig.update_yaxes(autorange=True)
    fig.show()
    # return fig
#-----------------------------------------------------------------------------------------------------------
def fit_fits_maps_to_grids_3d(observed_fits_files,
                              grid_dicts_3d,
                              x_axis_name,
                              y_axis_name,
                              z_axis_name,
                              x_axis_units='',
                              y_axis_units='',
                              z_axis_units='',
                              set_x_name='n',
                              set_y_name='FUV',
                              set_z_name=None,
                              set_mass_name='M',
                              obs_errors=None,
                              output_dir=None,
                              output_prefix='fitted',
                              error_fraction=0.2,
                              create_plots=True,
                              fig_dir_PATH=False,
                              or_PATH=False,
                              plot_format='pdf',
                              font_size=False,
                              fig_size=None,
                              cmap='plasma',
                              use_log_colorbar=False,
                              plot_contours=False,
                              plot_specific_contours=False,
                              plot_kde=True,
                              create_uncertainty_maps=True,
                              uncertainty_delta_chi2=1.0,
                              uncertainty_cmap='magma',
                              create_chi2_analysis=True,
                              chi2_analysis_pixel=None,
                              chi2_plot_results=True,
                              chi2_plot_projections=True,
                              chi2_plot_volume=False,
                              chi2_plot_species_slices=True,
                              chi2_species_slices_list=None,
                              chi2_species_slices_lines_only=True,
                              chi2_species_slices_max=8,
                              chi2_corner_cmap='viridis_r',
                              target_units='K km/s',
                              species_info=None,
                              create_ratios=False,
                              create_model_ratios=False):
    """
    Fit observed FITS intensity maps to 3D grid models (density, FUV/CRIR, mass).

    For each sky pixel, the function minimizes chi-squared over the full 3D
    parameter grid (x, y, mass or z) and writes three parameter maps. Optional
    Optional analysis reuses :func:`map_fit_extras.chi2_analysis_3d`
    for a representative spectrum (median footprint or one pixel).

    Parameters
    ----------
    observed_fits_files : dict
        Line name -> path to 2D observed intensity FITS map.
    obs_errors : dict, optional
        Per-line errors (FITS map path, 2D array, or scalar uniform error).
        Missing entries use ``error_fraction * |observed|`` (default 20%).
    error_fraction : float, optional
        Default fractional error when ``obs_errors`` is not supplied. Default 0.2.
    grid_dicts_3d : dict
        Line name -> 3D grid dict or FITS path. Each dict must contain
        ``'grid'`` with shape ``(n_z, ny, nx)`` and three coordinate meshgrids.
    x_axis_name, y_axis_name, z_axis_name : str
        Keys for x, y, and z (mass) meshgrids in each grid dict
        (e.g. ``'densities'``, ``'fuv_draine'``, ``'mass_values'``).
    set_x_name, set_y_name, set_mass_name : str, optional
        Labels passed to :func:`chi2_analysis_3d` for plots and printed output.
    set_z_name : str, optional
        Display label for the third axis in chi² / slice plots. If None, uses
        ``$\\zeta_{\\mathrm{H}}$`` when ``z_axis_name`` is CRIR-related, otherwise
        ``set_mass_name``.
    create_uncertainty_maps : bool, optional
        If True, also compute per-pixel uncertainty maps (profiled 1-sigma
        half-width in dex for each of x, y, z) plus chi^2_min and reduced chi^2
        maps, save them as FITS, and plot them. Default is True.
    uncertainty_delta_chi2 : float, optional
        Delta chi^2 threshold defining the per-parameter confidence interval.
        1.0 (default) is the 68% (1-sigma) interval for a single parameter;
        use 2.71 for 90% or 3.84 for 95%.
    uncertainty_cmap : str, optional
        Colormap for the uncertainty / chi^2 diagnostic maps. Default 'magma'.
    create_chi2_analysis : bool, optional
        Run ``chi2_analysis_3d`` on median footprint or ``chi2_analysis_pixel``.
    chi2_plot_species_slices : bool, optional
        Save three orthogonal slices showing ``model = observed`` contours for
        each line at the best-fit point. Default is True.
    chi2_species_slices_list : list of str, optional
        Lines/ratios to include on the slice plots. If None, uses intensity
        lines only (no ratios), up to ``chi2_species_slices_max``.
    create_model_ratios : bool, optional
        Build missing 3D model ratio grids via ``create_transition_ratios_3d``.
    create_ratios : bool, optional
        Build observed ratio maps from lines (2D) before fitting.

    Returns
    -------
    dict
        ``x_param_map``, ``y_param_map``, ``z_param_map``, FITS paths, and
        ``chi2_analysis`` (results dict from ``chi2_analysis_3d``, or None).
    """
    if not isinstance(observed_fits_files, dict) or not isinstance(grid_dicts_3d, dict):
        raise ValueError("observed_fits_files and grid_dicts_3d must be dictionaries")

    missing_grids = [line for line in observed_fits_files if line not in grid_dicts_3d]
    if missing_grids:
        raise ValueError(f"Missing 3D grids for lines: {missing_grids}")

    processed_grid_dicts = {}
    for line_name, grid_data in grid_dicts_3d.items():
        if isinstance(grid_data, str):
            print(f"{GREEN}Reading 3D grid FITS for {line_name}: {grid_data}{NC}")
            processed_grid_dicts[line_name] = _load_grid_from_fits_3d(
                grid_data, x_axis_name, y_axis_name, z_axis_name
            )
        else:
            processed_grid_dicts[line_name] = grid_data
    grid_dicts_3d = processed_grid_dicts

    if create_model_ratios:
        new_ratios = _create_model_ratios_3d(
            [n for n in observed_fits_files if '/' not in n],
            grid_dicts_3d,
        )
        if new_ratios:
            print(f"{GREEN}Added {len(new_ratios)} model ratio grid(s) from 3D lines.{NC}")
            grid_dicts_3d.update(new_ratios)

    num_lines = len(observed_fits_files)
    print(f"{GREEN}_Available lines: {list(observed_fits_files.keys())}{NC}")
    if num_lines <= 3:
        print(
            f"{ORANGE}Only {num_lines} line(s): 3D fit has 3 free parameters; "
            f"provide more lines or ratios to avoid degeneracy.{NC}"
        )

    observed_data, observed_errors, first_header, ny, nx = _read_observed_data(
        observed_fits_files, target_units, species_info, error_fraction, obs_errors
    )

    line_names_only = [n for n in observed_fits_files if '/' not in n]
    ratio_names_provided = [n for n in observed_fits_files if '/' in n]

    ratio_data = {}
    ratio_errors = {}
    if ratio_names_provided:
        for ratio_name in ratio_names_provided:
            fits_path = observed_fits_files[ratio_name]
            data, header = _read_fits_2d(fits_path)
            if data.shape != (ny, nx):
                raise ValueError(f"Ratio FITS {ratio_name} shape {data.shape} != ({ny}, {nx})")
            fits_unit = header.get('BUNIT', '')
            if fits_unit and target_units:
                data, _ = _convert_units_if_needed(
                    data, fits_unit, target_units, ratio_name, species_info
                )
            ratio_data[ratio_name] = data
            error_source = obs_errors.get(ratio_name) if obs_errors is not None else None
            ratio_errors[ratio_name] = _normalize_observed_error_map(
                error_source,
                data,
                ny,
                nx,
                error_fraction,
                ratio_name,
                target_units=target_units,
                species_info=species_info,
            )

    if create_ratios and not ratio_names_provided:
        created_ratios, created_errors = _create_ratios_from_data(
            line_names_only, observed_data, observed_errors, first_header,
            ny, nx, grid_dicts_3d, x_axis_name, y_axis_name, error_fraction
        )
        ratio_data.update(created_ratios)
        ratio_errors.update(created_errors)

    ratio_grids = {}
    for ratio_name in ratio_data:
        if ratio_name in grid_dicts_3d:
            ratio_grids[ratio_name] = grid_dicts_3d[ratio_name]

    all_data_names = list(line_names_only)
    all_observed_data = observed_data.copy()
    all_observed_errors = observed_errors.copy()
    if ratio_data:
        all_data_names.extend(list(ratio_data.keys()))
        all_observed_data.update(ratio_data)
        all_observed_errors.update(ratio_errors)

    first_grid = grid_dicts_3d[line_names_only[0]]
    x_mesh = _find_axis_mesh(
        first_grid, x_axis_name,
        ['densities', 'crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine'],
    )
    y_mesh = _find_axis_mesh(
        first_grid, y_axis_name,
        ['crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine', 'densities'],
    )
    z_mesh = _find_axis_mesh(
        first_grid, z_axis_name,
        ['mass_values', 'mass', 'crir_values', 'fuv_values', 'fuv_habing', 'fuv_draine'],
    )
    if x_mesh is None or y_mesh is None or z_mesh is None:
        raise ValueError(
            f"Could not find 3D axes (x={x_axis_name}, y={y_axis_name}, z={z_axis_name}). "
            f"Keys: {list(first_grid.keys())}"
        )

    X_mesh, Y_mesh, Z_mesh = _ensure_3d_meshgrids(x_mesh, y_mesh, z_mesh, first_grid['grid'].shape)

    # 1D parameter axes (grid shape is (n_z, n_y, n_x): x->axis2, y->axis1, z->axis0)
    x_values_1d = np.asarray(X_mesh[0, 0, :], dtype=float)
    y_values_1d = np.asarray(Y_mesh[0, :, 0], dtype=float)
    z_values_1d = np.asarray(Z_mesh[:, 0, 0], dtype=float)

    grid_arrays = {}
    for data_name in all_data_names:
        if data_name in line_names_only:
            source = grid_dicts_3d[data_name]
        elif data_name in ratio_grids:
            source = ratio_grids[data_name]
        else:
            print(f"{ORANGE}Warning: no 3D grid for {data_name}; skipping.{NC}")
            continue
        if 'grid' not in source:
            raise ValueError(f"Grid for {data_name} missing 'grid' key")
        grid_arrays[data_name] = source['grid']
        if grid_arrays[data_name].shape != X_mesh.shape:
            raise ValueError(
                f"Grid shape mismatch for {data_name}: "
                f"{grid_arrays[data_name].shape} != {X_mesh.shape}"
            )

    def _valid_pixel_mask(data_map, err_map):
        return np.isfinite(data_map) & np.isfinite(err_map) & (err_map > 0)

    candidate_ref_names = [
        n for n in all_data_names
        if n in grid_arrays and n in all_observed_data and n in all_observed_errors
    ]
    if not candidate_ref_names:
        raise ValueError("No valid reference map for output footprint.")

    ref_name = None
    ref_mask = None
    min_valid_count = None
    for name in candidate_ref_names:
        mask = _valid_pixel_mask(all_observed_data[name], all_observed_errors[name])
        valid_count = int(np.count_nonzero(mask))
        if min_valid_count is None or valid_count < min_valid_count:
            min_valid_count = valid_count
            ref_name = name
            ref_mask = mask

    ref_mask = ref_mask.astype(bool)
    n_ref_pixels = int(np.count_nonzero(ref_mask))
    print(f"{ORANGE}Reference footprint from {ref_name} ({n_ref_pixels} pixels).{NC}")

    print('-'*30)
    print(f"{BLUE}3D fitting {n_ref_pixels} pixels{NC}")
    print('-'*30)

    x_param_map = np.full((ny, nx), np.nan)
    y_param_map = np.full((ny, nx), np.nan)
    z_param_map = np.full((ny, nx), np.nan)

    if create_uncertainty_maps:
        x_sigma_map = np.full((ny, nx), np.nan)
        y_sigma_map = np.full((ny, nx), np.nan)
        z_sigma_map = np.full((ny, nx), np.nan)
        chi2_min_map = np.full((ny, nx), np.nan)
        chi2_red_map = np.full((ny, nx), np.nan)

    for i in tqdm(range(ny), desc="Processing rows"):
        for j in range(nx):
            if not ref_mask[i, j]:
                continue

            obs_values = {}
            obs_errs = {}
            valid_constraints = 0
            for data_name in all_data_names:
                if data_name not in grid_arrays:
                    continue
                obs_val = all_observed_data[data_name][i, j]
                obs_err = all_observed_errors[data_name][i, j]
                if (not np.isfinite(obs_val)) or (not np.isfinite(obs_err)) or obs_err <= 0:
                    continue
                obs_values[data_name] = obs_val
                obs_errs[data_name] = obs_err
                valid_constraints += 1

            if valid_constraints < 1:
                continue

            if create_uncertainty_maps:
                (x_param, y_param, z_param,
                 x_sigma, y_sigma, z_sigma,
                 chi2_min_pix, chi2_red_pix) = _fit_pixel_3d_with_uncertainty(
                    obs_values, obs_errs, grid_arrays, X_mesh, Y_mesh, Z_mesh,
                    x_values_1d, y_values_1d, z_values_1d, uncertainty_delta_chi2,
                )
                x_sigma_map[i, j] = x_sigma
                y_sigma_map[i, j] = y_sigma
                z_sigma_map[i, j] = z_sigma
                chi2_min_map[i, j] = chi2_min_pix
                chi2_red_map[i, j] = chi2_red_pix
            else:
                x_param, y_param, z_param = _fit_pixel_3d(
                    obs_values, obs_errs, grid_arrays, X_mesh, Y_mesh, Z_mesh
                )
            x_param_map[i, j] = x_param
            y_param_map[i, j] = y_param
            z_param_map[i, j] = z_param

    first_line = next(iter(observed_fits_files))
    fits_output_dir = (
        fig_dir_PATH if fig_dir_PATH else
        (output_dir if output_dir else os.path.dirname(os.path.abspath(observed_fits_files[first_line])))
    )
    os.makedirs(fits_output_dir, exist_ok=True)

    x_fits_name = f"{output_prefix}_{x_axis_name}.fits"
    y_fits_name = f"{output_prefix}_{y_axis_name}.fits"
    z_fits_name = f"{output_prefix}_{z_axis_name}.fits"
    x_fits_path = os.path.join(fits_output_dir, x_fits_name)
    y_fits_path = os.path.join(fits_output_dir, y_fits_name)
    z_fits_path = os.path.join(fits_output_dir, z_fits_name)

    x_header = _create_output_header(first_header, x_axis_name, x_axis_units, x_param_map)
    y_header = _create_output_header(first_header, y_axis_name, y_axis_units, y_param_map)
    z_header = _create_output_header(first_header, z_axis_name, z_axis_units, z_param_map)

    print('-'*30)
    print(f"{BLUE}Saving 3D-fit parameter FITS maps{NC}")
    print('-'*30)
    original_dir = os.getcwd()
    try:
        if fig_dir_PATH and or_PATH:
            os.chdir(fits_output_dir)
        fits.PrimaryHDU(data=x_param_map, header=x_header).writeto(x_fits_name, overwrite=True)
        fits.PrimaryHDU(data=y_param_map, header=y_header).writeto(y_fits_name, overwrite=True)
        fits.PrimaryHDU(data=z_param_map, header=z_header).writeto(z_fits_name, overwrite=True)
    finally:
        os.chdir(or_PATH if (fig_dir_PATH and or_PATH) else original_dir)

    print(f"{GREEN}Saved x map: {x_fits_path}{NC}")
    print(f"{GREEN}Saved y map: {y_fits_path}{NC}")
    print(f"{GREEN}Saved z map: {z_fits_path}{NC}")

    uncertainty_fits_paths = {}
    if create_uncertainty_maps:
        sigma_specs = [
            (f"{output_prefix}_{x_axis_name}_sigma_dex.fits", x_sigma_map, x_axis_name, 'dex',
             f'1-sigma uncertainty on log10({x_axis_name})'),
            (f"{output_prefix}_{y_axis_name}_sigma_dex.fits", y_sigma_map, y_axis_name, 'dex',
             f'1-sigma uncertainty on log10({y_axis_name})'),
            (f"{output_prefix}_{z_axis_name}_sigma_dex.fits", z_sigma_map, z_axis_name, 'dex',
             f'1-sigma uncertainty on log10({z_axis_name})'),
            (f"{output_prefix}_chi2_min.fits", chi2_min_map, 'chi2_min', '',
             'Minimum chi^2 per pixel'),
            (f"{output_prefix}_chi2_reduced.fits", chi2_red_map, 'chi2_reduced', '',
             'Reduced chi^2 per pixel (chi2_min / dof)'),
        ]

        print('-'*30)
        print(f"{BLUE}Saving 3D-fit uncertainty / chi^2 FITS maps{NC}")
        print('-'*30)
        original_dir = os.getcwd()
        try:
            if fig_dir_PATH and or_PATH:
                os.chdir(fits_output_dir)
            for fname, data_map, pname, punits, comment in sigma_specs:
                header = _create_output_header(first_header, pname, punits, data_map)
                header['COMMENT'] = comment
                fits.PrimaryHDU(data=data_map, header=header).writeto(fname, overwrite=True)
                full_path = os.path.join(fits_output_dir, fname)
                uncertainty_fits_paths[pname] = full_path
                print(f"{GREEN}Saved {pname} map: {full_path}{NC}")
        finally:
            os.chdir(or_PATH if (fig_dir_PATH and or_PATH) else original_dir)

    chi2_results = None
    if create_plots:
        plot_fitted_parameter_maps(
            x_param_map, y_param_map,
            x_axis_name, y_axis_name,
            x_axis_units, y_axis_units,
            z_param_map=z_param_map,
            z_axis_name=z_axis_name,
            z_axis_units=z_axis_units,
            set_x_name=set_x_name,
            set_y_name=set_y_name,
            set_z_name=set_z_name,
            set_mass_name=set_mass_name,
            fits_header=first_header,
            fig_dir_PATH=fig_dir_PATH,
            or_PATH=or_PATH,
            output_prefix=output_prefix,
            plot_format=plot_format,
            font_size=font_size if font_size else 19,
            fig_size=fig_size,
            cmap=cmap,
            use_log_colorbar=use_log_colorbar,
            plot_contours=plot_contours,
            plot_specific_contours=plot_specific_contours,
            plot_kde=plot_kde,
        )

    if create_plots and create_uncertainty_maps:
        conf_pct = ''
        if abs(uncertainty_delta_chi2 - 1.0) < 1e-6:
            conf_pct = ' (68%)'
        elif abs(uncertainty_delta_chi2 - 2.71) < 1e-2:
            conf_pct = ' (90%)'
        elif abs(uncertainty_delta_chi2 - 3.84) < 1e-2:
            conf_pct = ' (95%)'

        x_lab = _infer_axis_display_label(x_axis_name, set_x_name, set_mass_name=set_mass_name)
        y_lab = _infer_axis_display_label(y_axis_name, set_y_name, set_mass_name=set_mass_name)
        z_lab = _infer_axis_display_label(z_axis_name, set_z_name, set_mass_name=set_mass_name)
        unc_font = font_size if font_size else 19

        _plot_2d_diagnostic_map(
            x_sigma_map, rf'{x_lab} uncertainty{conf_pct}',
            r'$\sigma_{\log_{10}}$ [dex]', f'{output_prefix}_{x_axis_name}_sigma',
            fits_header=first_header, fig_dir_PATH=fig_dir_PATH, or_PATH=or_PATH,
            plot_format=plot_format, font_size=unc_font, fig_size=fig_size, cmap=uncertainty_cmap,
        )
        _plot_2d_diagnostic_map(
            y_sigma_map, rf'{y_lab} uncertainty{conf_pct}',
            r'$\sigma_{\log_{10}}$ [dex]', f'{output_prefix}_{y_axis_name}_sigma',
            fits_header=first_header, fig_dir_PATH=fig_dir_PATH, or_PATH=or_PATH,
            plot_format=plot_format, font_size=unc_font, fig_size=fig_size, cmap=uncertainty_cmap,
        )
        _plot_2d_diagnostic_map(
            z_sigma_map, rf'{z_lab} uncertainty{conf_pct}',
            r'$\sigma_{\log_{10}}$ [dex]', f'{output_prefix}_{z_axis_name}_sigma',
            fits_header=first_header, fig_dir_PATH=fig_dir_PATH, or_PATH=or_PATH,
            plot_format=plot_format, font_size=unc_font, fig_size=fig_size, cmap=uncertainty_cmap,
        )
        _plot_2d_diagnostic_map(
            chi2_red_map, r'Reduced $\chi^2$ (per pixel)',
            r'$\chi^2_\nu$', f'{output_prefix}_chi2_reduced',
            fits_header=first_header, fig_dir_PATH=fig_dir_PATH, or_PATH=or_PATH,
            plot_format=plot_format, font_size=unc_font, fig_size=fig_size, cmap=uncertainty_cmap,
        )

    if create_chi2_analysis:
        corner_obs, corner_err, corner_subtitle = _pixel_observations_for_chi2_analysis(
            all_observed_data,
            all_observed_errors,
            ref_mask,
            all_data_names,
            grid_arrays,
            pixel=chi2_analysis_pixel,
        )
        if corner_obs:
            print('-'*30)
            print(f"{BLUE}chi2_analysis_3d ({corner_subtitle}){NC}")
            print('-'*30)
            chi2_results = chi2_analysis_3d(
                grid_dicts_3d,
                observed_values=corner_obs,
                observed_errors=corner_err,
                set_x_name=set_x_name,
                set_y_name=set_y_name,
                set_mass_name=set_mass_name,
                set_z_name=set_z_name,
                plot_results=chi2_plot_results,
                plot_chi2_volume=chi2_plot_volume,
                plot_projections=chi2_plot_projections,
                plot_species_slices=chi2_plot_species_slices,
                species_slices_list=chi2_species_slices_list,
                species_slices_lines_only=chi2_species_slices_lines_only,
                species_slices_max=chi2_species_slices_max,
                colormap=chi2_corner_cmap,
                save_plot=bool(fig_dir_PATH),
                or_PATH=or_PATH,
                fig_dir_PATH=fig_dir_PATH,
                plot_type=plot_format,
            )
        else:
            print(f"{ORANGE}Warning: no valid data for chi2_analysis_3d; skipping.{NC}")

    result = {
        'x_param_map': x_param_map,
        'y_param_map': y_param_map,
        'z_param_map': z_param_map,
        'x_fits_path': x_fits_path,
        'y_fits_path': y_fits_path,
        'z_fits_path': z_fits_path,
        'chi2_analysis': chi2_results,
        'reference_line': ref_name,
        'data_names': all_data_names,
    }
    if create_uncertainty_maps:
        result.update({
            'x_sigma_map': x_sigma_map,
            'y_sigma_map': y_sigma_map,
            'z_sigma_map': z_sigma_map,
            'chi2_min_map': chi2_min_map,
            'chi2_reduced_map': chi2_red_map,
            'uncertainty_fits_paths': uncertainty_fits_paths,
        })
    return result
