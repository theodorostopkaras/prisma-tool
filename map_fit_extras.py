"""
Local helpers for FITS map fitting (vendored from KoSens3D).
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import numpy as np

BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
GREEN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m'
ORANGE = '\033[0;33m'

_CHI2_MAX_CONTRIB = 1e8
ZETA_H_LATEX = r'$\zeta_{\mathrm{H}}$'
ZETA_H_CONSOLE = 'zeta_H'



def convert_transition_names_func(names):
  """Pass-through: transition names are already spectroscopic in this tool."""
  return list(names)

def _spectroscopic_line_key_metadata(transition_name):
    """
    Read metadata from an **existing** spectroscopic line key ``Species(label)``.

    Used only to group lines by species and to order intra-molecular ratios. It
    does **not** construct, normalize, or rename transitions; those strings must
    already match keys in ``grid_dict`` (e.g. from SIMLINE column naming).

    Supported ``label`` patterns include simple rotational ``J-J'``, half-integer
    ``J/J'-J/J'``, fine-structure ``J_u K_u - J_l K_l``, and braced asymmetric
    labels where ``energy_level`` is inferred from the leading quantum number on
    the upper segment.

    Returns
    -------
    tuple
        ``(molecule, upper, lower, energy_level)``. ``upper``/``lower`` may be
        ``None`` when only a sort key is available (complex braced labels).

    Examples
    --------
    >>> _spectroscopic_line_key_metadata('CO(1-0)')
    ('CO', 1, 0, 1)

    >>> _spectroscopic_line_key_metadata('C+(3_2-1_2)')
    ('C+', 1.5, 0.5, 1.5)
    """
    if '(' not in transition_name or ')' not in transition_name:
        return None, None, None, None

    molecule = transition_name.split('(')[0]
    transition_part = transition_name.split('(')[1].split(')')[0].strip()
    if not transition_part:
        return None, None, None, None

    # Simple rotational: ``N-N`` or ``N/M-N/M`` (half-integer style).
    m_rot = re.match(r'^(\d+|\d+/\d+)-(\d+|\d+/\d+)$', transition_part)
    if m_rot:
        uq = Fraction(m_rot.group(1))
        lq = Fraction(m_rot.group(2))
        upper = int(uq) if uq.denominator == 1 else float(uq)
        lower = int(lq) if lq.denominator == 1 else float(lq)
        energy_level = float(uq)
        return molecule, upper, lower, energy_level

    # Fine structure ``J_u K_u - J_l K_l`` with exactly one underscore per side
    # (e.g. ``3_2-1_2``), not braced asymmetric labels.
    m_fs = re.match(r'^(\d+)_(\d+)-(\d+)_(\d+)$', transition_part)
    if m_fs and '{' not in transition_part:
        upper = float(f'{m_fs.group(1)}.{m_fs.group(2)}')
        lower = float(f'{m_fs.group(3)}.{m_fs.group(4)}')
        energy_level = upper
        return molecule, upper, lower, energy_level

    # Braced / multi-token asymmetric (e.g. ``10_{0-10} - 9_{1-9}``, ``5_{4} - 4_{4}``):
    # use leading integer or fraction on the upper segment for sorting.
    upper_seg = transition_part.split(' - ', 1)[0].strip()
    m_lead = re.match(r'^(\d+/\d+|\d+)', upper_seg)
    if m_lead:
        try:
            energy_level = float(Fraction(m_lead.group(1)))
            return molecule, None, None, energy_level
        except (ValueError, ZeroDivisionError):
            pass

    return None, None, None, None
#---------------------------------------------------------------------------------------------------------------------------------------

def parse_transition(transition_name):
    """
    Parse a spectroscopic line key for ratio ordering in ``grid_3d``.

    Thin wrapper around :func:`_spectroscopic_line_key_metadata` for simple
    rotational and fine-structure labels.
    """
    return _spectroscopic_line_key_metadata(transition_name)
########################################################################################################################################
#                                               Function for creating transition ratios
########################################################################################################################################

def create_transition_ratios(grid_dict, species_list=None,convert_transition_names=False):
    """
    Create line intensity ratio grids from molecular transition grids.

    This function automatically generates meaningful line ratios from a collection
    of molecular transition grids. It creates both intra-molecule ratios (e.g.,
    CO(2-1)/CO(1-0)) and inter-molecule ratios (e.g., C+(3/2-1/2)/CO(1-0)).

    Transition **names** in ratio keys are always the strings already present as
    keys in ``grid_dict``; nothing here invents new line labels. A small metadata
    helper (:func:`_spectroscopic_line_key_metadata`) only reads each key to obtain
    the species and an energy ordering for the ratio rules.

    Parameters
    ----------
    grid_dict : dict
        Dictionary containing grid data for each species, as returned by
        create_final_grids(). Each entry should have structure:
        {'grid': numpy.ndarray, 'densities': numpy.ndarray, 'crir_values': numpy.ndarray}
    species_list : list of str or None
        Optional whitelist of transition names. When ``None`` or empty, every
        non-ratio key in ``grid_dict`` is used (recommended so complex SIMLINE
        labels are not dropped). When provided, only transitions that resolve to
        a key in ``grid_dict`` are included.
    convert_transition_names : bool, optional
        If True, convert simple transition names (e.g., 'CO-0') to spectroscopic
        notation (e.g., 'CO(1-0)') before processing. Default is False.

    Returns
    -------
    dict
        Dictionary with ratio names as keys and grid data as values.
        Each entry has structure:
        {
            'ratio_name': {
                'grid': numpy.ndarray,  # 2D ratio values
                'densities': numpy.ndarray,  # Y-axis meshgrid
                'crir_values': numpy.ndarray  # X-axis meshgrid (or 'fuv_values')
            },
            ...
        }

    Notes
    -----
    Ratio creation follows these rules:

    **Intra-molecule ratios:**
    - Higher energy transition in numerator, lower in denominator
    - Example: CO(2-1)/CO(1-0), not CO(1-0)/CO(2-1)

    **Inter-molecule ratios:**
    - Uses molecule priority ordering (higher priority in numerator):
        C+ (7) > C (6) > HCO+ (5) > CO (4) > 13CO (3) > C18O (2) > H13CO+ (1)
    - Example: C+(3/2-1/2)/CO(1-0), not CO(1-0)/C+(3/2-1/2)

    Division by zero is handled by returning NaN for those grid points.

    Examples
    --------
    >>> ratio_grids = create_transition_ratios(my_grids)
    >>> ratio_grids = create_transition_ratios(
    ...     my_grids,
    ...     ['CO(1-0)', 'CO(2-1)', 'C(1-0)'],
    ... )
    >>> # Access CO(2-1)/CO(1-0) ratio
    >>> co_ratio = ratio_grids['CO(2-1)/CO(1-0)']['grid']
    """
    
    ratio_grids = {}
    
    # Determine the parameter keys (x/y axes) once so they're available even
    # when no intra-molecule ratios are created.
    if len(grid_dict) == 0:
        return ratio_grids
    first_item = next(iter(grid_dict.values()))
    item_keys = list(first_item.keys())

    def _spatial_keys(entry):
        keys = [k for k in entry if k != 'grid']
        if len(keys) >= 2:
            return keys[0], keys[1]
        return item_keys[1], item_keys[2]

    def _simple_token_from_spectroscopic_name(name):
        """Best-effort conversion: SPECIES(upper-lower) -> SPECIES-lower."""
        molecule, upper, lower, _ = _spectroscopic_line_key_metadata(name)
        if molecule is None:
            return None

        # Standard rotational ladders
        if isinstance(upper, (int, np.integer)) and isinstance(lower, (int, np.integer)):
            if upper == lower + 1:
                return f"{molecule}-{int(lower)}"
            return None

        # C+ fine-structure ladder: C+(3_2-1_2) <-> C+-0, C+(5_2-3_2) <-> C+-1, ...
        if molecule == 'C+' and isinstance(upper, float) and isinstance(lower, float):
            n_float = (2.0 * lower - 1.0) / 2.0
            n_round = int(round(n_float))
            if np.isclose(n_float, n_round) and np.isclose(upper, lower + 1.0):
                return f"{molecule}-{n_round}"
        return None

    def _resolve_grid_key(name):
        """
        Resolve transition name to an existing key in grid_dict.
        Tries exact key, converted key, and simple-token aliases.
        """
        if name in grid_dict:
            return name

        candidates = []
        converted_name = convert_transition_names_func([name])[0]
        candidates.append(converted_name)
        simple_from_name = _simple_token_from_spectroscopic_name(name)
        if simple_from_name:
            candidates.append(simple_from_name)
        simple_from_converted = _simple_token_from_spectroscopic_name(converted_name)
        if simple_from_converted:
            candidates.append(simple_from_converted)

        for candidate in candidates:
            if candidate in grid_dict:
                return candidate
        return None

    def _transition_keys_from_grids():
        """Single-line keys only (exclude existing ratio names containing ``/``)."""
        return [k for k in grid_dict if '/' not in str(k)]

    # Prefer keys already present in grid_dict so complex SIMLINE labels are kept.
    if not species_list:
        transition_keys = _transition_keys_from_grids()
    else:
        sl = (
            convert_transition_names_func(species_list)
            if convert_transition_names
            else list(species_list)
        )
        transition_keys = []
        seen = set()
        for entry in sl:
            key = _resolve_grid_key(entry)
            if key is None or key in seen or '/' in str(key):
                continue
            seen.add(key)
            transition_keys.append(key)

    all_transitions = []
    for key in transition_keys:
        if key not in grid_dict:
            continue
        try:
            molecule, upper, lower, energy_level = _spectroscopic_line_key_metadata(key)
        except Exception:
            continue
        if molecule is None or energy_level is None:
            continue
        all_transitions.append({
            'name': key,
            'molecule': molecule,
            'upper': upper,
            'lower': lower,
            'energy_level': energy_level,
            'grid_data': grid_dict[key],
        })

    # Create ratios within same molecule (higher energy / lower energy)
    molecules = {}
    for trans in all_transitions:
        mol = trans['molecule']
        if mol not in molecules:
            molecules[mol] = []
        molecules[mol].append(trans)
    
    for molecule, transitions in molecules.items():
        if len(transitions) > 1:
            # Sort by energy level (higher energy transitions first)
            transitions.sort(key=lambda x: x['energy_level'], reverse=True)
            
            # Create ratios: higher transition / lower transition
            for i in range(len(transitions)):
                for j in range(i+1, len(transitions)):
                    numerator = transitions[i]
                    denominator = transitions[j]
                    
                    # Verify higher/lower ordering
                    if numerator['energy_level'] > denominator['energy_level']:
                        # Create ratio name
                        ratio_name = f"{numerator['name']}/{denominator['name']}"
                        
                        # Calculate ratio grid
                        num_grid = numerator['grid_data']['grid']
                        den_grid = denominator['grid_data']['grid']
                        
                        # Avoid division by zero
                        ratio_grid = np.divide(num_grid, den_grid, 
                                             out=np.full_like(num_grid, np.nan), 
                                             where=(den_grid != 0) & (~np.isnan(den_grid)))
                        ax1, ax2 = _spatial_keys(numerator['grid_data'])
                        ratio_grids[ratio_name] = {
                            'grid': ratio_grid,
                            ax1: numerator['grid_data'][ax1],
                            ax2: numerator['grid_data'][ax2],
                        }
                        
                        print(f"  Created: {ratio_name}")
    
    # Define molecule priority (higher number = higher priority, goes in numerator)
    # Based on your examples: C+ > C > HCO+ > CO
    molecule_priority = {
        'C+'    : 13,
        'C'     : 12,
        'HCO+'  : 11,
        'CO'    : 10,
        '13CO'  : 9,
        'C18O'  : 8,
        'H13CO+': 7,
        'HCN'   : 6,
        'HNC'   : 5 ,
        'H3O+'  : 4 ,
        'N2H+'  : 3,
        'SO2'   : 2,
        'SO'    : 1,
        'AR'    : 0 
    }
    
    print(f"Molecule priority order (higher in numerator): {dict(sorted(molecule_priority.items(), key=lambda x: x[1], reverse=True))}")
    
    # Get unique molecules and create unique pairs to avoid duplicates
    unique_molecules = list(molecules.keys())
    
    # Process each unique pair of molecules only once
    for i, mol1 in enumerate(unique_molecules):
        for j, mol2 in enumerate(unique_molecules):
            if i < j:  # Only process unique pairs once (i < j ensures each pair processed only once)
                # Determine which molecule should be in numerator based on priority
                priority1 = molecule_priority.get(mol1, 0)
                priority2 = molecule_priority.get(mol2, 0)
                
                if priority1 > priority2:
                    numerator_transitions = molecules[mol1]
                    denominator_transitions = molecules[mol2]
                elif priority2 > priority1:
                    numerator_transitions = molecules[mol2]
                    denominator_transitions = molecules[mol1]
                else:
                    # Equal priority, skip to avoid duplicates
                    continue
                
                # Create ratios: higher_priority_molecule / lower_priority_molecule
                for num_trans in numerator_transitions:
                    for den_trans in denominator_transitions:
                        # Create ratio name
                        ratio_name = f"{num_trans['name']}/{den_trans['name']}"
                        
                        # Check if this ratio already exists (safety check)
                        if ratio_name in ratio_grids:
                            print(f"  Warning: {ratio_name} already exists, skipping duplicate")
                            continue
                        
                        # Calculate ratio grid
                        num_grid = num_trans['grid_data']['grid']
                        den_grid = den_trans['grid_data']['grid']
                        
                        # Avoid division by zero
                        ratio_grid = np.divide(num_grid, den_grid, 
                                             out=np.full_like(num_grid, np.nan), 
                                             where=(den_grid != 0) & (~np.isnan(den_grid)))
                        ax1, ax2 = _spatial_keys(num_trans['grid_data'])
                        ratio_grids[ratio_name] = {
                            'grid': ratio_grid,
                            ax1: num_trans['grid_data'][ax1],
                            ax2: num_trans['grid_data'][ax2],
                        }
                        
                        print(f"  Created: {ratio_name}")
    
    print(f"Total ratios created: {len(ratio_grids)}")
    return ratio_grids
########################################################################################################################################
#                                               Function for reading FITS files and creating grid structure
########################################################################################################################################
#TODO This function works pefectly for the density, fuv, cr as axis. It fails though if the axis are declination 
# right ascension. I have to change this feature here in order to take into account this information given in the fits 
# files header. This serves if the user would like to change the units in a observed map instead of a grid produced 
# by simulated data.

def _detect_3d_meshgrid_keys(grid_data):
    """
    Return dict keys for the (x, y, z) meshgrids in a species entry from create_final_grids_3d.

    Grids use indexing='ij' with shape (n_z, n_y, n_x): z varies along axis 0,
    y along axis 1, x along axis 2.
    """
    # coord_keys = list(next(iter(grid_data.values())).keys())
    # Get the name (key) and the contained dict for the first species in my_grids_3d
    first_species_name, first_species_dict = next(iter(grid_data.items()))
    coord_keys = list(key for key in first_species_dict.keys() if key != 'grid')
    
    # coord_keys = sub_keys
    if len(coord_keys) != 3:
        raise ValueError(f"Expected exactly 3 coordinate mesh keys, got {coord_keys}")

    x_key = y_key = z_key = None
    for key in coord_keys:
        mesh = grid_data[first_species_name][key]
        varies_axis0 = not np.isclose(mesh[0, 0, 0], mesh[-1, 0, 0], rtol=1e-12, atol=1e-19)
        varies_axis1 = not np.isclose(mesh[0, 0, 0], mesh[0, -1, 0], rtol=1e-12, atol=1e-19)
        varies_axis2 = not np.isclose(mesh[0, 0, 0], mesh[0, 0, -1], rtol=1e-12, atol=1e-19)
        if varies_axis0 and not varies_axis1 and not varies_axis2:
            z_key = key
        elif varies_axis1 and not varies_axis0 and not varies_axis2:
            y_key = key
        elif varies_axis2 and not varies_axis0 and not varies_axis1:
            x_key = key

    if x_key is None or y_key is None or z_key is None:
        raise ValueError(
            f"Could not classify x/y/z mesh keys for keys {coord_keys}; "
            f"detected x={x_key}, y={y_key}, z={z_key}"
        )
    return x_key, y_key, z_key

def create_transition_ratios_3d(species_list, grid_dict_3d, convert_transition_names=False):
    """
    Create 3D line intensity ratio grids from molecular transition grids.

    This function automatically generates meaningful line ratios from a collection
    of 3D molecular transition grids. It creates both intra-molecule ratios (e.g.,
    CO(2-1)/CO(1-0)) and inter-molecule ratios (e.g., C+(3/2-1/2)/CO(1-0)).

    Parameters
    ----------
    species_list : list of str
        List of transition names in spectroscopic notation
        (e.g., ['CO(1-0)', 'CO(2-1)', 'C(1-0)', 'C+(3_2-1_2)']).
    grid_dict_3d : dict
        Dictionary containing 3D grid data for each species, as returned by
        create_final_grids_3d(). Each entry should have structure:
        {'grid': numpy.ndarray (n_masses, ny, nx), 'densities': numpy.ndarray, 
         'crir_values': numpy.ndarray, 'mass_values': numpy.ndarray}
    convert_transition_names : bool, optional
        If True, convert simple transition names (e.g., 'CO-0') to spectroscopic
        notation (e.g., 'CO(1-0)') before processing. Default is False.

    Returns
    -------
    dict
        Dictionary with ratio names as keys and 3D grid data as values.
        Each entry has structure:
        {
            'ratio_name': {
                'grid': numpy.ndarray,        # 3D ratio values (n_masses, ny, nx)
                'densities': numpy.ndarray,   # 3D Y-axis meshgrid
                'crir_values': numpy.ndarray, # 3D X-axis meshgrid
                'mass_values': numpy.ndarray  # 3D mass meshgrid
            },
            ...
        }
    """
    
    ratio_grids_3d = {}
    
    # Determine the parameter keys (x/y/z axes) once
    if len(grid_dict_3d) == 0:
        return ratio_grids_3d
    # first_item = next(iter(grid_dict_3d.values()))
    x_key, y_key, z_key = _detect_3d_meshgrid_keys(grid_dict_3d)
    
    # Parse all transitions and create a flat list
    if convert_transition_names:
        species_list = convert_transition_names_func(species_list)
    
    all_transitions = []
    for species in species_list:
        molecule, upper, lower, energy_level = parse_transition(species)
        if molecule is not None and species in grid_dict_3d:
            all_transitions.append({
                'name'          : species,
                'molecule'      : molecule,
                'upper'         : upper,
                'lower'         : lower,
                'energy_level'  : energy_level,
                'grid_data'     : grid_dict_3d[species]
            })

    # Create ratios within same molecule (higher energy / lower energy)
    molecules = {}
    for trans in all_transitions:
        mol = trans['molecule']
        if mol not in molecules:
            molecules[mol] = []
        molecules[mol].append(trans)
    
    for molecule, transitions in molecules.items():
        if len(transitions) > 1:
            # Sort by energy level (higher energy transitions first)
            transitions.sort(key=lambda x: x['energy_level'], reverse=True)
            
            # Create ratios: higher transition / lower transition
            for i in range(len(transitions)):
                for j in range(i+1, len(transitions)):
                    numerator = transitions[i]
                    denominator = transitions[j]
                    
                    # Verify higher/lower ordering
                    if numerator['energy_level'] > denominator['energy_level']:
                        # Create ratio name
                        ratio_name = f"{numerator['name']}/{denominator['name']}"
                        
                        # Calculate ratio grid (3D) directly from stored line grids
                        # so results match manual line-by-line division.
                        num_grid = numerator['grid_data']['grid']
                        den_grid = denominator['grid_data']['grid']
                        
                        # Avoid invalid denominator values (zero/NaN) by marking as NaN.
                        ratio_grid = np.divide(
                            num_grid,
                            den_grid,
                            out=np.full_like(num_grid, np.nan),
                            where=(den_grid != 0) & (~np.isnan(den_grid))
                        )
                        
                        # Validate that coordinate grids have matching shapes
                        num_coord_y = numerator['grid_data'][y_key]
                        num_coord_x = numerator['grid_data'][x_key]
                        num_z = numerator['grid_data'][z_key]
                        if ratio_grid.shape != num_coord_y.shape or ratio_grid.shape != num_coord_x.shape or ratio_grid.shape != num_z.shape:
                            raise ValueError(f"{RED}Coordinate grid shape mismatch for ratio {ratio_name}: "
                                           f"ratio_grid={ratio_grid.shape}, coords={num_coord_y.shape}{NC}")
                        
                        ratio_grids_3d[ratio_name] = {
                            'grid': ratio_grid,
                            y_key: num_coord_y,
                            x_key: num_coord_x,
                            z_key: num_z,
                        }
                        
                        print(f"  Created 3D ratio: {ratio_name}")
    
    # Define molecule priority (higher number = higher priority, goes in numerator)
    molecule_priority = {
        'C+'    : 9,
        'C'     : 8,
        'HCO+'  : 7,
        'CO'    : 6,
        '13CO'  : 5,
        'C18O'  : 4,
        'H13CO+': 3,
        'HNC'   : 2,
        'HCN'   : 1 
    }
    
    # Get unique molecules and create unique pairs
    unique_molecules = list(molecules.keys())
    
    # Process each unique pair of molecules only once
    for i, mol1 in enumerate(unique_molecules):
        for j, mol2 in enumerate(unique_molecules):
            if i < j:
                priority1 = molecule_priority.get(mol1, 0)
                priority2 = molecule_priority.get(mol2, 0)
                
                if priority1 > priority2:
                    numerator_transitions = molecules[mol1]
                    denominator_transitions = molecules[mol2]
                elif priority2 > priority1:
                    numerator_transitions = molecules[mol2]
                    denominator_transitions = molecules[mol1]
                else:
                    continue
                
                # Create ratios: higher_priority_molecule / lower_priority_molecule
                for num_trans in numerator_transitions:
                    for den_trans in denominator_transitions:
                        ratio_name = f"{num_trans['name']}/{den_trans['name']}"
                        
                        if ratio_name in ratio_grids_3d:
                            continue
                        
                        num_grid = num_trans['grid_data']['grid']
                        den_grid = den_trans['grid_data']['grid']
                        
                        ratio_grid = np.divide(
                            num_grid,
                            den_grid,
                            out=np.full_like(num_grid, np.nan),
                            where=(den_grid != 0) & (~np.isnan(den_grid))
                        )
                        
                        # Validate that coordinate grids have matching shapes
                        num_coord_y = num_trans['grid_data'][y_key]
                        num_coord_x = num_trans['grid_data'][x_key]
                        num_z = num_trans['grid_data'][z_key]
                        if ratio_grid.shape != num_coord_y.shape or ratio_grid.shape != num_coord_x.shape or ratio_grid.shape != num_z.shape:
                            raise ValueError(f"{RED}Coordinate grid shape mismatch for ratio {ratio_name}: "
                                           f"ratio_grid={ratio_grid.shape}, coords={num_coord_y.shape}{NC}")
                        
                        ratio_grids_3d[ratio_name] = {
                            'grid': ratio_grid,
                            y_key: num_coord_y,
                            x_key: num_coord_x,
                            z_key: num_z,
                        }
                        
                        print(f"  Created 3D ratio: {ratio_name}")
    
    print(f"Total 3D ratios created: {len(ratio_grids_3d)}")
    return ratio_grids_3d
########################################################################################################################################
#                                           Create abundance 3D grids from dataframe
########################################################################################################################################

def _is_crir_axis_key(axis_key):
    z = str(axis_key).lower()
    return any(token in z for token in ('crir', 'cosray', 'zeta'))
#-----------------------------------------------------------------------------------------------------------

def _is_mass_axis_key(axis_key):
    return 'mass' in str(axis_key).lower()
#-----------------------------------------------------------------------------------------------------------

def _is_density_axis_key(axis_key):
    z = str(axis_key).lower()
    if z in ('n', 'densities', 'density', 'protdens'):
        return True
    return z.startswith('n_') or z.endswith('_n')
#-----------------------------------------------------------------------------------------------------------

def _is_fuv_axis_key(axis_key):
    z = str(axis_key).lower()
    return any(token in z for token in ('fuv', 'radm', 'habing', 'draine', 'g_0', 'g0', 'chi_0'))
# Physical units for the grid parameter axes, rendered inside math mode so that
# exponents display as proper superscripts.
_DIAGNOSTIC_AXIS_UNITS = (
    (_is_density_axis_key, r'\mathrm{cm^{-3}}'),
    (_is_fuv_axis_key,     r'\mathrm{Draine}'),
    (_is_crir_axis_key,    r'\mathrm{s^{-1}}'),
)
#-----------------------------------------------------------------------------------------------------------

def _is_generic_axis_label(label, axis_key=None):
    if label is None:
        return True
    text = str(label).strip().strip('$').replace(r'\mathrm{H}', 'h').lower()
    generic = {
        'm', 'mass', 'ζ', 'zeta', 'z', '', 'n', 'fuv', 'crir',
        'densities', 'fuv_values', 'crir_values', 'g_0', 'g0', 'chi_0',
    }
    if text in generic or label in ('M', 'ζ', 'FUV'):
        return True
    if axis_key is not None and text == str(axis_key).lower():
        return True
    return False
#-----------------------------------------------------------------------------------------------------------

def _append_axis_units(base, axis_units):
    if not axis_units or '[' in str(base):
        return base
    if str(base).startswith('$') and str(base).endswith('$'):
        inner = str(base)[1:-1]
        return rf'${inner}$ [{axis_units}]'
    return f'{base} [{axis_units}]'
#-----------------------------------------------------------------------------------------------------------

def _infer_axis_display_label(axis_key, set_name=None, axis_units='', set_mass_name='M'):
    """
    Resolve a matplotlib display label for a grid parameter axis key.

    Maps ``densities`` -> ``$n$``, FUV keys -> ``$G_0$``, CRIR keys ->
    ``$\\zeta_{\\mathrm{H}}$``, and mass keys -> ``set_mass_name``.
    """
    if set_name is not None and not _is_generic_axis_label(set_name, axis_key):
        return _append_axis_units(set_name, axis_units)
    if _is_crir_axis_key(axis_key):
        return _append_axis_units(ZETA_H_LATEX, axis_units)
    if _is_fuv_axis_key(axis_key):
        return _append_axis_units(r'$G_0$', axis_units)
    if _is_density_axis_key(axis_key):
        return _append_axis_units(r'$n$', axis_units)
    if _is_mass_axis_key(axis_key):
        base = set_name if set_name is not None else set_mass_name
        return _append_axis_units(base, axis_units)
    return _append_axis_units(str(axis_key), axis_units)
#-----------------------------------------------------------------------------------------------------------

def _third_axis_plain_label(label):
    """Plain label for console output (no broken LaTeX in the terminal)."""
    text = str(label)
    if ZETA_H_LATEX in text or (
        r'\zeta' in text and (r'\mathrm{H}' in text or r'_{\mathrm{H}}' in text)
    ):
        return ZETA_H_CONSOLE
    return text.strip('$')
#-----------------------------------------------------------------------------------------------------------

def _log10_third_axis_label(label):
    """Matplotlib label for log10(third axis parameter)."""
    text = str(label)
    if ZETA_H_LATEX in text or (
        r'\zeta' in text and (r'\mathrm{H}' in text or r'_{\mathrm{H}}' in text)
    ):
        return r'$\log_{10}(\zeta_{\mathrm{H}})$'
    if text.startswith('$') and text.endswith('$'):
        return rf'$\log_{{10}}({text[1:-1]})$'
    return rf'$\log_{{10}}({text})$'
#-----------------------------------------------------------------------------------------------------------

def _chi2_contribution(model_grid, obs_val, obs_err, max_contrib=_CHI2_MAX_CONTRIB):
    """Single-species chi² contribution with finite, capped values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        contrib = ((model_grid - obs_val) / obs_err) ** 2
    contrib = np.where(np.isfinite(contrib), contrib, max_contrib)
    return np.minimum(contrib, max_contrib)
#-----------------------------------------------------------------------------------------------------------

def _chi2_plot_surface(ax, X, Y, chi2_data, chi2_min, cmap, *, cap=25.0):
    """
    Plot a Δχ² heatmap with an explicit colour normalisation.

    Matplotlib ``contourf`` ignores ``vmin``/``vmax`` for the colourbar when the
    raw data span many orders of magnitude; passing a ``Normalize`` instance
    fixes the scale around the minimum.
    """
    from matplotlib.colors import Normalize

    reversed_cmap = cmap if isinstance(cmap, str) and cmap.endswith('_r') else f'{cmap}_r'
    delta = np.asarray(chi2_data, dtype=float) - chi2_min
    finite = delta[np.isfinite(delta)]
    vmax = cap
    if finite.size:
        vmax = min(cap, float(np.nanpercentile(finite, 95)))
    if vmax <= 0:
        vmax = cap

    norm = Normalize(vmin=0.0, vmax=vmax)
    levels = np.linspace(0.0, vmax, 51)
    mappable = ax.contourf(
        X, Y, delta, levels=levels, cmap=reversed_cmap, norm=norm, extend='max',
    )
    return mappable, delta, vmax
#-----------------------------------------------------------------------------------------------------------

def _detect_3d_axis_keys(grid_data):
    """
    Detect coordinate keys for x, y, and z axes from 3D meshgrids.

    Grid shape is (n_z, n_y, n_x). With indexing='ij':
    - Axis 0 varies along Z
    - Axis 1 varies along Y
    - Axis 2 varies along X

    Parameters
    ----------
    grid_data : dict
        Grid data dictionary with coordinate meshgrids.

    Returns
    -------
    tuple
        (x_key, y_key, z_key) - keys for x, y, and z coordinate meshgrids
    """
    item_keys = list(grid_data.keys())
    coord_keys = [k for k in item_keys if k != 'grid']

    if len(coord_keys) < 3:
        raise ValueError(f"Expected at least 3 coordinate keys, found {coord_keys}")

    x_key = None
    y_key = None
    z_key = None

    for key in coord_keys:
        mesh = grid_data[key]
        # Identify which axis this coordinate varies along.
        # Use finite-range checks (ptp) instead of fixed atol comparisons:
        # very small physical axes (e.g. CRIR ~1e-18..1e-14) can be
        # incorrectly treated as constant with atol=1e-12.
        axis0_line = mesh[:, 0, 0]
        axis1_line = mesh[0, :, 0]
        axis2_line = mesh[0, 0, :]
        range0 = np.nanmax(axis0_line) - np.nanmin(axis0_line)
        range1 = np.nanmax(axis1_line) - np.nanmin(axis1_line)
        range2 = np.nanmax(axis2_line) - np.nanmin(axis2_line)
        varies_axis0 = np.isfinite(range0) and (range0 > 0.0)
        varies_axis1 = np.isfinite(range1) and (range1 > 0.0)
        varies_axis2 = np.isfinite(range2) and (range2 > 0.0)

        if varies_axis0 and not varies_axis1 and not varies_axis2:
            z_key = key
        elif varies_axis1 and not varies_axis0 and not varies_axis2:
            y_key = key
        elif varies_axis2 and not varies_axis0 and not varies_axis1:
            x_key = key

    if z_key is None:
        for k in coord_keys:
            if k not in (x_key, y_key):
                z_key = k
                break
        if z_key is None:
            raise ValueError(f"Could not determine z axis key from {coord_keys}")

    if y_key is None or x_key is None:
        pool = [k for k in coord_keys if k != z_key]
        if len(pool) != 2:
            raise ValueError(
                f"Could not resolve x/y mesh keys for {coord_keys} (z_key={z_key})"
            )
        a, b = pool[0], pool[1]
        if y_key is None and x_key is None:
            y_key, x_key = a, b
        elif y_key is None:
            y_key = b if a == x_key else a
        else:
            x_key = b if a == y_key else a

    return x_key, y_key, z_key
#-----------------------------------------------------------------------------------------------------------

def _compute_chi2_grid(my_grids_3d, valid_species, observed_values, observed_errors):
    """
    Compute chi² grid over 3D parameter space.
    
    Parameters
    ----------
    my_grids_3d : dict
        Dictionary containing 3D grid data for each species
    valid_species : list
        List of species to include in chi² calculation
    observed_values : dict
        Observed values for each species
    observed_errors : dict
        Errors on observed values for each species
        
    Returns
    -------
    numpy.ndarray
        3D chi² grid with shape (n_mass, n_y, n_x)
    """
    first_species       = valid_species[0]
    n_mass, n_y, n_x    = my_grids_3d[first_species]['grid'].shape
    chi2_grid           = np.zeros((n_mass, n_y, n_x))
    
    for species in valid_species:
        model_grid = my_grids_3d[species]['grid']
        obs_val = observed_values[species]
        obs_err = observed_errors[species]
        chi2_grid += _chi2_contribution(model_grid, obs_val, obs_err)

    return chi2_grid
########################################################################################################################################
#                                               Find nearest contour points in 3D
########################################################################################################################################

def _check_at_boundary(log_mass, log_y, log_x, mass_log, y_log, x_log, tolerance=0.01):
    """
    Check if a point is at the boundary of the parameter space.
    
    Parameters
    ----------
    log_mass : array
        Log10 of mass values array
    log_y : array
        Log10 of y values array
    log_x : array
        Log10 of x values array
    mass_log : float
        Log10 of mass to check
    y_log : float
        Log10 of y to check
    x_log : float
        Log10 of x to check
    tolerance : float, optional
        Tolerance for boundary check. Default is 0.01.
        
    Returns
    -------
    bool
        True if point is at boundary, False otherwise
    """
    return (abs(mass_log - log_mass.min()) < tolerance or
            abs(mass_log - log_mass.max()) < tolerance or
            abs(y_log - log_y.min()) < tolerance or
            abs(y_log - log_y.max()) < tolerance or
            abs(x_log - log_x.min()) < tolerance or
            abs(x_log - log_x.max()) < tolerance)
#-----------------------------------------------------------------------------------------------------------

def chi2_analysis_3d(my_grids_3d,
                     observed_values,
                     observed_errors,
                     set_x_name              = 'n',
                     set_y_name              = 'FUV',
                     set_mass_name           = 'M',
                     set_z_name              = None,
                     plot_results            = True,
                     plot_chi2_volume        = False,
                     plot_projections        = True,
                     plot_species_slices     = False,
                     plot_species_surfaces   = False,
                     species_slices_list     = None,
                     species_slices_lines_only = True,
                     species_slices_max      = 8,
                     font_size               = None,
                     fig_size                = None,
                     colormap                = 'viridis',
                     save_plot               = False,
                     or_PATH                 = False,
                     fig_dir_PATH            = False,
                     plot_type               = 'pdf',
                     confidence_levels       = [0.68, 0.95, 0.99],
                     convert_transition_names = False,
                     save_plotly_to_html     = False,
                     plotly_output_dir       = 'plotly_figs'):
    """
    Perform chi-squared analysis in 3D parameter space to find optimal parameters.
    
    This function computes chi² over the full 3D parameter space (x, y, mass)
    in log space, finds the grid point with minimum chi², and visualizes
    the results at the optimal mass slice.
    
    The chi² is computed in log10 space:
        χ² = Σᵢ [(log10(model_i) - log10(observed_i)) / σ_log,i]²
    
    where the sum is over all species/transitions and σ_log is the
    propagated error in log space.
    
    Parameters
    ----------
    my_grids_3d : dict
        Dictionary containing 3D grid data with structure:
        my_grids_3d[species] = {
            'grid': 3D numpy array (n_masses, ny, nx),
            'densities': 3D meshgrid,
            'fuv_values' or 'crir_values': 3D meshgrid,
            'mass_values': 3D meshgrid
        }
    observed_values : dict or list
        Observed intensity values for each species. Can be:
        - dict: {species_name: value}
        - list: values in same order as species_list
    observed_errors : dict or list
        Errors on observed values. Same format as observed_values.
    species_list : list of str
        List of species names to include in chi² calculation.
    set_x_name : str, optional
        Label for x-axis. Default is 'n'.
    set_y_name : str, optional
        Label for y-axis. Default is 'FUV'.
    set_mass_name : str, optional
        Label for mass axis. Default is 'M'.
    plot_results : bool, optional
        Whether to create visualization plots. Default is True.
    plot_chi2_volume : bool, optional
        Whether to plot 3D chi² volume (requires plotly). Default is False.
    plot_projections : bool, optional
        Whether to plot marginalized projections. Default is True.
    font_size : int, optional
        Font size for plot labels. Default is 14.
    fig_size : tuple, optional
        Figure size (width, height). Default is (12, 10).
    colormap : str, optional
        Colormap for chi² visualization. Default is 'viridis'.
    save_plot : bool, optional
        Whether to save plots. Default is False.
    or_PATH : str, optional
        Original working directory.
    fig_dir_PATH : str, optional
        Directory to save plots.
    plot_type : str, optional
        File extension for saved plots. Default is 'pdf'.
    confidence_levels : list, optional
        Confidence levels for contours. Default is [0.68, 0.95, 0.99].
    convert_transition_names : bool, optional
        Whether to convert transition names. Default is False.
    plot_species_slices : bool, optional
        If True, save 2D slices showing ``model = observed`` contours for each
        species at the best-fit point (three orthogonal parameter planes).
        Default is False.
    plot_species_surfaces : bool, optional
        If True, also open an interactive Plotly iso-surface view. Default is False.
    species_slices_list : list of str, optional
        Subset of species/lines to draw on the slice plots. If None, uses
        intensity lines only (no ratios) up to ``species_slices_max``.
    species_slices_lines_only : bool, optional
        When ``species_slices_list`` is None, exclude ratio names (containing
        ``'/'``). Default is True.
    species_slices_max : int, optional
        Maximum number of species drawn on the slice plots. Default is 8.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'best_x': best-fit x parameter (linear scale)
        - 'best_y': best-fit y parameter (linear scale)
        - 'best_mass': best-fit mass value (linear scale)
        - 'best_x_log': best-fit x parameter (log scale)
        - 'best_y_log': best-fit y parameter (log scale)
        - 'best_mass_log': best-fit mass value (log scale)
        - 'chi2_min': minimum chi² value
        - 'chi2_reduced': reduced chi² (chi²_min / dof)
        - 'dof': degrees of freedom
        - 'chi2_grid': full 3D chi² grid
        - 'x_values': x coordinate array
        - 'y_values': y coordinate array
        - 'mass_values': mass coordinate array
        - 'confidence_intervals': dict with confidence intervals for each parameter
        - 'is_good_fit': boolean indicating if fit is acceptable (reduced chi² < 2)
        - 'model_values_at_best': dict of model values at best-fit point for each species
        - 'residuals': dict of (model - observed) / error for each species
    """
    from scipy.stats import chi2 as chi2_dist
    
    font_size = font_size if font_size else 18
    fig_size = fig_size if fig_size else (12, 10)
    
    # Convert transition names if needed
    species_list = my_grids_3d.keys()
    if convert_transition_names:
        species_list = convert_transition_names_func(species_list)
    
    # Validate species exist in grid
    valid_species = [s for s in species_list if s in my_grids_3d]
    if len(valid_species) == 0:
        print("ERROR: No valid species found in my_grids_3d")
        print(f"  Requested species: {species_list}")
        print(f"  Available species: {list(my_grids_3d.keys())}")
        return None
    
    # Convert observed values/errors to dict if list
    if isinstance(observed_values, list):
        observed_values = {s: v for s, v in zip(species_list, observed_values)}
    if isinstance(observed_errors, list):
        observed_errors = {s: e for s, e in zip(species_list, observed_errors)}
    
    # Filter to species with observations
    valid_species = [s for s in valid_species if s in observed_values and s in observed_errors]
    if len(valid_species) == 0:
        print("ERROR: No species with both grid data and observations")
        return None
    
    # Validate errors are positive
    for species in valid_species:
        if observed_errors[species] <= 0:
            print(f"ERROR: Error for {species} must be positive, got {observed_errors[species]}")
            return None
    
    # Get grid structure from first species
    first_species = valid_species[0]
    first_grid_data = my_grids_3d[first_species]
    
    # Detect coordinate keys
    x_key, y_key, z_key = _detect_3d_axis_keys(first_grid_data)
    third_axis_label = _infer_axis_display_label(z_key, set_z_name, set_mass_name=set_mass_name)
    set_z_name = third_axis_label
    set_mass_name = third_axis_label
    z_label_plain = _third_axis_plain_label(third_axis_label)
    z_log_label = _log10_third_axis_label(third_axis_label)
    
    # Extract coordinate meshgrids
    z_mesh = first_grid_data[z_key]
    Y_mesh = first_grid_data[y_key]
    X_mesh = first_grid_data[x_key]
    
    # Extract 1D arrays from meshgrid
    # For meshgrid with indexing='ij': shape is (n_mass, n_y, n_x)
    # mass_mesh[i, j, k] = mass_values[i] (constant in j, k)
    # Y_mesh[i, j, k] = y_values[j] (constant in i, k)
    # X_mesh[i, j, k] = x_values[k] (constant in i, j)
    z_values = z_mesh[:, 0, 0]
    y_values = Y_mesh[0, :, 0]
    x_values = X_mesh[0, 0, :]
    
    n_mass, n_y, n_x = first_grid_data['grid'].shape
    
    # Verify coordinate arrays match grid dimensions and are unique
    if len(z_values) != n_mass:
        print(f"WARNING: {set_z_name} coordinate length ({len(z_values)}) doesn't match grid dimension ({n_mass})")
    if len(y_values) != n_y:
        print(f"WARNING: Y coordinate length ({len(y_values)}) doesn't match grid dimension ({n_y})")
    if len(x_values) != n_x:
        print(f"WARNING: X coordinate length ({len(x_values)}) doesn't match grid dimension ({n_x})")
    
    # Verify values are unique and sorted (they should be from meshgrid)
    if len(np.unique(z_values)) != len(z_values):
        print(f"WARNING: {set_z_name} values are not unique!")
    if len(np.unique(y_values)) != len(y_values):
        print(f"WARNING: Y values are not unique!")
    if len(np.unique(x_values)) != len(x_values):
        print(f"WARNING: X values are not unique!")
    
    # Verify arrays are sorted
    if not np.all(np.diff(z_values) > 0):
        print(f"WARNING: {set_z_name} values are not sorted!")
    if not np.all(np.diff(y_values) > 0):
        print(f"WARNING: Y values are not sorted!")
    if not np.all(np.diff(x_values) > 0):
        print(f"WARNING: X values are not sorted!")
    
    # Compute chi² grid: χ² = Σᵢ [(model_i - observed_i) / error_i]²
    chi2_grid = _compute_chi2_grid(my_grids_3d, valid_species, observed_values, observed_errors)

    finite_chi2 = chi2_grid[np.isfinite(chi2_grid)]
    if finite_chi2.size:
        chi2_max = float(np.max(finite_chi2))
        chi2_med = float(np.median(finite_chi2))
        if chi2_max > max(1e5, 100.0 * chi2_med):
            print(
                f"{ORANGE}Warning: chi^2 grid spans a very large range "
                f"(median={chi2_med:.2e}, max={chi2_max:.2e}). "
                f"Check unit conversions and error estimates for all lines/ratios. "
                f"Per-line contributions are capped at {_CHI2_MAX_CONTRIB:.0e} for "
                f"numerical stability; plots use Δχ^2 around the minimum.{NC}"
            )

    # Find minimum on discrete grid
    valid_chi2 = np.where(np.isfinite(chi2_grid), chi2_grid, np.inf)
    min_idx = np.unravel_index(np.argmin(valid_chi2), chi2_grid.shape)
    best_mass_idx, best_y_idx, best_x_idx = min_idx
    chi2_min = chi2_grid[min_idx]
    
    best_z          = z_values[best_mass_idx]
    best_y          = y_values[best_y_idx]
    best_x          = x_values[best_x_idx]
    best_z_log      = np.log10(best_z)
    best_y_log      = np.log10(best_y)
    best_x_log      = np.log10(best_x)
    
    log_z           = np.log10(z_values)
    log_y           = np.log10(y_values)
    log_x           = np.log10(x_values)
    
    # Check if we're at a boundary
    at_boundary = []
    if _check_at_boundary(log_z, log_y, log_x, best_z_log, best_y_log, best_x_log):
        if abs(best_z_log - log_z.min()) < 0.01 or abs(best_z_log - log_z.max()) < 0.01:
            at_boundary.append(set_z_name)
        if abs(best_y_log - log_y.min()) < 0.01 or abs(best_y_log - log_y.max()) < 0.01:
            at_boundary.append(set_y_name)
        if abs(best_x_log - log_x.min()) < 0.01 or abs(best_x_log - log_x.max()) < 0.01:
            at_boundary.append(set_x_name)
    
    # Compute model values and residuals at best-fit grid point
    model_values_at_best = {}
    residuals = {}
    chi2_per_species = {}
    for species in valid_species:
        grid_val = my_grids_3d[species]['grid'][best_mass_idx, best_y_idx, best_x_idx]
        model_val = np.maximum(grid_val, 1e-30)
        obs_val = observed_values[species]
        obs_err = observed_errors[species]
        
        model_values_at_best[species] = model_val  # Store linear for display
        residuals[species] = (model_val - obs_val) / obs_err
        chi2_per_species[species] = residuals[species] ** 2
    
    # Degrees of freedom: N_observations - N_parameters
    n_obs = len(valid_species)
    n_params = 3
    dof = max(n_obs - n_params, 1)  # Ensure dof >= 1
    
    chi2_reduced = chi2_min / dof
    
    # Compute confidence intervals using delta chi²
    # For joint confidence regions with 3 parameters:
    # Δχ² = chi2_dist.ppf(confidence_level, df=3)
    delta_chi2_values = {}
    for conf in confidence_levels:
        delta_chi2_values[conf] = chi2_dist.ppf(conf, 3)
    
    confidence_intervals = {}
    for conf_level in confidence_levels:
        delta = delta_chi2_values[conf_level]
        threshold = chi2_min + delta
        within_conf = chi2_grid <= threshold
        
        # Find parameter ranges within confidence region
        # Marginalize over other dimensions to find range for each parameter
        z_in_conf = z_values[np.any(np.any(within_conf, axis=2), axis=1)]
        y_in_conf = y_values[np.any(np.any(within_conf, axis=2), axis=0)]
        x_in_conf = x_values[np.any(np.any(within_conf, axis=0), axis=0)]
        
        confidence_intervals[conf_level] = {
            'x_range': (x_in_conf.min(), x_in_conf.max()) if len(x_in_conf) > 0 else (best_x, best_x),
            'y_range': (y_in_conf.min(), y_in_conf.max()) if len(y_in_conf) > 0 else (best_y, best_y),
            'mass_range': (z_in_conf.min(), z_in_conf.max()) if len(z_in_conf) > 0 else (best_z, best_z)
        }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"3D Chi-Squared Analysis Results")
    print(f"{'='*70}")
    
    # Diagnostic: Show chi² at different masses (marginalized over x and y)
    print(f"\nChi² vs {z_label_plain} (marginalized over {set_x_name} and {set_y_name}):")
    chi2_by_mass = np.nanmin(np.nanmin(chi2_grid, axis=2), axis=1)
    for i, (z, chi2_m) in enumerate(zip(z_values, chi2_by_mass)):
        marker = " <-- MIN" if i == best_mass_idx else ""
        print(f"  {z_label_plain}={z:.4e} (log₁₀={np.log10(z):.2f}): χ²_min={chi2_m:.4f}{marker}")
    
    # Diagnostic: Check if observed values are achievable at different masses
    print(f"\nModel value ranges vs observed values at different {z_label_plain} values:")
    print(f"  {z_label_plain:<10} {'log₁₀(z)':<10} ", end="")
    for species in valid_species:
        print(f"{species[:15]:<18}", end="")
    print()
    print(f"  {'-'*10} {'-'*10} ", end="")
    for _ in valid_species:
        print(f"{'-'*18}", end="")
    print()
    
    # Check a few key masses
    check_masses = [0, len(z_values)//4, len(z_values)//2, len(z_values)*3//4, len(z_values)-1]
    for m_idx in check_masses:
        z = z_values[m_idx]
        if z < 0.1 or z >= 1000:
            z_str = f"{z:.2e}"
        else:
            z_str = f"{z:.2f}"
        print(f"  {z_str:<10} {np.log10(z):>9.2f} ", end="")
        for species in valid_species:
            grid_data = my_grids_3d[species]
            model_slice = grid_data['grid'][m_idx, :, :]
            model_min = np.nanmin(model_slice)
            model_max = np.nanmax(model_slice)
            obs_val = observed_values[species]
            in_range = "✓" if model_min <= obs_val <= model_max else "✗"
            print(f"{in_range} [{model_min:.1e},{model_max:.1e}]", end="")
        print()
    
    # Format mass for display
    if best_z < 0.1 or best_z >= 1000:
        z_str = f'{best_z:.2e}'
    else:
        z_str = f'{best_z:.2f}'
    
    print(f"\nBest-fit parameters:")
    print(f"  {set_x_name}: {best_x:.4e} (log₁₀: {np.log10(best_x):.4f})")
    print(f"  {set_y_name}: {best_y:.4e} (log₁₀: {np.log10(best_y):.4f})")
    print(f"  {z_label_plain}: {z_str} (log₁₀: {np.log10(best_z):.4f})")
    
    print(f"\nChi² statistics:")
    print(f"  χ²_min: {chi2_min:.4f}")
    print(f"  Number of observations: {n_obs}")
    print(f"  Number of parameters: {n_params}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Reduced χ² (χ²/dof): {chi2_reduced:.4f}")
    print(f"  Good fit (χ²_red < 2): {chi2_reduced < 2}")
    
    print(f"\nModel vs Observed at best-fit point:")
    print(f"  {'Species':<15} {'Observed':>12} {'Model':>12} {'Residual':>10} {'χ² contrib':>10}")
    print(f"  {'-'*60}")
    for species in valid_species:
        obs = observed_values[species]
        mod = model_values_at_best[species]
        res = residuals[species]
        chi2_c = chi2_per_species[species]
        print(f"  {species:<15} {obs:>12.4e} {mod:>12.4e} {res:>10.2f} {chi2_c:>10.2f}")
    
    if 0.68 in confidence_intervals:
        ci = confidence_intervals[0.68]
        print(f"\n68% confidence intervals (Δχ² = {delta_chi2_values[0.68]:.2f}):")
        print(f"  {set_x_name}: [{ci['x_range'][0]:.4e}, {ci['x_range'][1]:.4e}]")
        print(f"  {set_y_name}: [{ci['y_range'][0]:.4e}, {ci['y_range'][1]:.4e}]")
        print(f"  {z_label_plain}: [{ci['mass_range'][0]:.4e}, {ci['mass_range'][1]:.4e}]")
    
    if at_boundary:
        print(f"\nWARNING: Best-fit is at grid boundary for: {at_boundary}")
    
    print(f"{'='*70}\n")
    
    if plot_results or plot_projections or plot_chi2_volume or plot_species_slices or plot_species_surfaces:
        print(f"{ORANGE}Note: chi2_analysis plotting is disabled in Kosma-online-tool "
              f"(computation-only mode).{NC}")
    
    results = {
        'best_x': best_x,
        'best_y': best_y,
        'best_mass': best_z,
        'best_z': best_z,
        'best_x_log': best_x_log,
        'best_y_log': best_y_log,
        'best_mass_log': best_z_log,
        'best_z_log': best_z_log,
        'best_x_idx': best_x_idx,
        'best_y_idx': best_y_idx,
        'best_mass_idx': best_mass_idx,
        'chi2_min': chi2_min,
        'chi2_reduced': chi2_reduced,
        'dof': dof,
        'n_observations': n_obs,
        'chi2_grid': chi2_grid,
        'x_values': x_values,
        'y_values': y_values,
        'mass_values': z_values,
        'z_values': z_values,
        'z_key': z_key,
        'z_axis_name': set_z_name,
        'confidence_intervals': confidence_intervals,
        'delta_chi2_values': delta_chi2_values,
        'is_good_fit': chi2_reduced < 2,
        'model_values_at_best': model_values_at_best,
        'residuals': residuals,
        'chi2_per_species': chi2_per_species,
        'valid_species': valid_species
    }
    
    return results
########################################################################################################################################