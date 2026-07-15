"""
KOSMA-tau grid filename token parsing.

Standard models: ``<tag>_DD_MM_FF_ZZ_CC_AA`` (six parameter tokens), where
``<tag>`` is commonly ``Model100`` but may be ``pdr100`` or another run label.

Shorter encodings are expanded to the full six-token tuple internally:

* 5 tokens ``DD_MM_FF_ZZ_CC`` — ``atten = 0``
* 4 tokens ``DD_MM_FF_00`` — metallicity from the fourth token (often fixed
  ``00``); cosmic-ray rate and attenuation use defaults (no CR grid axis)
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

N_GRID_PARAMS = 6
MIN_GRID_PARAMS = 4
DEFAULT_ATTEN_TOKEN = 0
# ζ = 2×10⁻¹⁷ s⁻¹ when CR is omitted from filenames (decode: 10^(-token) s⁻¹).
DEFAULT_CRIR_RATE_S = 2e-17
DEFAULT_CRIR_TOKEN = -math.log10(DEFAULT_CRIR_RATE_S)
PARAM_KEYS = ('density', 'mass', 'fuv', 'metal', 'crir', 'atten')

# KoSens / map_fit meshgrid dict keys for each parameter axis.
PARAM_MESH_KEYS = {
    'density': 'densities',
    'mass': 'mass_values',
    'fuv': 'fuv_values',
    'metal': 'metal_values',
    'crir': 'crir_values',
    'atten': 'atten_values',
}

# Preferred 3-D fit axes when all three vary (classic PDR: n_H, FUV, ζ).
FIT_AXIS_PREFERRED = ('density', 'fuv', 'crir')


def varying_param_keys(axis_tokens) -> tuple:
    """Parameter keys with more than one token on disk."""
    return tuple(
        k for k in PARAM_KEYS
        if len(axis_tokens.get(k, []) or []) > 1
    )


def fixed_param_keys(axis_tokens) -> tuple:
    """Parameter keys with exactly one token (held fixed in the cube)."""
    return tuple(
        k for k in PARAM_KEYS
        if len(axis_tokens.get(k, []) or []) == 1
    )


def resolve_fit_axes(axis_tokens) -> tuple:
    """
    Choose (x, y, z) parameter keys for 3-D map fitting.

    Uses the PDR triple (density, fuv, crir) when all three vary; otherwise
    the first three varying parameters in ``PARAM_KEYS`` order, mirroring the
    slice-plane logic for two or one varying axis.
    """
    varying = varying_param_keys(axis_tokens)
    if not varying:
        raise ValueError(
            'No varying grid parameters — map fit requires a multi-dimensional grid.'
        )
    if all(k in varying for k in FIT_AXIS_PREFERRED):
        return FIT_AXIS_PREFERRED
    if len(varying) >= 3:
        return varying[0], varying[1], varying[2]
    fixed = fixed_param_keys(axis_tokens)
    if len(varying) == 2:
        sk = fixed[0] if fixed else varying[0]
        return varying[0], varying[1], sk
    others = [k for k in PARAM_KEYS if k != varying[0]]
    return varying[0], others[0], others[1]


def mesh_key_for_param(param_key: str) -> str:
    """Return the meshgrid dict key used by map_fit for a parameter."""
    key = PARAM_MESH_KEYS.get(param_key)
    if not key:
        raise KeyError(f'No mesh key mapping for parameter {param_key!r}')
    return key


def axis_tokens_from_tuples(token_tuples) -> dict:
    """Build ``{param_key: [token, ...]}`` from full model token tuples."""
    tuples = [tuple(t) for t in token_tuples if t is not None]
    if not tuples:
        return {k: [] for k in PARAM_KEYS}
    n = len(PARAM_KEYS)
    return {
        PARAM_KEYS[i]: sorted({t[i] for t in tuples if len(t) > i})
        for i in range(n)
    }


def model_core_from_simline_path(path: str) -> Optional[str]:
    """Return ``<tag>_DD_MM_…`` stem from a SIMLINE ``.smli`` / ``.smlc`` path."""
    base = os.path.basename(path)
    for ext in ('.smli', '.smlc'):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    for prefix in ('jtemp_', 'jerg_', 'tau_'):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    parts = base.split('_')
    tag_idx = find_model_tag_index(parts)
    if tag_idx is None:
        return None
    # Species suffix is everything after the parameter tokens.
    n_tok = param_token_count(parts, tag_idx)
    if n_tok <= 0:
        return None
    return '_'.join(parts[: tag_idx + 1 + n_tok])


def _parts_are_int(parts) -> bool:
    for part in parts:
        try:
            int(part)
        except (ValueError, TypeError):
            return False
    return True


def param_token_count(parts, model_idx: int) -> int:
    """Return 4, 5, or 6 numeric tokens after the model-tag segment, else 0."""
    start = model_idx + 1
    n_after = len(parts) - start
    for n in (N_GRID_PARAMS, N_GRID_PARAMS - 1, MIN_GRID_PARAMS):
        if n_after >= n and _parts_are_int(parts[start:start + n]):
            return n
    return 0


def expand_partial_tokens(numeric_tokens: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Expand 4- or 5-token filename runs to the canonical six grid tokens."""
    n = len(numeric_tokens)
    if n == N_GRID_PARAMS:
        return numeric_tokens
    if n == N_GRID_PARAMS - 1:
        return tuple(numeric_tokens) + (DEFAULT_ATTEN_TOKEN,)
    if n == MIN_GRID_PARAMS:
        # DD_MM_FF_ZZ — e.g. pdr100_20_-10_10_00 (density, mass, FUV, metallicity)
        dd, mm, ff, zz = numeric_tokens
        return (dd, mm, ff, zz, DEFAULT_CRIR_TOKEN, DEFAULT_ATTEN_TOKEN)
    return None


def find_model_tag_index(parts) -> Optional[int]:
    """
    Index of the model-tag segment immediately before grid parameter tokens.

    The tag may be ``Model100``, ``pdr100``, etc.  Detection is based on the
    following numeric pattern (4–6 tokens), not the tag prefix.
    """
    if not parts:
        return None
    candidates = []
    for i in range(len(parts)):
        if param_token_count(parts, i) >= MIN_GRID_PARAMS:
            candidates.append(i)
    if not candidates:
        return None
    for i in candidates:
        if str(parts[i]).startswith('Model'):
            return i
    return candidates[0]


def parse_model_tokens_from_parts(parts, model_idx: int) -> Optional[Tuple[int, ...]]:
    """
    Parse grid tokens after a model-tag segment and expand to six values.

    Supported layouts: ``DD_MM_FF_ZZ_CC_AA``, ``DD_MM_FF_ZZ_CC``, ``DD_MM_FF_ZZ``.
    """
    n_tok = param_token_count(parts, model_idx)
    if n_tok < MIN_GRID_PARAMS:
        return None
    start = model_idx + 1
    try:
        raw = tuple(int(parts[start + i]) for i in range(n_tok))
    except (ValueError, IndexError):
        return None
    return expand_partial_tokens(raw)


def simline_file_key(full_tokens, filename_token_count: int = 6):
    """Return the token tuple used to index SIMLINE files on disk."""
    tokens = tuple(full_tokens)
    if filename_token_count < N_GRID_PARAMS:
        return tokens[:filename_token_count]
    return tokens


def parse_model_tokens_from_stem(stem: str) -> Optional[Tuple[int, ...]]:
    """Parse parameter tokens from a filename stem or model folder name."""
    parts = str(stem).split('_')
    tag_idx = find_model_tag_index(parts)
    if tag_idx is None:
        return None
    return parse_model_tokens_from_parts(parts, tag_idx)
