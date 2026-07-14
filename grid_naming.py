"""
KOSMA-tau grid filename token parsing.

Standard models: ``Model<tag>_DD_MM_FF_ZZ_CC_AA`` (six parameter tokens).

Some non-attenuated runs omit the final ``_AA`` suffix
(``Model<tag>_DD_MM_FF_ZZ_CC``). Those are accepted with ``atten = 0``.
"""

from __future__ import annotations

from typing import Optional, Tuple

N_GRID_PARAMS = 6
DEFAULT_ATTEN_TOKEN = 0


def _parts_are_int(parts) -> bool:
    for part in parts:
        try:
            int(part)
        except (ValueError, TypeError):
            return False
    return True


def param_token_count(parts, model_idx: int) -> int:
    """Return 5 or 6 numeric tokens after the ``Model*`` segment, else 0."""
    start = model_idx + 1
    n_after = len(parts) - start
    if n_after >= N_GRID_PARAMS and _parts_are_int(parts[start:start + N_GRID_PARAMS]):
        return N_GRID_PARAMS
    if n_after >= N_GRID_PARAMS - 1 and _parts_are_int(parts[start:start + N_GRID_PARAMS - 1]):
        return N_GRID_PARAMS - 1
    return 0


def parse_model_tokens_from_parts(parts, model_idx: int) -> Optional[Tuple[int, ...]]:
    """
    Parse ``DD_MM_FF_ZZ_CC[_AA]`` tokens after a ``Model*`` name segment.

    Missing ``AA`` implies ``atten = 0`` (no attenuation tag in the filename).
    """
    n_tok = param_token_count(parts, model_idx)
    if n_tok < N_GRID_PARAMS - 1:
        return None
    start = model_idx + 1
    try:
        if n_tok == N_GRID_PARAMS:
            return tuple(int(parts[start + i]) for i in range(N_GRID_PARAMS))
        base = [int(parts[start + i]) for i in range(N_GRID_PARAMS - 1)]
        return tuple(base + [DEFAULT_ATTEN_TOKEN])
    except (ValueError, IndexError):
        return None


def parse_model_tokens_from_stem(stem: str) -> Optional[Tuple[int, ...]]:
    """Parse parameter tokens from a filename stem or ``Model*`` folder name."""
    parts = str(stem).split('_')
    model_idx = next((i for i, p in enumerate(parts) if p.startswith('Model')), None)
    if model_idx is None:
        return None
    return parse_model_tokens_from_parts(parts, model_idx)
