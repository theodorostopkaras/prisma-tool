"""
Cosmic-ray attenuation profiles (vendored from KoSens ``functions_for_cratten``).

Plots ζ_H2 vs H₂ column density with optional Padovani et al. (2018/2024)
reference bands (models 𝓛, 𝓗, 𝓤).
"""

from __future__ import annotations

import numpy as np

# ζ per H nucleus → ζ per H₂ molecule (KoSens / CR_atten_plot convention).
ZETA_H_TO_H2 = 20.0 / 13.0

DEFAULT_STOPPING_RATE = 1.0e20
DEFAULT_NH2_MIN = 1.0e19
DEFAULT_NH2_MAX = 1.0e25
DEFAULT_ZETA_MIN = 6.0e-19
DEFAULT_ZETA_MAX = 3.0e-14

_PADOVANI_L = np.array(
    [-2.93027572e2, 6.02083162e1, -4.72413075, 1.60518530e-1, -2.01065346e-3])
_PADOVANI_H = np.array(
    [-2.47467467e2, 4.60847057e1, -3.37884449, 1.09332535e-1, -1.32768917e-3])
_PADOVANI_U = np.array(
    [-2.25202661e2, 4.22778765e1, -3.09907676, 9.97659175e-2, -1.20564293e-3])

PADOVANI_MODELS = (
    ('L', _PADOVANI_L, 'blue', 'Model \\mathcal{L}'),
    ('H', _PADOVANI_H, 'green', 'Model \\mathcal{H}'),
    ('U', _PADOVANI_U, 'orange', 'Model \\mathcal{U}'),
)


def padovani_log_zeta(log_n_h2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log₁₀(ζ_H2 / s⁻¹) for Padovani models 𝓛, 𝓗, 𝓤 at log₁₀(N_H2)."""
    log_n = np.asarray(log_n_h2, dtype=float).reshape(-1)
    powers = log_n.reshape(-1, 1) ** np.arange(len(_PADOVANI_L))
    z_l = np.sum(_PADOVANI_L * powers, axis=1)
    z_h = np.sum(_PADOVANI_H * powers, axis=1)
    z_u = np.sum(_PADOVANI_U * powers, axis=1)
    return z_l, z_h, z_u


def padovani_zeta_h2(n_h2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ζ_H2 [s⁻¹] for Padovani models at N_H2 [cm⁻²]."""
    log_n = np.log10(np.asarray(n_h2, dtype=float))
    z_l, z_h, z_u = padovani_log_zeta(log_n)
    return 10.0 ** z_l, 10.0 ** z_h, 10.0 ** z_u


def padovani_reference_grid(n_min: float = DEFAULT_NH2_MIN,
                            n_max: float = DEFAULT_NH2_MAX,
                            n_points: int = 300):
    """Log-spaced N_H2 grid and Padovani ζ_H2 curves."""
    log_n = np.linspace(np.log10(n_min), np.log10(n_max), int(n_points))
    n_h2 = 10.0 ** log_n
    return n_h2, padovani_zeta_h2(n_h2)


def zeta_h2_from_cosray(cosray: np.ndarray) -> np.ndarray:
    """Convert KOSMA ζ_H profile to ζ_H2."""
    return np.asarray(cosray, dtype=float) * ZETA_H_TO_H2


def extrapolate_profile_loglog(x: np.ndarray, y: np.ndarray, x_max: float = DEFAULT_NH2_MAX,
                               n_tail: int = 5):
    """
    Extend a profile to ``x_max`` with a log-log linear fit on the last ``n_tail`` points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return x, y
    if np.nanmax(x) >= x_max:
        return x, y
    n_tail = min(int(n_tail), x.size)
    x_fit = x[-n_tail:]
    y_fit = y[-n_tail:]
    valid = (x_fit > 0) & (y_fit > 0) & np.isfinite(x_fit) & np.isfinite(y_fit)
    if valid.sum() < 2:
        return x, y
    slope, intercept = np.polyfit(np.log10(x_fit[valid]), np.log10(y_fit[valid]), 1)
    x_ext = float(x_max)
    y_ext = 10.0 ** (slope * np.log10(x_ext) + intercept)
    return np.append(x, x_ext), np.append(y, y_ext)


def attenuation_threshold_nh2(n_h2: np.ndarray, stopping_rate: float = DEFAULT_STOPPING_RATE):
    """
    N_H2 at the attenuation threshold (KoSens ``stopping_rate`` convention).

  ``cd_prof_h2[i] / stopping_rate > 1`` for the first depth index ``i``.
    """
    n_h2 = np.asarray(n_h2, dtype=float)
    if n_h2.size == 0 or stopping_rate <= 0:
        return None
    idx = 0
    for i, val in enumerate(n_h2):
        if val / stopping_rate > 1.0:
            idx = i
            break
        idx = 1
    if idx <= 0:
        return float(n_h2[0]) if n_h2.size else None
    return float(n_h2[idx - 1])
