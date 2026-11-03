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


def effective_column(n_h2, n_h):
    """H2-equivalent column N_eff = N(H2) + N(H)/2 = N_H/2 (KoSens attenuation onset).

    One H2 stops cosmic rays like two H atoms, so the atomic PDR layer counts too;
    attenuation starts where N_eff >= ``DEFAULT_STOPPING_RATE`` (Padovani et al. 2009).
    """
    return np.asarray(n_h2, dtype=float) + 0.5 * np.asarray(n_h, dtype=float)


def attenuated_mass_fraction(radius_cm, integrand_total_h, n_eff,
                             threshold: float = DEFAULT_STOPPING_RATE) -> float:
    """Fraction of the clump mass in which cosmic rays are attenuated (KoSens ``f_att``).

    ``f_att = ∫ 4πr² n_H [N_eff(r) >= threshold] dr / ∫ 4πr² n_H dr`` on the model's own
    shells (``integrand_total_h`` = 4πr² n_H), so 0 <= f_att <= 1.  All three arrays share
    one radius grid, ordered centre -> surface.
    """
    # ponytail: step mask on the shell grid, resolution-limited by KOSMA-tau's shell spacing
    w = np.asarray(integrand_total_h, dtype=float)
    inside = np.asarray(n_eff, dtype=float) >= threshold
    return float(np.trapezoid(w * inside, radius_cm) / np.trapezoid(w, radius_cm))
