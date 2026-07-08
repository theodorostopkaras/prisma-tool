"""Format SIMLINE transition tokens into spectroscopic labels."""

from __future__ import annotations

from fractions import Fraction


def format_smli_transition_label(raw_transition: str) -> str:
    """
    Format a SIMLINE transition token into a normalized human-readable label.

    Examples
    --------
    - ``'1--0'`` -> ``'1-0'``
    - ``'2.5_1--1.5_2'`` -> ``'5/2_{1} - 3/2_{2}'`` (half-integer *J* as a fraction)
    - ``'5_4--4_4'`` -> ``'5_{4} - 4_{4}'``
    - ``'10_0_10--9_1_9'`` -> ``'10_{0-10} - 9_{1-9}'``
    """
    if not raw_transition:
        return ''

    token = str(raw_transition).strip()
    if not token:
        return ''

    def _format_quantum_chunk(chunk):
        chunk_str = str(chunk).strip()
        if not chunk_str:
            return chunk_str
        if chunk_str.isdigit():
            return str(int(chunk_str))
        try:
            frac = Fraction(chunk_str)
        except (ValueError, ZeroDivisionError):
            return chunk_str
        if frac.denominator == 1:
            return str(frac.numerator)
        return f'{frac.numerator}/{frac.denominator}'

    cleaned = token.replace('--', '-')
    if cleaned.count('_') < 2 or '-' not in cleaned:
        if '-' not in cleaned:
            return cleaned
        upper_simple, lower_simple = cleaned.split('-', 1)
        upper_simple = _format_quantum_chunk(upper_simple)
        lower_simple = _format_quantum_chunk(lower_simple)
        return f'{upper_simple}-{lower_simple}'

    upper_part, lower_part = cleaned.split('-', 1)
    upper_bits = [_format_quantum_chunk(x) for x in upper_part.split('_') if x]
    lower_bits = [_format_quantum_chunk(x) for x in lower_part.split('_') if x]

    if len(upper_bits) == 2 and len(lower_bits) == 2:
        return (
            f"{upper_bits[0]}_{{{upper_bits[1]}}} - "
            f"{lower_bits[0]}_{{{lower_bits[1]}}}"
        )

    if len(upper_bits) == 3 and len(lower_bits) == 3:
        return (
            f"{upper_bits[0]}_{{{upper_bits[1]}-{upper_bits[2]}}} - "
            f"{lower_bits[0]}_{{{lower_bits[1]}-{lower_bits[2]}}}"
        )

    return cleaned
