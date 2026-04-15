"""Pure-numpy 1D Perlin noise, compatible with the `noise` library's pnoise1.

Reimplements the algorithm from Casey Duncan's `noise` C extension to allow
vectorized evaluation over numpy arrays, removing the C dependency.
"""

from __future__ import annotations

import numpy as np

# fmt: off
# Ken Perlin's permutation table, doubled to avoid index wrapping.
# From _noise.h PERM[] in the C noise library (caseman/noise).
# Note: the pure-Python noise.perlin.BaseNoise.permutation has a typo at
# index 180 (9 instead of 19). This table matches the authoritative C source.
_PERM = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
    69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,
    252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,
    168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,
    211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,
    216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,
    164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,
    126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,
    213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,
    253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,
    242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,
    192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
    69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,
    252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,
    168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,
    211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,
    216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,
    164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,
    126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,
    213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,
    253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,
    242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,
    192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
], dtype=np.int32)
# fmt: on


def _grad1(hash_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1D gradient function matching the C noise library's grad1().

    If bit 3 of hash is set, gradient is -1; otherwise gradient is (hash & 7) + 1.
    """
    g = np.where((hash_vals & 8) != 0, -1.0, (hash_vals & 7) + 1.0)
    return g * x


def pnoise1(
    x: float | np.ndarray,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    repeat: int = 1024,
    base: int = 0,
) -> float | np.ndarray:
    """1D Perlin noise, compatible with noise.pnoise1().

    Accepts both scalar and numpy array inputs. When given an array,
    returns an array of the same shape (vectorized evaluation).

    Args:
        x: Input coordinate(s).
        octaves: Number of fBm layers.
        persistence: Amplitude decay per octave.
        lacunarity: Frequency growth per octave.
        repeat: Tiling period in grid units.
        base: Offset into permutation table (selects noise pattern).

    Returns:
        Noise value(s) in approximately [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)

    total = np.zeros_like(x)
    freq = 1.0
    amp = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        xf = x * freq
        xi = np.floor(xf).astype(np.int64)
        rep = int(repeat * freq)

        i = ((xi % rep) & 255) + base
        ii = (((xi + 1) % rep) & 255) + base

        t = xf - np.floor(xf)
        fade = t * t * t * (t * (t * 6 - 15) + 10)

        g0 = _grad1(_PERM[i], t)
        g1 = _grad1(_PERM[ii], t - 1)

        total += (g0 + fade * (g1 - g0)) * 0.4 * amp
        max_amp += amp
        freq *= lacunarity
        amp *= persistence

    result = total / max_amp
    return float(result[0]) if scalar else result
