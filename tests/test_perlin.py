import numpy as np
import pytest

from tilke.perlin import pnoise1

try:
    import noise as _noise_c

    HAS_C_NOISE = True
except ImportError:
    HAS_C_NOISE = False


class TestPnoise1:
    @pytest.mark.skipif(not HAS_C_NOISE, reason="C noise library not installed")
    def test_matches_c_library_positive(self):
        """Numpy pnoise1 matches C noise.pnoise1 for positive values."""
        x_vals = np.linspace(0.1, 100, 1000)
        c_results = np.array([_noise_c.pnoise1(float(x)) for x in x_vals])
        np_results = pnoise1(x_vals)
        np.testing.assert_allclose(np_results, c_results, atol=1e-4)

    @pytest.mark.skipif(not HAS_C_NOISE, reason="C noise library not installed")
    def test_matches_c_library_negative(self):
        """Numpy pnoise1 matches C noise.pnoise1 for negative values."""
        x_vals = np.linspace(-100, -0.1, 1000)
        c_results = np.array([_noise_c.pnoise1(float(x)) for x in x_vals])
        np_results = pnoise1(x_vals)
        np.testing.assert_allclose(np_results, c_results, atol=1e-4)

    @pytest.mark.skipif(not HAS_C_NOISE, reason="C noise library not installed")
    def test_matches_c_library_multi_octave(self):
        """Multi-octave mode matches C library."""
        x_vals = np.linspace(-50, 50, 200)
        c_results = np.array([_noise_c.pnoise1(float(x), octaves=4) for x in x_vals])
        np_results = np.array([pnoise1(float(x), octaves=4) for x in x_vals])
        np.testing.assert_allclose(np_results, c_results, atol=1e-4)

    def test_scalar_returns_float(self):
        result = pnoise1(1.5)
        assert isinstance(result, float)

    def test_array_returns_ndarray(self):
        result = pnoise1(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_zero_is_zero(self):
        assert pnoise1(0.0) == 0.0

    def test_integer_is_zero(self):
        """Perlin noise at integer coordinates is always zero."""
        for i in range(-10, 11):
            assert abs(pnoise1(float(i))) < 1e-10
