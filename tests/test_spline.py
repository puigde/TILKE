import numpy as np

from tilke.spline import Spline, evaluate, get_int_ext_splines, make_spline


def _make_circle_spline() -> Spline:
    """Helper: create a circular spline for testing."""
    n = 50
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta).tolist()
    y = np.sin(theta).tolist()
    return make_spline([x, y], s=0)


class TestSpline:
    def test_make_spline(self):
        spline = _make_circle_spline()
        assert spline.t is not None
        assert len(spline.t) == 50
        assert spline.data.shape == (2, 50)

    def test_evaluate_scalar(self):
        spline = _make_circle_spline()
        point = evaluate(spline, spline.t[0])
        assert point.shape == (2,)

    def test_evaluate_array(self):
        spline = _make_circle_spline()
        tt = np.linspace(spline.t[0], spline.t[-1], 100)
        points = evaluate(spline, tt)
        assert points.shape == (100, 2)

    def test_evaluate_derivative(self):
        spline = _make_circle_spline()
        tt = np.linspace(spline.t[0], spline.t[-1], 100)
        d1 = evaluate(spline, tt, der=1)
        assert d1.shape == (100, 2)
        d2 = evaluate(spline, tt, der=2)
        assert d2.shape == (100, 2)

    def test_int_ext_splines(self):
        spline = _make_circle_spline()
        interior, exterior = get_int_ext_splines(spline, dist=0.2, stepsize=0.1)
        assert interior.t is not None
        assert exterior.t is not None
        mid_point = evaluate(spline, spline.t[0])
        int_point = evaluate(interior, interior.t[0])
        ext_point = evaluate(exterior, exterior.t[0])
        mid_r = np.linalg.norm(mid_point)
        int_r = np.linalg.norm(int_point)
        ext_r = np.linalg.norm(ext_point)
        assert int_r < mid_r or ext_r > mid_r
