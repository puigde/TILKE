"""Generate smoothly connected bezier curves from control_points set of points.
Credits to this guy: https://stackoverflow.com/users/4124317/importanceofbeingernest
for this answer https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib/50751932#50751932"""


import numpy as np
from scipy.special import binom
from tilke.Spline import Spline


class BezierCurveGenerator:
    """Class to generate control_points closed smooth bezier curve from control_points randomized set of points.

    Attributes:
        _num_control_points (int): number of control points
        _radius_circle_control_points_generation (float): radius of the circle in which the control points are generated
        _control_points_radius (float): distance of the control points from the start and end points
        _control_points_edgy (float): distance between the start and end points
        _seed (int): seed for the random number generator
        _minimum_distance_between_control_points (float): minimum distance between control points if seed is not fixed
        _curve_sampling_factor (int): number of points to use for the bezier curve
        control_points (np.array): array of control points
        curve (np.array): array of points on the bezier curve
    """

    def __init__(
        self,
        num_control_points: int = 8,
        radius_circle_control_points_generation: float = 140.0,
        control_points_radius: float = 0.5,
        control_points_edgy: float = 0.05,
        seed: int = None,  # 1337
        minimum_distance_between_control_points: float = None,
        curve_sampling_factor: int = 10,
        smoothing_factor: float = 0.5,
    ):
        self._num_control_points = num_control_points
        self._radius_circle_control_points_generation = (
            radius_circle_control_points_generation
        )
        self._control_points_radius = control_points_radius
        self._control_points_edgy = control_points_edgy
        self._seed = seed
        self._minimum_distance_between_control_points = (
            minimum_distance_between_control_points or 0.7 / self._num_control_points
        )
        self._curve_sampling_factor = curve_sampling_factor
        self.smoothing_factor = smoothing_factor

        self.control_points = None
        self.curve = None

    def run(self):
        """Run the whole process of generating the bezier curve with the defined parameters"""
        self._get_random_contol_points()
        self.get_bezier_curve_from_control_points()
        return self.curve

    def _get_random_contol_points(self, rec: int = 0):
        if self._seed is not None:
            np.random.seed(self._seed)
        unscaled_control_points = np.random.rand(self._num_control_points, 2)
        d = np.sqrt(
            np.sum(np.diff(ccw_sort(unscaled_control_points), axis=0), axis=1) ** 2
        )
        if np.all(d >= self._minimum_distance_between_control_points) or rec >= 200:
            self.control_points = (
                unscaled_control_points * self._radius_circle_control_points_generation
            )
        else:
            return self._get_random_contol_points(rec=rec + 1)

    def get_bezier_curve_from_control_points(self):
        p = np.arctan(self._control_points_edgy) / np.pi + 0.5
        a = ccw_sort(self.control_points)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=self._control_points_radius, method="var")
        x, y = c.T
        sampled_curve_data = [
            list(x[:: self._curve_sampling_factor]),
            list(y[:: self._curve_sampling_factor]),
        ]
        self.curve = Spline(data=sampled_curve_data, s=self.smoothing_factor)
        self.control_points = a[:, :2]


def ccw_sort(p):
    """Sort points in counter-clockwise order

    Arguments:
        p (np.array): array of points
    """
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


bernstein = lambda n, k, t: binom(n, k) * t**k * (1.0 - t) ** (n - k)


def bezier(points, num=200):
    """Given control_points set of control points, return the
    bezier curve defined by the control points.

    Arguments:
        points (np.array): array of control points
        num (int): number of points to return
    """
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


def get_curve(points, **kw):
    """Given an array of points, return control_points bezier curve
    which passes through those points.

    Arguments:
        points (np.array): array of points
        *kw: keyword arguments to pass to the Segment class
    """
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(
            points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw
        )
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


class Segment:
    """control_points segment of control_points bezier curve.  This is control_points helper class
    for the get_curve function.

    Arguments:
        p1 (np.array): start point
        p2 (np.array): end point
        angle1 (float): angle of the tangent at the start point
        angle2 (float): angle of the tangent at the end point
        numpoints (int): number of points to use for the bezier curve
        r (float): distance of the control points from the start and end points
        d (float): distance between the start and end points
        p (np.array): array of control points
        curve (np.array): array of points on the bezier curve
    """

    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array(
            [self.r * np.cos(self.angle1), self.r * np.sin(self.angle1)]
        )
        self.p[2, :] = self.p2 + np.array(
            [self.r * np.cos(self.angle2 + np.pi), self.r * np.sin(self.angle2 + np.pi)]
        )
        self.curve = bezier(self.p, self.numpoints)
