"""Bezier curve generation for random circuit middle lines.

Credits: https://stackoverflow.com/a/50751932 by ImportanceOfBeingErnest
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import binom

from tilke.spline import Spline, make_spline


@dataclass
class CurveGeneratorConfig:
    """Configuration for random middle curve generation.

    Attributes:
        num_control_points: Number of random control points.
        radius: Scale factor for control point coordinates.
        control_points_radius: Bezier handle length relative to segment.
        edginess: Controls sharpness of corners (0 = smooth, higher = sharper).
        seed: Random seed for reproducibility. None = random.
        min_control_point_distance: Minimum spacing between adjacent control points.
            Defaults to 0.7 / num_control_points if None.
        sampling_factor: Sub-sampling factor for the final curve.
        smoothing: Smoothing factor for the output spline.
    """

    num_control_points: int = 8
    radius: float = 140.0
    control_points_radius: float = 0.5
    edginess: float = 0.05
    seed: int | None = None
    min_control_point_distance: float | None = None
    sampling_factor: int = 10
    smoothing: float = 0.5


@dataclass
class Segment:
    """A cubic bezier segment between two points with tangent angles."""

    p1: np.ndarray
    p2: np.ndarray
    angle1: float
    angle2: float
    r: float = 0.3
    numpoints: int = 100
    curve: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        scaled_r = self.r * d
        p = np.zeros((4, 2))
        p[0] = self.p1
        p[3] = self.p2
        p[1] = self.p1 + scaled_r * np.array(
            [np.cos(self.angle1), np.sin(self.angle1)]
        )
        p[2] = self.p2 + scaled_r * np.array(
            [np.cos(self.angle2 + np.pi), np.sin(self.angle2 + np.pi)]
        )
        self.curve = bezier(p, self.numpoints)


def ccw_sort(points: np.ndarray) -> np.ndarray:
    """Sort 2D points in counter-clockwise order around their centroid."""
    d = points - np.mean(points, axis=0)
    angles = np.arctan2(d[:, 0], d[:, 1])
    return points[np.argsort(angles)]


def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
    """Bernstein basis polynomial B_{k,n}(t)."""
    return binom(n, k) * t**k * (1.0 - t) ** (n - k)


def bezier(points: np.ndarray, num: int = 200) -> np.ndarray:
    """Evaluate a bezier curve defined by control points.

    Args:
        points: Control points, shape (M, 2).
        num: Number of output points.

    Returns:
        Curve points, shape (num, 2).
    """
    n = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(n):
        curve += np.outer(bernstein(n - 1, i, t), points[i])
    return curve


def _build_bezier_segments(
    points: np.ndarray, r: float = 0.3
) -> tuple[list[Segment], np.ndarray]:
    """Build connected bezier segments through annotated control points.

    Args:
        points: Array of shape (N, 3) — columns are x, y, tangent_angle.
        r: Handle length relative to segment distance.

    Returns:
        (segments, concatenated_curve)
    """
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(
            p1=points[i, :2],
            p2=points[i + 1, :2],
            angle1=points[i, 2],
            angle2=points[i + 1, 2],
            r=r,
        )
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def generate_middle_curve(config: CurveGeneratorConfig | None = None) -> Spline:
    """Generate a random closed smooth middle curve for a circuit.

    Args:
        config: Generator configuration. Uses defaults if None.

    Returns:
        A periodic Spline representing the middle line.
    """
    if config is None:
        config = CurveGeneratorConfig()

    min_dist = config.min_control_point_distance or (0.7 / config.num_control_points)

    if config.seed is not None:
        np.random.seed(config.seed)

    # Generate control points with sufficient spacing
    control_points = None
    for _ in range(200):
        raw = np.random.rand(config.num_control_points, 2)
        diffs = np.sqrt(np.sum(np.diff(ccw_sort(raw), axis=0), axis=1) ** 2)
        if np.all(diffs >= min_dist):
            control_points = raw * config.radius
            break

    if control_points is None:
        # Fallback: accept whatever we got
        control_points = np.random.rand(config.num_control_points, 2) * config.radius

    # Compute tangent angles for smooth bezier connection
    p = np.arctan(config.edginess) / np.pi + 0.5
    a = ccw_sort(control_points)
    a = np.append(a, np.atleast_2d(a[0]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    ang = np.where(ang >= 0, ang, ang + 2 * np.pi)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)

    _, curve = _build_bezier_segments(a, r=config.control_points_radius)
    x, y = curve.T
    sampled = [
        list(x[:: config.sampling_factor]),
        list(y[:: config.sampling_factor]),
    ]
    return make_spline(sampled, s=config.smoothing)
