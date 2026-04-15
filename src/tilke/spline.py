"""Parametric spline curves for track representation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep


@dataclass
class Spline:
    """Parametric spline curve through a set of 2D points.

    Attributes:
        xt: scipy spline tck tuple for x-coordinates.
        yt: scipy spline tck tuple for y-coordinates.
        t: Arc-length parameter domain.
        data: Original control points, shape (2, N).
    """

    xt: tuple
    yt: tuple
    t: np.ndarray
    data: np.ndarray


def make_spline(
    data: list[list[float]] | np.ndarray,
    s: float = 0,
    k: int = 3,
) -> Spline:
    """Create a periodic Spline from 2D point data.

    Args:
        data: Control points as [[x0, x1, ...], [y0, y1, ...]] or (2, N) array.
        s: Smoothing factor for scipy splrep.
        k: Degree of the spline fit.
    """
    arr = np.asarray(data, dtype=np.float64)
    sq_diff = np.square(np.diff(arr))
    t_end = np.cumsum(np.sqrt(sq_diff[0] + sq_diff[1]))
    t = np.append([0.0], t_end)
    xt = splrep(x=t, y=arr[0], s=s, per=1, k=k)
    yt = splrep(x=t, y=arr[1], s=s, per=1, k=k)
    return Spline(xt=xt, yt=yt, t=t, data=arr)


def evaluate(spline: Spline, t: float | np.ndarray, der: int = 0) -> np.ndarray:
    """Evaluate spline (or its derivative) at parameter value(s) t.

    Args:
        spline: The spline to evaluate.
        t: Parameter value(s).
        der: Derivative order (0 = position, 1 = velocity, 2 = acceleration).

    Returns:
        Array of shape (N, 2) for array input, or (2,) for scalar input.
    """
    return np.array(
        [
            splev(t, spline.xt, der=der),
            splev(t, spline.yt, der=der),
        ]
    ).T


def get_int_ext_splines(
    spline: Spline,
    dist: float = 2.0,
    stepsize: float = 0.5,
    smoothing: float = 0.5,
    sampling_factor: int = 5,
) -> tuple[Spline, Spline]:
    """Compute interior and exterior offset curves from a middle spline.

    Args:
        spline: The middle curve.
        dist: Offset distance from the middle curve.
        stepsize: Parameter step for sampling the middle curve.
        smoothing: Smoothing factor for the offset splines.
        sampling_factor: Sub-sampling factor to reduce point count.

    Returns:
        (interior_spline, exterior_spline)
    """
    steps = np.linspace(spline.t[0], spline.t[-1], math.floor(spline.t[-1] / stepsize))
    pos = evaluate(spline, steps)  # (N, 2)
    vel = evaluate(spline, steps, der=1)  # (N, 2)

    scale = dist / np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)  # (N,)
    x_int = pos[:, 0] + scale * vel[:, 1]
    y_int = pos[:, 1] - scale * vel[:, 0]
    x_ext = pos[:, 0] - scale * vel[:, 1]
    y_ext = pos[:, 1] + scale * vel[:, 0]

    int_data = [x_int[::sampling_factor].tolist(), y_int[::sampling_factor].tolist()]
    ext_data = [x_ext[::sampling_factor].tolist(), y_ext[::sampling_factor].tolist()]
    return make_spline(int_data, s=smoothing), make_spline(ext_data, s=smoothing)


def plot_spline(
    spline: Spline,
    precision: int = 1000,
    cones: np.ndarray | None = None,
) -> None:
    """Plot a spline curve with optional cone markers.

    Args:
        spline: The spline to plot.
        precision: Number of evaluation points.
        cones: Optional (N, 2) array of cone positions to overlay.
    """
    tt = np.linspace(spline.t[0], spline.t[-1], precision)
    gamma = evaluate(spline, tt)
    plt.plot(gamma[:, 0], gamma[:, 1])
    if cones is not None and len(cones) > 0:
        plt.scatter(cones[:, 0], cones[:, 1], c="y")
