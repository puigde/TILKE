"""Image conversion utilities for circuit rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_cdt


def fig2data(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to an RGBA numpy array.

    Args:
        fig: The matplotlib figure to convert.

    Returns:
        RGBA image array of shape (H, W, 4).
    """
    plt.axis("off")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((w, h, 4))
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    return np.asarray(image)


def apply_distance_transform(image: np.ndarray) -> np.ndarray:
    """Apply a taxicab distance transform to a binary image.

    Args:
        image: A 2D numpy array (H, W) representing a black and white image.

    Returns:
        Distance transform of the input image, same shape.
    """
    return distance_transform_cdt(image, metric="taxicab")


def rgb_to_rgba(r: int, g: int, b: int) -> tuple[float, float, float, float]:
    """Convert 0-255 RGB values to 0-1 RGBA tuple with full opacity."""
    return (r / 255, g / 255, b / 255, 1.0)
