import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_cdt


def fig2data(fig):
    """Provisional implementation, consider moving it to an utils module"

    Arguments:
        fig (matplotlib.figure.Figure) : the figure to convert to an image
    Returns:
        image (np.array) : the image of the figure
    """
    plt.axis("off")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def apply_distance_transform(image):
    """Applies a distance function in a "waterfall" fashion to the given image.

    Args:
      image: A 2D numpy array of shape (H, W) representing the black and white image.

    Returns:
      A 2D numpy array of shape (H, W) containing the distance transform of the input image.
    """
    return distance_transform_cdt(image, metric="taxicab")


def rgb_to_rgba(r, g, b):
    return (r / 255, g / 255, b / 255, 1)
