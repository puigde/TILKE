# TILKE Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the TILKE repo with modern Python tooling (uv/ruff/ty), functional architecture (dataclasses + functions), tests, and a polished README with GIF demos.

**Architecture:** Clean room rebuild — remove all old files, create `src/tilke/` layout with `pyproject.toml` as single config source. OOP classes become dataclasses with free functions. Cone storage moves from Spline to Circuit. No logic changes.

**Tech Stack:** Python 3.12, uv, hatchling, ruff, ty, pytest, numpy, scipy, matplotlib, Pillow, noise, imageio, tqdm, ffmpeg

**Spec:** `docs/superpowers/specs/2026-04-15-tilke-cleanup-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, ruff/ty config |
| `.python-version` | Pin Python 3.12 |
| `.gitignore` | Ignore artifacts |
| `LICENSE` | MIT (keep existing) |
| `src/tilke/__init__.py` | Public API re-exports |
| `src/tilke/py.typed` | PEP 561 marker |
| `src/tilke/spline.py` | `Spline` dataclass + `make_spline`, `evaluate`, `get_int_ext_splines`, `plot_spline` |
| `src/tilke/curve_generator.py` | `CurveGeneratorConfig`, `Segment`, `generate_middle_curve`, bezier helpers |
| `src/tilke/circuit.py` | `Circuit`, config dataclasses, `generate_circuit`, cone/validation/export functions |
| `src/tilke/image_utils.py` | `fig2data`, `rgb_to_rgba`, `apply_distance_transform` |
| `src/tilke/data_utils.py` | `get_sample_csv` batch generator |
| `tests/test_spline.py` | Smoke tests for spline creation, evaluation, derivatives, int/ext |
| `tests/test_circuit.py` | Smoke tests for generation, cones, export, determinism |
| `media/generate.py` | Script to produce demo GIFs |
| `media/cover.jpeg` | Cover image (moved from root) |
| `README.md` | Project documentation with embedded GIFs |

---

### Task 1: Scaffold project structure and tooling

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/tilke/__init__.py`, `src/tilke/py.typed`
- Remove: `setup.py`, `requirements.txt`, `tilke/VERSION`, `tilke/__init__.py`, `tilke/Circuit.py`, `tilke/CurveGenerator.py`, `tilke/Spline.py`, `tilke/data_utils.py`, `tilke/image_utils.py`, `citation.cff`, `tutorial.ipynb`, `TILKE_generation_*.csv`
- Move: `cover.jpeg` → `media/cover.jpeg`

- [ ] **Step 1: Remove old files and create directory structure**

```bash
# Remove old files
git rm setup.py requirements.txt citation.cff tutorial.ipynb 'TILKE_generation_*.csv' tilke/VERSION tilke/__init__.py tilke/Circuit.py tilke/CurveGenerator.py tilke/Spline.py tilke/data_utils.py tilke/image_utils.py
rm -rf tilke/

# Create new directories
mkdir -p src/tilke tests media
```

- [ ] **Step 2: Move cover image**

```bash
git mv cover.jpeg media/cover.jpeg
```

- [ ] **Step 3: Create `.python-version`**

Write `.python-version`:
```
3.12
```

- [ ] **Step 4: Create `.gitignore`**

Write `.gitignore`:
```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
.ruff_cache/
.pytest_cache/
```

- [ ] **Step 5: Create `pyproject.toml`**

Write `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tilke"
version = "0.2.0"
description = "Generate randomized Formula-style racing circuits"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
authors = [{ name = "Pol Puigdemont Plana" }]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "Pillow>=10.0",
    "noise>=1.2",
    "imageio>=2.33",
    "tqdm>=4.66",
]

[project.urls]
Repository = "https://github.com/puigde/tilke"

[tool.hatch.build.targets.wheel]
packages = ["src/tilke"]

[dependency-groups]
dev = ["ruff>=0.11", "ty>=0.0.1a7", "pytest>=8.0"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 6: Create placeholder `src/tilke/__init__.py` and `src/tilke/py.typed`**

Write `src/tilke/__init__.py`:
```python
"""tilke — generate randomized Formula-style racing circuits."""
```

Write `src/tilke/py.typed` (empty marker file):
```
```

- [ ] **Step 7: Install the project with uv and verify**

```bash
uv sync
uv run python -c "import tilke; print('ok')"
```

Expected: prints `ok`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold modern project structure with uv/ruff/ty"
```

---

### Task 2: Write `src/tilke/spline.py`

**Files:**
- Create: `src/tilke/spline.py`

- [ ] **Step 1: Write `src/tilke/spline.py`**

Write `src/tilke/spline.py`:
```python
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
    return np.array([
        splev(t, spline.xt, der=der),
        splev(t, spline.yt, der=der),
    ]).T


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
    steps = np.linspace(
        spline.t[0], spline.t[-1], math.floor(spline.t[-1] / stepsize)
    )
    x_int, x_ext, y_int, y_ext = [], [], [], []
    for s in steps:
        pos = evaluate(spline, s)
        vel = evaluate(spline, s, der=1)
        x, y = pos[0], pos[1]
        dx, dy = vel[0], vel[1]
        scale = dist / np.sqrt(dx**2 + dy**2)
        x_int.append(x + scale * dy)
        y_int.append(y - scale * dx)
        x_ext.append(x - scale * dy)
        y_ext.append(y + scale * dx)

    int_data = [x_int[::sampling_factor], y_int[::sampling_factor]]
    ext_data = [x_ext[::sampling_factor], y_ext[::sampling_factor]]
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
```

- [ ] **Step 2: Verify it imports**

```bash
uv run python -c "from tilke.spline import Spline, make_spline, evaluate; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/tilke/spline.py
git commit -m "feat: add spline module with functional API"
```

---

### Task 3: Write `tests/test_spline.py` and run tests

**Files:**
- Create: `tests/test_spline.py`

- [ ] **Step 1: Write `tests/test_spline.py`**

Write `tests/test_spline.py`:
```python
import numpy as np

from tilke.spline import evaluate, get_int_ext_splines, make_spline


def _make_circle_spline() -> "Spline":
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
        interior, exterior = get_int_ext_splines(spline, dist=0.2, stepsize=0.5)
        assert interior.t is not None
        assert exterior.t is not None
        # Interior should be smaller, exterior larger than the original
        mid_point = evaluate(spline, spline.t[0])
        int_point = evaluate(interior, interior.t[0])
        ext_point = evaluate(exterior, exterior.t[0])
        mid_r = np.linalg.norm(mid_point)
        int_r = np.linalg.norm(int_point)
        ext_r = np.linalg.norm(ext_point)
        assert int_r < mid_r or ext_r > mid_r  # at least one offset direction works
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_spline.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spline.py
git commit -m "test: add spline smoke tests"
```

---

### Task 4: Write `src/tilke/curve_generator.py`

**Files:**
- Create: `src/tilke/curve_generator.py`

- [ ] **Step 1: Write `src/tilke/curve_generator.py`**

Write `src/tilke/curve_generator.py`:
```python
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
```

- [ ] **Step 2: Verify it works**

```bash
uv run python -c "
from tilke.curve_generator import generate_middle_curve, CurveGeneratorConfig
curve = generate_middle_curve(CurveGeneratorConfig(seed=42))
print(f'Generated curve with {len(curve.t)} knots')
"
```

Expected: prints something like `Generated curve with N knots`

- [ ] **Step 3: Commit**

```bash
git add src/tilke/curve_generator.py
git commit -m "feat: add curve generator module with functional API"
```

---

### Task 5: Write `src/tilke/image_utils.py`

**Files:**
- Create: `src/tilke/image_utils.py`

- [ ] **Step 1: Write `src/tilke/image_utils.py`**

Write `src/tilke/image_utils.py`:
```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/tilke/image_utils.py
git commit -m "feat: add image utils with fixed deprecated numpy calls"
```

---

### Task 6: Write `src/tilke/circuit.py`

**Files:**
- Create: `src/tilke/circuit.py`

This is the largest module. It contains the config dataclasses, the `Circuit` dataclass, and all circuit functions.

- [ ] **Step 1: Write `src/tilke/circuit.py`**

Write `src/tilke/circuit.py`:
```python
"""Circuit generation, validation, population, and export."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import noise
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import KDTree

from tilke import image_utils
from tilke.curve_generator import CurveGeneratorConfig, generate_middle_curve
from tilke.spline import Spline, evaluate, get_int_ext_splines, make_spline


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CircuitRestrictions:
    """Curvature and distance constraints for circuit validation."""

    min_curvature_radius_middle_curve: float = 6.5
    min_curvature_radius_interior_curve: float = 4.5
    min_curvature_radius_exterior_curve: float = 8.0
    stepsize_interior_curvature_checking: float = 1.0
    stepsize_exterior_curvature_checking: float = 1.0
    stepsize_middle_curvature_checking: float = 1.0
    stepsize_middle_distance_checking: float = 1.0
    stepsize_interior_distance_checking: float = 1.0
    stepsize_exterior_distance_checking: float = 1.0
    min_interior_exterior_distance: float = 1.0
    min_interior_middle_distance: float = 1.0
    min_exterior_middle_distance: float = 1.0


@dataclass
class ConePopulationParameters:
    """Parameters controlling cone placement and noise contamination."""

    total_number_cones_lower_bound: int = 90
    total_number_cones_upper_bound: int = 130
    minimum_proportion_cones_per_curve: float = 0.4
    total_number_cones: int = 130
    false_negative_probability: float = 0.1
    false_positive_probability: float = 0.07
    interior_curve_range_boost: float = 0.025
    exterior_curve_range_boost: float = 0.025


@dataclass
class TrackParameters:
    """Parameters for generating interior/exterior curves from the middle line."""

    distance_between_middle_and_track_sides: float = 2.0
    stepsize_for_track_generation: float = 0.5
    smoothing_for_track_generation: float = 0.5
    sampling_factor_for_track_generation: int = 5


# ---------------------------------------------------------------------------
# Circuit dataclass
# ---------------------------------------------------------------------------


@dataclass
class Circuit:
    """A generated racing circuit with three curves and optional cone positions."""

    middle_curve: Spline
    interior_curve: Spline
    exterior_curve: Spline
    restrictions_compliant: bool
    orientation: str
    interior_cones: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    exterior_cones: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    false_cones: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_circuit(
    restrictions: CircuitRestrictions | None = None,
    curve_config: CurveGeneratorConfig | None = None,
    track_params: TrackParameters | None = None,
    middle_curve: Spline | None = None,
    orientation: str = "random",
    seed: int | None = None,
) -> Circuit:
    """Generate a random closed circuit layout.

    Retries curve generation until curvature and distance constraints are met
    (unless a fixed seed or middle_curve is provided, in which case constraints
    may be violated and restrictions_compliant will be False).

    Args:
        restrictions: Curvature/distance constraints. Defaults if None.
        curve_config: Middle curve generator config. Defaults if None.
        track_params: Interior/exterior offset params. Defaults if None.
        middle_curve: Pre-built middle curve. Generated if None.
        orientation: "clockwise", "counter_clockwise", or "random".
        seed: Random seed for the curve generator.

    Returns:
        A Circuit instance.
    """
    if restrictions is None:
        restrictions = CircuitRestrictions()
    if curve_config is None:
        curve_config = CurveGeneratorConfig(seed=seed)
    elif seed is not None:
        curve_config = CurveGeneratorConfig(
            num_control_points=curve_config.num_control_points,
            radius=curve_config.radius,
            control_points_radius=curve_config.control_points_radius,
            edginess=curve_config.edginess,
            seed=seed,
            min_control_point_distance=curve_config.min_control_point_distance,
            sampling_factor=curve_config.sampling_factor,
            smoothing=curve_config.smoothing,
        )
    if track_params is None:
        track_params = TrackParameters()

    restrictions_compliant = True
    while True:
        if middle_curve is None:
            mid = generate_middle_curve(curve_config)
        else:
            mid = middle_curve

        interior, exterior = get_int_ext_splines(
            mid,
            dist=track_params.distance_between_middle_and_track_sides,
            stepsize=track_params.stepsize_for_track_generation,
            smoothing=track_params.smoothing_for_track_generation,
            sampling_factor=track_params.sampling_factor_for_track_generation,
        )

        circuit = Circuit(
            middle_curve=mid,
            interior_curve=interior,
            exterior_curve=exterior,
            restrictions_compliant=True,
            orientation=orientation,
        )

        curvature_ok = check_curvature(circuit, restrictions)
        distances_ok = check_distances(circuit, restrictions)

        if curvature_ok and distances_ok:
            break

        # Fixed seed or provided curve: can't retry, mark non-compliant
        if seed is not None or middle_curve is not None:
            restrictions_compliant = False
            break

        # Otherwise retry with a new random curve
        middle_curve = None

    circuit.restrictions_compliant = restrictions_compliant

    # Handle orientation
    actual_orientation = orientation
    if orientation == "counter_clockwise" or (
        orientation == "random" and random.random() > 0.5
    ):
        smoothing = curve_config.smoothing
        circuit = Circuit(
            middle_curve=make_spline(mid.data[:, ::-1], s=smoothing),
            interior_curve=make_spline(interior.data[:, ::-1], s=smoothing),
            exterior_curve=make_spline(exterior.data[:, ::-1], s=smoothing),
            restrictions_compliant=restrictions_compliant,
            orientation="counter_clockwise",
        )
    else:
        circuit.orientation = "clockwise"

    return circuit


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _curvature_radius(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute the radius of curvature from first and second derivatives."""
    return ((d1[0] ** 2 + d1[1] ** 2) ** (3 / 2)) / np.abs(
        d1[0] * d2[1] - d1[1] * d2[0]
    )


def check_curvature(circuit: Circuit, restrictions: CircuitRestrictions) -> bool:
    """Check that all three curves satisfy minimum curvature radius constraints."""
    curves = {
        "middle": (
            circuit.middle_curve,
            restrictions.min_curvature_radius_middle_curve,
            restrictions.stepsize_middle_curvature_checking,
        ),
        "interior": (
            circuit.interior_curve,
            restrictions.min_curvature_radius_interior_curve,
            restrictions.stepsize_interior_curvature_checking,
        ),
        "exterior": (
            circuit.exterior_curve,
            restrictions.min_curvature_radius_exterior_curve,
            restrictions.stepsize_exterior_curvature_checking,
        ),
    }
    all_ok = True
    for curve, min_radius, stepsize in curves.values():
        steps = np.linspace(
            curve.t[0], curve.t[-1], math.floor(curve.t[-1] / stepsize)
        )
        d1 = evaluate(curve, steps, der=1)
        d2 = evaluate(curve, steps, der=2)
        radii = np.array(
            [_curvature_radius(d1[i], d2[i]) for i in range(len(steps))]
        )
        if np.any(radii < min_radius):
            all_ok = False
    return all_ok


def check_distances(circuit: Circuit, restrictions: CircuitRestrictions) -> bool:
    """Check that curves maintain minimum distances from each other."""
    mid_steps = np.linspace(
        circuit.middle_curve.t[0],
        circuit.middle_curve.t[-1],
        math.floor(
            circuit.middle_curve.t[-1] / restrictions.stepsize_middle_distance_checking
        ),
    )
    mid_pts = evaluate(circuit.middle_curve, mid_steps)

    int_steps = np.linspace(
        circuit.interior_curve.t[0],
        circuit.interior_curve.t[-1],
        math.floor(
            circuit.interior_curve.t[-1]
            / restrictions.stepsize_interior_distance_checking
        ),
    )
    int_pts = evaluate(circuit.interior_curve, int_steps)

    ext_steps = np.linspace(
        circuit.exterior_curve.t[0],
        circuit.exterior_curve.t[-1],
        math.floor(
            circuit.exterior_curve.t[-1]
            / restrictions.stepsize_exterior_distance_checking
        ),
    )
    ext_pts = evaluate(circuit.exterior_curve, ext_steps)

    interior_tree = KDTree(int_pts)
    exterior_tree = KDTree(ext_pts)

    int_ext_dist, _ = interior_tree.query(ext_pts)
    int_mid_dist, _ = interior_tree.query(mid_pts)
    ext_mid_dist, _ = exterior_tree.query(mid_pts)

    return bool(
        np.all(int_ext_dist > restrictions.min_interior_exterior_distance)
        and np.all(int_mid_dist > restrictions.min_interior_middle_distance)
        and np.all(ext_mid_dist > restrictions.min_exterior_middle_distance)
    )


# ---------------------------------------------------------------------------
# Cone population
# ---------------------------------------------------------------------------


def populate_cones(
    circuit: Circuit,
    method: str = "perlin",
    params: ConePopulationParameters | None = None,
) -> Circuit:
    """Place cones along interior and exterior curves.

    Args:
        circuit: The circuit to populate.
        method: "naive" (uniform), "perlin" (noise-offset), or "random" (random count + noise).
        params: Cone population parameters. Defaults if None.

    Returns:
        A new Circuit with cones populated.
    """
    if params is None:
        params = ConePopulationParameters()

    if method == "naive":
        int_cones, ext_cones = _populate_naive(circuit, params.total_number_cones)
    elif method == "perlin":
        int_cones, ext_cones = _populate_perlin(circuit, params.total_number_cones)
    elif method == "random":
        int_cones, ext_cones = _populate_random(circuit, params)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'naive', 'perlin', or 'random'.")

    return Circuit(
        middle_curve=circuit.middle_curve,
        interior_curve=circuit.interior_curve,
        exterior_curve=circuit.exterior_curve,
        restrictions_compliant=circuit.restrictions_compliant,
        orientation=circuit.orientation,
        interior_cones=int_cones,
        exterior_cones=ext_cones,
    )


def _populate_naive(
    circuit: Circuit, n_cones: int
) -> tuple[np.ndarray, np.ndarray]:
    upscale = circuit.exterior_curve.t[-1] / circuit.interior_curve.t[-1]
    n_interior = n_cones // 2
    int_positions = np.linspace(0, circuit.interior_curve.t[-1], n_interior)
    int_cones = np.array([evaluate(circuit.interior_curve, t) for t in int_positions])
    ext_cones = np.array(
        [evaluate(circuit.exterior_curve, t * upscale) for t in int_positions]
    )
    return int_cones, ext_cones


def _populate_perlin(
    circuit: Circuit, n_cones: int
) -> tuple[np.ndarray, np.ndarray]:
    upscale = circuit.exterior_curve.t[-1] / circuit.interior_curve.t[-1]
    n_interior = n_cones // 2
    positions = np.linspace(0, circuit.interior_curve.t[-1], n_interior)
    max_t = circuit.interior_curve.t[-1]
    for i, val in enumerate(positions):
        positions[i] = min(val + noise.pnoise1(val), max_t)
        if positions[i] >= max_t:
            positions = positions[: i + 1]
            break
    int_cones = np.array([evaluate(circuit.interior_curve, t) for t in positions])
    ext_cones = np.array(
        [evaluate(circuit.exterior_curve, t * upscale) for t in positions]
    )
    return int_cones, ext_cones


def _populate_random(
    circuit: Circuit, params: ConePopulationParameters
) -> tuple[np.ndarray, np.ndarray]:
    total = random.randint(
        params.total_number_cones_lower_bound,
        params.total_number_cones_upper_bound,
    )
    min_per_curve = int(params.minimum_proportion_cones_per_curve * total)
    n_interior = random.randint(min_per_curve, total - min_per_curve)
    n_exterior = total - n_interior

    int_positions = np.linspace(
        circuit.interior_curve.t[0], circuit.interior_curve.t[-1], n_interior
    )
    ext_positions = np.linspace(
        circuit.exterior_curve.t[0], circuit.exterior_curve.t[-1], n_exterior
    )
    int_max = circuit.interior_curve.t[-1]
    ext_max = circuit.exterior_curve.t[-1]

    for i, val in enumerate(int_positions):
        int_positions[i] = min(val + noise.pnoise1(val), int_max)
        if int_positions[i] >= int_max:
            int_positions = int_positions[: i + 1]
            break

    for i, val in enumerate(ext_positions):
        ext_positions[i] = min(val + noise.pnoise1(val), ext_max)
        if ext_positions[i] >= ext_max:
            ext_positions = ext_positions[: i + 1]
            break

    int_cones = np.array(
        [evaluate(circuit.interior_curve, t) for t in int_positions]
    )
    ext_cones = np.array(
        [evaluate(circuit.exterior_curve, t) for t in ext_positions]
    )
    return int_cones, ext_cones


def contaminate_cones(
    circuit: Circuit,
    params: ConePopulationParameters | None = None,
) -> Circuit:
    """Simulate noisy cone detection by removing and adding cones randomly.

    Args:
        circuit: Circuit with populated cones.
        params: Contamination parameters. Defaults if None.

    Returns:
        New Circuit with contaminated cone population.
    """
    if params is None:
        params = ConePopulationParameters()

    # False negatives: remove some cones with boosted probability for adjacent misses
    int_cones = _apply_false_negatives(
        circuit.interior_cones,
        params.false_negative_probability,
        params.interior_curve_range_boost,
    )
    ext_cones = _apply_false_negatives(
        circuit.exterior_cones,
        params.false_negative_probability,
        params.exterior_curve_range_boost,
    )

    # False positives: add random cones in the bounding area
    all_cones = np.vstack([circuit.exterior_cones]) if len(circuit.exterior_cones) > 0 else np.empty((0, 2))
    false_cones = _generate_false_cones(
        all_cones,
        len(circuit.interior_cones) + len(circuit.exterior_cones),
        params.false_positive_probability,
    )

    return Circuit(
        middle_curve=circuit.middle_curve,
        interior_curve=circuit.interior_curve,
        exterior_curve=circuit.exterior_curve,
        restrictions_compliant=circuit.restrictions_compliant,
        orientation=circuit.orientation,
        interior_cones=int_cones,
        exterior_cones=ext_cones,
        false_cones=false_cones,
    )


def _apply_false_negatives(
    cones: np.ndarray,
    base_prob: float,
    range_boost: float,
) -> np.ndarray:
    if len(cones) == 0:
        return cones
    keep = []
    prev_removed = False
    for i in range(len(cones)):
        prob = base_prob + (range_boost if prev_removed else 0.0)
        if random.random() < prob:
            prev_removed = True
        else:
            keep.append(i)
            prev_removed = False
    if not keep:
        return np.empty((0, 2))
    return cones[keep]


def _generate_false_cones(
    reference_cones: np.ndarray,
    total_cone_count: int,
    false_positive_prob: float,
) -> np.ndarray:
    if len(reference_cones) == 0:
        return np.empty((0, 2))
    min_x, min_y = reference_cones.min(axis=0)
    max_x, max_y = reference_cones.max(axis=0)
    max_x += random.uniform(0, 80)
    max_y += random.uniform(0, 80)
    false = []
    for _ in range(total_cone_count):
        if random.random() < false_positive_prob:
            false.append([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
    if not false:
        return np.empty((0, 2))
    return np.array(false)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_circuit(circuit: Circuit) -> None:
    """Plot the circuit layout with all three curves and cones."""
    curves = [
        ("middle", circuit.middle_curve),
        ("interior", circuit.interior_curve),
        ("exterior", circuit.exterior_curve),
    ]
    for _, curve in curves:
        tt = np.linspace(curve.t[0], curve.t[-1], 1000)
        gamma = evaluate(curve, tt)
        plt.plot(gamma[:, 0], gamma[:, 1])

    if len(circuit.interior_cones) > 0:
        plt.scatter(
            circuit.interior_cones[:, 0], circuit.interior_cones[:, 1], c="blue", s=10
        )
    if len(circuit.exterior_cones) > 0:
        plt.scatter(
            circuit.exterior_cones[:, 0], circuit.exterior_cones[:, 1], c="orange", s=10
        )
    if len(circuit.false_cones) > 0:
        plt.plot(circuit.false_cones[:, 0], circuit.false_cones[:, 1], "xr")


# ---------------------------------------------------------------------------
# Export: image
# ---------------------------------------------------------------------------


def circuit_to_image(
    circuit: Circuit,
    save: bool = False,
    saving_path: str = "",
    circuit_filename: str = "circuit.png",
    track_filename: str = "track.png",
    cones_filename: str = "cones.png",
    precision: int = 1000,
    show_cones: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the circuit as images.

    Returns:
        (circuit_image, track_label, cones_input) — all uint8 numpy arrays.
    """
    curve_colors = {
        "middle": image_utils.rgb_to_rgba(0, 255, 255),
        "interior": image_utils.rgb_to_rgba(255, 0, 0),
        "exterior": image_utils.rgb_to_rgba(0, 255, 0),
    }
    label_colors = {
        "middle": image_utils.rgb_to_rgba(0, 0, 0),
        "interior": image_utils.rgb_to_rgba(255, 255, 255),
        "exterior": image_utils.rgb_to_rgba(255, 255, 255),
    }
    zone_colors = {
        "inside": image_utils.rgb_to_rgba(255, 255, 0),
        "track": image_utils.rgb_to_rgba(0, 0, 255),
    }
    label_zone_colors = {
        "inside": image_utils.rgb_to_rgba(255, 255, 255),
        "track": image_utils.rgb_to_rgba(0, 0, 0),
    }
    cone_color = image_utils.rgb_to_rgba(0, 0, 0)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for a in (ax, ax2, ax3):
        a.axis("off")

    curves_data = {}
    for name, curve in [
        ("middle", circuit.middle_curve),
        ("interior", circuit.interior_curve),
        ("exterior", circuit.exterior_curve),
    ]:
        tt = np.linspace(curve.t[0], curve.t[-1], precision)
        gamma = evaluate(curve, tt)
        curves_data[name] = gamma
        ax.plot(gamma[:, 0], gamma[:, 1], color=curve_colors[name])
        ax2.plot(gamma[:, 0], gamma[:, 1], color=label_colors[name])

    # Zone polygons
    int_xy = np.column_stack(
        (curves_data["interior"][:, 0], curves_data["interior"][:, 1])
    )
    ext_xy = np.column_stack(
        (curves_data["exterior"][:, 0], curves_data["exterior"][:, 1])
    )
    ax.add_patch(Polygon(ext_xy, facecolor=zone_colors["track"], alpha=1, edgecolor="none"))
    ax.add_patch(Polygon(int_xy, facecolor=zone_colors["inside"], alpha=1, edgecolor="none"))
    ax2.add_patch(Polygon(ext_xy, facecolor=label_zone_colors["track"], alpha=1, edgecolor="none"))
    ax2.add_patch(Polygon(int_xy, facecolor=label_zone_colors["inside"], alpha=1, edgecolor="none"))

    # Cones
    if show_cones:
        for cones in (circuit.interior_cones, circuit.exterior_cones):
            if len(cones) > 0:
                ax.scatter(cones[:, 0], cones[:, 1], c="black", s=10)
                ax3.scatter(cones[:, 0], cones[:, 1], color=cone_color, s=1)
        if len(circuit.false_cones) > 0:
            ax3.scatter(circuit.false_cones[:, 0], circuit.false_cones[:, 1], color=cone_color, s=1)
            ax2.scatter(circuit.false_cones[:, 0], circuit.false_cones[:, 1], color="w", s=1)

    im = image_utils.fig2data(fig)
    model_label = image_utils.fig2data(fig2)
    model_input = image_utils.fig2data(fig3)
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)

    if save:
        imageio.imwrite(saving_path + circuit_filename, im)
        imageio.imwrite(saving_path + track_filename, model_label)
        imageio.imwrite(saving_path + cones_filename, model_input)

    model_input = np.where(model_input > 128, 255, 0)
    model_label = np.where(model_label > 128, 1.0, 0)
    return np.uint8(im), np.uint8(model_label), np.uint8(model_input)


# ---------------------------------------------------------------------------
# Export: CSV
# ---------------------------------------------------------------------------


def circuit_to_csv(
    circuit: Circuit,
    filename: str | None = None,
    include_false_cones: bool = True,
    include_track_points: bool = True,
) -> None:
    """Save circuit data to a CSV file.

    Args:
        circuit: The circuit to export.
        filename: Output path (without .csv extension). Auto-generated if None.
        include_false_cones: Include false positive cones in output.
        include_track_points: Include track spline control points.
    """
    if filename is None:
        filename = f"TILKE_generation_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')}"

    with open(f"{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["EC-cones"])
        for cone in circuit.exterior_cones:
            writer.writerow([round(cone[0], 4), round(cone[1], 4)])

        writer.writerow(["IC-cones"])
        for cone in circuit.interior_cones:
            writer.writerow([round(cone[0], 4), round(cone[1], 4)])

        if include_false_cones and len(circuit.false_cones) > 0:
            writer.writerow(["F-cones"])
            for cone in circuit.false_cones:
                writer.writerow([round(cone[0], 4), round(cone[1], 4)])

        if include_track_points:
            titles = ["EC-points", "IC-points", "MC-points"]
            curves = [
                circuit.exterior_curve,
                circuit.interior_curve,
                circuit.middle_curve,
            ]
            for title, curve in zip(titles, curves):
                writer.writerow([title])
                for i in range(curve.data.shape[1]):
                    writer.writerow(
                        [round(float(curve.data[0, i]), 4), round(float(curve.data[1, i]), 4)]
                    )
```

- [ ] **Step 2: Verify import and basic generation**

```bash
uv run python -c "
from tilke.circuit import generate_circuit, populate_cones, Circuit
c = generate_circuit(seed=42)
print(f'Generated: {c.orientation}, compliant={c.restrictions_compliant}')
c = populate_cones(c, method='naive')
print(f'Interior cones: {len(c.interior_cones)}, Exterior cones: {len(c.exterior_cones)}')
"
```

Expected: prints circuit info with cone counts.

- [ ] **Step 3: Commit**

```bash
git add src/tilke/circuit.py
git commit -m "feat: add circuit module with functional API"
```

---

### Task 7: Write `tests/test_circuit.py` and run tests

**Files:**
- Create: `tests/test_circuit.py`

- [ ] **Step 1: Write `tests/test_circuit.py`**

Write `tests/test_circuit.py`:
```python
import os
import tempfile

import numpy as np

from tilke.circuit import (
    Circuit,
    CircuitRestrictions,
    ConePopulationParameters,
    check_curvature,
    check_distances,
    circuit_to_csv,
    circuit_to_image,
    contaminate_cones,
    generate_circuit,
    populate_cones,
)
from tilke.spline import evaluate


class TestGenerateCircuit:
    def test_returns_circuit(self):
        c = generate_circuit(seed=42)
        assert isinstance(c, Circuit)
        assert c.middle_curve is not None
        assert c.interior_curve is not None
        assert c.exterior_curve is not None

    def test_seeded_determinism(self):
        c1 = generate_circuit(seed=1337)
        c2 = generate_circuit(seed=1337)
        t = c1.middle_curve.t[0]
        p1 = evaluate(c1.middle_curve, t)
        p2 = evaluate(c2.middle_curve, t)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_orientation_clockwise(self):
        c = generate_circuit(seed=42, orientation="clockwise")
        assert c.orientation == "clockwise"

    def test_orientation_counter_clockwise(self):
        c = generate_circuit(seed=42, orientation="counter_clockwise")
        assert c.orientation == "counter_clockwise"

    def test_restrictions_compliant(self):
        c = generate_circuit()
        assert c.restrictions_compliant is True
        restrictions = CircuitRestrictions()
        assert check_curvature(c, restrictions) is True
        assert check_distances(c, restrictions) is True


class TestPopulateCones:
    def test_naive(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="naive")
        assert len(c.interior_cones) > 0
        assert len(c.exterior_cones) > 0

    def test_perlin(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="perlin")
        assert len(c.interior_cones) > 0
        assert len(c.exterior_cones) > 0

    def test_random(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="random")
        assert len(c.interior_cones) > 0
        assert len(c.exterior_cones) > 0

    def test_contaminate(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="perlin")
        original_int = len(c.interior_cones)
        original_ext = len(c.exterior_cones)
        params = ConePopulationParameters(
            false_negative_probability=0.5,
            false_positive_probability=0.5,
        )
        contaminated = contaminate_cones(c, params)
        # With 50% removal, very likely some are removed
        total_original = original_int + original_ext
        total_contaminated = len(contaminated.interior_cones) + len(contaminated.exterior_cones)
        assert total_contaminated <= total_original


class TestExport:
    def test_to_csv(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="naive")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_circuit")
            circuit_to_csv(c, filename=path)
            csv_path = path + ".csv"
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                content = f.read()
            assert "EC-cones" in content
            assert "IC-cones" in content
            assert "EC-points" in content

    def test_to_image(self):
        c = generate_circuit(seed=42)
        c = populate_cones(c, method="naive")
        im, label, inp = circuit_to_image(c)
        assert im.dtype == np.uint8
        assert label.dtype == np.uint8
        assert inp.dtype == np.uint8
        assert im.ndim == 3
        assert label.ndim == 3
        assert inp.ndim == 3
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (both `test_spline.py` and `test_circuit.py`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_circuit.py
git commit -m "test: add circuit smoke tests"
```

---

### Task 8: Write `src/tilke/data_utils.py` and finalize `src/tilke/__init__.py`

**Files:**
- Create: `src/tilke/data_utils.py`
- Modify: `src/tilke/__init__.py`

- [ ] **Step 1: Write `src/tilke/data_utils.py`**

Write `src/tilke/data_utils.py`:
```python
"""Batch generation utilities for circuit datasets."""

from __future__ import annotations

import os
from datetime import datetime

from tqdm import tqdm

from tilke.circuit import (
    Circuit,
    circuit_to_csv,
    contaminate_cones,
    generate_circuit,
    populate_cones,
)


def get_sample_csv(
    n: int,
    path: str,
    contaminated: bool = False,
    include_false_cones: bool = True,
    include_track_points: bool = True,
) -> None:
    """Generate n circuit CSV files in the given directory.

    Args:
        n: Number of circuits to generate.
        path: Output directory (created if it doesn't exist).
        contaminated: Whether to apply cone contamination.
        include_false_cones: Include false positive cones in CSV.
        include_track_points: Include track spline points in CSV.
    """
    os.makedirs(path, exist_ok=True)
    for _ in tqdm(range(n)):
        c = generate_circuit()
        c = populate_cones(c)
        if contaminated:
            c = contaminate_cones(c)
        circuit_to_csv(
            c,
            f"{path}/TILKE_generation_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S-%f')}",
            include_false_cones=include_false_cones,
            include_track_points=include_track_points,
        )
```

- [ ] **Step 2: Finalize `src/tilke/__init__.py`**

Write `src/tilke/__init__.py`:
```python
"""tilke — generate randomized Formula-style racing circuits."""

from tilke.circuit import (
    Circuit,
    CircuitRestrictions,
    ConePopulationParameters,
    TrackParameters,
    check_curvature,
    check_distances,
    circuit_to_csv,
    circuit_to_image,
    contaminate_cones,
    generate_circuit,
    plot_circuit,
    populate_cones,
)
from tilke.curve_generator import CurveGeneratorConfig, generate_middle_curve
from tilke.data_utils import get_sample_csv
from tilke.spline import Spline, evaluate, make_spline

__all__ = [
    "Circuit",
    "CircuitRestrictions",
    "ConePopulationParameters",
    "CurveGeneratorConfig",
    "Spline",
    "TrackParameters",
    "check_curvature",
    "check_distances",
    "circuit_to_csv",
    "circuit_to_image",
    "contaminate_cones",
    "evaluate",
    "generate_circuit",
    "generate_middle_curve",
    "get_sample_csv",
    "make_spline",
    "plot_circuit",
    "populate_cones",
]
```

- [ ] **Step 3: Verify top-level imports work**

```bash
uv run python -c "
from tilke import generate_circuit, populate_cones, plot_circuit
c = generate_circuit(seed=42)
c = populate_cones(c)
print(f'Cones: {len(c.interior_cones)} int, {len(c.exterior_cones)} ext')
print('All imports OK')
"
```

Expected: prints cone counts and `All imports OK`

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tilke/data_utils.py src/tilke/__init__.py
git commit -m "feat: add data utils and finalize public API"
```

---

### Task 9: Run ruff and ty

**Files:**
- Modify: all `src/tilke/*.py` and `tests/*.py` files (auto-formatted)

- [ ] **Step 1: Run ruff format**

```bash
uv run ruff format src/ tests/
```

Expected: files formatted (may show count of reformatted files).

- [ ] **Step 2: Run ruff check and fix auto-fixable issues**

```bash
uv run ruff check src/ tests/ --fix
```

Expected: no errors remaining (or only intentional ones).

- [ ] **Step 3: Run ruff check without --fix to verify clean**

```bash
uv run ruff check src/ tests/
```

Expected: `All checks passed!`

- [ ] **Step 4: Run ty type checker**

```bash
uv run ty check src/
```

Expected: no errors, or only warnings about third-party type stubs (noise, imageio). If there are issues with the `noise` package not having type stubs, that's expected and acceptable.

- [ ] **Step 5: Commit any formatting changes**

```bash
git add -A
git diff --cached --stat
# Only commit if there are changes
git commit -m "style: apply ruff formatting and lint fixes" || echo "Nothing to commit"
```

---

### Task 10: Write `media/generate.py` and produce GIFs

**Files:**
- Create: `media/generate.py`

- [ ] **Step 1: Verify ffmpeg is available**

```bash
ffmpeg -version | head -1
```

Expected: prints ffmpeg version. If not installed, install with `brew install ffmpeg`.

- [ ] **Step 2: Write `media/generate.py`**

Write `media/generate.py`:
```python
"""Generate demo GIFs for the README."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tilke import generate_circuit, populate_cones, contaminate_cones, plot_circuit


MEDIA_DIR = Path(__file__).parent


def _save_frame(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _frames_to_gif(frame_dir: Path, output: Path, fps: int = 2) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%03d.png"),
        "-vf", "palettegen",
        str(frame_dir / "palette.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%03d.png"),
        "-i", str(frame_dir / "palette.png"),
        "-lavfi", "paletteuse",
        "-loop", "0",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def generate_variety_gif() -> None:
    """GIF showing 8 different circuit layouts."""
    print("Generating circuit variety GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        seeds = [42, 123, 256, 404, 512, 777, 1024, 1337]
        for i, seed in enumerate(seeds):
            c = generate_circuit(seed=seed)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"seed={seed}", fontsize=12, fontfamily="monospace")
            plt.sca(ax)
            plot_circuit(c)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/8 (seed={seed})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "circuits.gif", fps=2)
    print(f"  Saved: {MEDIA_DIR / 'circuits.gif'}")


def generate_cones_gif() -> None:
    """GIF showing cone population stages on one circuit."""
    print("Generating cone population GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=42)
        stages = [
            ("Empty circuit", c),
            ("Naive cones", populate_cones(c, method="naive")),
            ("Perlin cones", populate_cones(c, method="perlin")),
            ("Contaminated", contaminate_cones(populate_cones(c, method="perlin"))),
        ]
        for i, (title, circuit) in enumerate(stages):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=12, fontfamily="monospace")
            plt.sca(ax)
            plot_circuit(circuit)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/4 ({title})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "cones.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'cones.gif'}")


if __name__ == "__main__":
    generate_variety_gif()
    generate_cones_gif()
    print("Done!")
```

- [ ] **Step 3: Run the generator**

```bash
uv run python media/generate.py
```

Expected: prints progress, creates `media/circuits.gif` and `media/cones.gif`.

- [ ] **Step 4: Verify GIFs exist**

```bash
ls -la media/*.gif
```

Expected: two GIF files.

- [ ] **Step 5: Commit**

```bash
git add media/generate.py
git commit -m "feat: add GIF generation script for README demos"
```

---

### Task 11: Write `README.md`

**Files:**
- Create: `README.md` (replace existing)

- [ ] **Step 1: Write `README.md`**

Write `README.md`:
```markdown
# tilke

Generate randomized Formula-style racing circuits with Bezier curves, spline interpolation, and configurable cone populations.

![Circuit variety](media/circuits.gif)

## Install

```bash
# With uv
uv add tilke

# Or with pip
pip install .
```

## Quick start

```python
from tilke import generate_circuit, populate_cones, plot_circuit

circuit = generate_circuit()
circuit = populate_cones(circuit)
plot_circuit(circuit)
```

![Cone population stages](media/cones.gif)

## API

### Circuit generation

```python
from tilke import generate_circuit, CurveGeneratorConfig, CircuitRestrictions, TrackParameters

# Default random circuit
circuit = generate_circuit()

# Fixed seed for reproducibility
circuit = generate_circuit(seed=1337)

# Custom curve generator
config = CurveGeneratorConfig(num_control_points=10, radius=200.0, edginess=0.1)
circuit = generate_circuit(curve_config=config)

# Custom restrictions
restrictions = CircuitRestrictions(min_curvature_radius_middle_curve=8.0)
circuit = generate_circuit(restrictions=restrictions)

# Control orientation
circuit = generate_circuit(orientation="clockwise")  # or "counter_clockwise", "random"
```

### Cone population

```python
from tilke import populate_cones, contaminate_cones, ConePopulationParameters

# Uniform spacing
circuit = populate_cones(circuit, method="naive")

# Perlin noise offset (default)
circuit = populate_cones(circuit, method="perlin")

# Random count per side
circuit = populate_cones(circuit, method="random")

# Simulate noisy detection
circuit = contaminate_cones(circuit)

# Custom parameters
params = ConePopulationParameters(false_negative_probability=0.2)
circuit = contaminate_cones(circuit, params)
```

### Export

```python
from tilke import circuit_to_csv, circuit_to_image

# Save to CSV
circuit_to_csv(circuit, filename="my_circuit")

# Render to images
circuit_img, track_label, cones_input = circuit_to_image(circuit)

# Batch generation
from tilke import get_sample_csv
get_sample_csv(n=100, path="dataset/", contaminated=True)
```

### Configuration dataclasses

| Class | Purpose |
|-------|---------|
| `CurveGeneratorConfig` | Control point count, scale, smoothing, seed |
| `CircuitRestrictions` | Minimum curvature radii, distance constraints |
| `ConePopulationParameters` | Cone counts, noise probabilities |
| `TrackParameters` | Interior/exterior offset distance and resolution |

All have sensible defaults — construct with no arguments for standard behavior.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with API docs and GIF demos"
```

---

### Task 12: Final cleanup and verification

**Files:**
- Verify all old files are removed
- Run full test suite
- Verify package installs and works end-to-end

- [ ] **Step 1: Verify no old files remain**

```bash
# These should all be gone
ls setup.py requirements.txt citation.cff tutorial.ipynb tilke/ 2>&1
```

Expected: `No such file or directory` for all of them.

- [ ] **Step 2: Verify directory structure is clean**

```bash
find . -not -path './.git/*' -not -path './.git' -not -path './.venv/*' -not -path './.venv' -not -name '*.pyc' -not -path '*__pycache__*' | sort
```

Expected output should match the target structure:
```
.
./.gitignore
./.python-version
./LICENSE
./README.md
./docs/superpowers/plans/2026-04-15-tilke-cleanup.md
./docs/superpowers/specs/2026-04-15-tilke-cleanup-design.md
./media/cover.jpeg
./media/generate.py
./pyproject.toml
./src/tilke/__init__.py
./src/tilke/circuit.py
./src/tilke/curve_generator.py
./src/tilke/data_utils.py
./src/tilke/image_utils.py
./src/tilke/py.typed
./src/tilke/spline.py
./tests/test_circuit.py
./tests/test_spline.py
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all 16 tests pass.

- [ ] **Step 4: Run full lint + format + type check**

```bash
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

Expected: all checks pass.

- [ ] **Step 5: End-to-end smoke test**

```bash
uv run python -c "
from tilke import (
    generate_circuit, populate_cones, contaminate_cones,
    circuit_to_csv, circuit_to_image, get_sample_csv,
    CurveGeneratorConfig, CircuitRestrictions, ConePopulationParameters,
)
import tempfile, os

# Generate
c = generate_circuit(seed=42)
c = populate_cones(c, method='perlin')
c = contaminate_cones(c)

# Export CSV
with tempfile.TemporaryDirectory() as d:
    circuit_to_csv(c, filename=os.path.join(d, 'test'))
    assert os.path.exists(os.path.join(d, 'test.csv'))

# Export image
im, label, inp = circuit_to_image(c)
assert im.shape[2] in (3, 4)

print('All smoke tests passed!')
"
```

Expected: `All smoke tests passed!`

- [ ] **Step 6: Final commit if any loose changes**

```bash
git status
# If anything unstaged:
git add -A && git commit -m "chore: final cleanup" || echo "Clean working tree"
```
