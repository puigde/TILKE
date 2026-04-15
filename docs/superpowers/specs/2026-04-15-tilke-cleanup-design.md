# TILKE Cleanup — Design Spec

**Date:** 2026-04-15
**Scope:** Phase 1 — modernize tooling, restructure, functional rewrite, tests, README + media

Phase 2 (numpy vectorization, `noise` dependency reduction) is out of scope but this cleanup provides the foundation for it.

---

## 1. Project structure

Clean room rebuild inside the existing `TILKE/` git repo. All old files removed, new structure:

```
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .python-version         # 3.12
├── media/
│   ├── cover.jpeg
│   └── generate.py
├── src/
│   └── tilke/
│       ├── __init__.py
│       ├── circuit.py
│       ├── curve_generator.py
│       ├── spline.py
│       ├── image_utils.py
│       ├── data_utils.py
│       └── py.typed
└── tests/
    ├── test_circuit.py
    └── test_spline.py
```

## 2. Tooling

**`pyproject.toml`** is the single config source:

- Build system: `hatchling`
- Python: `>=3.12`
- Dependencies: `numpy`, `scipy`, `matplotlib`, `Pillow`, `noise`, `imageio`, `tqdm`
- Dev dependencies (via `[dependency-groups]`): `ruff`, `ty`, `pytest`
- `[tool.ruff]` section for linting + formatting
- `[tool.ty]` section for type checking

**`.python-version`**: `3.12`

**Removed files:**
- `setup.py`
- `requirements.txt`
- `tilke/VERSION`
- `citation.cff`
- `tutorial.ipynb`
- `TILKE_generation_*.csv`

## 3. Functional rewrite

Architecture shift from OOP classes-with-methods to dataclasses + free functions. No logic changes.

### `spline.py`

`Spline` becomes a pure data container. Cone storage removed (it's a circuit concern). Operations become free functions:

```python
@dataclass
class Spline:
    xt: tuple       # scipy spline tck for x
    yt: tuple       # scipy spline tck for y
    t: np.ndarray   # parameter domain
    data: np.ndarray

def evaluate(spline: Spline, t, der: int = 0) -> np.ndarray: ...
def derivative(spline: Spline) -> Spline: ...
def get_int_ext_splines(spline: Spline, dist, stepsize, smoothing, sampling_factor) -> tuple[Spline, Spline]: ...
def plot_spline(spline: Spline, precision=1000, cones=None) -> None: ...
```

### `curve_generator.py`

Generator config becomes a dataclass. Generation becomes a single function. Helper classes/functions stay module-level:

```python
@dataclass
class CurveGeneratorConfig:
    num_control_points: int = 8
    radius: float = 140.0
    control_points_radius: float = 0.5
    edginess: float = 0.05
    seed: int | None = None
    min_control_point_distance: float | None = None
    sampling_factor: int = 10
    smoothing: float = 0.5

def generate_middle_curve(config: CurveGeneratorConfig) -> Spline: ...

# Module-level helpers
def ccw_sort(points: np.ndarray) -> np.ndarray: ...
def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray: ...
def bezier(points: np.ndarray, num: int = 200) -> np.ndarray: ...

@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    angle1: float
    angle2: float
    curve: np.ndarray  # bezier curve points, computed via __post_init__
```

### `circuit.py`

Circuit becomes a dataclass. All methods become free functions. Cone population method selection via a `method` parameter instead of three separate methods:

```python
@dataclass
class CircuitRestrictions:
    min_curvature_radius_middle_curve: float = 6.5
    min_curvature_radius_interior_curve: float = 4.5
    min_curvature_radius_exterior_curve: float = 8
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
    distance_between_middle_and_track_sides: float = 2
    stepsize_for_track_generation: float = 0.5
    smoothing_for_track_generation: float = 0.5
    sampling_factor_for_track_generation: int = 5

@dataclass
class Circuit:
    middle_curve: Spline
    interior_curve: Spline
    exterior_curve: Spline
    interior_cones: list[np.ndarray]
    exterior_cones: list[np.ndarray]
    false_cones: list[np.ndarray]
    restrictions_compliant: bool
    orientation: str

def generate_circuit(
    restrictions: CircuitRestrictions | None = None,
    curve_config: CurveGeneratorConfig | None = None,
    cone_params: ConePopulationParameters | None = None,
    track_params: TrackParameters | None = None,
    middle_curve: Spline | None = None,
    orientation: str = "random",
    seed: int | None = None,
) -> Circuit: ...

def check_curvature(circuit: Circuit, restrictions: CircuitRestrictions) -> bool: ...
def check_distances(circuit: Circuit, restrictions: CircuitRestrictions) -> bool: ...
def populate_cones(circuit: Circuit, method: str = "perlin") -> Circuit: ...
def contaminate_cones(circuit: Circuit, params: ConePopulationParameters) -> Circuit: ...
def plot_circuit(circuit: Circuit) -> None: ...
def circuit_to_image(circuit: Circuit, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def circuit_to_csv(circuit: Circuit, path: str, ...) -> None: ...
```

### `image_utils.py`

Minimal fixes only:
- `np.fromstring` → `np.frombuffer`
- `.tostring()` → `.tobytes()`

### `data_utils.py`

Updated to use the new functional API (`generate_circuit`, `populate_cones`, etc.).

### `__init__.py`

Thin re-exports of public API:
```python
from tilke.circuit import (
    Circuit, CircuitRestrictions, ConePopulationParameters, TrackParameters,
    generate_circuit, populate_cones, contaminate_cones, plot_circuit,
    circuit_to_image, circuit_to_csv,
)
from tilke.curve_generator import CurveGeneratorConfig, generate_middle_curve
from tilke.data_utils import get_sample_csv
```

## 4. Code quality fixes

| Issue | Fix |
|-------|-----|
| Bare `except:` in `to_image` | Catch specific exception or remove |
| `np.fromstring` (deprecated) | `np.frombuffer` |
| `.tostring()` (deprecated) | `.tobytes()` |
| Lambda in loop (`curve_radius`) | Named function |
| `for i in range(len(...))` | enumerate / comprehension |
| Module-level lambda `bernstein` | `def bernstein(...)` |
| `f = lambda ang:` | `np.where` or named function |
| Typo `_get_random_contol_points` | `_get_random_control_points` |
| Typo `inteior_tree` | `interior_tree` |
| Recursive control point gen | While loop |
| `self` documented as argument | Remove |

Ruff handles remaining style (import sorting, trailing whitespace, quote consistency, etc.).

## 5. Tests

**`tests/test_circuit.py`:**

| Test | Validates |
|------|-----------|
| `test_generate_circuit` | Returns valid Circuit with three curves |
| `test_generate_circuit_seeded` | Same seed → same output |
| `test_generate_circuit_orientations` | clockwise, counter_clockwise, random all work |
| `test_populate_cones_methods` | naive, perlin, random all produce cones |
| `test_contaminate_cones` | Adds false cones, removes some true ones |
| `test_circuit_to_csv` | Writes CSV with expected sections |
| `test_circuit_to_image` | Returns three arrays with correct shape/dtype |
| `test_restrictions_compliant` | Default circuit passes validation |

**`tests/test_spline.py`:**

| Test | Validates |
|------|-----------|
| `test_evaluate` | Evaluates without error, correct shape |
| `test_derivative` | Returns valid Spline |
| `test_int_ext_splines` | Generates interior/exterior from middle |

Run: `uv run pytest`

## 6. README and media

**`media/generate.py`** produces two GIFs via matplotlib frames + ffmpeg:

1. **Circuit variety** — 8-10 circuits with different seeds, one frame each, looping
2. **Cone population** — one circuit shown empty → naive → perlin → contaminated

Run: `uv run python media/generate.py`

**`README.md`** structure:

1. Title + one-line description
2. Variety GIF
3. Install instructions (`uv add tilke` / `pip install .`)
4. Quick start code snippet (functional API)
5. Cone population GIF
6. API reference (brief, covers all public functions and config dataclasses)
7. License (MIT)
