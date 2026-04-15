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
