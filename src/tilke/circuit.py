"""Circuit generation, validation, population, and export."""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import KDTree
from tqdm import tqdm

from tilke.curve_generator import CurveGeneratorConfig, generate_middle_curve
from tilke.perlin import pnoise1
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
# Helpers
# ---------------------------------------------------------------------------


def _rgba(r: int, g: int, b: int) -> tuple[float, float, float, float]:
    return (r / 255, g / 255, b / 255, 1.0)


def _fig2data(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


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
        mid = generate_middle_curve(curve_config) if middle_curve is None else middle_curve

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
    if orientation == "counter_clockwise" or (orientation == "random" and random.random() > 0.5):
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


def check_curvature(circuit: Circuit, restrictions: CircuitRestrictions) -> bool:
    """Check that all three curves satisfy minimum curvature radius constraints."""
    r = restrictions
    curves = [
        (
            circuit.middle_curve,
            r.min_curvature_radius_middle_curve,
            r.stepsize_middle_curvature_checking,
        ),
        (
            circuit.interior_curve,
            r.min_curvature_radius_interior_curve,
            r.stepsize_interior_curvature_checking,
        ),
        (
            circuit.exterior_curve,
            r.min_curvature_radius_exterior_curve,
            r.stepsize_exterior_curvature_checking,
        ),
    ]
    for curve, min_radius, stepsize in curves:
        steps = np.linspace(curve.t[0], curve.t[-1], math.floor(curve.t[-1] / stepsize))
        d1 = evaluate(curve, steps, der=1)  # (N, 2)
        d2 = evaluate(curve, steps, der=2)  # (N, 2)
        speed_cubed = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
        cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        radii = speed_cubed / cross
        if np.any(radii < min_radius):
            return False
    return True


def check_distances(circuit: Circuit, restrictions: CircuitRestrictions) -> bool:
    """Check that curves maintain minimum distances from each other."""
    mid_steps = np.linspace(
        circuit.middle_curve.t[0],
        circuit.middle_curve.t[-1],
        math.floor(circuit.middle_curve.t[-1] / restrictions.stepsize_middle_distance_checking),
    )
    mid_pts = evaluate(circuit.middle_curve, mid_steps)

    int_steps = np.linspace(
        circuit.interior_curve.t[0],
        circuit.interior_curve.t[-1],
        math.floor(circuit.interior_curve.t[-1] / restrictions.stepsize_interior_distance_checking),
    )
    int_pts = evaluate(circuit.interior_curve, int_steps)

    ext_steps = np.linspace(
        circuit.exterior_curve.t[0],
        circuit.exterior_curve.t[-1],
        math.floor(circuit.exterior_curve.t[-1] / restrictions.stepsize_exterior_distance_checking),
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


def _populate_naive(circuit: Circuit, n_cones: int) -> tuple[np.ndarray, np.ndarray]:
    upscale = circuit.exterior_curve.t[-1] / circuit.interior_curve.t[-1]
    n_interior = n_cones // 2
    int_positions = np.linspace(0, circuit.interior_curve.t[-1], n_interior)
    int_cones = evaluate(circuit.interior_curve, int_positions)
    ext_cones = evaluate(circuit.exterior_curve, int_positions * upscale)
    return int_cones, ext_cones


def _apply_perlin_offsets(positions: np.ndarray, max_t: float) -> np.ndarray:
    """Apply Perlin noise offsets to cone positions, truncating at max_t."""
    offsets = pnoise1(positions)
    positions = np.clip(positions + offsets, positions[0], max_t)
    # Truncate at first position that hits max_t
    over = np.where(positions >= max_t)[0]
    if len(over) > 0:
        positions = positions[: over[0] + 1]
        positions[-1] = max_t
    return positions


def _populate_perlin(circuit: Circuit, n_cones: int) -> tuple[np.ndarray, np.ndarray]:
    upscale = circuit.exterior_curve.t[-1] / circuit.interior_curve.t[-1]
    n_interior = n_cones // 2
    positions = np.linspace(0, circuit.interior_curve.t[-1], n_interior)
    positions = _apply_perlin_offsets(positions, circuit.interior_curve.t[-1])
    int_cones = evaluate(circuit.interior_curve, positions)
    ext_cones = evaluate(circuit.exterior_curve, positions * upscale)
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

    int_positions = _apply_perlin_offsets(int_positions, circuit.interior_curve.t[-1])
    ext_positions = _apply_perlin_offsets(ext_positions, circuit.exterior_curve.t[-1])

    int_cones = evaluate(circuit.interior_curve, int_positions)
    ext_cones = evaluate(circuit.exterior_curve, ext_positions)
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
    all_cones = circuit.exterior_cones if len(circuit.exterior_cones) > 0 else np.empty((0, 2))
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
    max_x += random.uniform(0, 10)
    max_y += random.uniform(0, 10)
    # Determine how many false cones to generate (binomial draw)
    n_false = np.random.binomial(total_cone_count, false_positive_prob)
    if n_false == 0:
        return np.empty((0, 2))
    return np.column_stack(
        [
            np.random.uniform(min_x, max_x, n_false),
            np.random.uniform(min_y, max_y, n_false),
        ]
    )


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
        plt.scatter(circuit.interior_cones[:, 0], circuit.interior_cones[:, 1], c="blue", s=10)
    if len(circuit.exterior_cones) > 0:
        plt.scatter(circuit.exterior_cones[:, 0], circuit.exterior_cones[:, 1], c="orange", s=10)
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
        "middle": _rgba(0, 255, 255),
        "interior": _rgba(255, 0, 0),
        "exterior": _rgba(0, 255, 0),
    }
    label_colors = {
        "middle": _rgba(0, 0, 0),
        "interior": _rgba(255, 255, 255),
        "exterior": _rgba(255, 255, 255),
    }
    zone_colors = {
        "inside": _rgba(255, 255, 0),
        "track": _rgba(0, 0, 255),
    }
    label_zone_colors = {
        "inside": _rgba(255, 255, 255),
        "track": _rgba(0, 0, 0),
    }
    cone_color = _rgba(0, 0, 0)

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
    int_xy = np.column_stack((curves_data["interior"][:, 0], curves_data["interior"][:, 1]))
    ext_xy = np.column_stack((curves_data["exterior"][:, 0], curves_data["exterior"][:, 1]))
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

    im = _fig2data(fig)
    model_label = _fig2data(fig2)
    model_input = _fig2data(fig3)
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
            for title, curve in zip(titles, curves, strict=True):
                writer.writerow([title])
                for i in range(curve.data.shape[1]):
                    writer.writerow(
                        [round(float(curve.data[0, i]), 4), round(float(curve.data[1, i]), 4)]
                    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


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
