"""Generate demo GIFs for the README."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tilke import generate_circuit, populate_cones, contaminate_cones, ConePopulationParameters
from tilke.spline import evaluate


MEDIA_DIR = Path(__file__).parent


def _circuit_bounds(circuit, pad: float = 5.0) -> tuple[float, float, float, float]:
    """Compute axis limits from the exterior curve with padding."""
    tt = np.linspace(circuit.exterior_curve.t[0], circuit.exterior_curve.t[-1], 500)
    pts = evaluate(circuit.exterior_curve, tt)
    xmin, ymin = pts.min(axis=0) - pad
    xmax, ymax = pts.max(axis=0) + pad
    # Make square
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2
    return cx - half, cx + half, cy - half, cy + half


def _plot_circuit(ax, circuit, show_middle: bool = False) -> None:
    """Plot a circuit on the given axes."""
    curves = [circuit.interior_curve, circuit.exterior_curve]
    if show_middle:
        curves.insert(0, circuit.middle_curve)
    for curve in curves:
        tt = np.linspace(curve.t[0], curve.t[-1], 1000)
        gamma = evaluate(curve, tt)
        ax.plot(gamma[:, 0], gamma[:, 1], color="#444", linewidth=1.2)

    if len(circuit.interior_cones) > 0:
        ax.scatter(circuit.interior_cones[:, 0], circuit.interior_cones[:, 1],
                   c="#2563eb", s=12, zorder=5)
    if len(circuit.exterior_cones) > 0:
        ax.scatter(circuit.exterior_cones[:, 0], circuit.exterior_cones[:, 1],
                   c="#f97316", s=12, zorder=5)
    if len(circuit.false_cones) > 0:
        ax.scatter(circuit.false_cones[:, 0], circuit.false_cones[:, 1],
                   c="#ef4444", marker="x", s=15, zorder=5)


def _make_frame(circuit, title: str | None = None, bounds=None,
                show_middle: bool = False) -> plt.Figure:
    """Create a single frame figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontfamily="monospace", pad=8)
    _plot_circuit(ax, circuit, show_middle=show_middle)
    if bounds:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    return fig


def _save_frame(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _frames_to_gif(frame_dir: Path, output: Path, fps: int = 2) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%03d.png"),
        "-vf", "palettegen",
        "-update", "1",
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
    """GIF showing 10 different random circuit layouts."""
    print("Generating circuit variety GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        seeds = [42, 123, 256, 404, 512, 777, 888, 1024, 1337, 2048]
        for i, seed in enumerate(seeds):
            c = generate_circuit(seed=seed)
            bounds = _circuit_bounds(c)
            fig = _make_frame(c, bounds=bounds)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/{len(seeds)}")
        _frames_to_gif(frame_dir, MEDIA_DIR / "circuits.gif", fps=2)
    print(f"  Saved: {MEDIA_DIR / 'circuits.gif'}")


def generate_quickstart_gif() -> None:
    """GIF showing a circuit being generated and populated."""
    print("Generating quick start GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=777)
        bounds = _circuit_bounds(c)
        fig = _make_frame(c, title="generate_circuit()", bounds=bounds)
        _save_frame(fig, frame_dir / "frame_000.png")
        c_cones = populate_cones(c, method="perlin")
        fig = _make_frame(c_cones, title="populate_cones(circuit)", bounds=bounds)
        _save_frame(fig, frame_dir / "frame_001.png")
        _frames_to_gif(frame_dir, MEDIA_DIR / "quickstart.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'quickstart.gif'}")


def generate_seed_gif() -> None:
    """GIF showing the same seed always produces the same circuit."""
    print("Generating seed GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        seeds = [42, 1337, 777, 256]
        for i, seed in enumerate(seeds):
            c = generate_circuit(seed=seed)
            c = populate_cones(c, method="perlin")
            bounds = _circuit_bounds(c)
            fig = _make_frame(c, title=f"seed={seed}", bounds=bounds)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/{len(seeds)}")
        _frames_to_gif(frame_dir, MEDIA_DIR / "seeds.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'seeds.gif'}")


def generate_cones_gif() -> None:
    """GIF comparing the three cone placement methods."""
    print("Generating cone methods GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=42)
        bounds = _circuit_bounds(c)
        methods = [
            ("populate_cones(c, method='naive')", "naive"),
            ("populate_cones(c, method='perlin')", "perlin"),
            ("populate_cones(c, method='random')", "random"),
        ]
        for i, (title, method) in enumerate(methods):
            c_pop = populate_cones(c, method=method)
            fig = _make_frame(c_pop, title=title, bounds=bounds)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/3 ({method})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "cones.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'cones.gif'}")


def generate_contamination_gif() -> None:
    """GIF showing clean cones vs soft contaminated detection."""
    print("Generating contamination GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=42)
        bounds = _circuit_bounds(c)
        c_clean = populate_cones(c, method="perlin")
        # Soft contamination for a subtle, realistic demo
        soft_params = ConePopulationParameters(
            false_negative_probability=0.05,
            false_positive_probability=0.03,
        )
        c_noisy = contaminate_cones(c_clean, soft_params)
        for i, (title, circuit) in enumerate([
            ("Clean cones", c_clean),
            ("Simulated detection", c_noisy),
        ]):
            fig = _make_frame(circuit, title=title, bounds=bounds)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/2 ({title})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "contamination.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'contamination.gif'}")


if __name__ == "__main__":
    generate_variety_gif()
    generate_quickstart_gif()
    generate_seed_gif()
    generate_cones_gif()
    generate_contamination_gif()
    print("Done!")
