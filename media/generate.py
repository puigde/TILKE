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
    # Do NOT use bbox_inches="tight" — it changes output dimensions per frame,
    # which causes ffmpeg paletteuse to fail when frames differ in size.
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
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            ax.axis("off")
            plt.sca(ax)
            plot_circuit(c)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/{len(seeds)}")
        _frames_to_gif(frame_dir, MEDIA_DIR / "circuits.gif", fps=2)
    print(f"  Saved: {MEDIA_DIR / 'circuits.gif'}")


def generate_quickstart_gif() -> None:
    """GIF showing a single circuit being generated and populated."""
    print("Generating quick start GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=777)
        # Frame 1: empty circuit
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.axis("off")
        plt.sca(ax)
        plot_circuit(c)
        _save_frame(fig, frame_dir / "frame_000.png")
        # Frame 2: with cones
        c_cones = populate_cones(c, method="perlin")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.axis("off")
        plt.sca(ax)
        plot_circuit(c_cones)
        _save_frame(fig, frame_dir / "frame_001.png")
        _frames_to_gif(frame_dir, MEDIA_DIR / "quickstart.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'quickstart.gif'}")


def generate_cones_gif() -> None:
    """GIF comparing the three cone placement methods."""
    print("Generating cone methods GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=42)
        methods = [
            ("method='naive'", "naive"),
            ("method='perlin'", "perlin"),
            ("method='random'", "random"),
        ]
        for i, (title, method) in enumerate(methods):
            c_pop = populate_cones(c, method=method)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontfamily="monospace", pad=10)
            plt.sca(ax)
            plot_circuit(c_pop)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/3 ({method})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "cones.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'cones.gif'}")


def generate_contamination_gif() -> None:
    """GIF showing clean cones vs contaminated detection."""
    print("Generating contamination GIF...")
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)
        c = generate_circuit(seed=42)
        c_clean = populate_cones(c, method="perlin")
        c_noisy = contaminate_cones(c_clean)
        for i, (title, circuit) in enumerate([
            ("Clean cones", c_clean),
            ("Simulated detection", c_noisy),
        ]):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontfamily="monospace", pad=10)
            plt.sca(ax)
            plot_circuit(circuit)
            _save_frame(fig, frame_dir / f"frame_{i:03d}.png")
            print(f"  Frame {i + 1}/2 ({title})")
        _frames_to_gif(frame_dir, MEDIA_DIR / "contamination.gif", fps=1)
    print(f"  Saved: {MEDIA_DIR / 'contamination.gif'}")


if __name__ == "__main__":
    generate_variety_gif()
    generate_quickstart_gif()
    generate_cones_gif()
    generate_contamination_gif()
    print("Done!")
