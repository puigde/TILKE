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
