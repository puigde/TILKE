"""Batch generation utilities for circuit datasets."""

from __future__ import annotations

import os
from datetime import datetime

from tqdm import tqdm

from tilke.circuit import (
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
