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
        c1 = generate_circuit(seed=1337, orientation="clockwise")
        c2 = generate_circuit(seed=1337, orientation="clockwise")
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
        c = generate_circuit(orientation="clockwise")
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
