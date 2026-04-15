# TILKE Phase 2 — Vectorization + Noise Replacement

**Date:** 2026-04-15
**Scope:** Replace `noise` C dependency with pure-numpy Perlin noise, vectorize hot loops

---

## 1. Replace `noise` dependency with `src/tilke/perlin.py`

Pure-numpy 1D Perlin noise (~40 lines). Algorithm from the C `noise` library:
- Ken Perlin's 256-entry permutation table (doubled to 512)
- 1D gradient: `grad1(hash, x) = ((hash & 7) + 1) * x` or `-x` if bit 3 set
- Quintic fade: `6t^5 - 15t^4 + 10t^3`
- Linear interpolation + 0.4 scale factor
- fBm loop for multi-octave (our code only uses 1 octave)

Key advantage: accepts numpy arrays, so inherently vectorized.

Validate against C `noise.pnoise1` for 1000 random values before removing dependency.

## 2. Vectorize hot loops

| Location | Current | Fix |
|----------|---------|-----|
| `spline.py:get_int_ext_splines` | Per-step loop calling evaluate twice per step | Single vectorized evaluate call + array normal computation |
| `circuit.py:check_curvature` | List comprehension computing per-point curvature | Vectorized cross product + norm computation |
| `circuit.py:_populate_naive` | Per-cone evaluate loop | Single `evaluate(curve, all_positions)` |
| `circuit.py:_populate_perlin/random` | Scalar pnoise1 in loop + per-cone evaluate | Vectorized pnoise1 + single evaluate |
| `circuit.py:_generate_false_cones` | Per-cone random loop | `np.random.uniform` batch |

Not vectorized: `_apply_false_negatives` (sequential dependency on prev_removed).

## 3. Testing

- `tests/test_perlin.py`: validate numpy pnoise1 against C noise.pnoise1 (float32 tolerance)
- Existing 16 tests as regression guard
- Remove `noise` from pyproject.toml dependencies after validation

## 4. Files

| File | Change |
|------|--------|
| `src/tilke/perlin.py` | New — pure numpy pnoise1 |
| `src/tilke/spline.py` | Vectorize get_int_ext_splines |
| `src/tilke/circuit.py` | Vectorize + replace noise import |
| `tests/test_perlin.py` | New — validate against C library |
| `pyproject.toml` | Remove noise dependency |
