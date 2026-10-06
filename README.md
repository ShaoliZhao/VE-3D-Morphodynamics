# VEC 3D Morphodynamics Lab

Reference baselines for Task 2 that model tissue geometry as geometry, rather than regressing absolute coordinates from embryos measured in unrelated frames.

The package provides four composable pieces:

1. rotation/translation-invariant relational descriptors;
2. entropic coupling based on expression and relational geometry;
3. interpolation of normalized shape with physical scale handled separately.
4. chirality-aware signatures that distinguish proper rotations from biological mirror reversal.

Inputs and outputs are NumPy arrays, so users retain control of their AnnData pipeline and the official board contract.

## Install and run

```bash
python3 -m pip install -e .
python3 examples/make_synthetic.py
vec-morph interpolate examples/synthetic_pair.npz examples/prediction.npz --alpha 0.5
python3 examples/mirror_demo.py
python3 -m unittest discover -s tests -v
```

The input archive contains `coords_a`, `coords_b`, `expr_a` and `expr_b`. The output contains `coords`, `expr`, `coupling` and diagnostic metadata. Coordinates remain in physical units through an independently interpolated RMS radius.

## Why these baselines

Task 2 scores expression change, cell-state distribution, tissue shape/scale and local spatial organisation. Absolute coordinate MSE is inappropriate because each embryo has its own frame. Copying or uniformly scaling the last point cloud preserves neither developmental shape change nor expression-neighborhood coupling.

The coupling uses a fused cost:

`cost = expression_weight * expression_distance + geometry_weight * relational_distance`

Relational descriptors contain normalized radius and local neighbor-distance quantiles. They survive rigid changes of frame. A Sinkhorn coupling creates a soft target correspondence, which is aligned to the source with a proper-rotation Kabsch transform before interpolation. Shape is interpolated after RMS normalization; scale is interpolated in log space.

## Commands

```bash
vec-morph describe cloud.npz
vec-morph interpolate endpoints.npz prediction.npz --alpha 0.5
vec-morph forecast-scale stages.npz --target-time 10.5
```

`describe` accepts an archive containing `coords`. `forecast-scale` expects `times` and `scales` and reports a low-capacity log-linear extrapolation.

## Scientific boundaries

This is a transparent baseline, not a full morphogenesis model. The soft coupling is inferred from endpoints and is not cell lineage. Barycentric transport can smooth multimodal states. Local distance descriptors cannot identify biological laterality. Extrapolating scale does not extrapolate shape. The method should therefore be compared with copy-last, expression-only coupling and shuffled expression-position controls.

See [method details](docs/method.md), [chirality](docs/chirality.md), [experiment plan](docs/experiments.md), [paper notes](references/reading.md) and the [Community Award draft](SUBMISSION_TEXT.md).

MIT licensed. No challenge data are included.
