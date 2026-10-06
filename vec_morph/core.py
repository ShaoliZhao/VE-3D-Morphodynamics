"""Small, explicit geometry baselines for unregistered 3D cell populations."""
from __future__ import annotations

import numpy as np


def _matrix(value, columns=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"expected a 2D matrix{f' with {columns} columns' if columns else ''}")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("matrix must be nonempty and finite")
    return array


def rms_radius(coords) -> float:
    coords = _matrix(coords, 3)
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def normalize_shape(coords):
    coords = _matrix(coords, 3)
    center = coords.mean(axis=0)
    scale = rms_radius(coords)
    if scale <= 1e-12:
        raise ValueError("point cloud has zero spatial extent")
    return (coords - center) / scale, center, scale


def _pairwise_sq(a, b):
    return np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T, 0.0)


def relational_features(coords, k: int = 8):
    """Normalized radius and nearest-neighbor distance quantiles per cell."""
    shape, _, _ = normalize_shape(coords)
    n = len(shape)
    if n < 2:
        raise ValueError("at least two cells are required")
    distances = np.sqrt(_pairwise_sq(shape, shape))
    np.fill_diagonal(distances, np.inf)
    ordered = np.sort(distances, axis=1)
    k = max(1, min(int(k), n - 1))
    indices = np.unique(np.clip(np.array([0, k // 3, (2 * k) // 3, k - 1]), 0, k - 1))
    local = ordered[:, indices]
    radius = np.sqrt(np.sum(shape * shape, axis=1, keepdims=True))
    return np.concatenate([radius, local], axis=1)


def _standardize(matrix):
    matrix = _matrix(matrix)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


def fused_cost(coords_a, coords_b, expr_a, expr_b, expression_weight=0.5, geometry_weight=0.5, k=8):
    coords_a, coords_b = _matrix(coords_a, 3), _matrix(coords_b, 3)
    expr_a, expr_b = _matrix(expr_a), _matrix(expr_b)
    if len(coords_a) != len(expr_a) or len(coords_b) != len(expr_b):
        raise ValueError("coordinate and expression row counts must match")
    if expr_a.shape[1] != expr_b.shape[1]:
        raise ValueError("expression dimensions must match")
    if expression_weight < 0 or geometry_weight < 0 or expression_weight + geometry_weight <= 0:
        raise ValueError("cost weights must be nonnegative with a positive sum")
    combined = np.vstack([expr_a, expr_b])
    combined = _standardize(combined)
    ea, eb = combined[: len(expr_a)], combined[len(expr_a) :]
    ga, gb = relational_features(coords_a, k), relational_features(coords_b, k)
    expression = _pairwise_sq(ea, eb) / max(expr_a.shape[1], 1)
    geometry = _pairwise_sq(ga, gb) / max(ga.shape[1], 1)
    total = expression_weight + geometry_weight
    return (expression_weight * expression + geometry_weight * geometry) / total


def sinkhorn(cost, regularization=0.1, iterations=300, tolerance=1e-9):
    cost = _matrix(cost)
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    shifted = cost - cost.min()
    kernel = np.exp(-shifted / regularization)
    kernel = np.maximum(kernel, 1e-300)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(int(iterations)):
        previous = u.copy()
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if np.max(np.abs(u - previous)) < tolerance:
            break
    coupling = (u[:, None] * kernel) * v[None, :]
    coupling /= coupling.sum()
    return coupling


def proper_rotation(source, target):
    source, target = _matrix(source, 3), _matrix(target, 3)
    if source.shape != target.shape:
        raise ValueError("alignment matrices must have the same shape")
    covariance = (target - target.mean(0)).T @ (source - source.mean(0))
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def interpolate_population(coords_a, coords_b, expr_a, expr_b, alpha, coupling=None, **cost_kwargs):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    coords_a, coords_b = _matrix(coords_a, 3), _matrix(coords_b, 3)
    expr_a, expr_b = _matrix(expr_a), _matrix(expr_b)
    if coupling is None:
        coupling = sinkhorn(fused_cost(coords_a, coords_b, expr_a, expr_b, **cost_kwargs))
    coupling = _matrix(coupling)
    if coupling.shape != (len(coords_a), len(coords_b)):
        raise ValueError("coupling shape mismatch")
    row_mass = coupling.sum(axis=1, keepdims=True)
    conditional = coupling / np.maximum(row_mass, 1e-300)
    target_shape, _, scale_b = normalize_shape(coords_b)
    source_shape, center_a, scale_a = normalize_shape(coords_a)
    target_on_source = conditional @ target_shape
    rotation = proper_rotation(target_on_source, source_shape)
    target_on_source = (target_on_source - target_on_source.mean(0)) @ rotation
    interpolated_shape = (1 - alpha) * source_shape + alpha * target_on_source
    interpolated_shape -= interpolated_shape.mean(0)
    shape_scale = rms_radius(interpolated_shape)
    if shape_scale > 1e-12:
        interpolated_shape /= shape_scale
    physical_scale = float(np.exp((1 - alpha) * np.log(scale_a) + alpha * np.log(scale_b)))
    center_b = coords_b.mean(axis=0)
    center = (1 - alpha) * center_a + alpha * center_b
    coords = center + physical_scale * interpolated_shape
    target_expr = conditional @ expr_b
    expr = np.maximum((1 - alpha) * expr_a + alpha * target_expr, 0.0)
    return {
        "coords": coords,
        "expr": expr,
        "coupling": coupling,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "predicted_scale": physical_scale,
    }


def forecast_log_scale(times, scales, target_time):
    times, scales = np.asarray(times, float), np.asarray(scales, float)
    if times.ndim != 1 or scales.ndim != 1 or len(times) != len(scales) or len(times) < 2:
        raise ValueError("times and scales must be equal 1D arrays with at least two values")
    if not np.isfinite(times).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("times must be finite and scales finite positive")
    design = np.column_stack([np.ones_like(times), times])
    coefficients, *_ = np.linalg.lstsq(design, np.log(scales), rcond=None)
    return float(np.exp(np.array([1.0, float(target_time)]) @ coefficients))
