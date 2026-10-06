import unittest

import numpy as np

from vec_morph.core import (
    forecast_log_scale,
    interpolate_population,
    proper_rotation,
    relational_features,
    sinkhorn,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.coords = rng.normal(size=(24, 3)) * np.array([2.0, 1.0, 0.5])

    def test_relational_features_ignore_rigid_motion_and_scale(self):
        angle = 0.8
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        transformed = 3.4 * self.coords @ rotation + np.array([9.0, -4.0, 2.0])
        np.testing.assert_allclose(
            relational_features(self.coords, k=6),
            relational_features(transformed, k=6),
            atol=1e-7,
        )

    def test_proper_rotation_has_positive_determinant(self):
        target = self.coords.copy()
        target[:, 0] *= -1
        rotation = proper_rotation(target, self.coords)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)


class TransportTests(unittest.TestCase):
    def test_sinkhorn_has_uniform_marginals(self):
        cost = np.arange(15, dtype=float).reshape(3, 5) / 10
        coupling = sinkhorn(cost, regularization=0.5, iterations=2_000)
        np.testing.assert_allclose(coupling.sum(axis=1), np.full(3, 1 / 3), atol=1e-7)
        np.testing.assert_allclose(coupling.sum(axis=0), np.full(5, 1 / 5), atol=1e-7)

    def test_interpolation_endpoints_and_scale(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(12, 3))
        b = 2.5 * rng.normal(size=(18, 3)) + 3
        xa = np.abs(rng.normal(size=(12, 4)))
        xb = np.abs(rng.normal(size=(18, 4)))
        start = interpolate_population(a, b, xa, xb, alpha=0)
        middle = interpolate_population(a, b, xa, xb, alpha=0.5)
        np.testing.assert_allclose(start["coords"], a, atol=1e-7)
        np.testing.assert_allclose(start["expr"], xa, atol=1e-7)
        self.assertAlmostEqual(
            middle["predicted_scale"],
            np.sqrt(middle["scale_a"] * middle["scale_b"]),
            places=9,
        )

    def test_log_scale_forecast_recovers_exponential_growth(self):
        times = np.array([8.0, 8.5, 9.0, 9.5])
        scales = 1.2 * np.exp(0.35 * times)
        self.assertAlmostEqual(forecast_log_scale(times, scales, 10.0), 1.2 * np.exp(3.5), places=8)


if __name__ == "__main__":
    unittest.main()
