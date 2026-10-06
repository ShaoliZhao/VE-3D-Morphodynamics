import unittest
import numpy as np
from vec_chirality.core import chirality_signature, mirror


class ChiralityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]], float)
        self.w = np.eye(5)
    def test_reflection_flips_sign(self):
        np.testing.assert_allclose(chirality_signature(self.x, self.w), -chirality_signature(mirror(self.x), self.w))
    def test_proper_rotation_preserves_sign(self):
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))
        if np.linalg.det(q) < 0: q[:, 0] *= -1
        transformed = 2.7 * self.x @ q + [4, -2, 8]
        np.testing.assert_allclose(chirality_signature(self.x, self.w), chirality_signature(transformed, self.w), atol=1e-10)


if __name__ == "__main__": unittest.main()
