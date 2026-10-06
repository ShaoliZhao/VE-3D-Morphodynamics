"""Create two deliberately unregistered synthetic developmental snapshots."""
from pathlib import Path

import numpy as np


def main():
    output = Path(__file__).resolve().parent / "synthetic_pair.npz"
    rng = np.random.default_rng(11)
    coords_a = rng.normal(size=(120, 3)) * np.array([1.0, 0.55, 0.35])
    angle = 1.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    coords_b = (rng.normal(size=(150, 3)) * np.array([1.45, 0.75, 0.42])) @ rotation
    coords_b += np.array([4.0, -2.0, 1.0])
    expr_a = np.column_stack([np.exp(-coords_a[:, 0] ** 2), coords_a[:, 1] ** 2, coords_a[:, 2] + 1])
    local_b = (coords_b - coords_b.mean(0)) @ rotation.T
    expr_b = np.column_stack([np.exp(-local_b[:, 0] ** 2), local_b[:, 1] ** 2, local_b[:, 2] + 1.5])
    expr_a, expr_b = np.maximum(expr_a, 0), np.maximum(expr_b, 0)
    np.savez(output, coords_a=coords_a, coords_b=coords_b, expr_a=expr_a, expr_b=expr_b)
    print(output)


if __name__ == "__main__":
    main()
