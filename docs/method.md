# Method

Task 2 asks for an intermediate or future 3D cell population from unregistered observations. A baseline therefore has to separate three problems that are often conflated: correspondence, orientation, and physical growth.

## 1. Shape normalization

Each point cloud is centered and divided by its root-mean-square radius. Translation and global scale are stored, not discarded. This produces a dimensionless shape for correspondence and retains one interpretable growth variable.

## 2. Relation-aware descriptors

For every cell, the library computes its normalized distance to the population center and selected nearest-neighbor distances. These descriptors are unchanged by translation, rotation, and uniform scale. They provide geometric context without assuming that the two samples share a coordinate frame.

## 3. Fused matching cost

Expression features are standardized jointly across the two snapshots. Pairwise expression distance and pairwise descriptor distance are averaged with user-controlled nonnegative weights. Entropic optimal transport then produces a soft coupling with uniform marginals. The result supports unequal cell counts and avoids pretending that one cell has a known exact successor.

## 4. Alignment and interpolation

The later normalized shape is transported onto earlier cells using the conditional coupling. A proper Kabsch rotation aligns it to the earlier normalized shape. Linear interpolation is applied only after this alignment. The physical RMS radius is interpolated geometrically, while population centers are interpolated linearly.

## 5. Extrapolation baseline

For future size, `forecast_log_scale` fits a linear model to log RMS radius versus developmental time. It is deliberately low-capacity and can serve as a control for learned growth models. Full future shape extrapolation remains an open modeling step; this repository does not hide that uncertainty behind a visually plausible animation.

## Boundaries

This method does not recover lineage, cell division, tissue topology, or mechanics. It is a reproducible baseline and diagnostic scaffold. Stronger models should compare against its correspondence quality, invariance, scale prediction, and held-out interpolation error.
