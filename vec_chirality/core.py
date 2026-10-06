import numpy as np


def _coords(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 4 or not np.isfinite(x).all(): raise ValueError("coords must be finite n x 3")
    return x


def weighted_centroid(coords, weights):
    coords = _coords(coords); weights = np.asarray(weights, float)
    if weights.shape != (len(coords),) or np.any(weights < 0) or weights.sum() <= 0: raise ValueError("invalid weights")
    return (coords * weights[:, None]).sum(0) / weights.sum()


def signed_tetrahedron(a, b, c, d):
    a, b, c, d = map(lambda x: np.asarray(x, float), (a, b, c, d))
    return float(np.dot(b-a, np.cross(c-a, d-a)) / 6.0)


def chirality_signature(coords, landmark_weights):
    coords = _coords(coords); weights = np.asarray(landmark_weights, float)
    if weights.ndim != 2 or weights.shape[1] != len(coords) or weights.shape[0] < 4: raise ValueError("need at least four landmark weight rows")
    centers = np.array([weighted_centroid(coords, row) for row in weights])
    scale = np.sqrt(np.mean(np.sum((coords - coords.mean(0))**2, axis=1)))
    values = []
    for i in range(1, len(centers)-2):
        values.append(signed_tetrahedron(centers[0], centers[i], centers[i+1], centers[i+2]) / max(scale**3, 1e-12))
    return np.asarray(values)


def mirror(coords, axis=0):
    result = _coords(coords).copy(); result[:, axis] *= -1
    return result


def chirality_agreement(reference, prediction):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape: raise ValueError("signature shape mismatch")
    valid = (np.abs(reference) > 1e-8) & (np.abs(prediction) > 1e-8)
    return float(np.mean(np.sign(reference[valid]) == np.sign(prediction[valid]))) if valid.any() else float("nan")
