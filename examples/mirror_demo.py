import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vec_chirality.core import chirality_signature, mirror

coords = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]], float)
weights = np.eye(5)
original = chirality_signature(coords, weights)
reflected = chirality_signature(mirror(coords), weights)
print(json.dumps({"original": original.tolist(), "mirror": reflected.tolist(), "sign_flipped": bool(np.allclose(original, -reflected))}, indent=2))
