import numpy as np


def normalize_vector(vector, eps=1e-8):
    """Normalize a vector and return None for near-zero vectors."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return None
    return (vector / norm).astype(np.float32)
