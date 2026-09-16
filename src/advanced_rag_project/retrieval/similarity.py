import numpy as np

def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    return float(
        np.dot(a,b)
        / (np.linalg.norm(a) * np.linalg.norm(b))
    )
