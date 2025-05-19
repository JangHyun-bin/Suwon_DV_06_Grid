import numpy as np
from noise import pnoise2

def generate_perlin(h: int, w: int, scale: float) -> np.ndarray:
    n = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            n[y, x] = pnoise2(x * scale, y * scale)
    return (n + 1) * 0.5 