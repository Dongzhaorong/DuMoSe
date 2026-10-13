"""Paper-described 3D flips, rotations and intensity gamma jitter."""
import random
import numpy as np
from scipy.ndimage import rotate

def augment_roi(x):
    for ax in range(3):
        if random.random() < 0.5:
            x = np.flip(x, axis=ax)
    if random.random() < 0.5:
        plane = random.choice([(0, 1), (0, 2), (1, 2)])
        angle = random.uniform(-180, 180)
        x = rotate(x, angle, axes=plane, reshape=False, order=1, mode="constant", cval=0.0)
    gamma = random.uniform(0.85, 1.15)
    x = np.clip(x, 0, 1) ** gamma
    return np.ascontiguousarray(x, dtype=np.float32)
