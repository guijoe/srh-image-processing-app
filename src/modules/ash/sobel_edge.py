import numpy as np
from modules.ash.filtering_utils import manual_convolve
def apply_sobel_edge_detection(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    if image.ndim == 3 and (image.shape[2] == 3 or image.shape[2] == 4):
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        gray = image.astype(float)
    edges_x = manual_convolve(gray, Kx)
    edges_y = manual_convolve(gray, Ky)
    magnitude = np.sqrt(edges_x**2 + edges_y**2)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
    return magnitude
