import numpy as np
from modules.ash.sobel_edge import apply_sobel_edge_detection
def apply_anime_filter(image, color_levels=5.0, edge_threshold=0.2):
    img_float = image.astype(float)
    max_val = np.max(img_float) if np.max(img_float) > 0 else 1.0
    normalized = img_float / max_val
    multiplied = normalized * color_levels
    stepped = np.floor(multiplied)
    quantized = stepped / color_levels
    cartoon_colors = quantized * max_val
    edges = apply_sobel_edge_detection(image)
    edge_mask = np.where(edges > edge_threshold, 0.0, 1.0)
    if img_float.ndim == 3:
        channels = img_float.shape[2]
        result = np.zeros_like(img_float)
        for c in range(min(3, channels)):
            result[:, :, c] = cartoon_colors[:, :, c] * edge_mask
        if channels == 4:
            result[:, :, 3] = img_float[:, :, 3]
        return result
    else:
        return cartoon_colors * edge_mask
