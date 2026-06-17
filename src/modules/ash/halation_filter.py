import numpy as np
from modules.ash.gaussian_blur import apply_gaussian_blur
def apply_halation_filter(image, threshold=0.7, intensity=0.8):
    img_float = image.astype(float)
    max_val = np.max(img_float) if np.max(img_float) > 0 else 1.0
    base = img_float / max_val
    highlights = np.maximum(base - threshold, 0.0)
    high_max = np.max(highlights)
    if high_max > 0:
        highlights = highlights / high_max
    if highlights.ndim == 3 and (highlights.shape[2] == 3 or highlights.shape[2] == 4):
        highlights[:, :, 0] *= 1.0                    
        highlights[:, :, 1] *= 0.4                                                    
        highlights[:, :, 2] *= 0.0                         
    blurred_highlights = apply_gaussian_blur(highlights * max_val, 3.0)
    result = img_float + (blurred_highlights * intensity)
    return np.clip(result, 0, max_val)
