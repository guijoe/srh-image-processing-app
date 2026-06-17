import numpy as np
from modules.ash.gaussian_blur import apply_gaussian_blur
def apply_glow_filter(image, intensity=0.6, blur_amount=2.0):
    img_float = image.astype(float)
    max_val = np.max(img_float) if np.max(img_float) > 0 else 1.0
    base_image = img_float / max_val
    blurred_glow = apply_gaussian_blur(base_image, blur_amount)
    blurred_glow = blurred_glow * intensity
    screen_blend = 1.0 - (1.0 - base_image) * (1.0 - blurred_glow)
    result = screen_blend * max_val
    return np.clip(result, 0, max_val)
