import numpy as np
def apply_gamma_transformation(image, gamma):
    img_float = image.astype(float)
    max_pixel = np.max(img_float)
    if max_pixel > 0:
        normalized = img_float / max_pixel
        corrected = np.power(normalized, gamma)
        result = corrected * max_pixel
        return result
    return img_float
