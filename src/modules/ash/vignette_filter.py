import numpy as np
def apply_vignette_filter(image, strength=1.0):
    img_float = image.astype(float)
    h = image.shape[0]
    w = image.shape[1]
    center_y = h / 2.0
    center_x = w / 2.0
    max_dist = np.sqrt(center_y**2 + center_x**2)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist_y = y_coords - center_y
    dist_x = x_coords - center_x
    distance = np.sqrt(dist_y**2 + dist_x**2)
    ratio = distance / max_dist
    darkness = np.power(ratio, strength * 2.0)
    mask = 1.0 - darkness
    mask = np.clip(mask, 0.0, 1.0)
    if img_float.ndim == 3:
        channels = img_float.shape[2]
        result = np.zeros_like(img_float)
        for c in range(min(3, channels)):
            result[:, :, c] = img_float[:, :, c] * mask
        if channels == 4:
            result[:, :, 3] = img_float[:, :, 3] 
        return result
    else:
        return img_float * mask
