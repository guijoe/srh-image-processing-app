import numpy as np
def apply_fisheye_filter(image, distortion=-0.3):
    h = image.shape[0]
    w = image.shape[1]
    center_y = h / 2.0
    center_x = w / 2.0
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    y_norm = (y_coords - center_y) / center_y
    x_norm = (x_coords - center_x) / center_x
    radius = np.sqrt(x_norm**2 + y_norm**2)
    new_radius = radius * (1.0 + distortion * radius**2)
    x_curved = x_norm * (new_radius / (radius + 0.00001)) * center_x + center_x
    y_curved = y_norm * (new_radius / (radius + 0.00001)) * center_y + center_y
    x_int = np.clip(x_curved.astype(int), 0, w - 1)
    y_int = np.clip(y_curved.astype(int), 0, h - 1)
    if image.ndim == 3:
        channels = image.shape[2]
        result = np.zeros_like(image)
        for c in range(channels):
            result[:, :, c] = image[:, :, c][y_int, x_int]
        return result
    else:
        return image[y_int, x_int]
