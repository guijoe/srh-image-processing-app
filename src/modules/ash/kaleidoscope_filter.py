import numpy as np
def apply_kaleidoscope(image, segments=6.0):
    img_float = image.astype(float)
    h = image.shape[0]
    w = image.shape[1]
    center_y = h / 2.0
    center_x = w / 2.0
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    y_norm = y_coords - center_y
    x_norm = x_coords - center_x
    radius = np.sqrt(x_norm**2 + y_norm**2)
    angle = np.arctan2(y_norm, x_norm)
    slice_angle = np.pi / segments
    angle = np.abs((angle % (2.0 * slice_angle)) - slice_angle)
    new_x = radius * np.cos(angle) + center_x
    new_y = radius * np.sin(angle) + center_y
    x_int = np.clip(new_x.astype(int), 0, w - 1)
    y_int = np.clip(new_y.astype(int), 0, h - 1)
    return image[y_int, x_int]
