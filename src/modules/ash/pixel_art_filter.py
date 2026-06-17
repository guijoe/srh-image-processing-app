import numpy as np
def apply_pixel_art(image, pixel_size=8, color_levels=4.0):
    img_float = image.astype(float)
    h = image.shape[0]
    w = image.shape[1]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    y_snap = (y_coords // pixel_size) * pixel_size
    x_snap = (x_coords // pixel_size) * pixel_size
    pixelated_img = img_float[y_snap, x_snap]
    max_val = np.max(pixelated_img) if np.max(pixelated_img) > 0 else 1.0
    normalized = pixelated_img / max_val
    multiplied = normalized * color_levels
    stepped = np.floor(multiplied)
    quantized = stepped / color_levels
    result = quantized * max_val
    return result
