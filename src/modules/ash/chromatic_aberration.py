import numpy as np
def apply_chromatic_aberration(image, shift_amount=5):
    img_float = image.astype(float)
    shift = int(shift_amount)
    if shift == 0 or img_float.ndim != 3 or img_float.shape[2] < 3:
        return image 
    h = image.shape[0]
    w = image.shape[1]
    channels = image.shape[2]
    result = np.zeros_like(img_float)
    r_channel = img_float[:, :, 0]
    new_r = np.zeros((h, w), dtype=float)
    if shift < w:
        new_r[:, :-shift] = r_channel[:, shift:]
    result[:, :, 0] = new_r
    result[:, :, 1] = img_float[:, :, 1]
    b_channel = img_float[:, :, 2]
    new_b = np.zeros((h, w), dtype=float)
    if shift < w:
        new_b[:, shift:] = b_channel[:, :-shift]
    result[:, :, 2] = new_b
    if channels == 4:
        result[:, :, 3] = img_float[:, :, 3]
    return result
