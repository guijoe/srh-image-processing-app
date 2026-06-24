import numpy as np
def apply_vintage_filter(image, intensity=1.0):
    img_float = image.astype(float)
    if img_float.ndim == 3 and (img_float.shape[2] == 3 or img_float.shape[2] == 4):
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]
        new_r = (r * 0.393) + (g * 0.769) + (b * 0.189)
        new_g = (r * 0.349) + (g * 0.686) + (b * 0.168)
        new_b = (r * 0.272) + (g * 0.534) + (b * 0.131)
        final_r = (new_r * intensity) + (r * (1.0 - intensity))
        final_g = (new_g * intensity) + (g * (1.0 - intensity))
        final_b = (new_b * intensity) + (b * (1.0 - intensity))
        max_val = np.max(img_float) if np.max(img_float) > 1.0 else 1.0
        final_r = np.clip(final_r, 0, max_val)
        final_g = np.clip(final_g, 0, max_val)
        final_b = np.clip(final_b, 0, max_val)
        if img_float.shape[2] == 4:
            alpha = img_float[:, :, 3]
            return np.stack([final_r, final_g, final_b, alpha], axis=-1)
        else:
            return np.stack([final_r, final_g, final_b], axis=-1)
    else:
        return img_float
