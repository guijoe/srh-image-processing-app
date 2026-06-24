import numpy as np
from modules.ash.filtering_utils import manual_convolve
def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * (ax**2) / (sigma**2))
    kernel = np.outer(gauss, gauss)
    total_sum = np.sum(kernel)
    return kernel / total_sum
def apply_gaussian_blur(image, sigma):
    size = int(6 * sigma)
    if size % 2 == 0:
        size = size + 1
    kernel = create_gaussian_kernel(size, sigma)
    img_float = image.astype(float)
    if img_float.ndim == 3 and (img_float.shape[2] == 3 or img_float.shape[2] == 4):
        num_channels = img_float.shape[2]
        channel_results = []
        for c in range(num_channels):
            channel_data = img_float[:, :, c]
            filtered_channel = manual_convolve(channel_data, kernel)
            channel_results.append(filtered_channel)
        return np.stack(channel_results, axis=-1)
    else:
        return manual_convolve(img_float, kernel)
