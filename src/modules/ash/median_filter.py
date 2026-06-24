import numpy as np
from modules.ash.filtering_utils import manual_pad
def apply_median_filter(image, size):
    def do_filter_2d(img_2d, s):
        temp_img = np.squeeze(img_2d)
        h, w = temp_img.shape
        pad = s // 2
        padded = manual_pad(temp_img, pad, pad)
        result = np.zeros((h, w), dtype=img_2d.dtype)
        for i in range(h):
            for j in range(w):
                square = padded[i : i + s, j : j + s]
                result[i, j] = np.median(square)
        return result
    if image.ndim == 3 and (image.shape[2] == 3 or image.shape[2] == 4):
        num_channels = image.shape[2]
        channel_results = []
        for c in range(num_channels):
            channel_data = image[:, :, c]
            filtered_channel = do_filter_2d(channel_data, size)
            channel_results.append(filtered_channel)
        return np.stack(channel_results, axis=-1)
    else:
        return do_filter_2d(image, size)
