import numpy as np
def manual_pad(image, pad_height, pad_width):
    if image.ndim == 2:
        h, w = image.shape
        new_h = h + 2 * pad_height
        new_w = w + 2 * pad_width
        padded = np.zeros((new_h, new_w), dtype=image.dtype)
        padded[pad_height : pad_height + h, pad_width : pad_width + w] = image
        return padded
    elif image.ndim == 3:
        h, w, c = image.shape
        new_h = h + 2 * pad_height
        new_w = w + 2 * pad_width
        padded = np.zeros((new_h, new_w, c), dtype=image.dtype)
        padded[pad_height : pad_height + h, pad_width : pad_width + w, :] = image
        return padded
    return image
def manual_correlate(image, kernel):
    img_h = image.shape[0]
    img_w = image.shape[1]
    ker_h = kernel.shape[0]
    ker_w = kernel.shape[1]
    pad_h = ker_h // 2
    pad_w = ker_w // 2
    padded_img = manual_pad(image, pad_h, pad_w)
    output = np.zeros((img_h, img_w), dtype=float)
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i : i + ker_h, j : j + ker_w]
            multiplied = region * kernel
            output[i, j] = np.sum(multiplied)
    return output
def manual_convolve(image, kernel):
    flipped = np.flipud(kernel)
    flipped = np.fliplr(flipped)
    return manual_correlate(image, flipped)
