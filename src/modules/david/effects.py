import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        return (
            0.2989 * image[:, :, 0] +
            0.5870 * image[:, :, 1] +
            0.1140 * image[:, :, 2]
        )
    return image.copy()


def normalize(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float64)
    max_val = img.max()

    if max_val > 1.0:
        img = img / max_val

    return img


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image.copy()


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)

    for ki in range(kh):
        for kj in range(kw):
            weight = kernel[ki, kj]

            if weight == 0:
                continue

            region = padded[ki: ki + h, kj: kj + w]
            output += weight * region

    return output


def apply_sepia(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    img = ensure_rgb(image)

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])

    sepia_img = np.einsum('ijk,lk->ijl', img[:, :, :3], sepia_matrix)
    sepia_img = np.clip(sepia_img, 0.0, 1.0)

    result = img[:, :, :3] * (1.0 - intensity) + sepia_img * intensity

    return np.clip(result, 0.0, 1.0)


def apply_channel_swap(image: np.ndarray, swap_mode: str = "R<->B") -> np.ndarray:
    img = ensure_rgb(image)
    result = img[:, :, :3].copy()

    if swap_mode == "R<->B":
        result[:, :, 0] = img[:, :, 2]
        result[:, :, 2] = img[:, :, 0]

    elif swap_mode == "R<->G":
        result[:, :, 0] = img[:, :, 1]
        result[:, :, 1] = img[:, :, 0]

    elif swap_mode == "G<->B":
        result[:, :, 1] = img[:, :, 2]
        result[:, :, 2] = img[:, :, 1]

    elif swap_mode == "Rotate":
        result[:, :, 0] = img[:, :, 2]
        result[:, :, 1] = img[:, :, 0]
        result[:, :, 2] = img[:, :, 1]

    return result


def apply_vignette(image: np.ndarray, strength: float = 0.8, radius: float = 1.0) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    y_coords = np.linspace(-1.0, 1.0, h)
    x_coords = np.linspace(-1.0, 1.0, w)

    xx, yy = np.meshgrid(x_coords, y_coords)

    distance = np.sqrt(xx ** 2 + yy ** 2)

    sigma = radius
    gaussian_mask = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))

    vignette_mask = 1.0 - strength * (1.0 - gaussian_mask)

    if img.ndim == 3:
        img = img * vignette_mask[:, :, np.newaxis]
    else:
        img = img * vignette_mask

    return np.clip(img, 0.0, 1.0)


def apply_negative(image: np.ndarray) -> np.ndarray:
    return 1.0 - image


def apply_emboss(image: np.ndarray, direction: str = "Top-Left", strength: float = 1.0) -> np.ndarray:
    kernels = {
        "Top-Left": np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2],
        ], dtype=np.float64),

        "Top-Right": np.array([
            [ 0, -1, -2],
            [ 1,  1, -1],
            [ 2,  1,  0],
        ], dtype=np.float64),

        "Bottom-Left": np.array([
            [ 0,  1,  2],
            [-1,  1,  1],
            [-2, -1,  0],
        ], dtype=np.float64),

        "Bottom-Right": np.array([
            [ 2,  1,  0],
            [ 1,  1, -1],
            [ 0, -1, -2],
        ], dtype=np.float64),
    }

    kernel = kernels.get(direction, kernels["Top-Left"])
    kernel = kernel * strength

    gray = to_grayscale(normalize(image))

    embossed = convolve2d(gray, kernel)
    embossed = embossed + 0.5

    result = np.clip(embossed, 0.0, 1.0)

    if image.ndim == 3 and image.shape[2] >= 3:
        return np.stack([result, result, result], axis=-1)

    return result


def apply_polar_transform(image: np.ndarray, direction: str = "To Polar") -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    is_color = (img.ndim == 3)

    cx, cy = w / 2.0, h / 2.0
    max_radius = np.sqrt(cx ** 2 + cy ** 2)

    out_rows, out_cols = np.mgrid[0:h, 0:w]

    if direction == "To Polar":
        radius = (out_rows / h) * max_radius
        angle  = (out_cols / w) * 2.0 * np.pi

        src_cols = cx + radius * np.cos(angle)
        src_rows = cy + radius * np.sin(angle)

    else:
        dx = out_cols - cx
        dy = out_rows - cy

        radius = np.sqrt(dx ** 2 + dy ** 2)
        angle  = np.arctan2(dy, dx)

        src_cols = ((angle + np.pi) / (2.0 * np.pi)) * w
        src_rows = (radius / max_radius) * h

    src_col_idx = np.clip(src_cols.astype(np.int32), 0, w - 1)
    src_row_idx = np.clip(src_rows.astype(np.int32), 0, h - 1)

    if is_color:
        output = img[src_row_idx, src_col_idx, :]
    else:
        output = img[src_row_idx, src_col_idx]

    return np.clip(output, 0.0, 1.0)


def apply_glitch(image: np.ndarray, intensity: float = 0.5, seed: int = 42) -> np.ndarray:
    img = ensure_rgb(image)
    result = img[:, :, :3].copy()
    h, w = result.shape[:2]

    rng = np.random.default_rng(seed)

    num_shifted_rows = int(h * intensity * 0.3)
    rows_to_shift = rng.integers(0, h, size=num_shifted_rows)

    for row in rows_to_shift:
        max_shift = max(1, int(w * intensity * 0.4))
        shift = rng.integers(-max_shift, max_shift + 1)
        result[row] = np.roll(result[row], shift, axis=0)

    channel_shift = int(w * intensity * 0.05)

    if channel_shift > 0:
        result[:, :, 0] = np.roll(result[:, :, 0], channel_shift, axis=1)
        result[:, :, 2] = np.roll(result[:, :, 2], -channel_shift, axis=1)

    num_blocks = int(5 * intensity) + 1

    for _ in range(num_blocks):
        block_h = rng.integers(int(h * 0.05), max(int(h * 0.20), 2))
        block_w = rng.integers(int(w * 0.10), max(int(w * 0.40), 2))

        src_y = rng.integers(0, max(h - block_h, 1))
        src_x = rng.integers(0, max(w - block_w, 1))

        dst_y = rng.integers(0, max(h - block_h, 1))
        dst_x = rng.integers(0, max(w - block_w, 1))

        result[dst_y : dst_y + block_h, dst_x : dst_x + block_w] = (
            result[src_y : src_y + block_h, src_x : src_x + block_w]
        )

    return np.clip(result, 0.0, 1.0)
