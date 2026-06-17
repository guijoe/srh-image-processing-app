import numpy as np
from modules.ash.filtering_utils import manual_pad
def apply_oil_painting(image, radius=2):
    img_float = image.astype(float)
    h, w = img_float.shape[:2]
    if img_float.ndim != 3 or img_float.shape[2] < 3:
        return image
    padded = manual_pad(img_float, radius, radius)
    result = np.zeros_like(img_float)
    gray = 0.299 * padded[:,:,0] + 0.587 * padded[:,:,1] + 0.114 * padded[:,:,2]
    for i in range(h):
        for j in range(w):
            y = i + radius
            x = j + radius
            regions = [
                (y - radius, y + 1, x - radius, x + 1),           
                (y - radius, y + 1, x, x + radius + 1),            
                (y, y + radius + 1, x - radius, x + 1),              
                (y, y + radius + 1, x, x + radius + 1)                
            ]
            min_variance = float('inf')
            best_color = [0, 0, 0]
            for (y1, y2, x1, x2) in regions:
                region_gray = gray[y1:y2, x1:x2]
                variance = np.var(region_gray)
                if variance < min_variance:
                    min_variance = variance
                    best_color[0] = np.mean(padded[y1:y2, x1:x2, 0])
                    best_color[1] = np.mean(padded[y1:y2, x1:x2, 1])
                    best_color[2] = np.mean(padded[y1:y2, x1:x2, 2])
            result[i, j, 0] = best_color[0]
            result[i, j, 1] = best_color[1]
            result[i, j, 2] = best_color[2]
    if img_float.shape[2] == 4:
        result[:,:,3] = img_float[:,:,3]
    return result
