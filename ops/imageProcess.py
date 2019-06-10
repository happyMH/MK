import numpy as np
import math

from PIL import Image

def imageMerge(images):
    num = images.shape[0]
    size = int(math.sqrt(num))

    image = np.zeros((size * images.shape[1], size * images.shape[2], images.shape[3]),dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / size)
        j = index % size
        image[i * images.shape[1]:(i + 1) * images.shape[1], j * images.shape[2]:(j + 1) * images.shape[2], :] = img
    return image


def UndoNormalizedImg(normalizedImg):
    image = normalizedImg * 127.5 + 127.5
    return Image.fromarray(image.astype(np.uint8))
