import cv2
import numpy as np


def collector(count):
    img = np.zeros((100, 100), dtype=np.float)
    for i in range(count):
        img[i % 100, 0] = img[i % 100, 0] + 1
    return img[0,0]
