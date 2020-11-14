import cv2
import numpy
cimport numpy
cdef numpy.ndarray arr

cpdef collector(unsigned long long count):
    cdef unsigned long long i
    img = numpy.zeros((100, 100), dtype=numpy.float)
    for i in range(count):
        img[i % 100, 0] = img[i % 100, 0] + 1
    return img[0,0]
