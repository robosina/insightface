from ctypes import *
import numpy as np
import pickle
import time
import mxnet as mx
from mxnet import ndarray as nd
objects = []
with (open("CYTHON_FACE_SUITE_TEST.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
objects = objects[0]
data = objects[0]
_feat_stride_fpn = objects[1]
use_landmarks = objects[2]
net_outs = objects[3]
_num_anchors = objects[4]
_anchors_fpn = objects[5]
threshold = objects[6]
decay4 = objects[7]
im_scale = objects[8]
vote = objects[9]
final_dets_list = objects[10]
final_landmarks_list = objects[11]
image_info = objects[12]


lib = cdll.LoadLibrary('/home/isv/Documents/insightface/fpnlib/cmake-build-debug/libfpnlib.so')
fun = lib.getdata
fun(c_int(3),
    c_void_p(np.array(_feat_stride_fpn,dtype=np.intc).ctypes.data),
    len(_feat_stride_fpn),
    c_bool(use_landmarks))