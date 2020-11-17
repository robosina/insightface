from ctypes import *
import numpy as np
lib = cdll.LoadLibrary('/home/isv/qt_projects/dlib2/libdlib2.so')

fun = lib.getdata
_feat_stride_fpn = [32,16,8]
use_landmark=True
fun(c_int(3), c_void_p(np.array(_feat_stride_fpn,dtype=np.intc).ctypes.data), len(_feat_stride_fpn),
    c_bool(use_landmark))