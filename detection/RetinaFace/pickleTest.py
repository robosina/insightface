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

# print(objects)
t6 = time.time()
npNet_outs = [out.asnumpy() for out in net_outs]
t7 = time.time()
print("T7 (Convert from mxnet to numpy) => %.2f" % (t7 - t6))
