# %%cylandmarks_list=None --link-args=-fopenmp
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
import numpy as np
cimport numpy as np
import math
cimport cython
# tag: run
# tag: openmp
from libcpp.string cimport string
from libcpp cimport bool
from cython.parallel import prange,threadid
from libcpp.vector cimport vector
# @cython.boundscheck(False)
cdef float get_scores(float[:,::1] a) nogil:
  # Function abbreviated for simplicity
  return 0.0
cpdef postprocessing(self, data, list net_outs, image_info,int data_shape0,feat_stride_fpn,bool use_landmark,
                     dict _num_anchors,dict _anchors_fpn,
                     threshold, im_scale, final_dets_list, final_landmarks_list):
    cpdef int _idx, s, stride,feat_size=len(feat_stride_fpn),i,A,K
    cpdef int idx,output_index
    cdef float [:,:,:,:] n_scores,n_box_deltas
    cdef string stride_str="stride",_key,which_stride
    cpdef list proposals_list
    cpdef list scores_list
    cpdef list landmarks_list

    # for output_index in prange(data_shape0, nogil=True):
        # STRIDE='stride'
    for output_index in range(data_shape0):
        for _idx in range(feat_size):
        # for _idx, s in enumerate(feat_stride_fpn):
            _key = stride_str + str(feat_stride_fpn[_idx])
            stride = s
            if use_landmark:
                idx = _idx * 3
            else:
                idx = _idx * 2
            # print(len(net_outs))
            scores = net_outs[idx][output_index]  # MY CHANGE
        #     # print(scores.shape)
            n_scores = scores.reshape((1, *scores.shape))
            print(stride_str)
            which_stride= _num_anchors[stride_str]
            n_scores = n_scores[:, which_stride:, :, :] # TODO : I assume all stride in each fpn level is 2
            idx += 1
            bbox_deltas = net_outs[idx][output_index]  # MY CHANGE
            n_bbox_deltas = bbox_deltas.reshape((1, *bbox_deltas.shape))

            height, width = n_bbox_deltas.shape[2], n_bbox_deltas.shape[3]
            A = _num_anchors[which_stride] # TODO : I assume all stride in each fpn level is 2
            K = height * width
            anchors_fpn = _anchors_fpn[stride_str]
        #     anchors = anchors_plane(height, width, stride, anchors_fpn)
        #     anchors = anchors.reshape((K * A, 4))
        #
        #     scores = self._clip_pad(scores, (height, width))
        #     scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        #
        #     bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
        #     bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
        #     bbox_pred_len = bbox_deltas.shape[3] // A
        #     bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        #
        #     proposals = self.bbox_pred(anchors, bbox_deltas)
        #     proposals = clip_boxes(proposals, image_info[:2])
        #
        #     scores_ravel = scores.ravel()  # We Ravel The Scores For All Of The BBoxes
        #     order = np.where(scores_ravel >= threshold)[0]  # We Only Pick The Best Scores By Threshold
        #     proposals = proposals[order, :]
        #     scores = scores[order]
        #     if stride == 4 and self.decay4 < 1.0:
        #         scores *= self.decay4
        #
        #     proposals[:, 0:4] /= im_scale
        #
        #     proposals_list.append(proposals)
        #     scores_list.append(scores)
        #
        #     if not self.vote and self.use_landmarks:
        #         idx += 1
        #         landmark_deltas = net_outs[idx][output_index].asnumpy()
        #         landmark_deltas = landmark_deltas.reshape((1, *landmark_deltas.shape))  # MY CHANGE
        #
        #         landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
        #         landmark_pred_len = landmark_deltas.shape[1] // A
        #         landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
        #         landmarks = self.landmark_pred(anchors, landmark_deltas)
        #         landmarks = landmarks[order, :]
        #
        #         landmarks[:, :, 0:2] /= im_scale
        #         landmarks_list.append(landmarks)
        #
        # proposals = np.vstack(proposals_list)
        # landmarks = None
        # if proposals.shape[0] == 0:
        #     if self.use_landmarks:
        #         landmarks = np.zeros((0, 5, 2))
        #     final_dets_list.append(np.zeros((0, 5)))
        #     final_landmarks_list.append(landmarks)
        #     continue
        # scores = np.vstack(scores_list)
        #
        # scores_ravel = scores.ravel()
        # order = scores_ravel.argsort()[::-1]
        #
        # proposals = proposals[order, :]
        # scores = scores[order]
        # if not self.vote and self.use_landmarks:
        #     landmarks = np.vstack(landmarks_list)
        #     landmarks = landmarks[order].astype(np.float32, copy=False)
        #
        # pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        # if not self.vote:
        #     keep = self.nms(pre_det)
        #     det = np.hstack((pre_det, proposals[:, 4:]))
        #     det = det[keep, :]
        #     if self.use_landmarks:
        #         landmarks = landmarks[keep]
        # else:
        #     det = np.hstack((pre_det, proposals[:, 4:]))
        #     det = self.bbox_vote(det)
        #
        # final_dets_list.append(det)
        # final_landmarks_list.append(landmarks)
    return final_dets_list,final_landmarks_list
