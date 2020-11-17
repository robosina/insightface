# %%cython -+ -c=-fopenmp --link-args=-fopenmp
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
import numpy as np
cimport numpy as np
import math
cimport cython
# tag: run
# tag: openmp

from cython.parallel import prange,threadid
from libcpp.vector cimport vector
@cython.boundscheck(False)
cpdef postprocessing(, data, net_outs, image_info, threshold, im_scale, final_dets_list, final_landmarks_list):
    cdef unsigned char output_index
    cdef list proposals_list
    cdef list scores_list
    cdef list landmarks_list
    cdef int number_of_batch=data.shape[0]
    cdef str _key
    cdef unsigned char stride
    # cdef cnp.ndarray scores
    cdef unsigned int prod
    cdef int NUM_THREADS,tid

    NUM_THREADS = data.shape[0]
    cdef vector[100] final_landmarks_list_vec,final_dets_list_vec
    cdef np.ndarray [float,ndim=3] landmarks
    for output_index in prange(NUM_THREADS,nogil=True,num_threads = NUM_THREADS):
        tid = threadid()
    # for output_index in range(data.shape[0]):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        for _idx, s in enumerate(selfObj._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if selfObj.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            scores = net_outs[idx][output_index].asnumpy()  # MY CHANGE
            scores = scores.reshape((1, *scores.shape))

            scores = scores[:, selfObj._num_anchors['stride%s' % s]:, :, :]

            idx += 1
            bbox_deltas = net_outs[idx][output_index].asnumpy()  # MY CHANGE
            bbox_deltas = bbox_deltas.reshape((1, *bbox_deltas.shape))

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = selfObj._num_anchors['stride%s' % s]
            K = height * width
            anchors_fpn = selfObj._anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))

            scores = selfObj._clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = selfObj._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = selfObj.bbox_pred(anchors, bbox_deltas)
            proposals = clip_boxes(proposals, image_info[:2])

            scores_ravel = scores.ravel()  # We Ravel The Scores For All Of The BBoxes
            order = np.where(scores_ravel >= threshold)[0]  # We Only Pick The Best Scores By Threshold
            proposals = proposals[order, :]
            scores = scores[order]
            if stride == 4 and selfObj.decay4 < 1.0:
                scores *= selfObj.decay4

            proposals[:, 0:4] /= im_scale

            proposals_list.append(proposals)
            scores_list.append(scores)

            if not selfObj.vote and selfObj.use_landmarks:
                idx += 1
                landmark_deltas = net_outs[idx][output_index].asnumpy()
                landmark_deltas = landmark_deltas.reshape((1, *landmark_deltas.shape))  # MY CHANGE

                landmark_deltas = selfObj._clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmarks = selfObj.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]

                landmarks[:, :, 0:2] /= im_scale
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if selfObj.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            final_dets_list.append(np.zeros((0, 5)))
            final_landmarks_list.append(landmarks)
            continue
        scores = np.vstack(scores_list)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        if not selfObj.vote and selfObj.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        if not selfObj.vote:
            keep = selfObj.nms(pre_det)
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            if selfObj.use_landmarks:
                landmarks = landmarks[keep]
        else:
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = selfObj.bbox_vote(det)

        final_dets_list_vec[tid]=det
        final_landmarks_list_vec[tid]=landmarks

    return final_dets_list, final_landmarks_list
