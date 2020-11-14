import time
from RetinaFace.retinaface import RetinaFace
import cv2
import numpy as np
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
cimport numpy as np
from libcpp cimport bool
cdef bint boolean_variable = True

cpdef detectFaces(self,np.float do_flip,list scales, image,threshold):
    cdef list proposals_list = []
    cdef list scores_list = []
    cdef list landmarks_list = []
    cdef list flips = [0]
    if do_flip:
        flips = [0, 1]

    cdef float im_scale
    cdef char flip
    cdef unsigned short h
    cdef unsigned short w
    for im_scale in scales:
        for flip in flips:
            if im_scale != 1.0:
                im = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
            else:
                im = image.copy()
            if flip:
                im = im[:, ::-1, :]
            if self.nocrop:
                if im.shape[0] % 32 == 0:
                    h = im.shape[0]
                else:
                    h = (im.shape[0] // 32 + 1) * 32
                if im.shape[1] % 32 == 0:
                    w = im.shape[1]
                else:
                    w = (im.shape[1] // 32 + 1) * 32
                _im = np.zeros((h, w, 3), dtype=np.float32)
                _im[0:im.shape[0], 0:im.shape[1], :] = im
                im = _im
            else:
                im = im.astype(np.float32)

            # self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
            # im_info = [im.shape[0], im.shape[1], im_scale]
            im_info = [im.shape[0], im.shape[1]]
            im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
            for i in range(3):
                im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / \
                                        self.pixel_stds[2 - i]

            data = self.nd.array(im_tensor)
            db = self.mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])

            self.model.forward(db, is_train=False)
            net_out = self.model.get_outputs()

            # post_nms_topN = self._rpn_post_nms_top_n
            # min_size_dict = self._rpn_min_size_fpn

            # Enumerate Feature Pyramind Network (FPN) By Strides
            for _idx, s in enumerate(self._feat_stride_fpn):
                # if len(scales)>1 and s==32 and im_scale==scales[-1]:
                #  continue
                _key = 'stride%s' % s
                stride = int(s)
                # if self.vote and stride==4 and len(scales)>2 and (im_scale==scales[0]):
                #  continue
                if self.use_landmarks:
                    idx = _idx * 3
                else:
                    idx = _idx * 2
                # print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
                scores = net_out[idx].asnumpy()

                # print(scores.shape)
                # print('scores',stride, scores.shape, file=sys.stderr)
                scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]

                idx += 1
                bbox_deltas = net_out[idx].asnumpy()

                # if DEBUG:
                #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                #    print 'scale: {}'.format(im_info[2])

                # _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
                height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                A = self._num_anchors['stride%s' % s]
                K = height * width
                anchors_fpn = self._anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                # print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
                anchors = anchors.reshape((K * A, 4))

                scores = self._clip_pad(scores, (height, width))
                scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                bbox_pred_len = bbox_deltas.shape[3] // A
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                proposals = self.bbox_pred(anchors, bbox_deltas)
                proposals = clip_boxes(proposals, im_info[:2])

                scores_ravel = scores.ravel()  # We Ravel The Scores For All Of The BBoxes

                order = np.where(scores_ravel >= threshold)[0]  # We Only Pick The Best Scores By Threshold

                proposals = proposals[order, :]
                scores = scores[order]
                if stride == 4 and self.decay4 < 1.0:
                    scores *= self.decay4
                if flip:
                    oldx1 = proposals[:, 0].copy()
                    oldx2 = proposals[:, 2].copy()
                    proposals[:, 0] = im.shape[1] - oldx2 - 1
                    proposals[:, 2] = im.shape[1] - oldx1 - 1

                proposals[:, 0:4] /= im_scale

                proposals_list.append(proposals)
                scores_list.append(scores)

                if not self.vote and self.use_landmarks:
                    idx += 1
                    landmark_deltas = net_out[idx].asnumpy()
                    landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
                    landmark_pred_len = landmark_deltas.shape[1] // A
                    landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape(
                        (-1, 5, landmark_pred_len // 5))
                    # print(landmark_deltas.shape, landmark_deltas)
                    landmarks = self.landmark_pred(anchors, landmark_deltas)
                    landmarks = landmarks[order, :]

                    if flip:
                        landmarks[:, :, 0] = im.shape[1] - landmarks[:, :, 0] - 1
                        order = [1, 0, 2, 4, 3]
                        flandmarks = landmarks.copy()
                        for idx, a in enumerate(order):
                            flandmarks[:, idx, :] = landmarks[:, a, :]
                        landmarks = flandmarks
                    landmarks[:, :, 0:2] /= im_scale
                    landmarks_list.append(landmarks)

    proposals = np.vstack(proposals_list)
    landmarks = None
    if proposals.shape[0] == 0:
        if self.use_landmarks:
            landmarks = np.zeros((0, 5, 2))
        return np.zeros((0, 5)), landmarks
    scores = np.vstack(scores_list)
    # print('shapes', proposals.shape, scores.shape)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]
    if not self.vote and self.use_landmarks:
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
    if not self.vote:
        keep = self.nms(pre_det)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]
    else:
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = self.bbox_vote(det)

    end = time.time()
    # logger.debug("Single Detection Time => %s" % (end - start))
    return det, landmarks  # det Contains 4 BBox Points + 1 Score Point, Landmarks Contains 5 Points
