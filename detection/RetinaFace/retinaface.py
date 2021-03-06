from __future__ import print_function
import sys
import pickle
import os
from ctypes import *
import datetime
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from ctypes import *
import numpy as np
import csv
# from rcnn import config
from rcnn.logger import logger
# from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps

from rcnn.cython.bbox_flitering import postprocessing


# from rcnn.cython.detect_high import detectFaces


class RetinaFace:
    def __init__(self, prefix, epoch, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4=0.5, vote=False):
        import mxnet as mx
        from mxnet import ndarray as nd
        self.mx = mx
        self.nd = nd
        self.ctx_id = ctx_id
        self.network = network
        self.decay4 = decay4
        self.nms_threshold = nms
        self.vote = vote
        self.nocrop = nocrop
        self.debug = False
        self.fpn_keys = []
        self.anchor_cfg = None
        pixel_means = [0.0, 0.0, 0.0]
        pixel_stds = [1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.preprocess = False
        _ratio = (1.,)
        fmc = 3
        if network == 'ssh' or network == 'vgg':
            pixel_means = [103.939, 116.779, 123.68]
            self.preprocess = True
        elif network == 'net3':
            _ratio = (1.,)
        elif network == 'net3a':
            _ratio = (1., 1.5)
        elif network == 'net6':  # like pyramidbox or s3fd
            fmc = 6
        elif network == 'net5':  # retinaface
            fmc = 5
        elif network == 'net5a':
            fmc = 5
            _ratio = (1., 1.5)
        elif network == 'net4':
            fmc = 4
        elif network == 'net4a':
            fmc = 4
            _ratio = (1., 1.5)
        elif network == 'x5':
            fmc = 5
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        elif network == 'x3':
            fmc = 3
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        elif network == 'x3a':
            fmc = 3
            _ratio = (1., 1.5)
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        else:
            assert False, 'network setting error %s' % network

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {
                    'SCALES': (32, 16),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '16': {
                    'SCALES': (8, 4),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '8': {
                    'SCALES': (2, 1),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
            }
        elif fmc == 4:
            self._feat_stride_fpn = [32, 16, 8, 4]
            self.anchor_cfg = {
                '32': {
                    'SCALES': (32, 16),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '16': {
                    'SCALES': (8, 4),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '8': {
                    'SCALES': (2, 1),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '4': {
                    'SCALES': (2, 1),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
            }
        elif fmc == 6:
            self._feat_stride_fpn = [128, 64, 32, 16, 8, 4]
            self.anchor_cfg = {
                '128': {
                    'SCALES': (32,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '64': {
                    'SCALES': (16,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '32': {
                    'SCALES': (8,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '16': {
                    'SCALES': (4,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '8': {
                    'SCALES': (2,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '4': {
                    'SCALES': (1,),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
            }
        elif fmc == 5:
            self._feat_stride_fpn = [64, 32, 16, 8, 4]
            self.anchor_cfg = {}
            _ass = 2.0 ** (1.0 / 3)
            _basescale = 1.0
            for _stride in [4, 8, 16, 32, 64]:
                key = str(_stride)
                value = {
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                }
                scales = []
                for _ in range(3):
                    scales.append(_basescale)
                    _basescale *= _ass
                value['SCALES'] = tuple(scales)
                self.anchor_cfg[key] = value

        print(self._feat_stride_fpn, self.anchor_cfg)

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        dense_anchor = False
        # self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
        self._anchors_fpn = dict(
            zip(
                self.fpn_keys,
                generate_anchors_fpn(dense_anchor=dense_anchor,
                                     cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(
            zip(self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        # self._bbox_pred = nonlinear_pred
        # self._landmark_pred = landmark_pred
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        if self.ctx_id >= 0:
            self.ctx = mx.gpu(self.ctx_id)
            self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        else:
            self.ctx = mx.cpu()
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        print('means', self.pixel_means)
        self.use_landmarks = False
        if len(sym) // len(self._feat_stride_fpn) >= 3:
            self.use_landmarks = True
        print('use_landmarks', self.use_landmarks)
        self.cascade = 0
        if float(len(sym)) // len(self._feat_stride_fpn) > 3.0:
            self.cascade = 1
        print('cascade', self.cascade)
        # self.bbox_stds = [0.1, 0.1, 0.2, 0.2]
        # self.landmark_std = 0.1
        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.landmark_std = 1.0

        if self.debug:
            c = len(sym) // len(self._feat_stride_fpn)
            sym = sym[(c * 0):]
            self._feat_stride_fpn = [32, 16, 8]
        print('sym size:', len(sym))

        image_size = (640, 640)
        self.model = mx.mod.Module(symbol=sym,
                                   context=self.ctx,
                                   label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0],
                                               image_size[1]))],
                        for_training=False)
        self.model.set_params(arg_params, aux_params)

    def get_input(self, img):
        im = img.astype(np.float32)
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[
            0,
            i, :, :] = (im[:, :, 2 - i] / self.pixel_scale -
                        self.pixel_means[2 - i]) / self.pixel_stds[2 - i]
        # if self.debug:
        #  timeb = datetime.datetime.now()
        #  diff = timeb - timea
        #  print('X2 uses', diff.total_seconds(), 'seconds')
        data = nd.array(im_tensor)
        return data

    def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        # print('in_detect', threshold, scales, do_flip, do_nms)
        proposals_list = []
        scores_list = []
        landmarks_list = []
        strides_list = []
        timea = datetime.datetime.now()
        flips = [0]
        if do_flip:
            flips = [0, 1]

        imgs = [img]
        if isinstance(img, list):
            imgs = img
        for img in imgs:
            for im_scale in scales:
                for flip in flips:
                    if im_scale != 1.0:
                        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img.copy()
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
                    if self.debug:
                        timeb = datetime.datetime.now()
                        diff = timeb - timea
                        print('X1 uses', diff.total_seconds(), 'seconds')
                    # self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
                    # im_info = [im.shape[0], im.shape[1], im_scale]
                    im_info = [im.shape[0], im.shape[1]]
                    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
                    for i in range(3):
                        im_tensor[0, i, :, :] = (
                                                        im[:, :, 2 - i] / self.pixel_scale -
                                                        self.pixel_means[2 - i]) / self.pixel_stds[2 - i]
                    if self.debug:
                        timeb = datetime.datetime.now()
                        diff = timeb - timea
                        print('X2 uses', diff.total_seconds(), 'seconds')
                    data = nd.array(im_tensor)
                    ############preprocess finished here#########################

                    db = mx.io.DataBatch(data=(data,),
                                         provide_data=[('data', data.shape)])
                    if self.debug:
                        timeb = datetime.datetime.now()
                        diff = timeb - timea
                        print('X3 uses', diff.total_seconds(), 'seconds')
                    self.model.forward(db, is_train=False)
                    net_out = self.model.get_outputs()
                    ##############model finished here############################

                    # post_nms_topN = self._rpn_post_nms_top_n
                    # min_size_dict = self._rpn_min_size_fpn

                    sym_idx = 0

                    for _idx, s in enumerate(self._feat_stride_fpn):
                        # if len(scales)>1 and s==32 and im_scale==scales[-1]:
                        #  continue
                        _key = 'stride%s' % s
                        stride = int(s)
                        is_cascade = False
                        if self.cascade:
                            is_cascade = True
                        # if self.vote and stride==4 and len(scales)>2 and (im_scale==scales[0]):
                        #  continue
                        # print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
                        scores = net_out[sym_idx].asnumpy()
                        if self.debug:
                            timeb = datetime.datetime.now()
                            diff = timeb - timea
                            print('A uses', diff.total_seconds(), 'seconds')
                        # print(scores.shape)
                        # print('scores',stride, scores.shape, file=sys.stderr)
                        scores = scores[:, self._num_anchors['stride%s' %
                                                             s]:, :, :]

                        bbox_deltas = net_out[sym_idx + 1].asnumpy()

                        # if DEBUG:
                        #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                        #    print 'scale: {}'.format(im_info[2])

                        # _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
                        height, width = bbox_deltas.shape[
                                            2], bbox_deltas.shape[3]

                        A = self._num_anchors['stride%s' % s]
                        K = height * width
                        anchors_fpn = self._anchors_fpn['stride%s' % s]
                        anchors = anchors_plane(height, width, stride,
                                                anchors_fpn)
                        # print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
                        anchors = anchors.reshape((K * A, 4))
                        # print('num_anchors', self._num_anchors['stride%s'%s], file=sys.stderr)
                        # print('HW', (height, width), file=sys.stderr)
                        # print('anchors_fpn', anchors_fpn.shape, file=sys.stderr)
                        # print('anchors', anchors.shape, file=sys.stderr)
                        # print('bbox_deltas', bbox_deltas.shape, file=sys.stderr)
                        # print('scores', scores.shape, file=sys.stderr)

                        # scores = self._clip_pad(scores, (height, width))
                        scores = scores.transpose((0, 2, 3, 1)).reshape(
                            (-1, 1))

                        # print('pre', bbox_deltas.shape, height, width)
                        # bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                        # print('after', bbox_deltas.shape, height, width)
                        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                        bbox_pred_len = bbox_deltas.shape[3] // A
                        # print(bbox_deltas.shape)
                        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                        bbox_deltas[:,
                        0::4] = bbox_deltas[:, 0::
                                               4] * self.bbox_stds[0]
                        bbox_deltas[:,
                        1::4] = bbox_deltas[:, 1::
                                               4] * self.bbox_stds[1]
                        bbox_deltas[:,
                        2::4] = bbox_deltas[:, 2::
                                               4] * self.bbox_stds[2]
                        bbox_deltas[:,
                        3::4] = bbox_deltas[:, 3::
                                               4] * self.bbox_stds[3]
                        proposals = self.bbox_pred(anchors, bbox_deltas)

                        # print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
                        if is_cascade:
                            cascade_sym_num = 0
                            cls_cascade = False
                            bbox_cascade = False
                            __idx = [3, 4]
                            if not self.use_landmarks:
                                __idx = [2, 3]
                            for diff_idx in __idx:
                                if sym_idx + diff_idx >= len(net_out):
                                    break
                                body = net_out[sym_idx + diff_idx].asnumpy()
                                if body.shape[1] // A == 2:  # cls branch
                                    if cls_cascade or bbox_cascade:
                                        break
                                    else:
                                        cascade_scores = body[:, self.
                                                                     _num_anchors[
                                                                     'stride%s' %
                                                                     s]:, :, :]
                                        cascade_scores = cascade_scores.transpose(
                                            (0, 2, 3, 1)).reshape((-1, 1))
                                        # scores = (scores+cascade_scores)/2.0
                                        scores = cascade_scores  # TODO?
                                        cascade_sym_num += 1
                                        cls_cascade = True
                                        # print('find cascade cls at stride', stride)
                                elif body.shape[1] // A == 4:  # bbox branch
                                    cascade_deltas = body.transpose(
                                        (0, 2, 3, 1)).reshape(
                                        (-1, bbox_pred_len))
                                    cascade_deltas[:, 0::
                                                      4] = cascade_deltas[:, 0::
                                                                             4] * self.bbox_stds[
                                                               0]
                                    cascade_deltas[:, 1::
                                                      4] = cascade_deltas[:, 1::
                                                                             4] * self.bbox_stds[
                                                               1]
                                    cascade_deltas[:, 2::
                                                      4] = cascade_deltas[:, 2::
                                                                             4] * self.bbox_stds[
                                                               2]
                                    cascade_deltas[:, 3::
                                                      4] = cascade_deltas[:, 3::
                                                                             4] * self.bbox_stds[
                                                               3]
                                    proposals = self.bbox_pred(
                                        proposals, cascade_deltas)
                                    cascade_sym_num += 1
                                    bbox_cascade = True
                                    # print('find cascade bbox at stride', stride)

                        proposals = clip_boxes(proposals, im_info[:2])

                        # if self.vote:
                        #  if im_scale>1.0:
                        #    keep = self._filter_boxes2(proposals, 160*im_scale, -1)
                        #  else:
                        #    keep = self._filter_boxes2(proposals, -1, 100*im_scale)
                        #  if stride==4:
                        #    keep = self._filter_boxes2(proposals, 12*im_scale, -1)
                        #    proposals = proposals[keep, :]
                        #    scores = scores[keep]

                        # keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
                        # proposals = proposals[keep, :]
                        # scores = scores[keep]
                        # print('333', proposals.shape)
                        if stride == 4 and self.decay4 < 1.0:
                            scores *= self.decay4

                        scores_ravel = scores.ravel()
                        # print('__shapes', proposals.shape, scores_ravel.shape)
                        # print('max score', np.max(scores_ravel))
                        order = np.where(scores_ravel >= threshold)[0]
                        # _scores = scores_ravel[order]
                        # _order = _scores.argsort()[::-1]
                        # order = order[_order]
                        proposals = proposals[order, :]
                        scores = scores[order]
                        if flip:
                            oldx1 = proposals[:, 0].copy()
                            oldx2 = proposals[:, 2].copy()
                            proposals[:, 0] = im.shape[1] - oldx2 - 1
                            proposals[:, 2] = im.shape[1] - oldx1 - 1

                        proposals[:, 0:4] /= im_scale

                        proposals_list.append(proposals)
                        scores_list.append(scores)
                        if self.nms_threshold < 0.0:
                            _strides = np.empty(shape=(scores.shape),
                                                dtype=np.float32)
                            _strides.fill(stride)
                            strides_list.append(_strides)

                        if not self.vote and self.use_landmarks:
                            landmark_deltas = net_out[sym_idx + 2].asnumpy()
                            # landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
                            landmark_pred_len = landmark_deltas.shape[1] // A
                            landmark_deltas = landmark_deltas.transpose(
                                (0, 2, 3, 1)).reshape(
                                (-1, 5, landmark_pred_len // 5))
                            landmark_deltas *= self.landmark_std
                            # print(landmark_deltas.shape, landmark_deltas)
                            landmarks = self.landmark_pred(
                                anchors, landmark_deltas)
                            landmarks = landmarks[order, :]

                            if flip:
                                landmarks[:, :,
                                0] = im.shape[1] - landmarks[:, :,
                                                   0] - 1
                                # for a in range(5):
                                #  oldx1 = landmarks[:, a].copy()
                                #  landmarks[:,a] = im.shape[1] - oldx1 - 1
                                order = [1, 0, 2, 4, 3]
                                flandmarks = landmarks.copy()
                                for idx, a in enumerate(order):
                                    flandmarks[:, idx, :] = landmarks[:, a, :]
                                    # flandmarks[:, idx*2] = landmarks[:,a*2]
                                    # flandmarks[:, idx*2+1] = landmarks[:,a*2+1]
                                landmarks = flandmarks
                            landmarks[:, :, 0:2] /= im_scale
                            # landmarks /= im_scale
                            # landmarks = landmarks.reshape( (-1, landmark_pred_len) )
                            landmarks_list.append(landmarks)
                            # proposals = np.hstack((proposals, landmarks))
                        if self.use_landmarks:
                            sym_idx += 3
                        else:
                            sym_idx += 2
                        if is_cascade:
                            sym_idx += cascade_sym_num

        if self.debug:
            timeb = datetime.datetime.now()
            diff = timeb - timea
            print('B uses', diff.total_seconds(), 'seconds')
        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            if self.nms_threshold < 0.0:
                return np.zeros((0, 6)), landmarks
            else:
                return np.zeros((0, 5)), landmarks
        scores = np.vstack(scores_list)
        # print('shapes', proposals.shape, scores.shape)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        # if config.TEST.SCORE_THRESH>0.0:
        #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
        #  order = order[:_count]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.nms_threshold < 0.0:
            strides = np.vstack(strides_list)
            strides = strides[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        if self.nms_threshold > 0.0:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                    copy=False)
            if not self.vote:
                keep = self.nms(pre_det)
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = det[keep, :]
                if self.use_landmarks:
                    landmarks = landmarks[keep]
            else:
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = self.bbox_vote(det)
        elif self.nms_threshold < 0.0:
            det = np.hstack(
                (proposals[:, 0:4], scores, strides)).astype(np.float32,
                                                             copy=False)
        else:
            det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                copy=False)

        if self.debug:
            timeb = datetime.datetime.now()
            diff = timeb - timea
            print('C uses', diff.total_seconds(), 'seconds')
        return det, landmarks

    def detectD2(self, image, threshold=0.5, scales=[1.0], do_flip=False):
        return detectFaces(self, do_flip, scales, image, threshold)

    def detect_batch_cython(self, images, scales=[1.0], threshold=0.5, do_flip=False):
        t0 = time.time()
        im_scale = scales[0]
        if im_scale != 1.0:
            images = [cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC) for image
                      in images]

        images = np.array(images, dtype=np.float32)

        t1 = time.time()
        print("T1 (Numpy Array Initialization) => %.3f" % float(t1 - t0))

        image_info = [images[0].shape[0], images[0].shape[1]]
        image_tensors = np.empty((len(images), 3, images[0].shape[0], images[0].shape[1]))

        t2 = time.time()
        print("T2 (Numpy Empty Array Initialization) => %.3f" % float(t2 - t1))

        # for j, image_tensor in enumerate(image_tensors):
        #     for i in range(3):
        #         image_tensor[i, :, :] = (images_copy[j][:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i]

        mstd1 = self.pixel_means[2] / self.pixel_stds[2]
        mstd2 = self.pixel_means[1] / self.pixel_stds[1]
        mstd3 = self.pixel_means[0] / self.pixel_stds[0]
        for j, (image_copy, image_tensor) in enumerate(zip(images, image_tensors)):
            image_tensor[0, :, :] = (image_copy[:, :, 2] / self.pixel_stds[2]) - mstd1
            image_tensor[1, :, :] = (image_copy[:, :, 1] / self.pixel_stds[1]) - mstd2
            image_tensor[2, :, :] = (image_copy[:, :, 0] / self.pixel_stds[0]) - mstd3

        t3 = time.time()
        print("T3 (Image Tensor Copying) => %.3f" % float(t3 - t2))

        data = self.nd.array(image_tensors)
        db = self.mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])

        t4 = time.time()
        print("T4 (MXNet Array Initialization) => %.3f" % float(t4 - t3))

        self.model.forward(db, is_train=False)
        net_outs = self.model.get_outputs()

        t5 = time.time()
        print("T5 (Model Forward) => %.2f" % float(t5 - t4))

        final_dets_list = []
        final_landmarks_list = []

        # nda.wait_to_read()
        t6 = time.time()
        print("T1 + T2 + T3 + T4 + T5 => %.2f" % float(t6 - t0))
        # # TODO Remove Later
        # pickle_data = [data, self._feat_stride_fpn, self.use_landmarks,
        #                net_outs, self._num_anchors, self._anchors_fpn,
        #                threshold, self.decay4, im_scale, self.vote,
        #                final_dets_list, final_landmarks_list, image_info]
        # pickle.dump(pickle_data, open('./ctest.pkl', "wb"))
        # TODO Remove Later
        for k in range(3):
            t6 = time.time()
            npNet_outs = [out.asnumpy() for out in net_outs]
            t7 = time.time()
            print("\033[1;31mT7 (Convert from mxnet to numpy) => %.2f\033[0m" % (t7 - t6))

        for i in range(3):
            #     npNet_outs.append(net_outs[i].asnumpy())
            final_dets_list, final_landmarks_list = postprocessing(self, data, npNet_outs, image_info, data.shape[0],
                                                                   self._feat_stride_fpn, self.use_landmarks,
                                                                   self._num_anchors,self._anchors_fpn,
                                                                   threshold, im_scale, final_dets_list,
                                                                   final_landmarks_list)

            t8 = time.time()
            print("\033[1;32mT8 (Landmark & BBox Processing) => %.2f\033[0m" % (t8 - t7))
        return final_dets_list, final_landmarks_list

    def detect_batch(self, images, scales=[1.0], threshold=0.5, do_flip=False):
        t0 = time.time()
        im_scale = scales[0]
        if im_scale != 1.0:
            images = [cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC) for image
                      in images]

        images = np.array(images, dtype=np.float32)

        t1 = time.time()
        print("T1 (Numpy Array Initialization) => %.3f" % float(t1 - t0))

        image_info = [images[0].shape[0], images[0].shape[1]]
        image_tensors = np.empty((len(images), 3, images[0].shape[0], images[0].shape[1]))

        t2 = time.time()
        print("T2 (Numpy Empty Array Initialization) => %.3f" % float(t2 - t1))

        # for j, image_tensor in enumerate(image_tensors):
        #     for i in range(3):
        #         image_tensor[i, :, :] = (images_copy[j][:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i]

        mstd1 = self.pixel_means[2] / self.pixel_stds[2]
        mstd2 = self.pixel_means[1] / self.pixel_stds[1]
        mstd3 = self.pixel_means[0] / self.pixel_stds[0]
        for j, (image_copy, image_tensor) in enumerate(zip(images, image_tensors)):
            image_tensor[0, :, :] = (image_copy[:, :, 2] / self.pixel_stds[2]) - mstd1
            image_tensor[1, :, :] = (image_copy[:, :, 1] / self.pixel_stds[1]) - mstd2
            image_tensor[2, :, :] = (image_copy[:, :, 0] / self.pixel_stds[0]) - mstd3

        t3 = time.time()
        print("T3 (Image Tensor Copying) => %.3f" % float(t3 - t2))

        data = self.nd.array(image_tensors)
        db = self.mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])

        t4 = time.time()
        print("T4 (MXNet Array Initialization) => %.3f" % float(t4 - t3))

        self.model.forward(db, is_train=False)
        net_outs = self.model.get_outputs()

        t5 = time.time()
        print("T5 (Model Forward) => %.2f" % float(t5 - t4))

        final_dets_list = []
        final_landmarks_list = []

        t6 = time.time()
        print("T1 + T2 + T3 + T4 + T5 => %.2f" % float(t6 - t0))

        to_save = {'data': data, '_feat_stride_fpn': self._feat_stride_fpn}

        # write that dictionary to a file
        with open('my_log.csv', 'w+') as f:
            w = csv.DictWriter(f, to_save.keys())
            w.writeheader()
            w.writerow(to_save)

        # # TODO Remove Later
        # pickle_data = [data, self._feat_stride_fpn, self.use_landmarks,
        #                net_outs, self._num_anchors, self._anchors_fpn,
        #                threshold, self.decay4, im_scale, self.vote,
        #                final_dets_list, final_landmarks_list, image_info]
        # pickle.dump(pickle_data, open('./CYTHON_FACE_SUITE_TEST.pkl', "wb"))
        # TODO Remove Later

        # lib = cdll.LoadLibrary('/home/isv/qt_projects/dlib2/libdlib2.so')
        # fun = lib.getdata
        # fun(c_int(data.shape[0]),c_void_p)

        for output_index in range(data.shape[0]):
            proposals_list = []
            scores_list = []
            landmarks_list = []

            for _idx, s in enumerate(self._feat_stride_fpn):
                _key = 'stride%s' % s
                stride = int(s)
                if self.use_landmarks:
                    idx = _idx * 3
                else:
                    idx = _idx * 2
                scores = net_outs[idx][output_index].asnumpy()  # MY CHANGE
                # print(scores.shape)
                scores = scores.reshape((1, *scores.shape))

                scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]

                idx += 1
                bbox_deltas = net_outs[idx][output_index].asnumpy()  # MY CHANGE
                bbox_deltas = bbox_deltas.reshape((1, *bbox_deltas.shape))

                height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                A = self._num_anchors['stride%s' % s]
                K = height * width
                anchors_fpn = self._anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = anchors.reshape((K * A, 4))

                scores = self._clip_pad(scores, (height, width))
                scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

                bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                bbox_pred_len = bbox_deltas.shape[3] // A
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

                proposals = self.bbox_pred(anchors, bbox_deltas)
                proposals = clip_boxes(proposals, image_info[:2])

                scores_ravel = scores.ravel()  # We Ravel The Scores For All Of The BBoxes
                order = np.where(scores_ravel >= threshold)[0]  # We Only Pick The Best Scores By Threshold
                proposals = proposals[order, :]
                scores = scores[order]
                if stride == 4 and self.decay4 < 1.0:
                    scores *= self.decay4

                proposals[:, 0:4] /= im_scale

                proposals_list.append(proposals)
                scores_list.append(scores)

                if not self.vote and self.use_landmarks:
                    idx += 1
                    landmark_deltas = net_outs[idx][output_index].asnumpy()
                    landmark_deltas = landmark_deltas.reshape((1, *landmark_deltas.shape))  # MY CHANGE

                    landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
                    landmark_pred_len = landmark_deltas.shape[1] // A
                    landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                    landmarks = self.landmark_pred(anchors, landmark_deltas)
                    landmarks = landmarks[order, :]

                    landmarks[:, :, 0:2] /= im_scale
                    landmarks_list.append(landmarks)

            proposals = np.vstack(proposals_list)
            landmarks = None
            if proposals.shape[0] == 0:
                if self.use_landmarks:
                    landmarks = np.zeros((0, 5, 2))
                final_dets_list.append(np.zeros((0, 5)))
                final_landmarks_list.append(landmarks)
                continue
            scores = np.vstack(scores_list)

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

            final_dets_list.append(det)
            final_landmarks_list.append(landmarks)

        t7 = time.time()
        print("T7 (Landmark & BBox Processing) => %.2f" % (t7 - t6))
        return final_dets_list, final_landmarks_list

    # detectFaces(self, do_flip, scales, image, threshold):
    def detectD(self, image, threshold=0.5, scales=[1.0], do_flip=False):
        start = time.time()
        # print('in_detect', threshold, scales, do_flip, do_nms)
        proposals_list = []
        scores_list = []
        landmarks_list = []
        flips = [0]
        if do_flip:
            flips = [0, 1]

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
        logger.debug("Single Detection Time => %s" % (end - start))
        return det, landmarks  # det Contains 4 BBox Points + 1 Score Point, Landmarks Contains 5 Points

    def detect_center(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        det, landmarks = self.detect(img, threshold, scales, do_flip)
        if det.shape[0] == 0:
            return None, None
        bindex = 0
        if det.shape[0] > 1:
            img_size = np.asarray(img.shape)[0:2]
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                           det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared *
                               2.0)  # some extra weight on the centering
        bbox = det[bindex, :]
        landmark = landmarks[bindex, :, :]
        return bbox, landmark

    @staticmethod
    def check_large_pose(landmark, bbox):
        assert landmark.shape == (5, 2)
        assert len(bbox) == 4

        def get_theta(base, x, y):
            vx = x - base
            vy = y - base
            vx[1] *= -1
            vy[1] *= -1
            tx = np.arctan2(vx[1], vx[0])
            ty = np.arctan2(vy[1], vy[0])
            d = ty - tx
            d = np.degrees(d)
            # print(vx, tx, vy, ty, d)
            # if d<-1.*math.pi:
            #  d+=2*math.pi
            # elif d>math.pi:
            #  d-=2*math.pi
            if d < -180.0:
                d += 360.
            elif d > 180.0:
                d -= 360.0
            return d

        landmark = landmark.astype(np.float32)

        theta1 = get_theta(landmark[0], landmark[3], landmark[2])
        theta2 = get_theta(landmark[1], landmark[2], landmark[4])
        # print(va, vb, theta2)
        theta3 = get_theta(landmark[0], landmark[2], landmark[1])
        theta4 = get_theta(landmark[1], landmark[0], landmark[2])
        theta5 = get_theta(landmark[3], landmark[4], landmark[2])
        theta6 = get_theta(landmark[4], landmark[2], landmark[3])
        theta7 = get_theta(landmark[3], landmark[2], landmark[0])
        theta8 = get_theta(landmark[4], landmark[1], landmark[2])
        # print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
        left_score = 0.0
        right_score = 0.0
        up_score = 0.0
        down_score = 0.0
        if theta1 <= 0.0:
            left_score = 10.0
        elif theta2 <= 0.0:
            right_score = 10.0
        else:
            left_score = theta2 / theta1
            right_score = theta1 / theta2
        if theta3 <= 10.0 or theta4 <= 10.0:
            up_score = 10.0
        else:
            up_score = max(theta1 / theta3, theta2 / theta4)
        if theta5 <= 10.0 or theta6 <= 10.0:
            down_score = 10.0
        else:
            down_score = max(theta7 / theta5, theta8 / theta6)
        mleft = (landmark[0][0] + landmark[3][0]) / 2
        mright = (landmark[1][0] + landmark[4][0]) / 2
        box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        ret = 0
        if left_score >= 3.0:
            ret = 1
        if ret == 0 and left_score >= 2.0:
            if mright <= box_center[0]:
                ret = 1
        if ret == 0 and right_score >= 3.0:
            ret = 2
        if ret == 0 and right_score >= 2.0:
            if mleft >= box_center[0]:
                ret = 2
        if ret == 0 and up_score >= 2.0:
            ret = 3
        if ret == 0 and down_score >= 5.0:
            ret = 4
        return ret, left_score, right_score, up_score, down_score

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _filter_boxes2(boxes, max_size, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            keep = np.where(np.minimum(ws, hs) < max_size)[0]
        elif min_size > 0:
            keep = np.where(np.maximum(ws, hs) > min_size)[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N 4 * num_classes]
      """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
        return pred
        # preds = []
        # for i in range(landmark_deltas.shape[1]):
        #  if i%2==0:
        #    pred = (landmark_deltas[:,i]*widths + ctr_x)
        #  else:
        #    pred = (landmark_deltas[:,i]*heights + ctr_y)
        #  preds.append(pred)
        # preds = np.vstack(preds).transpose()
        # return preds

    def bbox_vote(self, det):
        # order = det[:, 4].ravel().argsort()[::-1]
        # det = det[order, :]
        if det.shape[0] == 0:
            return np.zeros((0, 5))
            # dets = np.array([[10, 10, 20, 20, 0.002]])
            # det = np.empty(shape=[0, 5])
        dets = None
        while det.shape[0] > 0:
            if dets is not None and dets.shape[0] >= 750:
                break
            # IOU
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # nms
            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    try:
                        dets = np.row_stack((dets, det_accu))
                    except:
                        dets = det_accu
                continue
            det_accu[:,
            0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:],
                                              (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            if dets is None:
                dets = det_accu_sum
            else:
                dets = np.row_stack((dets, det_accu_sum))
        dets = dets[0:750, :]
        return dets
