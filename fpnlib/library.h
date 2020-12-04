#ifndef FPNLIB_LIBRARY_H
#define FPNLIB_LIBRARY_H

#include "stdio.h"
#include "stdbool.h"

void hello(void);

int returnIndex(bool use_landmark, int _idx) {
    if (use_landmark) {
        return _idx * 3;
    } else {
        return _idx * 2;
    }
}

void getdata(int batchNumber,
             void *_feat_stride_fpn, int length__feat_stride_fpn,
             bool use_landmark) {
    printf("C side:%d\n", batchNumber);
    printf("length:%d\n", length__feat_stride_fpn);
    printf("use landmark is:%d\n", use_landmark);
    int *_feat_stride_fpnR = (int *) _feat_stride_fpn;
    for (int output_index = 0; output_index < batchNumber; output_index++) {
        //for loop on feat size
        for (int _idx = 0; _idx < length__feat_stride_fpn; _idx++) {
            int stride = _feat_stride_fpnR[_idx];
            int idx = returnIndex(use_landmark, _idx);
        }
        return;
    }
}

#endif //FPNLIB_LIBRARY_H
