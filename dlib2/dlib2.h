#ifndef DLIB2_H
#define DLIB2_H

#include "dlib2_global.h"
#include <iostream>
extern "C"{
DLIB2_EXPORT void getdata(int batchNumber,
                          void* _feat_stride_fpn,int length__feat_stride_fpn,
                          bool use_landmark){
    std::cout<<"C side:"<<batchNumber<<std::endl;
    std::cout<<"length:"<<length__feat_stride_fpn<<std::endl;
    std::cout<<"use landmark is:"<<use_landmark<<std::endl;
    int* _feat_stride_fpnR=(int*) _feat_stride_fpn;
    for(int output_index=0;output_index<batchNumber;output_index++){
        for(int idx=0;idx<length__feat_stride_fpn;idx++){
            int stride = _feat_stride_fpnR[idx];
            std::string _key = "stride"+std::to_string(stride);

        }
        return ;
    }
}
}

#endif // DLIB2_H
