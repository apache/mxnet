/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_blob.h
 * \brief convert mshadow/tensor to caffe/blob 
 * \author Haoran Wang 
*/
#ifndef PLUGIN_CAFFE_CAFFE_BLOB_H_
#define PLUGIN_CAFFE_CAFFE_BLOB_H_

#include<caffe/blob.hpp>
#include<mshadow/tensor.h>
#include<mshadow/tensor_blob.h>
#include<vector>

namespace mxnet {
namespace op {


using mshadow::TBlob;
using mshadow::TShape;
using mshadow::index_t;
using caffe::Blob;
// Declare Memory Type for Caffe blob
namespace caffememtype {
enum caffeMemoryTypes {Data, Grad, Non};
}  // caffememtype

// implementation of tensor to blob, called by TensorToBlob
template<typename Device>
void SetDataGradToBlob(caffememtype::caffeMemoryTypes memType,
                       std::vector<Blob<float>*>::iterator blob,
                       std::vector<TBlob>::const_iterator itr);

TShape Vector2TShape(const std::vector<int> &vec_int); 

std::vector<int> TShape2Vector(const TShape &tshape);

/**
 * \brief The interface to convert mxnet's tensor to caffe's blob
 * \brief called in caffe_operator_inl.h
 */
template<typename Device>
void TBlob2CaffeBlob(caffememtype::caffeMemoryTypes memType,
                     std::vector<Blob<float>*>::iterator blob,
                     std::vector<TBlob>::const_iterator tblob,
                     int n=1) {
  for (int i = 0; i < n; ++i, ++blob, ++tblob) {
    (*blob)->Reshape(TShape2Vector((*tblob).shape_));
    SetDataGradToBlob<Device>(memType, blob, tblob);
  }
}

void InitCaffeBlobs(std::vector<Blob<float>*>& v, size_t n_num);

void DelCaffeBlobs(std::vector<Blob<float>*>& v, size_t n_num);

}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_CAFFE_CAFFE_BLOB_H_
