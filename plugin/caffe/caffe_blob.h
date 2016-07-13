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
void SetDataGradToBlob(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       TBlob *tblob);

TShape Vector2TShape(const std::vector<int> &vec_int); 

std::vector<int> TShape2Vector(const TShape &tshape);

/**
 * \brief The interface to convert mxnet's tensor to caffe's blob
 * \brief called in caffe_operator_inl.h
 */
template<typename Device>
void TensorToBlob(Blob<float> *blob,
                  caffememtype::caffeMemoryTypes memType0,
                  TBlob *tblob0,
                  caffememtype::caffeMemoryTypes memType1 = caffememtype::Non,
                  TBlob *tblob1 = NULL) {
  blob->Reshape(TShape2Vector(tblob0->shape_));
  SetDataGradToBlob<Device>(blob, memType0, tblob0);
  if ((memType1 != caffememtype::Non) && (tblob1 != NULL))
    SetDataGradToBlob<Device>(blob, memType1, tblob1);
}

}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_CAFFE_CAFFE_BLOB_H_
