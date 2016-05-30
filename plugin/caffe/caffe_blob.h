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
#include<vector>

namespace mxnet {
namespace op {

using caffe::Blob;
// Declare Memory Type for Caffe blob
namespace caffememtype {
enum caffeMemoryTypes {Data, Grad, Non};
}  // caffememtype

// implementation of tensor to blob, called by TensorToBlob
template<typename Device, int dimension>
void SetDataGradToBlob(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Device, dimension> *tensor);

/**
 * \brief The interface to convert mxnet's tensor to caffe's blob
 * \brief called in caffe_operator_inl.h
 */
template<typename Device, int dimension>
void TensorToBlob(Blob<float> *blob,
                  caffememtype::caffeMemoryTypes memType0,
                  ::mshadow::Tensor<Device, dimension> *tensor0,
                  caffememtype::caffeMemoryTypes memType1 = caffememtype::Non,
                  ::mshadow::Tensor<Device, dimension> *tensor1 = NULL) {
  std::vector<int> shape;
  shape.resize(dimension);
  // In Caffe's blob, shape[0] is batch size. shape[1] is length of dimension
  for (int i = 0; i < dimension; ++i) {
    shape[i] = tensor0->shape_[i];
    if (tensor1 != NULL)
      CHECK_EQ(tensor0->shape_[i], tensor1->shape_[i]);
  }
  blob->Reshape(shape);
  SetDataGradToBlob<Device, dimension>(blob, memType0, tensor0);
  if ((memType1 != caffememtype::Non) && (tensor1 != NULL))
    SetDataGradToBlob<Device, dimension>(blob, memType1, tensor1);
}

}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_CAFFE_CAFFE_BLOB_H_
