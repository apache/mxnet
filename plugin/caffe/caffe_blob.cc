/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_blob.cc
 * \brief Implementations of SetDataGradToBlob given various device/dimension
 * \author Haoran Wang 
*/
#include "caffe_blob.h"
namespace mxnet {
namespace op {

typedef ::mshadow::cpu Mcpu;
typedef ::mshadow::gpu Mgpu;


template<>
void SetDataGradToBlob<Mgpu, 1>(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mgpu, 1> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mgpu, 2>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mgpu, 2> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mgpu, 3>(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mgpu, 3> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mgpu, 4>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mgpu, 4> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mcpu, 1>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mcpu, 1> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mcpu, 2>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mcpu, 2> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mcpu, 3>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mcpu, 3> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<Mcpu, 4>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<Mcpu, 4> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

}  // namespace op
}  // namespace mxnet
