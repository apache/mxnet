/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_blob.cc
 * \brief Implementations of SetDataGradToBlob given various device/dimension
 * \author Haoran Wang 
*/
#include "caffe_blob.h"
namespace mxnet {
namespace op {

using ::mshadow::cpu;
using ::mshadow::gpu;


template<>
void SetDataGradToBlob<gpu, 1>(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<gpu, 1> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<gpu, 2>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<gpu, 2> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<gpu, 3>(Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<gpu, 3> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<gpu, 4>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<gpu, 4> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_gpu_data(data_ptr);
  else
    blob->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<cpu, 1>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<cpu, 1> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<cpu, 2>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<cpu, 2> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<cpu, 3>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<cpu, 3> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<cpu, 4>(::caffe::Blob<float> *blob,
                       caffememtype::caffeMemoryTypes memType,
                       ::mshadow::Tensor<cpu, 4> *tensor) {
  float *data_ptr = tensor->dptr_;
  if (memType == caffememtype::Data)
    blob->set_cpu_data(data_ptr);
  else
    blob->set_cpu_diff(data_ptr);
}

}  // namespace op
}  // namespace mxnet
