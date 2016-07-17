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
void SetDataGradToBlob<cpu, float>(caffememtype::caffeMemoryTypes memType,
                            std::vector<Blob<float>*>::iterator blob,
                            std::vector<TBlob>::const_iterator itr) {
  float *data_ptr = (float*)(*itr).dptr_;
  if (memType == caffememtype::Data)
    (*blob)->set_cpu_data(data_ptr);
  else
    (*blob)->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<cpu, double>(caffememtype::caffeMemoryTypes memType,
                            std::vector<Blob<double>*>::iterator blob,
                            std::vector<TBlob>::const_iterator itr) {
  double *data_ptr = (double*)(*itr).dptr_;
  if (memType == caffememtype::Data)
    (*blob)->set_cpu_data(data_ptr);
  else
    (*blob)->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<gpu, float>(caffememtype::caffeMemoryTypes memType,
                            std::vector<Blob<float>*>::iterator blob,
                            std::vector<TBlob>::const_iterator itr) {
  float *data_ptr = (float*)(*itr).dptr_;
  if (memType == caffememtype::Data)
    (*blob)->set_gpu_data(data_ptr);
  else
    (*blob)->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<gpu, double>(caffememtype::caffeMemoryTypes memType,
                            std::vector<Blob<double>*>::iterator blob,
                            std::vector<TBlob>::const_iterator itr) {
  double *data_ptr = (double*)(*itr).dptr_;
  if (memType == caffememtype::Data)
    (*blob)->set_gpu_data(data_ptr);
  else
    (*blob)->set_gpu_diff(data_ptr);
}

TShape Vector2TShape(const std::vector<int> &vec_int) {
  TShape res;
  std::vector<index_t> vec_indx;
  for (index_t i = 0; i < vec_int.size(); ++i)
    vec_indx.push_back(vec_int[i]);
  // 0-dim represents scalar in caffe
  if (vec_int.size() == 0)
    vec_indx.push_back(1);
  res = vec_indx;
  return res;
}

std::vector<int> TShape2Vector(const TShape &tshape) {
  std::vector<int> s;
  for (index_t i =0 ; i < tshape.ndim(); ++i)
    s.push_back(tshape[i]);
  return s;
}

}  // namespace op
}  // namespace mxnet
