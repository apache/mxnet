/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_blob.cc
 * \brief Implementations of SetDataGradToBlob given various device/dimension
 * \author Haoran Wang 
*/
#include "caffe_blob.h"
namespace mxnet {
namespace op {
namespace caffe {

template<>
void SetDataGradToBlob<mshadow::cpu, float>(caffeMemoryTypes memType,
                            std::vector<::caffe::Blob<float>*>::iterator blob,
                            std::vector<mshadow::TBlob>::const_iterator itr) {
  float *data_ptr = reinterpret_cast<float*>((*itr).dptr_);
  if (memType == Data)
    (*blob)->set_cpu_data(data_ptr);
  else
    (*blob)->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<mshadow::cpu, double>(caffeMemoryTypes memType,
                            std::vector<::caffe::Blob<double>*>::iterator blob,
                            std::vector<mshadow::TBlob>::const_iterator itr) {
  double *data_ptr = reinterpret_cast<double*>((*itr).dptr_);
  if (memType == Data)
    (*blob)->set_cpu_data(data_ptr);
  else
    (*blob)->set_cpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<mshadow::gpu, float>(caffeMemoryTypes memType,
                            std::vector<::caffe::Blob<float>*>::iterator blob,
                            std::vector<mshadow::TBlob>::const_iterator itr) {
  float *data_ptr = reinterpret_cast<float*>((*itr).dptr_);
  if (memType == Data)
    (*blob)->set_gpu_data(data_ptr);
  else
    (*blob)->set_gpu_diff(data_ptr);
}

template<>
void SetDataGradToBlob<mshadow::gpu, double>(caffeMemoryTypes memType,
                            std::vector<::caffe::Blob<double>*>::iterator blob,
                            std::vector<mshadow::TBlob>::const_iterator itr) {
  double *data_ptr = reinterpret_cast<double*>((*itr).dptr_);
  if (memType == Data)
    (*blob)->set_gpu_data(data_ptr);
  else
    (*blob)->set_gpu_diff(data_ptr);
}

mshadow::TShape Vector2TShape(const std::vector<int> &vec_int) {
  mshadow::TShape res;
  std::vector<mshadow::index_t> vec_indx;
  for (int i = 0; i < vec_int.size(); ++i)
    vec_indx.push_back(vec_int[i]);
  // 0-dim represents scalar in caffe
  if (vec_int.size() == 0)
    vec_indx.push_back(1);
  res = vec_indx;
  return res;
}

std::vector<int> TShape2Vector(const mshadow::TShape &tshape) {
  std::vector<int> s;
  for (int i =0 ; i < tshape.ndim(); ++i)
    s.push_back(tshape[i]);
  return s;
}

}  // namespace caffe
}  // namespace op
}  // namespace mxnet
