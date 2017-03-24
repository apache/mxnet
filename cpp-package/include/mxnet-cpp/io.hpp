/*!
*  Copyright (c) 2016 by Contributors
* \file operator.hpp
* \brief implementation of data iter
* \author Zhang Chen
*/
#ifndef MXNETCPP_IO_HPP
#define MXNETCPP_IO_HPP

#include <string>
#include <vector>
#include "mxnet-cpp/io.h"

namespace mxnet {
namespace cpp {

MXDataIterMap *MXDataIter::mxdataiter_map_ = new MXDataIterMap;

MXDataIter::MXDataIter(const std::string &mxdataiter_type) {
  creator_ = mxdataiter_map_->GetMXDataIterCreator(mxdataiter_type);
  blob_ptr_ = std::make_shared<MXDataIterBlob>(nullptr);
}

void MXDataIter::BeforeFirst() {
  int r = MXDataIterBeforeFirst(blob_ptr_->handle_);
  CHECK_EQ(r, 0);
}

bool MXDataIter::Next() {
  int out;
  int r = MXDataIterNext(blob_ptr_->handle_, &out);
  CHECK_EQ(r, 0);
  return out;
}

NDArray MXDataIter::GetData() {
  NDArrayHandle handle;
  int r = MXDataIterGetData(blob_ptr_->handle_, &handle);
  CHECK_EQ(r, 0);
  return NDArray(handle);
}

NDArray MXDataIter::GetLabel() {
  NDArrayHandle handle;
  int r = MXDataIterGetLabel(blob_ptr_->handle_, &handle);
  CHECK_EQ(r, 0);
  return NDArray(handle);
}

int MXDataIter::GetPadNum() {
  int out;
  int r = MXDataIterGetPadNum(blob_ptr_->handle_, &out);
  CHECK_EQ(r, 0);
  return out;
}
std::vector<int> MXDataIter::GetIndex() {
  uint64_t *out_index, out_size;
  int r = MXDataIterGetIndex(blob_ptr_->handle_, &out_index, &out_size);
  CHECK_EQ(r, 0);
  std::vector<int> ret;
  for (uint64_t i = 0; i < out_size; ++i) {
    ret.push_back(out_index[i]);
  }
  return ret;
}

MXDataIter MXDataIter::CreateDataIter() {
  std::vector<const char *> param_keys;
  std::vector<const char *> param_values;

  for (auto &data : params_) {
    param_keys.push_back(data.first.c_str());
    param_values.push_back(data.second.c_str());
  }

  MXDataIterCreateIter(creator_, param_keys.size(), param_keys.data(),
                       param_values.data(), &blob_ptr_->handle_);
  return *this;
}

// MXDataIter MNIst

}  // namespace cpp
}  // namespace mxnet

#endif /* end of include guard: MXNETCPP_IO_HPP */

