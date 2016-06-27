/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_util.h
 * \brief Caffe Operator Registry
 * \author Haoran Wang 
*/
#include "caffe_operator_util.h"

namespace mxnet {
namespace op {

CaffeOpInitEntry& CaffeOpInitEntry::SetInNum(int in_num) {
  in_num_ = in_num;
  return *this;
}

CaffeOpInitEntry& CaffeOpInitEntry::SetOutNum(int out_num) {
  out_num_ = out_num;
  return *this;
}

CaffeOpInitEntry& CaffeOpInitRegistry::__REGISTER__(const char* name_str,
                                                    pFunc f) {
  std::string name(name_str);
  CHECK(fmap_.count(name) == 0) << "Caffe initial param of " << name << " already existed";
  CaffeOpInitEntry *e = new CaffeOpInitEntry(name, f);
  fmap_[name] = e;
  return *e;
}

CaffeOpInitRegistry* CaffeOpInitRegistry::Get() {
  static CaffeOpInitRegistry inst;
  return &inst;
}

CaffeOpInitRegistry::~CaffeOpInitRegistry() {
  for (auto kv : fmap_) {
    delete kv.second;
  }
}

}  // namespace op
}  // namespace mxnet
