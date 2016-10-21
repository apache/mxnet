/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkl_memory.cc
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_MKL_MEMORY_H_
#define MXNET_MKL_MEMORY_H_

#include <string>
#include <vector>
#include <memory>


namespace mxnet {
// Base class
struct PrvMemDescr {
  virtual void convert_from_prv(void* cpu_ptr) = 0;
  virtual void convert_to_prv(void* cpu_ptr) = 0;
  virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual void* prv_ptr() = 0;
  // returns true for matching layouts
  virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual size_t prv_count() = 0;
  virtual size_t prv_size() = 0;
  // This might help using prv_ptr_ by different accelerators/engines
  enum PrvDescrType {
    PRV_DESCR_MKL2017,
    PRV_DESCR_MKLDNN
  };
  virtual PrvDescrType get_descr_type() = 0;
};

struct MKLMemHolder {
 public:
  virtual std::shared_ptr<PrvMemDescr> get_prv_descriptor() = 0;
};

}  // namespace mxnet
#endif  // MXNET_MKL_MEMORY_H_
