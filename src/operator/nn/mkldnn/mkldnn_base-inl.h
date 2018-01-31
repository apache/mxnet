/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
* \file mkldnn_base-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*         zhengda1936@gmail.com
*
*******************************************************************************/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include "mkldnn.hpp"
#include "mxnet/ndarray.h"
#include "mxnet/resource.h"
#include "mxnet/op_attr_types.h"
using namespace mkldnn;
namespace mxnet {
extern bool EnableMkldnnWarnGenerated();
// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
  static CpuEngine *Get() {
    // I's thread-safe in C++11.
    static thread_local CpuEngine myInstance;
    return &myInstance;
  }
  CpuEngine(CpuEngine const &) = delete;             // Copy construct
  CpuEngine(CpuEngine &&) = delete;                  // Move construct
  CpuEngine &operator=(CpuEngine const &) = delete;  // Copy assign
  CpuEngine &operator=(CpuEngine &&) = delete;       // Move assign

  mkldnn::engine &get_engine() { return _cpu_engine; }

 protected:
  CpuEngine() : _cpu_engine(mkldnn::engine::cpu, 0) {}
  ~CpuEngine() {}

 private:
  mkldnn::engine _cpu_engine;
};

// type enumerator
template <typename T>
struct data_type_enum {};

template <>
struct data_type_enum<float> {
  enum { type = mkldnn::memory::data_type::f32 };
};

template <>
struct data_type_enum<int32_t> {
  enum { type = mkldnn::memory::data_type::s32 };
};

template <>
struct data_type_enum<int16_t> {
  enum { type = mkldnn::memory::data_type::s16 };
};

template <>
struct data_type_enum<int8_t> {
  enum { type = mkldnn::memory::data_type::s8 };
};

template <>
struct data_type_enum<uint8_t> {
  enum { type = mkldnn::memory::data_type::u8 };
};

static inline bool SupportMKLDNNArray(int dtype, const TShape &shape) {
  int ndim = shape.ndim();
  bool support = ndim == 1 || ndim == 2 || ndim == 4;
  support = support && (dtype == mshadow::kFloat32 || dtype == mshadow::kInt32
                        || dtype == mshadow::kInt8 || dtype == mshadow::kUint8);
  return support;
}

static inline bool SupportStorageMKLDNN(int stype) {
  return stype == kDefaultStorage;
}

static inline bool SupportMKLDNN(int dtype, const TShape &shape) {
  int ndim = shape.ndim();
  return dtype == mshadow::kFloat32 && (ndim == 1 || ndim == 2 || ndim == 4);
}

static inline bool SupportMKLDNN(const NDArray &input) {
  return SupportMKLDNN(input.dtype(), input.shape())
      && SupportStorageMKLDNN(input.storage_type());
}

static inline bool SupportMKLDNNConv(const NDArray &input) {
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

/*
 * This is to align address to a certain alignment.
 */
void *AlignMem(void *mem, size_t size, size_t alignment, size_t *space);

namespace op {
struct ActivationParam;
bool SupportMKLDNNAct(const op::ActivationParam& param);
}

static int GetTypeSize(int dtype) {
  int size = -1;
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    size = sizeof(DType);
  });
  return size;
}

static inline size_t GetArraySize(const NDArray &arr) {
  return arr.shape().Size() * GetTypeSize(arr.dtype());
}

static inline mkldnn::memory::data_type get_mkldnn_type(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return mkldnn::memory::data_type::f32;
    case mshadow::kInt32:
      return mkldnn::memory::data_type::s32;
    case mshadow::kInt8:
      return mkldnn::memory::data_type::s8;
    case mshadow::kUint8:
      return mkldnn::memory::data_type::u8;
    default:
      LOG(FATAL) << "unknown type for MKLDNN";
      return mkldnn::memory::data_type::data_undef;
  }
}

inline static mkldnn::memory::desc GetMemDesc(const NDArray &arr, int ndim) {
  mkldnn::memory::dims dims(ndim);
  for (size_t i = 0; i < dims.size(); i++) dims[i] = arr.shape()[i];
  return mkldnn::memory::desc{dims, get_mkldnn_type(arr.dtype()),
                              mkldnn::memory::format::any};
}

inline static mkldnn::memory::desc GetMemDesc(const NDArray &arr) {
  return GetMemDesc(arr, arr.shape().ndim());
}

inline static mkldnn::memory::desc GetWeightDesc(const NDArray &arr,
                                                 int num_groups) {
  if (num_groups == 1) {
    return GetMemDesc(arr);
  } else {
    CHECK_EQ(arr.shape().ndim(), 4U);
    mkldnn::memory::dims tz = mkldnn::memory::dims{ num_groups,
      static_cast<int>(arr.shape()[0] / num_groups),
      static_cast<int>(arr.shape()[1]),
      static_cast<int>(arr.shape()[2]),
      static_cast<int>(arr.shape()[3])};
    return mkldnn::memory::desc{tz, get_mkldnn_type(arr.dtype()),
                                mkldnn::memory::format::any};
  }
}

typedef std::shared_ptr<mkldnn::memory> mkldnn_mem_ptr;
typedef std::shared_ptr<const mkldnn::memory> mkldnn_mem_const_ptr;

/*
 * This is to manage the temporary memory provided by MXNet for operators.
 * The temp memory is mainly used to keep the reordered data. In an operator, we
 * may need multiple pieces of memory for them. But MXNet can only provide
 * a single piece of memory. This class is to help break the temporary memory
 * from MXNet to store the reordered data.
 * The amount of temporary memory used in an operator depends on the layout of
 * input arrays and the operator. It's difficult to calculate it manually, so
 * the class also estimate the amount of memory automatically.
 */
class TmpMemMgr {
  // This points to the memory buffer where we can allocate temp memory.
  char *curr_mem;
  // The total size of the temp memory.
  size_t mem_size;
  // This contains the current available memory size.
  size_t curr_size;
  // This estimate the required temp memory size in an operator.
  size_t est_size;
  const size_t alignment = 4096;

 public:
  static TmpMemMgr *Get() {
    static thread_local TmpMemMgr mgr;
    return &mgr;
  }

  TmpMemMgr() {
    Reset();
    est_size = 0;
    mem_size = 0;
  }

  void Reset() {
    curr_mem = nullptr;
    curr_size = 0;
    // We don't reset est_size and mem_size because est_size contains the
    // estimated temp memory size from the last run and mem_size contains the
    // memroy size allocated in the last run.
  }

  void Init(const Resource &r) {
    // If the last time, if we estimate that we need more memory, we should the
    // larger memory size.
    mem_size = std::max(mem_size, est_size);
    if (mem_size > 0) {
      // Let's allocate some extra memory. If we don't use some of them all the time,
      // the OS won't physically allocate pages for them any way.
      this->curr_size = mem_size * 2;
      this->curr_mem = static_cast<char *>(r.get_host_space_internal(this->curr_size));
    }
    // reset est_size, so we can start to estimate the temp memory size.
    this->est_size = 0;
  }

  mkldnn::memory *Alloc(const mkldnn::memory::primitive_desc &pd);
};

class MKLDNNStream {
  std::vector<mkldnn::primitive> net;
  // Here we hold all memory related to the operators in the stream.
  std::vector<std::shared_ptr<const mkldnn::memory> > mem_holder;

 public:
  static MKLDNNStream *Get() {
    static thread_local MKLDNNStream stream;
    return &stream;
  }

  void RegisterPrim(const mkldnn::primitive &prim) { net.push_back(prim); }

  void RegisterMem(std::shared_ptr<const mkldnn::memory> mem) {
    mem_holder.push_back(mem);
  }

  bool HasOps() const {
    return !net.empty();
  }

  void Submit() {
    if (!net.empty())
      mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    net.clear();
    mem_holder.clear();
    TmpMemMgr::Get()->Reset();
  }
};

class MKLDNNOpSignature {
  std::vector<int> eles;
  uint64_t hash;

 public:
  MKLDNNOpSignature() {
    hash = 0;
  }

  explicit MKLDNNOpSignature(uint64_t hash) {
    this->hash = hash;
  }

  /*
   * We provide different methods to add signature to an op.
   * For operations, such as convolutin and fully connected, which determines
   * the optimal data layout for the op, we only need to use the shape and data
   * type to sign the op. For other operations, such as activation, which uses
   * whatever layout in the input array, we have to use the shape, the data type
   * and the layout to sign the op.
   */

  void AddSign(const mkldnn::memory &mem) {
    auto desc = mem.get_primitive_desc().desc();
    hash = hash * 2 + desc.data.format;
    eles.push_back(desc.data.format);
    hash = hash * 2 + desc.data.data_type;
    eles.push_back(desc.data.data_type);
    for (int i = 0; i < desc.data.ndims; i++) {
      hash = hash * 2 + desc.data.dims[i];
      eles.push_back(desc.data.dims[i]);
    }
  }

  void AddSign(const std::vector<NDArray> &arrs) {
    for (auto &arr : arrs) {
      AddSign(arr);
    }
  }

  void AddSign(const NDArray &arr) {
    if (arr.IsMKLDNNData()) {
      AddSign(*(arr.GetMKLDNNData()));
    } else {
      hash = hash * 2 + arr.dtype();
      eles.push_back(arr.dtype());
      AddSign(arr.shape());
    }
  }

  void AddSign(const TShape &shape) {
    for (size_t i = 0; i < shape.ndim(); i++) {
      hash = hash * 2 + shape[i];
      eles.push_back(shape[i]);
    }
  }

  void AddSign(int val) {
    hash = hash * 2 + val;
    eles.push_back(val);
  }

  bool operator==(const MKLDNNOpSignature &sign) const {
    if (hash != sign.hash)
      return false;
    if (eles.size() != sign.eles.size())
      return false;
    for (size_t i = 0; i < eles.size(); i++)
      if (eles[i] != sign.eles[i])
        return false;
    return true;
  }

  uint64_t GetHash() const {
    return hash;
  }
};

struct MKLDNNOpHash {
  size_t operator()(const MKLDNNOpSignature &sign) const {
    return sign.GetHash();
  }
};

template<typename ParamType>
class MKLDNNParamOpSign: public MKLDNNOpSignature {
  const ParamType param;

  static size_t hash(const ParamType &param) {
    std::hash<ParamType> fn;
    return fn(param);
  }

 public:
  explicit MKLDNNParamOpSign(const ParamType &_param): MKLDNNOpSignature(
      hash(_param)), param(_param) {
  }

  bool operator==(const MKLDNNParamOpSign<ParamType> &sign) const {
    const MKLDNNOpSignature &this_upper = *this;
    const MKLDNNOpSignature &other_upper = sign;
    return this_upper == other_upper && param == sign.param;
  }
};

enum OutDataOp {
  Noop,
  CopyBack,
  AddBack,
};

typedef std::pair<OutDataOp, mkldnn::memory *> mkldnn_output_t;

/*
 * These two functions try to create MKLDNN memory in an NDArray based on `req'.
 * The difference is that the first function can create MKLDNN memory with
 * special layouts in an NDArray, while the second one can only create MKLDNN
 * memory with default layouts.
 * If these two functions are used, we have to call CommitOutput to write
 * the output back to the output NDArray.
 */
mkldnn_output_t CreateMKLDNNMem(const NDArray &arr,
                                const mkldnn::memory::primitive_desc &desc,
                                OpReqType req);
mkldnn_output_t CreateMKLDNNWeightGrad(const NDArray &arr,
                                       const mkldnn::memory::primitive_desc &desc,
                                       OpReqType req);
/* This function has to be used with one of the functions above. */
void CommitOutput(const NDArray &arr, const mkldnn_output_t &res);

static inline void InvalidateOutputs(const std::vector<NDArray> &arrs,
                                     const std::vector<OpReqType> &reqs) {
  for (size_t i = 0; i < arrs.size(); i++) {
    if (reqs[i] == kWriteTo || reqs[i] == kNullOp) {
      const_cast<NDArray &>(arrs[i]).InvalidateMKLDNNData();
    }
  }
}

const mkldnn::memory *GetWeights(const NDArray &arr,
                                 const mkldnn::memory::primitive_desc &target_pd,
                                 int num_groups);

mkldnn_memory_format_t GetDefaultFormat(mkldnn::memory::desc desc);
mkldnn_memory_format_t GetDefaultFormat(int num_dims);
mkldnn::memory::primitive_desc GetPrimitiveDesc(mkldnn::memory::primitive_desc pd,
                                                mkldnn_memory_format_t format);

void FallBackCompute(FCompute fn, const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs);

/*
 * This class is used to check the correctness of MKLDNN operators.
 */
class OpCheck {
  std::vector<mxnet::NDArray> inputs;
  std::vector<mxnet::NDArray> outputs;
  bool backward;
  size_t num_checks;

 public:
  OpCheck(bool backward, size_t num_checks) {
    this->backward = backward;
    this->num_checks = num_checks;
  }

  void Init(const std::vector<mxnet::NDArray> &inputs_,
          const std::vector<mxnet::NDArray> &outputs_);

  void Run(mxnet::FCompute fn, const nnvm::NodeAttrs &attrs,
           const mxnet::OpContext &ctx,
           const std::vector<mxnet::NDArray> &inputs_,
           const std::vector<mxnet::OpReqType> &req,
           const std::vector<mxnet::NDArray> &outputs_);
};

#define MKLDNN_OPCHECK_INIT(backward, num_checks, inputs, outputs)  \
    static bool debug = dmlc::GetEnv("MXNET_MKLDNN_DEBUG", false);  \
    OpCheck check(backward, num_checks);                            \
    if (debug) check.Init(inputs, outputs);

#define MKLDNN_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs)    \
    if (debug) check.Run(fn, attrs, ctx, inputs, req, outputs);

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BASE_INL_H_
