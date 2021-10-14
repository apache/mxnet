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
 * \file dnnl_base-inl.h
 * \brief
 * \author young.jin.kim@intel.com
 *         ashok.emani@intel.com
 *         deepthi.karkada@intel.com
 *         louis.feng@intel.com
 *         adam.d.straw@intel.com
 *         zhengda1936@gmail.com
 *
 *******************************************************************************/

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_BASE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_BASE_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnnl.hpp"
#include "mxnet/graph_attr_types.h"
#include "mxnet/ndarray.h"
#include "mxnet/op_attr_types.h"
#include "mxnet/resource.h"

#define DNNL_REAL_TYPE_SWITCH(type, DType, ...)   \
  switch (type) {                                 \
    case mshadow::kFloat32: {                     \
      typedef float DType;                        \
      { __VA_ARGS__ }                             \
    } break;                                      \
    case mshadow::kBfloat16: {                    \
      typedef mshadow::bfloat::bf16_t DType;      \
      { __VA_ARGS__ }                             \
    } break;                                      \
    default:                                      \
      LOG(FATAL) << "Unknown type enum " << type; \
  }

namespace mxnet {

// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine {
 public:
  static CpuEngine* Get() {
    // I's thread-safe in C++11.
    // ensure same dnnl engine is used across threads
    static CpuEngine myInstance;
    return &myInstance;
  }
  CpuEngine(CpuEngine const&) = delete;             // Copy construct
  CpuEngine(CpuEngine&&)      = delete;             // Move construct
  CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
  CpuEngine& operator=(CpuEngine&&) = delete;       // Move assign

  dnnl::engine& get_engine() {
    return _cpu_engine;
  }

 protected:
  CpuEngine() : _cpu_engine(dnnl::engine::kind::cpu, 0) {}
  ~CpuEngine() {}

 private:
  dnnl::engine _cpu_engine;
};

// type enumerator
template <typename T>
struct data_type_enum {};

template <>
struct data_type_enum<float> {
  enum { type = static_cast<unsigned int>(dnnl::memory::data_type::f32) };
};

template <>
struct data_type_enum<mshadow::bfloat::bf16_t> {
  enum { type = static_cast<unsigned int>(dnnl::memory::data_type::bf16) };
};

template <>
struct data_type_enum<int32_t> {
  enum { type = static_cast<unsigned int>(dnnl::memory::data_type::s32) };
};

template <>
struct data_type_enum<int8_t> {
  enum { type = static_cast<unsigned int>(dnnl::memory::data_type::s8) };
};

template <>
struct data_type_enum<uint8_t> {
  enum { type = static_cast<unsigned int>(dnnl::memory::data_type::u8) };
};

static inline bool SupportDNNLArray(int dtype, const mxnet::TShape& shape) {
  int ndim     = shape.ndim();
  bool support = ndim == 1 || ndim == 2 || ndim == 4;
  support      = support &&
            (dtype == mshadow::kFloat32 || dtype == mshadow::kInt32 || dtype == mshadow::kInt8 ||
             dtype == mshadow::kUint8 || dtype == mshadow::kBfloat16);
  return support;
}

static inline bool SupportStorageDNNL(int stype) {
  return stype == kDefaultStorage;
}

static inline bool SupportDNNL(int dtype, const mxnet::TShape& shape) {
  int ndim = shape.ndim();
  if (ndim == 0 || shape.Size() == 0) {
    // DNNL currently does not support 0-dim Tensor and 0-size Tensor
    return false;
  }
  return (dtype == mshadow::kFloat32 || dtype == mshadow::kBfloat16) &&
         (ndim == 1 || ndim == 2 || ndim == 4);
}

static inline bool IsDNNLType(int dtype) {
  return dtype == mshadow::kFloat32 || dtype == mshadow::kInt8 || dtype == mshadow::kUint8 ||
         dtype == mshadow::kBfloat16;
}

static inline bool SupportDNNL(const NDArray& input) {
  return SupportDNNL(input.dtype(), input.shape()) && SupportStorageDNNL(input.storage_type());
}

static inline bool DNNLEnvSet() {
  static bool is_dnnl_enabled = dmlc::GetEnv("MXNET_ONEDNN_ENABLED", true);
  return is_dnnl_enabled;
}

static inline int GetDNNLCacheSize() {
  static int dnnl_cache_size = dmlc::GetEnv("MXNET_ONEDNN_CACHE_NUM", -1);
  return dnnl_cache_size;
}

// TODO(alex): (MXNET-1075) Will remove env variable and calculate cache size during runtime
template <typename S, typename I, typename H>
static typename std::unordered_map<S, I, H>::iterator AddToCache(std::unordered_map<S, I, H>* cache,
                                                                 const S& key,
                                                                 const I& item) {
  int dnnl_cache_size = GetDNNLCacheSize();
  if (dnnl_cache_size != -1 && static_cast<int>(cache->size()) > dnnl_cache_size)
    cache->erase(cache->begin());
  auto ins_return = cache->insert(std::pair<S, I>(key, item));
  CHECK(ins_return.second);
  return ins_return.first;
}

/*
 * This is to align address to a certain alignment.
 */
void* AlignMem(void* mem, size_t size, size_t alignment, size_t* space);

namespace op {
struct ActivationParam;
struct LeakyReLUParam;
struct ConvolutionParam;
struct DeconvolutionParam;
struct SoftmaxParam;
struct SoftmaxOutputParam;
struct ReshapeParam;
struct LayerNormParam;
struct NumpyReduceAxesParam;
bool SupportDNNLAct(const ActivationParam& param);
bool SupportDNNLAct(const ActivationParam& param, const NDArray& input);
bool SupportDNNLLeakyRelu(const LeakyReLUParam& param);
bool SupportDNNLLeakyRelu(const LeakyReLUParam& param, const NDArray& input);
bool SupportQuantizedDNNLAct(const ActivationParam& param);
bool SupportDNNLConv(const ConvolutionParam& params, const NDArray& input);
bool SupportDNNLDeconv(const DeconvolutionParam& params, const NDArray& input);
bool SupportDNNLSoftmax(const SoftmaxParam& param, const NDArray& input, const NDArray& output);
bool SupportDNNLLogSoftmax(const SoftmaxParam& param, const NDArray& input, const NDArray& output);
bool SupportDNNLSoftmaxOutput(const SoftmaxOutputParam& param);
bool SupportDNNLTranspose(const NDArray& data);
bool SupportDNNLBatchDot(const std::vector<NDArray>& inputs, const NDArray& output);
bool SupportDNNLLayerNorm(const LayerNormParam& param, const std::vector<NDArray>& inputs);
bool SupportDNNLReshape(const NDArray& input, const NDArray& output);
bool SupportDNNLReduce(const NDArray& input,
                       const NDArray& output,
                       const NumpyReduceAxesParam& param);
}  // namespace op

static int GetTypeSize(int dtype) {
  int size = -1;
  MSHADOW_TYPE_SWITCH(dtype, DType, { size = sizeof(DType); });
  return size;
}

static inline size_t GetArraySize(const NDArray& arr) {
  if (arr.IsDNNLData()) {
    return arr.GetDNNLData()->get_desc().get_size();
  }
  return arr.shape().Size() * GetTypeSize(arr.dtype());
}

static inline dnnl::memory::data_type get_dnnl_type(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return dnnl::memory::data_type::f32;
    case mshadow::kBfloat16:
      return dnnl::memory::data_type::bf16;
    case mshadow::kInt32:
      return dnnl::memory::data_type::s32;
    case mshadow::kInt8:
      return dnnl::memory::data_type::s8;
    case mshadow::kUint8:
      return dnnl::memory::data_type::u8;
    default:
      LOG(FATAL) << "unknown type for DNNL :" << static_cast<int>(dtype);
      return dnnl::memory::data_type::undef;
  }
}

template <typename T>
static inline dnnl::memory::data_type get_dnnl_type() {
  return static_cast<dnnl::memory::data_type>(data_type_enum<T>::type);
}

static inline dnnl_data_type_t get_dnnl_type_t(int dtype) {
  return static_cast<dnnl_data_type_t>(get_dnnl_type(dtype));
}

template <typename T>
static inline dnnl_data_type_t get_dnnl_type_t() {
  return static_cast<dnnl_data_type_t>(data_type_enum<T>::type);
}

static inline int get_mxnet_type(dnnl_data_type_t dtype) {
  auto dnnl_dtype = static_cast<dnnl::memory::data_type>(dtype);
  switch (dnnl_dtype) {
    case dnnl::memory::data_type::f32:
      return mshadow::kFloat32;
    case dnnl::memory::data_type::bf16:
      return mshadow::kBfloat16;
    case dnnl::memory::data_type::s32:
      return mshadow::kInt32;
    case dnnl::memory::data_type::s8:
      return mshadow::kInt8;
    case dnnl::memory::data_type::u8:
      return mshadow::kUint8;
    default:
      LOG(FATAL) << "unknown DNNL type";
      return mshadow::kFloat32;
  }
}

static inline size_t GetMemDescSize(const dnnl::memory::desc& md) {
  if (md.data.ndims == 0)
    return 0;

  size_t ret = 1;
  for (int i = 0; i < md.data.ndims; i++) {
    ret *= md.data.dims[i];
  }

  ret *= mshadow::mshadow_sizeof(get_mxnet_type(md.data.data_type));
  return ret;
}

inline static dnnl::memory::desc GetMemDesc(const NDArray& arr, int dtype = -1) {
  int ndim = arr.shape().ndim();
  dnnl::memory::dims dims(ndim);
  dtype = (dtype == -1) ? arr.dtype() : dtype;
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = arr.shape()[i];
  return dnnl::memory::desc{dims, get_dnnl_type(dtype), dnnl::memory::format_tag::any};
}

inline static bool ChooseBRGEMMImpl(const dnnl::memory::dims& weight_dims, size_t batch_size) {
  // Conditions based on measurement results done on CLX8280
  // https://github.com/apache/incubator-mxnet/pull/20533
  return weight_dims[0] >= 1024 && weight_dims[1] >= 1024 && batch_size >= 16384 &&
         weight_dims[0] % 64 == 0 && weight_dims[1] % 64 == 0;
}

inline static dnnl::memory::desc GetFCWeightDesc(const NDArray& arr,
                                                 size_t batch_size,
                                                 int dtype = -1) {
  int ndim = arr.shape().ndim();
  dnnl::memory::dims dims(ndim);
  dtype = (dtype == -1) ? arr.dtype() : dtype;
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = arr.shape()[i];
  auto format = dnnl::memory::format_tag::any;
  // for batch 256 alexnet benchmark test
  const bool force_fc_ab_format = dmlc::GetEnv("MXNET_ONEDNN_FORCE_FC_AB_FORMAT", false);
  if (dims.size() == 2) {
    if (force_fc_ab_format || !ChooseBRGEMMImpl(dims, batch_size)) {
      format = dnnl::memory::format_tag::ab;
    }
  }

  return dnnl::memory::desc{dims, get_dnnl_type(dtype), format};
}

inline static dnnl::memory::desc GetWeightDesc(const NDArray& arr,
                                               int num_groups,
                                               bool quantized = false) {
  int dtype = quantized ? mshadow::kInt8 : arr.dtype();
  if (num_groups == 1) {
    return GetMemDesc(arr, dtype);
  } else {
    const auto ndim = arr.shape().ndim();
    CHECK((ndim == 3) || (ndim == 4) || (ndim == 5))
        << "DNNL weight currently supports 3d or 4d or 5d layout";
    auto tz = dnnl::memory::dims{0};
    int N = 0, C = 1, H = 2, W = 3;
    int D = -1;
    if (ndim == 5) {
      D = 2;
      H = 3;
      W = 4;
    }
    switch (ndim) {
      case 3:
        tz = dnnl::memory::dims{
            num_groups, arr.shape()[N] / num_groups, arr.shape()[C], arr.shape()[H]};
        break;
      case 4:
        tz = dnnl::memory::dims{num_groups,
                                arr.shape()[N] / num_groups,
                                arr.shape()[C],
                                arr.shape()[H],
                                arr.shape()[W]};
        break;
      case 5:
        tz = dnnl::memory::dims{num_groups,
                                arr.shape()[N] / num_groups,
                                arr.shape()[C],
                                arr.shape()[D],
                                arr.shape()[H],
                                arr.shape()[W]};
    }
    return dnnl::memory::desc{tz, get_dnnl_type(dtype), dnnl::memory::format_tag::any};
  }
}

inline static bool CheckDNNLInputArrayIsView(const std::vector<NDArray>& inputs) {
  for (const auto& in : inputs) {
    if (in.IsView() && in.IsDNNLData()) {
      return true;
    }
  }
  return false;
}

typedef std::shared_ptr<dnnl::memory> dnnl_mem_ptr;
typedef std::shared_ptr<const dnnl::memory> dnnl_mem_const_ptr;

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
  char* curr_mem;
  // The total size of the temp memory.
  size_t mem_size;
  // This contains the current available memory size.
  size_t curr_size;
  // This estimate the required temp memory size in an operator.
  size_t est_size;
  const size_t alignment = kDNNLAlign;

 public:
  static TmpMemMgr* Get() {
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local TmpMemMgr mgr;
#else
    static MX_THREAD_LOCAL TmpMemMgr mgr;
#endif
    return &mgr;
  }

  TmpMemMgr() {
    Reset();
    est_size = 0;
    mem_size = 0;
  }

  void Reset() {
    curr_mem  = nullptr;
    curr_size = 0;
    // We don't reset est_size and mem_size because est_size contains the
    // estimated temp memory size from the last run and mem_size contains the
    // memroy size allocated in the last run.
  }

  void Init(const Resource& r) {
    // If the last time, if we estimate that we need more memory, we should the
    // larger memory size.
    mem_size = std::max(mem_size, est_size);
    if (mem_size > 0) {
      // Let's allocate some extra memory. If we don't use some of them all the time,
      // the OS won't physically allocate pages for them any way.
      this->curr_size = mem_size * 2;
      this->curr_mem  = static_cast<char*>(r.get_host_space_internal(this->curr_size));
    }
    // reset est_size, so we can start to estimate the temp memory size.
    this->est_size = 0;
  }

  dnnl::memory* Alloc(const dnnl::memory::desc& md);
};

typedef std::unordered_map<int, dnnl::memory> dnnl_args_map_t;
class DNNLStream {
  std::vector<std::pair<dnnl::primitive, dnnl_args_map_t> > net_prim_args;
  // Here we hold all memory related to the operators in the stream.
  std::vector<std::shared_ptr<const dnnl::memory> > mem_holder;
  dnnl::stream s;

 public:
  static DNNLStream* Get();

  DNNLStream() : s(CpuEngine::Get()->get_engine()) {}

  void RegisterPrimArgs(const dnnl::primitive& prim, const dnnl_args_map_t& args) {
    net_prim_args.emplace_back(prim, args);
  }

  void RegisterMem(std::shared_ptr<const dnnl::memory> mem) {
    mem_holder.push_back(mem);
  }

  bool HasOps() const {
    return !net_prim_args.empty();
  }

  /*
   * After submitting dnnl operations for execution, we need to
   * clean up memory held by the stream. However, sometimes users
   * might want to separate dnnl execution and memory cleanup.
   */
  void Submit(bool cleanup = true) {
    if (!net_prim_args.empty()) {
      for (auto& v : net_prim_args) {
        v.first.execute(s, v.second);
      }
      net_prim_args.clear();
    }
    if (cleanup)
      Cleanup();
  }

  void Cleanup() {
    mem_holder.clear();
    TmpMemMgr::Get()->Reset();
  }
};

enum OutDataOp {
  Noop,
  CopyBack,
  AddBack,
};

typedef std::pair<OutDataOp, dnnl::memory*> dnnl_output_t;
void DNNLMemoryCopy(const dnnl::memory& mem, const dnnl::memory* this_mem);

/*
 * Here we want to get DNNL memory whose desc is exactly the same as
 * the given one. operator== can't guarantee that. == can return true even if
 * the formats are different. I need to double check its format.
 */
static inline dnnl::memory* GetDNNLExact(const dnnl::memory* mem, const dnnl::memory::desc& desc) {
  dnnl::memory::desc src_desc = mem->get_desc();
  if (desc == src_desc) {
    return const_cast<dnnl::memory*>(mem);
  } else {
    std::shared_ptr<dnnl::memory> ret(
        new dnnl::memory(desc, CpuEngine::Get()->get_engine(), mem->get_data_handle()));
    DNNLStream::Get()->RegisterMem(ret);
    return ret.get();
  }
}

/*
 * These two functions try to create DNNL memory in an NDArray based on `req'.
 * The difference is that the first function can create DNNL memory with
 * special layouts in an NDArray, while the second one can only create DNNL
 * memory with default layouts.
 * Also an optional in_arr parameter can be passed in the first function with
 * the kWriteInPlace req to validate if dnnl can support write in place;
 * otherwise new memory will be written to an copied back onto out_arr.
 * If these two functions are used, we have to call CommitOutput to write
 * the output back to the output NDArray.
 */
dnnl_output_t CreateDNNLMem(const NDArray& out_arr,
                            const dnnl::memory::desc& desc,
                            OpReqType req,
                            const NDArray* in_arr = nullptr);
dnnl_output_t CreateDNNLWeightGrad(const NDArray& out_arr,
                                   const dnnl::memory::desc& desc,
                                   OpReqType req);
/* This function has to be used with one of the functions above. */
void CommitOutput(const NDArray& arr, const dnnl_output_t& res);

const dnnl::memory* GetWeights(const NDArray& arr, int num_groups);

const dnnl::memory* GetWeights(const NDArray& arr,
                               const dnnl::memory::desc& target_md,
                               int num_groups);

bool IsDefaultFormat(const dnnl::memory::desc& desc);
bool IsDNNL(const dnnl::memory::desc& desc);

dnnl_format_tag_t GetDefaultFormat(const dnnl::memory::desc& md);
dnnl_format_tag_t GetDefaultFormat(int num_dims);
dnnl::memory::desc GetDesc(const dnnl::memory::desc& md, const dnnl_format_tag_t& format);

inline bool same_shape(const mxnet::TShape& shape, const dnnl_dims_t dims, int ndims) {
  if (shape.ndim() != ndims)
    return false;
  for (int i = 0; i < ndims; i++)
    if (shape[i] != dims[i])
      return false;
  return true;
}

inline bool same_shape(const dnnl::memory::desc& desc1, const dnnl::memory::desc& desc2) {
  if (desc1.data.ndims != desc2.data.ndims)
    return false;
  for (int i = 0; i < desc1.data.ndims; i++)
    if (desc1.data.dims[i] != desc2.data.dims[i])
      return false;
  return true;
}

inline bool same_shape(const mxnet::TShape& shape, int dtype, const dnnl::memory::desc& desc) {
  return same_shape(shape, desc.data.dims, desc.data.ndims) &&
         get_dnnl_type(dtype) == desc.data.data_type;
}

/*
 * There is a large overhead of getting dnnl::memory::desc from
 * dnnl::memory. This class is created to cache the metadata of dnnl memory
 * to provide a much more lightweight method to access them.
 */
class DNNLMemory {
  std::shared_ptr<dnnl::memory> mem;
  dnnl::memory::desc desc;
  size_t size;  // The number of bytes.

 public:
  DNNLMemory(dnnl::memory::desc md, void* addr) : desc(md) {
    mem.reset(new dnnl::memory(md, CpuEngine::Get()->get_engine(), addr));
    size = desc.get_size();
  }

  explicit DNNLMemory(std::shared_ptr<dnnl::memory> mem) : desc(mem->get_desc()) {
    this->mem = mem;
    size      = desc.get_size();
  }

  void SetDataHandle(void* handle) {
    mem->set_data_handle(handle);
  }

  void* GetDataHandle() const {
    return mem->get_data_handle();
  }

  std::shared_ptr<dnnl::memory> GetMem() const {
    return mem;
  }

  dnnl::memory* GetRaw() const {
    return mem.get();
  }

  size_t GetSize() const {
    return size;
  }

  dnnl::memory::desc GetDesc() const {
    return mem->get_desc();
  }

  dnnl::memory::desc GetDesc(
      dnnl_format_tag_t format,
      dnnl::memory::data_type data_type = dnnl::memory::data_type::undef) const {
    dnnl::memory::dims dims(desc.data.dims, desc.data.dims + desc.data.ndims);
    dnnl::memory::data_type cpp_type =
        (data_type == dnnl::memory::data_type::undef) ?
            static_cast<dnnl::memory::data_type>(desc.data.data_type) :
            data_type;
    dnnl::memory::desc data_md(dims, cpp_type, static_cast<dnnl::memory::format_tag>(format));
    return data_md;
  }

  dnnl_format_tag_t GetDefaultFormat() const {
    return mxnet::GetDefaultFormat(desc);
  }

  bool IsDNNL() const {
    return mxnet::IsDNNL(desc);
  }

  bool SameFormat(dnnl::memory::desc md) const {
    return mem->get_desc() == md;
  }

  bool SameFormat(const mxnet::TShape& shape, int dtype) const {
    return same_shape(shape, dtype, desc);
  }

  void ReorderTo(dnnl::memory* other) const {
    dnnl::stream s(CpuEngine::Get()->get_engine());
    dnnl::reorder(*mem, *other).execute(s, *mem, *other);
  }
};

// reorder dnnl src to dst format dtype
void ReorderTo(const dnnl::memory* src, const dnnl::memory* dst);

template <typename Compute, typename AttrState>
void FallBackCompute(Compute fn,
                     const AttrState& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs);

/*
 * This class is used to check the correctness of DNNL operators.
 */
class OpCheck {
  std::vector<mxnet::NDArray> inputs;
  std::vector<mxnet::NDArray> outputs;
  bool backward;
  size_t num_checks;

 public:
  OpCheck(bool backward, size_t num_checks) {
    this->backward   = backward;
    this->num_checks = num_checks;
  }

  void Init(const std::vector<mxnet::NDArray>& inputs_,
            const std::vector<mxnet::NDArray>& outputs_);

  void Run(mxnet::FCompute fn,
           const nnvm::NodeAttrs& attrs,
           const mxnet::OpContext& ctx,
           const std::vector<mxnet::NDArray>& inputs_,
           const std::vector<mxnet::OpReqType>& req,
           const std::vector<mxnet::NDArray>& outputs_);

  void CopyResult(const std::vector<mxnet::NDArray>& outputs_, const std::vector<size_t>& indice);
};

bool DNNLStorageType(const nnvm::NodeAttrs& attrs,
                     const int dev_mask,
                     bool support_dnnl,
                     DispatchMode* dispatch_mode,
                     std::vector<int>* in_attrs,
                     std::vector<int>* out_attrs);

#define DNNL_OPCHECK_INIT(backward, num_checks, inputs, outputs) \
  static bool debug = dmlc::GetEnv("MXNET_ONEDNN_DEBUG", false); \
  OpCheck check(backward, num_checks);                           \
  if (debug)                                                     \
    check.Init(inputs, outputs);

#define DNNL_OPCHECK_RUN(fn, attrs, ctx, inputs, req, outputs) \
  if (debug)                                                   \
    check.Run(fn, attrs, ctx, inputs, req, outputs);
#define DNNL_OPCHECK_COPY_RESULT(outputs, indice) \
  if (debug)                                      \
    check.CopyResult(outputs, indice);

struct DNNLPostEltwiseParam {
  dnnl::algorithm alg = dnnl::algorithm::undef;
  float scale         = 1.f;
  float alpha         = 0.f;
  float beta          = 1.f;
};

void DNNLRun(mxnet::FComputeEx fn,
             const nnvm::NodeAttrs& attrs,
             const mxnet::OpContext& ctx,
             const std::vector<mxnet::NDArray>& inputs_,
             const std::vector<mxnet::OpReqType>& req,
             const std::vector<mxnet::NDArray>& outputs_);

using FComputeExUnary = std::function<void(const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const NDArray& input,
                                           const OpReqType& req,
                                           const NDArray& output)>;

void DNNLRun(FComputeExUnary fn,
             const nnvm::NodeAttrs& attrs,
             const mxnet::OpContext& ctx,
             const mxnet::NDArray& inputs_,
             const mxnet::OpReqType& req,
             const mxnet::NDArray& outputs_);

}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_BASE_INL_H_
