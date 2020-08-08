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

/*!
 * Copyright (c) 2019 by Contributors
 * \file lib_api.h
 * \brief APIs to interact with libraries
 * This API specifies function prototypes to
 * register custom ops, partitioner, and passes
 * for library authors
 * See example/extension/lib_custom_op/README.md
 * See example/extension/lib_subgraph/README.md
 * See example/extension/lib_pass/README.md
 */

#ifndef MXNET_LIB_API_H_
#define MXNET_LIB_API_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <functional>
#include <random>

#if defined(__NVCC__)
  #include <curand_kernel.h>
#endif

/* Make sure to update the version number everytime you make changes */
#define MX_LIBRARY_VERSION 8

/*!
 * \brief For loading multiple custom op libraries in Linux, exporting same symbol multiple
 * times may lead to undefined behaviour, so we need to set symbol visibility to hidden
 * see https://labjack.com/news/simple-cpp-symbol-visibility-demo for details
 */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  #define PRIVATE_SYMBOL
#else
  #define PRIVATE_SYMBOL  __attribute__ ((visibility ("hidden")))
#endif

/*
 * Import from DLPack https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
 */
#ifndef DLPACK_VERSION
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current version of dlpack */
#define DLPACK_VERSION 020

/*! \brief DLPACK_DLL prefix for windows */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
  #endif
  /*!
   * \brief The device type in DLContext.
   */
  typedef enum {
    /*! \brief CPU device */
    kDLCPU = 1,
    /*! \brief CUDA GPU device */
    kDLGPU = 2,
    /*!
     * \brief Pinned CUDA GPU device by cudaMallocHost
     * \note kDLCPUPinned = kDLCPU | kDLGPU
     */
    kDLCPUPinned = 3,
    /*! \brief OpenCL devices. */
    kDLOpenCL = 4,
    /*! \brief Vulkan buffer for next generation graphics. */
    kDLVulkan = 7,
    /*! \brief Metal for Apple GPU. */
    kDLMetal = 8,
    /*! \brief Verilog simulator buffer */
    kDLVPI = 9,
    /*! \brief ROCm GPUs for AMD GPUs */
    kDLROCM = 10,
    /*!
     * \brief Reserved extension device type,
     * used for quickly test extension device
     * The semantics can differ depending on the implementation.
     */
    kDLExtDev = 12,
  } DLDeviceType;

  /*!
   * \brief A Device context for Tensor and operator.
   */
  typedef struct {
    /*! \brief The device type used in the device. */
    DLDeviceType device_type;
    /*! \brief The device index */
    int device_id;
  } DLContext;

  /*!
   * \brief The type code options DLDataType.
   */
  typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
  } DLDataTypeCode;

  /*!
   * \brief The data type the tensor can hold.
   *
   *  Examples
   *   - float: type_code = 2, bits = 32, lanes=1
   *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
   *   - int8: type_code = 0, bits = 8, lanes=1
   */
  typedef struct {
    /*!
     * \brief Type code of base types.
     * We keep it uint8_t instead of DLDataTypeCode for minimal memory
     * footprint, but the value should be one of DLDataTypeCode enum values.
     * */
    uint8_t code;
    /*!
     * \brief Number of bits, common choices are 8, 16, 32.
     */
    uint8_t bits;
    /*! \brief Number of lanes in the type, used for vector types. */
    uint16_t lanes;
  } DLDataType;

  /*!
   * \brief Plain C Tensor object, does not manage memory.
   */
  typedef struct {
    /*!
     * \brief The opaque data pointer points to the allocated data. This will be
     * CUDA device pointer or cl_mem handle in OpenCL. This pointer is always
     * aligns to 256 bytes as in CUDA.
     *
     * For given DLTensor, the size of memory required to store the contents of
     * data is calculated as follows:
     *
     * \code{.c}
     * static inline size_t GetDataSize(const DLTensor* t) {
     *   size_t size = 1;
     *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
     *     size *= t->shape[i];
     *   }
     *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
     *   return size;
     * }
     * \endcode
     */
    void* data;
    /*! \brief The device context of the tensor */
    DLContext ctx;
    /*! \brief Number of dimensions */
    int ndim;
    /*! \brief The data type of the pointer*/
    DLDataType dtype;
    /*! \brief The shape of the tensor */
    int64_t* shape;
    /*!
     * \brief strides of the tensor (in number of elements, not bytes)
     *  can be nullptr, indicating tensor is compact and row-majored.
     */
    int64_t* strides;
    /*! \brief The offset in bytes to the beginning pointer to data */
    uint64_t byte_offset;
  } DLTensor;
#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif

namespace mxnet {
namespace ext {

/*!
 * \brief Tensor data type, consistent with mshadow data type
 */
enum MXDType {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kUNSET = 100,
};

/*
 * MXTensor storage type.
 */
enum MXStorageType {
  // dense
  kDefaultStorage = 0,
  // row sparse
  kRowSparseStorage = 1,
  // csr
  kCSRStorage = 2,
};

/*!
 * \brief Context info passing from MXNet OpContext
 * dev_type is string repr of supported context, currently only "cpu" and "gpu"
 * dev_id is the device index where the tensor locates
 */
struct MXContext {
  MXContext() : dev_type("error"), dev_id(-1) {}
  explicit MXContext(std::string dev_type_, int dev_id_)
    : dev_type(dev_type_), dev_id(dev_id_) {}
  explicit MXContext(const char* dev_type_, int dev_id_)
    : dev_type(dev_type_), dev_id(dev_id_) {}
  static MXContext CPU() { return MXContext("cpu", 0); }
  static MXContext GPU() { return MXContext("gpu", 0); }
  static MXContext CPU(int dev_id) { return MXContext("cpu", dev_id); }
  static MXContext GPU(int dev_id) { return MXContext("gpu", dev_id); }

  std::string dev_type;
  int dev_id;
};

enum MXReturnValue {
  MX_FAIL = 0,
  MX_SUCCESS = 1,
};

// For sparse tensors, read/write the data from NDarray via pointers.
struct MXSparse {
  // Pointer to data.
  void *data{nullptr};
  // length of (non-zero) data.
  int64_t data_len;

  // To store aux data for sparse.
  // For CSR, indices stores the col index of non-zero elements.
  // For row sparse, indices store row index of rows which have non-zero elements.
  int64_t* indices;
  int64_t indices_len;

  // For CSR, indptr gives the start and end index of data for each row.
  // For row sparse, indptr is not used.
  int64_t* indptr = nullptr;
  int64_t indptr_len;

  void set(void *data_ptr, const int64_t* dims, int ndims, void *idx,
          int64_t num_idx, void *idx_ptr = nullptr, int64_t num_idx_ptr = 0) {
    data = data_ptr;
    // If CSR, num of non-zero elemets is num_idx,
    // If row sparse, num of elements is num_idx * width.
    data_len = num_idx;
    if (!idx_ptr) {
      for (int i = 1; i < ndims; ++i)
         data_len *= dims[i];
    }

    indices = reinterpret_cast<int64_t*>(idx);
    indices_len = num_idx;

    if (idx_ptr) {
      indptr = reinterpret_cast<int64_t*>(idx_ptr);
      indptr_len = num_idx_ptr;
    }
  }
};

/*!
 * \brief Tensor data structure used by custom operator
 */
struct MXTensor {
  MXTensor() : data_ptr(nullptr), dtype(kUNSET), verID(0), stype(kDefaultStorage) {}
  MXTensor(const MXTensor& oth) : data_ptr(oth.data_ptr), shape(oth.shape),
    dtype(oth.dtype), verID(oth.verID), ctx(oth.ctx), stype(oth.stype) {
    setDLTensor();
  }
  MXTensor(void *data_ptr, const std::vector<int64_t> &shape, MXDType dtype,
           size_t vID, MXContext mx_ctx, MXStorageType stype = kDefaultStorage)
  : data_ptr(data_ptr), shape(shape), dtype(dtype), verID(vID), ctx(mx_ctx), stype(stype) {
    setDLTensor();
  }

  /*! \brief populate internal tensor fields */
  void setTensor(void *dptr, MXDType type, const int64_t* dims, int ndims,
                 size_t vID, MXContext mx_ctx, MXStorageType storage_type) {
    data_ptr = dptr; dtype = type; verID = vID; ctx = mx_ctx; stype = storage_type;
    shape.clear();
    for (int j = 0; j < ndims; j++) {
      shape.push_back(dims[j]);
    }
    setDLTensor();
  }

  /*! \brief populate DLTensor fields */
  void setDLTensor() {
    dltensor.data = data_ptr;
    dltensor.ndim = shape.size();
    dltensor.shape = const_cast<int64_t*>(shape.data());
    dltensor.strides = nullptr;
    dltensor.byte_offset = 0;
    dltensor.dtype.lanes = 1;
    dltensor.ctx.device_id = ctx.dev_id;
    if (ctx.dev_type == "cpu")
      dltensor.ctx.device_type = kDLCPU;
    else if (ctx.dev_type == "gpu")
      dltensor.ctx.device_type = kDLGPU;
    else if (ctx.dev_type == "opencl")
      dltensor.ctx.device_type = kDLOpenCL;
    else if (ctx.dev_type == "vulcan")
      dltensor.ctx.device_type = kDLVulkan;
    else if (ctx.dev_type == "metal")
      dltensor.ctx.device_type = kDLMetal;
    else if (ctx.dev_type == "vpi")
      dltensor.ctx.device_type = kDLVPI;
    else if (ctx.dev_type == "rocm")
      dltensor.ctx.device_type = kDLROCM;
    else
      dltensor.ctx.device_type = kDLExtDev;
    switch (dtype) {
    case kFloat32:
      dltensor.dtype.code = kDLFloat;
      dltensor.dtype.bits = 32;
      break;
    case kFloat64:
      dltensor.dtype.code = kDLFloat;
      dltensor.dtype.bits = 64;
      break;
    case kFloat16:
      dltensor.dtype.code = kDLFloat;
      dltensor.dtype.bits = 16;
      break;
    case kUint8:
      dltensor.dtype.code = kDLUInt;
      dltensor.dtype.bits = 8;
      break;
    case kInt32:
      dltensor.dtype.code = kDLInt;
      dltensor.dtype.bits = 32;
      break;
    case kInt8:
      dltensor.dtype.code = kDLInt;
      dltensor.dtype.bits = 8;
      break;
    case kInt64:
      dltensor.dtype.code = kDLInt;
      dltensor.dtype.bits = 64;
      break;
    default:
      dltensor.dtype.code = 0;
      dltensor.dtype.bits = 0;
      throw std::runtime_error("Error! Invalid dtype flag: "
                               + std::to_string(static_cast<int>(dtype))
                               + " when constructing MXTensor");
    }
  }

  /*! \brief helper function to cast data pointer */
  template<typename data_type>
  inline data_type* data() {
    return reinterpret_cast<data_type*>(data_ptr);
  }

  /*! \brief helper function to get data size */
  inline int64_t size() const {
    int64_t size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }
    return size;
  }

  /*! \brief helper function to compare two MXTensors */
  inline bool isSame(const MXTensor &oth) const {
    return data_ptr == oth.data_ptr &&
           dtype == oth.dtype &&
           verID == oth.verID &&
           ctx.dev_type == oth.ctx.dev_type &&
           ctx.dev_id == oth.ctx.dev_id &&
           shape == oth.shape &&
           stype == oth.stype;
  }

  // For dense, data_ptr points to 1D flattened tensor data
  // For sparse, data_ptr points to MXSparse
  void *data_ptr;

  // shape is in [2,3,4] format to represent high-dim tensor
  std::vector<int64_t> shape;

  // type can only be MXDType enum types
  MXDType dtype;

  // version number updated if the tensor has changed since the last use by custom op
  size_t verID;

  // context of MXTensor representing which device the tensor data is located
  MXContext ctx;

  // corresponding DLTensor repr of MXTensor
  // easy way to reuse functions taking DLTensor
  DLTensor dltensor;

  // storage type
  MXStorageType stype;
};

/*! \brief resource malloc function to allocate memory inside Forward/Backward functions */
typedef void* (*xpu_malloc_t)(void*, int);
/*! \brief sparse alloc function to allocate memory inside Forward/Backward functions */
typedef void (*sparse_malloc_t)(void*, int, int, int, void**, int64_t**, int64_t**);
/*! \brief resource malloc function to allocate ndarrays for graph passes */
typedef void (*nd_malloc_t)(const void* _ndarray_alloc, const int64_t* shapes, int num_shapes,
                            const char* dev_str, int dev_id, int dtype, const char* name,
                            int isArg, void** data);
/*! \brief GPU stream pointer, is void* when not compiled with CUDA */
#if defined(__NVCC__)
  typedef cudaStream_t mx_stream_t;
  typedef curandStatePhilox4_32_10_t mx_gpu_rand_t;
#else
  typedef void* mx_stream_t;
  typedef void* mx_gpu_rand_t;
#endif
typedef std::mt19937 mx_cpu_rand_t;

/*! \brief MXNet initialized random states for each device, used for parallelism */
/* Each thread should generate random number unique sequence out of different states */
#define MX_NUM_CPU_RANDOM_STATES 1024
#define MX_NUM_GPU_RANDOM_STATES 32768

/* \brief Class to help allocate new args/aux params in graph passes */
class PassResource {
 public:
  PassResource(std::unordered_map<std::string, MXTensor>* new_args,
               std::unordered_map<std::string, MXTensor>* new_aux,
               nd_malloc_t nd_malloc, const void* nd_alloc)
    : new_args_(new_args), new_aux_(new_aux), nd_malloc_(nd_malloc), nd_alloc_(nd_alloc) {}
  // allocate new arg param, adds to args map, returns newly allocated tensor
  MXTensor* alloc_arg(const std::string& name, const std::vector<int64_t>& shapes,
                      const MXContext &ctx, MXDType dtype) const {
    void* data;
    nd_malloc_(nd_alloc_, shapes.data(), shapes.size(), ctx.dev_type.c_str(), ctx.dev_id,
               dtype, name.c_str(), 1, &data);
    MXTensor tensor(data, shapes, dtype, 0, ctx, kDefaultStorage);
    (*new_args_)[name] = tensor;
    return &(new_args_->at(name));
  }
  // allocate new aux param, adds to aux map, returns newly allocated tensor
  MXTensor* alloc_aux(const std::string& name, const std::vector<int64_t>& shapes,
                      const MXContext &ctx, MXDType dtype) const {
    void* data;
    nd_malloc_(nd_alloc_, shapes.data(), shapes.size(), ctx.dev_type.c_str(), ctx.dev_id,
               dtype, name.c_str(), 0, &data);
    MXTensor tensor(data, shapes, dtype, 0, ctx, kDefaultStorage);
    (*new_aux_)[name] = tensor;
    return &(new_aux_->at(name));
  }

 private:
  std::unordered_map<std::string, MXTensor>* new_args_;
  std::unordered_map<std::string, MXTensor>* new_aux_;
  nd_malloc_t nd_malloc_;
  const void* nd_alloc_;
};

/*!
 * \brief provide resource APIs memory allocation mechanism to Forward/Backward functions
 */
class OpResource {
 public:
  OpResource(xpu_malloc_t cpu_malloc_fp, void* cpu_alloc_fp,
             xpu_malloc_t gpu_malloc_fp, void* gpu_alloc_fp, void* stream,
             sparse_malloc_t sparse_malloc_fp, void* sparse_alloc_fp,
             void* rng_cpu_states, void* rng_gpu_states)
    : cpu_malloc(cpu_malloc_fp), gpu_malloc(gpu_malloc_fp),
      cpu_alloc(cpu_alloc_fp), gpu_alloc(gpu_alloc_fp), cuda_stream(stream),
      sparse_malloc(sparse_malloc_fp), sparse_alloc(sparse_alloc_fp),
      rand_cpu_states(rng_cpu_states), rand_gpu_states(rng_gpu_states) {}

  /*! \brief allocate cpu memory controlled by MXNet */
  void* alloc_cpu(int size) const {
    return cpu_malloc(cpu_alloc, size);
  }

  /*! \brief allocate gpu memory controlled by MXNet */
  void* alloc_gpu(int size) const {
    return gpu_malloc(gpu_alloc, size);
  }

  /*! \brief return the cuda stream object with correct type */
  mx_stream_t get_cuda_stream() const {
    return static_cast<mx_stream_t>(cuda_stream);
  }

  /*! \brief allocate sparse memory controlled by MXNet */
  void alloc_sparse(MXSparse* sparse, int index, int indices_len, int indptr_len = 0) const {
    sparse_malloc(sparse_alloc, index, indices_len, indptr_len,
                   &(sparse->data), &(sparse->indices), &(sparse->indptr));
  }

  /*! \brief get pointer to initialized and seeded random number states located on CPU */
  /* Access each state by states[id], but this id should be <= MX_NUM_CPU_RANDOM_STATES */
  mx_cpu_rand_t* get_cpu_rand_states() const {
    return static_cast<mx_cpu_rand_t*>(rand_cpu_states);
  }

  /*! \brief get pointer to initialized and seeded random number states located on GPU */
  /* Access each state by states[id], but this id should be <= MX_NUM_GPU_RANDOM_STATES */
  /* Note that if you are using cpu build, it will return a nullptr */
  mx_gpu_rand_t* get_gpu_rand_states() const {
    return static_cast<mx_gpu_rand_t*>(rand_gpu_states);
  }

 private:
  /*! \brief allocation lambda function */
  xpu_malloc_t cpu_malloc, gpu_malloc;
  /*! \brief lambda function to return allocated memory handle */
  void *cpu_alloc, *gpu_alloc;
  /*! \brief cuda stream passed from MXNet */
  void *cuda_stream;
  /*! \brief sparse allocation lambda function */
  sparse_malloc_t sparse_malloc;
  /*! \brief lambda function to return allocated sparse memory handle */
  void *sparse_alloc;
  /*! \brief cpu and gpu rng fully inited and seeded states */
  void *rand_cpu_states, *rand_gpu_states;
};

/*! \brief attribute key to help passing serialized subgraph through subgraph op attribute */
#define MX_STR_SUBGRAPH_SYM_JSON "subgraph_sym_json"
/*! \brief dtype attribute key for ops after type propagation */
#define MX_STR_DTYPE "__ext_dtype__"
/*! \brief shape attribute key for ops after shape propagation */
#define MX_STR_SHAPE "__ext_shape__"
/*! \brief extra input attribute key for ops */
#define MX_STR_EXTRA_INPUTS "__ext_extra_inputs__"

/* \brief get shape value from list of shapes string
 *
 * Examples:
 *
 * getShapeAt("[[1]]", 0) returns "[1]"
 * getShapeAt("[[1],[2,3]]", 1) returns "[2,3]"
 */
std::string getShapeAt(const std::string& shape, unsigned index) {
  int idx = 1;  // start at 1 to skip the first square bracket [
  // find the beginning of the output shape for the particular output index
  for (unsigned x=0; x < index; x++)
    idx = shape.find("[", idx+1);
  int stop = shape.find("]", idx);  // find stop index for this output shape
  // add this shape to the list
  return shape.substr(idx, stop-idx+1);
}

/* \brief get dtype value from list of dtypes string
 *
 * Examples:
 *
 * getDtypeAt("[1]", 0) returns "1"
 * getDtypeAt("[1,2]", 1) returns "2" 
 */
std::string getDtypeAt(const std::string& dtype, unsigned index) {
  // find the beginning of the output dtype for the particular output index
  int idx = 0;
  for (unsigned x=0; x < index; x++)
    idx = dtype.find(",", idx+1);
  int stop = dtype.find(",", idx+1);  // find stop index for this output dtype
  if (stop == -1) stop = dtype.find("]", idx+1);
  return dtype.substr(idx+1, stop-idx-1);
}

/*!
 * \brief Json utility to parse serialized subgraph symbol
 */
/*! \brief Types of JSON objects */
enum JsonType {ERR, STR, NUM, LIST, MAP};

/*! \brief definition of JSON objects */
struct JsonVal {
  JsonVal() : type(ERR), num(-1), str("") {}  // default constructor
  // construct a JSON object by type
  explicit JsonVal(JsonType t) : type(t), num(-1), str("") {}
  // construct a string JSON object
  explicit JsonVal(std::string s) : type(STR), num(-1), str(s) {}
  // construct a number JSON object
  explicit JsonVal(int n) : type(NUM), num(n), str(std::to_string(n)) {}
  // complex constructor
  JsonVal(JsonType t, int n, std::string s) : type(t), num(n), str(s) {}
  bool operator<(const JsonVal &o) const {
    // for string JSON objects compare the string
    if (type == STR) return type == o.type && str < o.str;
    // for number JSON objects compare the number
    if (type == NUM) return type == o.type && num < o.num;
    // for list JSON objects, compare the size of list, and then each object in the list
    if (type == LIST) {
      if (list.size() != o.list.size()) return false;
      for (unsigned int i=0; i< list.size(); i++)
        if (list[i] < o.list[i])
          return false;  // if we find an object that doesnt match return
      return true;  // all objects in lists matched
    }
    // for map JSON objects, compare the size of map, and then each key/value in the maps
    if (type == MAP) {
      if (map.size() != o.map.size()) return false;
      for (auto &item : map) {
        // if one map is missing a key in another return
        if (o.map.find(item.first) == o.map.end()) return false;
        if (item.second < o.map.at(item.first)) return false;
      }
      return true;
    }
    return type < o.type;
  }

  // convert JSON object back to JSON-compatible string
  std::string dump() const {
    std::string ret;
    switch (type) {
    case ERR:
      ret = "json(Error)";
      break;
    case STR:
      ret = "\"" + str + "\"";
      break;
    case NUM:
      ret = str;
      break;
    case LIST:
      ret = "[";
      for (unsigned i=0; i < list.size(); i++) {
        auto &item = list[i];
        ret += item.dump();
        if (i < list.size()-1)
          ret += ",";
      }
      ret += "]";
      break;
    case MAP:
      ret = "{";
      unsigned cnt = 0;
      for (auto &item : map) {
        ret += item.first.dump() + " : " + item.second.dump();
        if (cnt++ < map.size()-1)
          ret += ",";
      }
      ret += "}";
      break;
    }
    return ret;
  }
  // convert JSON-compatible string to JSON object
  static JsonVal parse(const std::string& json) {
    unsigned int idx = 0;
    return JsonVal::parse(json, &idx);
  }
  // parse a string JSON object
  static JsonVal parse_string(const std::string& json, unsigned int* idx) {
    JsonVal ret(STR);
    while (*idx < json.size()) {
      if (json[*idx] == '"') {
        ++(*idx);
        return ret;
      } else {
        ret.str += json[*idx];
        ++(*idx);
      }
    }
    std::cout << "Error! Unable to parse string" << std::endl;
    return JsonVal();
  }
  // parse a number JSON object
  static JsonVal parse_num(const std::string& json, unsigned int* idx) {
    JsonVal ret(NUM);
    while (*idx < json.size()) {
      if (json[*idx] >= '0' && json[*idx] <= '9') {
        ret.str += json[*idx];
        ++(*idx);
      } else {
        break;
      }
    }
    ret.num = std::stoi(ret.str);
    return ret;
  }
  // parse a list of JSON objects
  static JsonVal parse_list(const std::string& json, unsigned int* idx) {
    JsonVal ret(LIST);
    while (*idx < json.size()) {
      if (json[*idx] == ']') {
        ++(*idx);
        return ret;
      } else {
        JsonVal item = JsonVal::parse(json, idx);
        if (item.type != ERR)
          ret.list.push_back(item);
      }
    }
    std::cout << "Error! Unable to parse list" << std::endl;
    return JsonVal();
  }
  // parse a map of JSON objects
  static JsonVal parse_map(const std::string& json, unsigned int* idx) {
    JsonVal ret(MAP), key;
    while (*idx < json.size()) {
      if (json[*idx] == '}') {
        ++(*idx);
        return ret;
      } else {
        JsonVal item = JsonVal::parse(json, idx);
        if (key.type == ERR) {
          key = item;
        } else {
          ret.map[key] = item;
          key.type = ERR;
        }
      }
    }
    std::cout << "Error! Unable to parse map" << std::endl;
    return JsonVal();
  }
  // generic parse function
  static JsonVal parse(const std::string& json, unsigned int *idx) {
    JsonVal ret;
    while (*idx < json.size()) {
      if (json[*idx] == '"') {
        ++(*idx);
        ret = JsonVal::parse_string(json, idx);
      } else if (json[*idx] >= '0' && json[*idx] <= '9') {
        ret = JsonVal::parse_num(json, idx);
      } else if (json[*idx] == '[') {
        ++(*idx);
        ret = JsonVal::parse_list(json, idx);
      } else if (json[*idx] == '{') {
        ++(*idx);
        ret = JsonVal::parse_map(json, idx);
      } else if (json[*idx] == ']' || json[*idx] == '}') {return ret;}
      if (ret.type != ERR) return ret;
      ++(*idx);
    }
    return ret;
  }
  // debug function to convert data structure to a debugstring
  std::string toString() const {
    std::string ret;
    switch (type) {
    case ERR:
      ret = "json(Error)";
      break;
    case STR:
      ret = "json(STR:" + str + ")";
      break;
    case NUM:
      ret = "json(INT:" + str + ")";
      break;
    case LIST:
      ret = "json(LIST:[";
      for (auto &item : list)
        ret += item.toString() + ",";
      ret += "])";
      break;
    case MAP:
      ret = "json(MAP:{";
      for (auto &item : map)
        ret += item.first.toString() + " : " + item.second.toString() + ",";
      ret += "})";
      break;
    }
    return ret;
  }
  JsonType type;
  int num;
  std::string str;
  std::vector<JsonVal> list;
  std::map<JsonVal, JsonVal> map;
};

/*!
 * \brief Graph utility to parse serialized subgraph symbol
 */
class Node;
class Graph;

// Representation of an input/output to a node
struct NodeEntry {
  Node* node;  // other node thats producing/consuming inputs/outputs
  int entry;  // entry index from other node (ie. output index from producing node)
};

// Representation of a node in the graph
class Node {
 public:
  Node() {tensor = nullptr;}
  // internally set passResource to enable tensor allocation for graph passes
  void _setPassResource(PassResource* res_) {res = res_;}
  /* \brief allocate an arg tensor for this node */
  void alloc_arg(const std::vector<int64_t>& shapes,
                 const MXContext &ctx, MXDType dtype) {
    if (!res)
      throw std::runtime_error("Node not initialized. Cannot use alloc_arg outside of graph passes.");
    tensor = res->alloc_arg(name, shapes, ctx, dtype);
  }
  /* \brief allocate an aux tensor for this node */
  void alloc_aux(const std::vector<int64_t>& shapes,
                 const MXContext &ctx, MXDType dtype) {
    if (!res)
      throw std::runtime_error("Node not initialized. Cannot use alloc_aux outside of graph passes.");
    tensor = res->alloc_aux(name, shapes, ctx, dtype);
  }
  std::string op;  // operator name (ie. Convolution)
  std::string name;  // unique node name (ie. conv_0 or conv_1)
  MXTensor* tensor;  // tensor data for input nodes
  std::vector<NodeEntry> inputs;  // set of inputs to the node
  std::vector<NodeEntry> outputs;  // set of outputs from the node
  std::vector<Graph*> subgraphs;  // set of subgraphs within this node
  std::unordered_map<std::string, std::string> attrs;  // node attributes
 private:
  PassResource* res;
};

// Representation of the graph
class Graph {
 public:
  Graph() : res(nullptr) {}
  /* \brief deleted nodes when deleting the graph */
  ~Graph() {
    for (int i = 0; i < nodes.size(); i++)
      delete nodes[i];
  }

  /* \brief create a graph object from an unparsed string */
  static Graph* fromString(const std::string& json) {
    JsonVal val = JsonVal::parse(json);
    return fromJson(val);
  }

  /* \brief create a graph object from a parsed JSON object */
  static Graph* fromJson(JsonVal val) {
    // get nodes list
    JsonVal nodes = val.map[JsonVal("nodes")];
    Graph *g = new Graph();

    std::map<int, Node*> nodeMap;
    // loop over nodes
    for (int i = 0; i < nodes.list.size(); i++) {
      Node* n = new Node();
      g->nodes.push_back(n);
      JsonVal node = nodes.list[i];

      // set the op info
      n->op = node.map[JsonVal("op")].str;
      n->name = node.map[JsonVal("name")].str;

      // if op is null its an input to the graph
      if (n->op.compare("null") == 0)
        g->inputs.push_back(n);

      // set attrs
      JsonVal attributes = node.map[JsonVal("attrs")];
      for (auto& kv : attributes.map) {
        n->attrs[kv.first.str] = kv.second.str;
      }

      // set subgraphs, parsing each into a graph
      if (node.map.count(JsonVal("subgraphs")) > 0) {
        JsonVal subgraphs = node.map[JsonVal("subgraphs")];
        for (auto &subgraph : subgraphs.list) {
          n->subgraphs.push_back(fromJson(subgraph));
        }
      }

      // set node inputs
      JsonVal node_inputs = node.map[JsonVal("inputs")];
      n->inputs.resize(node_inputs.list.size());
      for (int j = 0; j < node_inputs.list.size(); j++) {
        JsonVal input = node_inputs.list[j];
        NodeEntry& entry = n->inputs[j];
        // get pointer to other node
        entry.node = nodeMap[input.list[0].num];
        // get the other node's output index
        entry.entry = input.list[1].num;
        // set other nodes output as connected to this node
        entry.node->outputs.push_back({n, j});
      }
      nodeMap[i] = n;
    }

    // set graph level outputs
    JsonVal& heads = val.map[JsonVal("heads")];
    g->outputs.resize(heads.list.size());
    for (int i = 0; i < heads.list.size(); i++) {
      JsonVal head = heads.list[i];
      g->outputs[i].node = nodeMap[head.list[0].num];
      g->outputs[i].entry = head.list[1].num;
    }

    // add all attributes to the graph
    for (auto& kv : val.map) {
      if (kv.first.str.compare("nodes") != 0 &&
         kv.first.str.compare("heads") != 0 &&
         kv.first.str.compare("node_row_ptr") != 0 &&
         kv.first.str.compare("arg_nodes") != 0) {
        g->attrs[kv.first.str] = kv.second;
      }
    }
    return g;
  }

  /* \brief convert graph object back to JSON object */
  JsonVal toJson() {
    // top level object is a map
    JsonVal val(MAP);

    // add attributes
    for (auto& kv : attrs) {
      val.map[JsonVal(kv.first)] = kv.second;
    }

    // sort graph nodes in topological order, create mapping of node to index
    std::map<Node*, int> nodeMap;
    std::vector<Node*> sorted = topological_sort();
    // nodes are in reverse topological order in the vector (back is first)
    // so loop from end to front over the vector 'sorted'
    for (int i = sorted.size()-1; i >= 0; i--) {
      nodeMap[sorted[i]] = sorted.size()-1-i;
    }

    // create node_row_ptr entry
    val.map[JsonVal("node_row_ptr")] = JsonVal(LIST);
    JsonVal& node_row_ptr = val.map[JsonVal("node_row_ptr")];
    for (int i = 0; i < nodes.size(); i++)
      node_row_ptr.list.push_back(JsonVal(i));

    // add all input nodes
    val.map[JsonVal("arg_nodes")] = JsonVal(LIST);
    JsonVal& arg_nodes = val.map[JsonVal("arg_nodes")];
    for (int i = 0; i < inputs.size(); i++)
      arg_nodes.list.push_back(JsonVal(nodeMap[inputs[i]]));

    // add all output nodes
    val.map[JsonVal("heads")] = JsonVal(LIST);
    JsonVal& heads = val.map[JsonVal("heads")];
    for (int i = 0; i < outputs.size(); i++) {
      heads.list.push_back(JsonVal(LIST));
      JsonVal& out = heads.list[i];
      out.list.push_back(JsonVal(nodeMap[outputs[i].node]));
      out.list.push_back(JsonVal(outputs[i].entry));
      out.list.push_back(JsonVal(0));
    }

    // add all graph nodes
    val.map[JsonVal("nodes")] = JsonVal(LIST);
    JsonVal& nodes_ = val.map[JsonVal("nodes")];
    for (int i = sorted.size()-1; i >= 0; i--) {
      // each node is a map
      nodes_.list.push_back(JsonVal(MAP));
      Node* n = sorted[i];
      JsonVal& n_ = nodes_.list[nodes_.list.size()-1];

      n_.map[JsonVal("op")] = JsonVal(n->op);
      n_.map[JsonVal("name")] = JsonVal(n->name);
      n_.map[JsonVal("inputs")] = JsonVal(LIST);

      // add inputs for this node
      JsonVal& inputs_ = n_.map[JsonVal("inputs")];
      for (int j = 0; j < n->inputs.size(); j++) {
        inputs_.list.push_back(JsonVal(LIST));
        NodeEntry& entry = n->inputs[j];
        JsonVal& in = inputs_.list[j];
        in.list.push_back(JsonVal(nodeMap[entry.node]));
        in.list.push_back(JsonVal(entry.entry));
        in.list.push_back(JsonVal(0));
      }

      // add subgraphs for this node, convert each back to JSON
      if (n->subgraphs.size() > 0) {
        n_.map[JsonVal("subgraphs")] = JsonVal(LIST);
        JsonVal &subgraphs_ = n_.map[JsonVal("subgraphs")];
        for (Graph *subgraph : n->subgraphs) {
          subgraphs_.list.push_back(subgraph->toJson());
        }
      }

      // add attributes for this node
      n_.map[JsonVal("attrs")] = JsonVal(MAP);
      JsonVal& attrs_ = n_.map[JsonVal("attrs")];
      for (auto& kv : n->attrs) {
        attrs_.map[JsonVal(kv.first)] = JsonVal(kv.second);
      }
    }
    return val;
  }

  /* \brief convert graph object to JSON string */
  std::string toString() {
    return toJson().dump();
  }

  /* \brief visits a node "n" */
  void _dfs_util(Node* n, std::unordered_set<Node*>* to_visit,
                 std::function<void(Node*)> handler) const {
    to_visit->erase(n);  // remove node now that we're visiting it
    for (NodeEntry& e : n->outputs) {
      Node* o = e.node;
      if (to_visit->count(o) != 0) {
        _dfs_util(o, to_visit, handler);  // visit neighbor
      }
    }
    handler(n);  // post-order visit this node
  }

  /* \brief post-order DFS graph traversal */
  void DFS(std::function<void(Node*)> handler) const {
    std::unordered_set<Node*> to_visit;
    // put all nodes in set to visit
    for (auto& n : nodes)
      to_visit.insert(n);
    // visit all inputs first
    for (auto& i : inputs)
      if (to_visit.count(i) != 0)
        _dfs_util(i, &to_visit, handler);
    // visit any nodes left
    while (to_visit.size() > 0)
      _dfs_util(*(to_visit.begin()), &to_visit, handler);
  }

  /* \brief sort graph nodes in topological order */
  std::vector<Node*> topological_sort() const {
    std::vector<Node*> sorted;
    auto handler = [&](Node* n) {
      sorted.push_back(n);  // when visiting each node, add it in order to the vector
    };
    DFS(handler);
    return sorted;
  }

  /* \brief print out graph details */
  void print(int indent = 0) const {
    std::string space = "";
    for (int i = 0; i < indent; i++) space+=" ";

    std::cout << space << "########### Graph #############" << std::endl;
    std::cout << space << "attributes: " << std::endl;
    for (auto &kv : attrs)
      std::cout << space << "\t" << kv.first << " : " << kv.second.str << std::endl;
    std::cout << space << "inputs: " << inputs.size() << std::endl;
    std::cout << space << "outputs: " << outputs.size() << std::endl;
    std::cout << space << "nodes: " << nodes.size() << std::endl;
    std::vector<Node*> sorted = topological_sort();
    // loop over each node and print out its inputs/outputs
    for (int i = sorted.size()-1; i >= 0; i--) {
      std::cout << space << "Node: " << sorted[i]->name << std::endl;
      for (int j = 0; j < sorted[i]->inputs.size(); j++) {
        std::cout << space << "\tInput: " << sorted[i]->inputs[j].node->name << " "
                  << sorted[i]->inputs[j].entry << std::endl;
      }
      for (int j = 0; j < sorted[i]->outputs.size(); j++) {
        std::cout << space << "\tOutput: " << sorted[i]->outputs[j].node->name << " "
                  << sorted[i]->outputs[j].entry << std::endl;
      }
      if (sorted[i]->subgraphs.size() > 0) {
        for (auto &subgraph : sorted[i]->subgraphs) {
          std::cout << space << "\tSubgraph:" << std::endl;
          subgraph->print(indent+2);
        }
      }
    }
    std::cout << space << "###############################" << std::endl;
  }

  /* \brief add a new node to this graph */
  Node* addNode(const std::string& name, const std::string& op) {
    Node* n = new Node();
    n->name = name;
    n->op = op;
    if (res)
      n->_setPassResource(res);
    return n;
  }
  /* \brief get node at index in graph */
  Node* getNode(size_t idx) {
    return nodes[idx];
  }
  /* \brief get const node at index in const graph */
  const Node* getNode(size_t idx) const {
    return nodes.at(idx);
  }
  /* \brief get attribute on graph */
  const JsonVal& getAttr(const std::string& key) const {
    return attrs.at(key);
  }
  /* \brief get number of nodes in the graph */
  size_t size() const {
    return nodes.size();
  }
  // internally set passResource to enable tensor allocation for graph passes
  void _setPassResource(PassResource* res_) {res = res_;}
  // internally set arg/aux params when available
  void _setParams(std::unordered_map<std::string, mxnet::ext::MXTensor>* args,
                  std::unordered_map<std::string, mxnet::ext::MXTensor>* aux) {
    // set params for each input node
    for (Node* node : inputs) {
      if (args->count(node->name) > 0)
        node->tensor = &args->at(node->name);
      else if (aux->count(node->name) > 0)
        node->tensor = &aux->at(node->name);
    }

    if (res) {
      // set passResource for each node
      for (Node* node : nodes) {
        node->_setPassResource(res);
      }
    }
  }

  std::vector<Node*> inputs;
  std::vector<NodeEntry> outputs;
  std::map<std::string, JsonVal> attrs;

 private:
  std::vector<Node*> nodes;
  PassResource* res;
};

/* \brief An abstract class for library authors creating custom
 * partitioners. Optional, can just implement supportedOps instead
 */
class CustomOpSelector {
 public:
  /* \brief Select a node to include in subgraph, return true to include node
   * nodeID - index of node in graph
   */
  virtual bool Select(int nodeID) = 0;
  /* \brief Select an input node from current node to include in subgraph
   * return true to include node
   * nodeID - index of node in graph
   * input_nodeID - index of input node in graph
   */
  virtual bool SelectInput(int nodeID, int input_nodeID) = 0;
  /* \brief Select an output node from current node to include in subgraph
   * return true to include node
   * nodeID - index of node in graph
   * output_nodeID - index of output node in graph
   */
  virtual bool SelectOutput(int nodeID, int output_nodeID) = 0;
  /* \brief Review nodes to include in subgraph
   * return set of candidate nodes to keep in subgraph
   * candidates - indices of nodes to include in subgraph
   * keep - indices of nodes to keep in subgraph
   */
  virtual void Filter(const std::vector<int>& candidates,
                      std::vector<int>* keep) {
    keep->insert(keep->end(), candidates.begin(), candidates.end());
  }
  /* \brief Reset any selector state, called after growing subgraph, before filter
   * Called after finished calling SelectInput/SelectOutput and growing subgraph
   */
  virtual void Reset() {}
};

/*!
 * \brief An abstract class for library authors creating stateful op
 * custom library should override Forward and destructor, and has an
 * option to implement Backward
 */
class CustomStatefulOp {
 public:
  virtual MXReturnValue Forward(std::vector<MXTensor>* inputs,
                                std::vector<MXTensor>* outputs,
                                const OpResource& op_res) = 0;
  virtual MXReturnValue Backward(std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& op_res) {
    std::cout << "Error! Operator does not support backward" << std::endl;
    return MX_FAIL;
  }
};

/*! \brief StatefulOp wrapper class to pass to backend OpState */
class CustomStatefulOpWrapper {
 public:
  explicit CustomStatefulOpWrapper(CustomStatefulOp* inst) : instance(inst) {}
  CustomStatefulOp* get_instance() { return instance; }
 private:
  CustomStatefulOp* instance;
};

/*! \brief Custom Operator function templates */
typedef MXReturnValue (*fcomp_t)(const std::unordered_map<std::string,
                                                          std::string>& attributes,
                                 std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& res);
typedef MXReturnValue (*parseAttrs_t)(const std::unordered_map<std::string,
                                                               std::string>& attributes,
                                      int* num_inputs, int* num_outputs);
typedef MXReturnValue (*inferType_t)(const std::unordered_map<std::string,
                                                               std::string>& attributes,
                                     std::vector<int>* in_types,
                                     std::vector<int>* out_types);
typedef MXReturnValue (*inferSType_t)(const std::unordered_map<std::string,
                                                               std::string>& attributes,
                                      std::vector<int>* in_storage_types,
                                      std::vector<int>* out_storage_types);
typedef MXReturnValue (*inferShape_t)(const std::unordered_map<std::string,
                                                               std::string>& attributes,
                                      std::vector<std::vector<unsigned int> >* in_shapes,
                                      std::vector<std::vector<unsigned int> >* out_shapes);
typedef MXReturnValue (*mutateInputs_t)(const std::unordered_map<std::string,
                                                                 std::string>& attributes,
                                        std::vector<int>* input_indices);
typedef MXReturnValue (*createOpState_t)(const std::unordered_map<std::string,
                                                                  std::string>& attributes,
                                         CustomStatefulOp**);

/*!
 * \brief Class to hold custom operator registration
 */
class CustomOp {
 public:
  explicit CustomOp(const char* op_name) : name(op_name),
    parse_attrs(NULL), infer_type(NULL), infer_storage_type(NULL), infer_shape(NULL),
    mutate_inputs(NULL), isSGop(false) {}
  CustomOp& setForward(fcomp_t fcomp, const char* ctx) {
    if (forward_ctx_map.count(ctx) > 0)
      raiseDuplicateContextError();
    forward_ctx_map[ctx] = fcomp;
    return *this;
  }
  CustomOp& setBackward(fcomp_t fgrad, const char* ctx) {
    if (backward_ctx_map.count(ctx) > 0)
      raiseDuplicateContextError();
    backward_ctx_map[ctx] = fgrad;
    return *this;
  }
  CustomOp& setParseAttrs(parseAttrs_t func) {
    parse_attrs = func;
    return *this;
  }
  CustomOp& setInferType(inferType_t func) {
    infer_type = func;
    return *this;
  }
  CustomOp& setInferSType(inferSType_t func) {
    infer_storage_type = func;
    return *this;
  }
  CustomOp& setInferShape(inferShape_t func) {
    infer_shape = func;
    return *this;
  }
  CustomOp& setMutateInputs(mutateInputs_t func) {
    mutate_inputs = func;
    return *this;
  }
  CustomOp& setCreateOpState(createOpState_t func, const char* ctx) {
    if (create_op_ctx_map.count(ctx) > 0)
      raiseDuplicateContextError();
    create_op_ctx_map[ctx] = func;
    return *this;
  }
  CustomOp& setIsSubgraphOp() {
    isSGop = true;
    return *this;
  }
  void mapToVector() {
    for (auto kv : forward_ctx_map) {
      forward_ctx_cstr.push_back(kv.first);
      forward_fp.push_back(kv.second);
    }
    for (auto kv : backward_ctx_map) {
      backward_ctx_cstr.push_back(kv.first);
      backward_fp.push_back(kv.second);
    }
    for (auto kv : create_op_ctx_map) {
      create_op_ctx_cstr.push_back(kv.first);
      create_op_fp.push_back(kv.second);
    }
  }
  ~CustomOp() {}

  /*! \brief operator name */
  const char* name;

  /*! \brief operator functions */
  parseAttrs_t parse_attrs;
  inferType_t infer_type;
  inferSType_t infer_storage_type;
  inferShape_t infer_shape;
  mutateInputs_t mutate_inputs;
  bool isSGop;

  /*! \brief vector repr of ctx map to be easily loaded from c_api */
  std::vector<const char*> forward_ctx_cstr, backward_ctx_cstr, create_op_ctx_cstr;
  std::vector<fcomp_t> forward_fp, backward_fp;
  std::vector<createOpState_t> create_op_fp;

 private:
  void raiseDuplicateContextError() {
    std::string op_name_str(name);
    throw std::runtime_error(
      "Error! Error! Cannot register multiple functions under same context for operator '"
      + op_name_str + "'");
  }

  /*! \brief dedup context maps - static string ctx to custom function */
  std::unordered_map<const char*, fcomp_t> forward_ctx_map, backward_ctx_map;
  std::unordered_map<const char*, createOpState_t> create_op_ctx_map;
};

/*! \brief Custom Pass Create function template */
typedef MXReturnValue (*graphPass_t)(mxnet::ext::Graph* graph,
                                     const std::unordered_map<std::string, std::string>& options);

/*!
 * \brief An abstract class for graph passes
 */
class CustomPass {
 public:
  CustomPass() : name("ERROR") {}
  explicit CustomPass(const char* pass_name)
    : name(pass_name) {}
  CustomPass& setBody(graphPass_t fn) {
    pass = fn;
    return *this;
  }

  /*! \brief pass name */
  const char* name;
  /*! \brief pass function */
  graphPass_t pass;
};

/*! \brief Custom Subgraph Create function template */
typedef MXReturnValue (*supportedOps_t)(const mxnet::ext::Graph *graph, std::vector<int>* ids,
                                        const std::unordered_map<std::string,
                                                                 std::string>& options);
typedef MXReturnValue (*createSelector_t)(const mxnet::ext::Graph *graph,
                                          CustomOpSelector** sel_inst,
                                          const std::unordered_map<std::string,
                                                                   std::string>& options);
typedef MXReturnValue (*reviewSubgraph_t)(const mxnet::ext::Graph *subgraph, int subgraph_id,
                                          bool* accept,
                                          const std::unordered_map<std::string,
                                                                   std::string>& options);

/*!
 * \brief An abstract class for subgraph property
 */
class CustomPartitioner {
 public:
  CustomPartitioner() : name("ERROR") {}
  explicit CustomPartitioner(const char* backend_name) :
    name(backend_name) {}
  CustomPartitioner& addStrategy(const char* prop_name,
                                 const char* sg_name) {
    strategies.push_back(prop_name);
    op_names.push_back(sg_name);
    return *this;
  }
  CustomPartitioner& setSupportedOps(const char* prop_name, supportedOps_t fn) {
    supported_map[std::string(prop_name)] = fn;
    return *this;
  }
  CustomPartitioner& setCreateSelector(const char* prop_name, createSelector_t fn) {
    selector_map[std::string(prop_name)] = fn;
    return *this;
  }
  CustomPartitioner& setReviewSubgraph(const char* prop_name, reviewSubgraph_t fn) {
    review_map[std::string(prop_name)] = fn;
    return *this;
  }
  supportedOps_t getSupportedOps(int stg_id) {
    std::string prop(strategies[stg_id]);
    if (supported_map.count(prop) > 0)
      return supported_map[prop];
    else
      return nullptr;
  }
  createSelector_t getCreateSelector(int stg_id) {
    std::string prop(strategies[stg_id]);
    if (selector_map.count(prop) > 0)
      return selector_map[prop];
    else
      return nullptr;
  }
  reviewSubgraph_t getReviewSubgraph(int stg_id) {
    std::string prop(strategies[stg_id]);
    if (review_map.count(prop) > 0)
      return review_map[prop];
    else
      return nullptr;
  }

  /*! \brief partitioner name */
  const char* name;
  std::map<std::string, supportedOps_t> supported_map;
  std::map<std::string, createSelector_t> selector_map;
  std::map<std::string, reviewSubgraph_t> review_map;
  /*! \brief strategy names */
  std::vector<const char*> strategies;
  /*! \brief subgraph operator name */
  std::vector<const char*> op_names;
};

/*!
 * \brief Registry class to registers things (ops, properties)
 *        Singleton class
 */
template <class T>
class Registry {
 public:
  /*!
   * \brief get singleton pointer to class
   * \returns pointer to class
   */
  static Registry* get() PRIVATE_SYMBOL {
    static Registry inst;
    return &inst;
  }
  /*!
   * \brief add a new entry
   * \returns new object associated with registered name
   */
  T& add(const char* name) {
    T *entry = new T(name);
    entries.push_back(entry);
    return *entry;
  }
  int size() {
    return entries.size();
  }
  T& get(int idx) {
    return *(entries.at(idx));
  }

 private:
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {}
  /*! \brief map of entries in registry */
  std::vector<T*> entries;
};

/*!
 * \brief Macros to help with string concat
 * Annoyingly, the concat_ and concat macros are necessary to
 * be able to use __COUNTER__ in an identifier name
 */
#define MX_STR_CONCAT_(__a, __b) __a ## __b
#define MX_STR_CONCAT(__a, __b) MX_STR_CONCAT_(__a, __b)

/*! \brief convert a token to a string */
#define MX_STRINGIFY(x) #x
#define MX_TOSTRING(x) MX_STRINGIFY(x)

/*! \brief declare a variable with custom name */
#define MX_REGISTER_NAME_(Name) MXNet ## _CustomOp ## _
#define MX_REGISTER_DEF_(Name) CustomOp MX_REGISTER_NAME_(Name)

#define MX_REGISTER_PROP_NAME_(Name) MXNet ## _CustomSubProp ## _
#define MX_REGISTER_PROP_DEF_(Name) CustomPartitioner MX_REGISTER_PROP_NAME_(Name)

#define MX_REGISTER_PASS_NAME_(Name) MXNet ## _CustomPass ## _
#define MX_REGISTER_PASS_DEF_(Name) CustomPass MX_REGISTER_PASS_NAME_(Name)

/*! \brief assign a var to a value */
#define REGISTER_OP(Name) MX_STR_CONCAT(MX_REGISTER_DEF_(Name), __COUNTER__) = \
    Registry<CustomOp>::get()->add(MX_TOSTRING(Name))

#define REGISTER_PARTITIONER(Name) \
  MX_STR_CONCAT(MX_REGISTER_PROP_DEF_(Name), __COUNTER__) = \
    Registry<CustomPartitioner>::get()->add(MX_TOSTRING(Name))

#define REGISTER_PASS(Name) \
  MX_STR_CONCAT(MX_REGISTER_PASS_DEF_(Name), __COUNTER__) = \
    Registry<CustomPass>::get()->add(MX_TOSTRING(Name))

/* -------------- BELOW ARE CTYPE FUNCTIONS PROTOTYPES --------------- */

/*!
 * \brief Following are the C type APIs implemented in the external library
 * Each API has a #define string that is used to lookup the function in the library
 * Followed by the function declaration
 */
#define MXLIB_OPREGSIZE_STR "_opRegSize"
typedef int (*opRegSize_t)(void);

#define MXLIB_OPREGGET_STR "_opRegGet"
typedef int (*opRegGet_t)(int idx, const char** name, int *isSGop,
                          const char*** forward_ctx, mxnet::ext::fcomp_t** forward_fp,
                          int* forward_count, const char*** backward_ctx,
                          mxnet::ext::fcomp_t** backward_fp, int* backward_count,
                          const char*** create_op_ctx, mxnet::ext::createOpState_t** create_op_fp,
                          int* create_op_count, mxnet::ext::parseAttrs_t* parse,
                          mxnet::ext::inferType_t* type, mxnet::ext::inferSType_t* stype,
                          mxnet::ext::inferShape_t* shape, mxnet::ext::mutateInputs_t* mutate);

#define MXLIB_OPCALLFREE_STR "_opCallFree"
typedef int (*opCallFree_t)(void* ptr);

#define MXLIB_OPCALLPARSEATTRS_STR "_opCallParseAttrs"
typedef int (*opCallParseAttrs_t)(parseAttrs_t parseAttrs, const char* const* keys,
                                  const char* const* vals, int num,
                                  int* num_in, int* num_out);

#define MXLIB_OPCALLINFERSHAPE_STR "_opCallInferShape"
typedef int (*opCallInferShape_t)(inferShape_t inferShape, const char* const* keys,
                                  const char* const* vals, int num,
                                  unsigned int** inshapes, int* indims, int num_in,
                                  unsigned int*** mod_inshapes, int** mod_indims,
                                  unsigned int*** outshapes, int** outdims, int num_out);

#define MXLIB_OPCALLINFERTYPE_STR "_opCallInferType"
typedef int (*opCallInferType_t)(inferType_t inferType, const char* const* keys,
                                 const char* const* vals, int num,
                                 int* intypes, int num_in, int* outtypes, int num_out);

#define MXLIB_OPCALLINFERSTYPE_STR "_opCallInferSType"
typedef int (*opCallInferSType_t)(inferSType_t inferSType, const char* const* keys,
                                 const char* const* vals, int num,
                                 int* intypes, int num_in, int* outtypes, int num_out);

#define MXLIB_OPCALLFCOMP_STR "_opCallFCompute"
typedef int (*opCallFComp_t)(fcomp_t fcomp, const char* const* keys,
                             const char* const* vals, int num,
                             const int64_t** inshapes, int* indims,
                             void** indata, int* intypes,
                             size_t* inIDs, const char** indev_type,
                             int* indev_id, int num_in,
                             const int64_t** outshapes, int* outdims,
                             void** outdata, int* outtypes,
                             size_t* outIDs, const char** outdev_type,
                             int* outdev_id, int num_out,
                             xpu_malloc_t cpu_malloc, void* cpu_alloc,
                             xpu_malloc_t gpu_malloc, void* gpu_alloc, void* cuda_stream,
                             sparse_malloc_t sparse_malloc, void* sparse_alloc,
                             int* instypes, int* outstypes,
                             void** in_indices, void** out_indices,
                             void** in_indptr, void** out_indptr,
                             int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                             int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                             void* rng_cpu_states, void* rng_gpu_states);

#define MXLIB_OPCALLMUTATEINPUTS_STR "_opCallMutateInputs"
typedef int (*opCallMutateInputs_t)(mutateInputs_t mutate, const char* const* keys,
                                    const char* const* vals, int num,
                                    int** mutate_indices, int* indices_size);

#define MXLIB_OPCALLCREATEOPSTATE_STR "_opCallCreateOpState"
typedef int (*opCallCreateOpState_t)(createOpState_t create_op, const char* const* keys,
                                     const char* const* vals, int num,
                                     void** state_op);

#define MXLIB_OPCALLFSTATEFULCOMP_STR "_opCallFStatefulCompute"
typedef int (*opCallFStatefulComp_t)(int is_forward, void* state_op,
                                     const int64_t** inshapes, int* indims,
                                     void** indata, int* intypes,
                                     size_t* inIDs, const char** indev_type,
                                     int* indev_id, int num_in,
                                     const int64_t** outshapes, int* outdims,
                                     void** outdata, int* outtypes,
                                     size_t* outIDs, const char** outdev_type,
                                     int* outdev_id, int num_out,
                                     xpu_malloc_t cpu_malloc, void* cpu_alloc,
                                     xpu_malloc_t gpu_malloc, void* gpu_alloc, void* stream,
                                     sparse_malloc_t sparse_malloc, void* sparse_alloc,
                                     int* instypes, int* outstypes,
                                     void** in_indices, void** out_indices,
                                     void** in_indptr, void** out_indptr,
                                     int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                                     int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                                     void* rng_cpu_states, void* rng_gpu_states);

#define MXLIB_PARTREGSIZE_STR "_partRegSize"
typedef int (*partRegSize_t)(void);

#define MXLIB_PARTREGGETCOUNT_STR "_partRegGetCount"
typedef int (*partRegGetCount_t)(int idx, const char** name);

#define MXLIB_PARTREGGET_STR "_partRegGet"
typedef void (*partRegGet_t)(int part_idx, int stg_idx, const char** strategy,
                             supportedOps_t* supportedOps, createSelector_t* createSelector,
                             reviewSubgraph_t* reviewSubgraph, const char** op_name);

#define MXLIB_PARTCALLSUPPORTEDOPS_STR "_partCallSupportedOps"
typedef int (*partCallSupportedOps_t)(supportedOps_t supportedOps, const char *json,
                                      int num_ids, int *ids, const char* const* opt_keys,
                                      const char* const* opt_vals, int num_opts);

#define MXLIB_PARTCALLCREATESELECTOR_STR "_partCallCreateSelector"
typedef int (*partCallCreateSelector_t)(createSelector_t createSelector, const char *json,
                                        void** selector, const char* const* opt_keys,
                                        const char* const* opt_vals, int num_opts);

#define MXLIB_PARTCALLSELECT_STR "_partCallSelect"
typedef void (*partCallSelect_t)(void* sel_inst, int nodeID, int* selected);

#define MXLIB_PARTCALLSELECTINPUT_STR "_partCallSelectInput"
typedef void (*partCallSelectInput_t)(void* sel_inst, int nodeID, int input_nodeID,
                                  int* selected);

#define MXLIB_PARTCALLSELECTOUTPUT_STR "_partCallSelectOutput"
typedef void (*partCallSelectOutput_t)(void* sel_inst, int nodeID, int output_nodeID,
                                   int* selected);

#define MXLIB_PARTCALLFILTER_STR "_partCallFilter"
typedef void (*partCallFilter_t)(void* sel_inst, int* candidates, int num_candidates,
                             int** keep, int* num_keep);

#define MXLIB_PARTCALLRESET_STR "_partCallReset"
typedef void (*partCallReset_t)(void* sel_inst);

#define MXLIB_PARTCALLREVIEWSUBGRAPH_STR "_partCallReviewSubgraph"
typedef int (*partCallReviewSubgraph_t)(reviewSubgraph_t reviewSubgraph, const char *json,
                                        int subgraph_id, int *accept, const char* const* opt_keys,
                                        const char* const* opt_vals, int num_opts,
                                        char*** attr_keys, char*** attr_vals, int *num_attrs,
                                        const char* const* arg_names, int num_args,
                                        void* const* arg_data, const int64_t* const* arg_shapes,
                                        const int* arg_dims, const int* arg_types,
                                        const size_t* arg_IDs, const char* const* arg_dev_type,
                                        const int* arg_dev_id,
                                        const char* const* aux_names, int num_aux,
                                        void* const* aux_data, const int64_t* const* aux_shapes,
                                        const int* aux_dims, const int* aux_types,
                                        const size_t* aux_IDs, const char* const* aux_dev_type,
                                        const int* aux_dev_id);

#define MXLIB_PASSREGSIZE_STR "_passRegSize"
typedef int (*passRegSize_t)(void);

#define MXLIB_PASSREGGET_STR "_passRegGet"
typedef void (*passRegGet_t)(int pass_idx, graphPass_t* graphPass, const char** pass_name);

#define MXLIB_PASSCALLGRAPHPASS_STR "_passCallGraphPass"
typedef int (*passCallGraphPass_t)(graphPass_t graphPass, const char *in_graph,
                                   char** out_graph, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts,
                                   const char* pass_name, const char* const* arg_names,
                                   int num_args, void* const* arg_data,
                                   const int64_t* const* arg_shapes, const int* arg_dims,
                                   const int* arg_types, const size_t* arg_IDs,
                                   const char* const* arg_dev_type, const int* arg_dev_id,
                                   const char* const* aux_names, int num_aux,
                                   void* const* aux_data, const int64_t* const* aux_shapes,
                                   const int* aux_dims, const int* aux_types,
                                   const size_t* aux_IDs, const char* const* aux_dev_type,
                                   const int* aux_dev_id, nd_malloc_t nd_malloc,
                                   const void* nd_alloc);

#define MXLIB_INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int version);

#define MXLIB_OPVERSION_STR "_opVersion"
typedef int (*opVersion_t)();

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#define MX_INT_RET  __declspec(dllexport) int __cdecl
#define MX_VOID_RET __declspec(dllexport) void __cdecl
#else
#define MX_INT_RET  int
#define MX_VOID_RET void
#endif

}  // namespace ext
}  // namespace mxnet

extern "C" {
  /*! \brief returns MXNet library version */
  MX_INT_RET _opVersion() {
    return MX_LIBRARY_VERSION;
  }

  /*! \brief returns number of ops registered in this library */
  MX_INT_RET _opRegSize() {
    return mxnet::ext::Registry<mxnet::ext::CustomOp>::get()->size();
  }

  /*! \brief returns operator registration at specified index */
  MX_VOID_RET _opRegGet(int idx, const char** name, int *isSGop,
                        const char*** forward_ctx, mxnet::ext::fcomp_t** forward_fp,
                        int* forward_count, const char*** backward_ctx,
                        mxnet::ext::fcomp_t** backward_fp, int* backward_count,
                        const char*** create_op_ctx, mxnet::ext::createOpState_t** create_op_fp,
                        int* create_op_count, mxnet::ext::parseAttrs_t* parse,
                        mxnet::ext::inferType_t* type, mxnet::ext::inferSType_t* stype,
                        mxnet::ext::inferShape_t* shape, mxnet::ext::mutateInputs_t* mutate) {
    mxnet::ext::CustomOp &op = mxnet::ext::Registry<mxnet::ext::CustomOp>::get()->get(idx);
    *name = op.name;
    *parse = op.parse_attrs;
    *type = op.infer_type;
    *stype = op.infer_storage_type;
    *shape = op.infer_shape;
    *mutate = op.mutate_inputs;
    *isSGop = op.isSGop;
    op.mapToVector();
    *forward_ctx = op.forward_ctx_cstr.data();
    *forward_fp = op.forward_fp.data();
    *forward_count = op.forward_fp.size();
    *backward_ctx = op.backward_ctx_cstr.data();
    *backward_fp = op.backward_fp.data();
    *backward_count = op.backward_fp.size();
    *create_op_ctx = op.create_op_ctx_cstr.data();
    *create_op_fp = op.create_op_fp.data();
    *create_op_count = op.create_op_fp.size();
  }

  /*! \brief calls free from the external library for library allocated arrays */
  MX_VOID_RET _opCallFree(void* ptr) {
    free(ptr);
  }

  /*! \brief returns status of calling parse attributes function for operator from library */
  MX_INT_RET _opCallParseAttrs(mxnet::ext::parseAttrs_t parseAttrs, const char* const* keys,
                               const char* const* vals, int num,
                               int* num_in, int* num_out) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    return parseAttrs(attrs, num_in, num_out);
  }

  /*! \brief returns status of calling inferShape function for operator from library */
  MX_INT_RET _opCallInferShape(mxnet::ext::inferShape_t inferShape, const char* const* keys,
                               const char* const* vals, int num,
                               unsigned int** inshapes, int* indims, int num_in,
                               unsigned int*** mod_inshapes, int** mod_indims,
                               unsigned int*** outshapes, int** outdims, int num_out) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of shapes for inputs
    std::vector<std::vector<unsigned int> > in_shapes(num_in);
    for (int i = 0; i < num_in; i++) {
      for (int j = 0; j < indims[i]; j++) {
        in_shapes[i].push_back(inshapes[i][j]);
      }
    }

    // create a vector of shapes for outputs
    std::vector<std::vector<unsigned int> > out_shapes(num_out);

    int retval = inferShape(attrs, &in_shapes, &out_shapes);
    if (!retval) return retval;

    // allocate space for modified input dims, shape
    *mod_indims = static_cast<int*>(malloc (num_in * sizeof(int)));
    *mod_inshapes = static_cast<unsigned**>(malloc (num_in * sizeof(unsigned*)));

    // copy modified input shapes
    for (int i = 0; i < num_in; i++) {
      (*mod_indims)[i] = in_shapes[i].size();
      (*mod_inshapes)[i] = static_cast<unsigned*>(malloc ((*mod_indims)[i] * sizeof(unsigned)));
      for (int j = 0; j < (*mod_indims)[i]; j++) {
        (*mod_inshapes)[i][j] = in_shapes[i][j];
      }
    }

    // allocate space for output dims, shape
    *outdims = static_cast<int*>(malloc (num_out * sizeof(int)));
    *outshapes = static_cast<unsigned**>(malloc (num_out * sizeof(unsigned*)));

    // copy output shapes
    for (int i = 0; i < num_out; i++) {
      (*outdims)[i] = out_shapes[i].size();
      (*outshapes)[i] = static_cast<unsigned*>(malloc ((*outdims)[i] * sizeof(unsigned)));
      for (int j = 0; j < (*outdims)[i]; j++) {
        (*outshapes)[i][j] = out_shapes[i][j];
      }
    }

    return retval;
  }

  /*! \brief returns status of calling inferType function for operator from library */
  MX_INT_RET _opCallInferType(mxnet::ext::inferType_t inferType, const char* const* keys,
                              const char* const* vals, int num,
                              int* intypes, int num_in, int* outtypes, int num_out) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of types for inputs
    std::vector<int> in_types(num_in);
    for (int i = 0; i < num_in; i++) {
      in_types[i] = intypes[i];
    }

    // create a vector of types for outputs
    std::vector<int> out_types(num_out, -1);

    int retval = inferType(attrs, &in_types, &out_types);
    if (!retval)
      return retval;

    // copy modified input types
    for (int i = 0; i < num_in; i++) {
      intypes[i] = in_types[i];
    }
    // copy output types
    for (int i = 0; i < num_out; i++) {
      outtypes[i] = out_types[i];
    }

    return retval;
  }

  /*! \brief returns status of calling inferSType function for operator from library */
  MX_INT_RET _opCallInferSType(mxnet::ext::inferSType_t inferSType, const char* const* keys,
                               const char* const* vals, int num,
                               int* instypes, int num_in, int* outstypes, int num_out) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of types for inputs
    std::vector<int> in_stypes(num_in);
    for (int i = 0; i < num_in; i++) {
      in_stypes[i] = instypes[i];
    }

    // create a vector of types for outputs
    std::vector<int> out_stypes(num_out, -1);

    int retval = inferSType(attrs, &in_stypes, &out_stypes);

    if (!retval)
      return retval;

    // copy modified input storage types
    for (int i = 0; i < num_in; i++) {
      instypes[i] = in_stypes[i];
    }
    // copy output storage types
    for (int i = 0; i < num_out; i++) {
      outstypes[i] = out_stypes[i];
    }

    return retval;
  }

  /*! \brief returns status of calling Forward/Backward function for operator from library */
  MX_INT_RET _opCallFCompute(mxnet::ext::fcomp_t fcomp, const char* const* keys,
                             const char* const* vals,
                             int num, const int64_t** inshapes, int* indims, void** indata,
                             int* intypes, size_t* inIDs, const char** indev_type, int* indev_id,
                             int num_in, const int64_t** outshapes, int* outdims, void** outdata,
                             int* outtypes, size_t* outIDs, const char** outdev_type,
                             int* outdev_id, int num_out, mxnet::ext::xpu_malloc_t cpu_malloc,
                             void* cpu_alloc,
                             mxnet::ext::xpu_malloc_t gpu_malloc, void* gpu_alloc,
                             void* cuda_stream,
                             mxnet::ext::sparse_malloc_t sparse_malloc, void* sparse_alloc,
                             int* instypes, int* outstypes, void** in_indices, void** out_indices,
                             void** in_indptr, void** out_indptr,
                             int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                             int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                             void* rng_cpu_states, void* rng_gpu_states) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of tensors for inputs
    std::vector<mxnet::ext::MXTensor> inputs(num_in);
    // create a vector for sparse inputs
    std::vector<mxnet::ext::MXSparse> in_sparse(num_in);

    for (int i = 0; i < num_in; i++) {
      // Dense representation.
      if (instypes[i] == 0) {
        inputs[i].setTensor(indata[i], (mxnet::ext::MXDType)intypes[i], inshapes[i], indims[i],
                            inIDs[i], mxnet::ext::MXContext(indev_type[i], indev_id[i]),
                            mxnet::ext::kDefaultStorage);
      } else {
        // Sparse representation.
        mxnet::ext::MXStorageType type;
        if (instypes[i] == 1) {
          type = mxnet::ext::kRowSparseStorage;
          in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i], in_indices_shapes[i]);
        } else {
          type = mxnet::ext::kCSRStorage;
          in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i],
                           in_indices_shapes[i], in_indptr[i], in_indptr_shapes[i]);
        }
        inputs[i].setTensor(reinterpret_cast<void*>(&in_sparse[i]), (mxnet::ext::MXDType)intypes[i],
                            inshapes[i], indims[i], inIDs[i],
                            mxnet::ext::MXContext(indev_type[i], indev_id[i]), type);
      }
    }

    // create a vector of tensors for outputs
    std::vector<mxnet::ext::MXTensor> outputs(num_out);
    std::vector<mxnet::ext::MXSparse> out_sparse(num_out);

    for (int i = 0; i < num_out; i++) {
      // Dense representation.
      if (outstypes[i] == 0) {
        outputs[i].setTensor(outdata[i], (mxnet::ext::MXDType)outtypes[i], outshapes[i], outdims[i],
                             outIDs[i], mxnet::ext::MXContext(outdev_type[i], outdev_id[i]),
                             mxnet::ext::kDefaultStorage);
      } else {
        // Sparse representation.
        mxnet::ext::MXStorageType type;
        if (outstypes[i] == 1) {
          type = mxnet::ext::kRowSparseStorage;
          out_sparse[i].set(outdata[i], outshapes[i], outdims[i],
                            out_indices[i], out_indices_shapes[i]);
        } else {
          type = mxnet::ext::kCSRStorage;
          out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                            out_indices_shapes[i], out_indptr[i], out_indptr_shapes[i]);
        }
        outputs[i].setTensor(reinterpret_cast<void*>(&out_sparse[i]),
                             (mxnet::ext::MXDType)outtypes[i],
                             outshapes[i], outdims[i], outIDs[i],
                             mxnet::ext::MXContext(outdev_type[i], outdev_id[i]), type);
      }
    }

    mxnet::ext::OpResource res(cpu_malloc, cpu_alloc, gpu_malloc, gpu_alloc,
                               cuda_stream, sparse_malloc, sparse_alloc,
                               rng_cpu_states, rng_gpu_states);
    return fcomp(attrs, &inputs, &outputs, res);
  }

  /*! \brief returns status of calling mutateInputs function for operator from library */
  MX_INT_RET _opCallMutateInputs(mxnet::ext::mutateInputs_t mutate, const char* const* keys,
                                 const char* const* vals, int num,
                                 int** mutate_indices, int* indices_size) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of mutate input indices
    std::vector<int> mut_ind;

    int retval = mutate(attrs, &mut_ind);
    if (!retval)
      return retval;

    // output the input indices
    *indices_size = mut_ind.size();
    *mutate_indices = static_cast<int*>(malloc (*indices_size * sizeof(int)));
    for (int i = 0; i < *indices_size; i++) {
      (*mutate_indices)[i] = mut_ind[i];
    }

    return retval;
  }

  /*! \brief returns status of calling createStatefulOp function for operator from library */
  MX_INT_RET _opCallCreateOpState(mxnet::ext::createOpState_t create_op, const char* const* keys,
                                  const char* const* vals, int num,
                                  void** state_op) {
    // create map of attributes from list
    std::unordered_map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // void pointer to hold custom state op instance created in custom library
    // eventually state_op pointer is populated by instance from custom library
    mxnet::ext::CustomStatefulOp** op_ptr =
      reinterpret_cast<mxnet::ext::CustomStatefulOp**>(state_op);
    return create_op(attrs, op_ptr);
  }

  /*! \brief returns status of calling Stateful Forward/Backward for operator from library */
  MX_INT_RET _opCallFStatefulCompute(int is_forward, void* state_op, const int64_t** inshapes,
                                     int* indims, void** indata, int* intypes, size_t* inIDs,
                                     const char** indev_type, int* indev_id, int num_in,
                                     const int64_t** outshapes, int* outdims, void** outdata,
                                     int* outtypes, size_t* outIDs, const char** outdev_type,
                                     int* outdev_id, int num_out,
                                     mxnet::ext::xpu_malloc_t cpu_malloc,
                                     void* cpu_alloc, mxnet::ext::xpu_malloc_t gpu_malloc,
                                     void* gpu_alloc,
                                     void* stream, mxnet::ext::sparse_malloc_t sparse_malloc,
                                     void* sparse_alloc, int* instypes, int* outstypes,
                                     void** in_indices, void** out_indices, void** in_indptr,
                                     void** out_indptr, int64_t* in_indices_shapes,
                                     int64_t* out_indices_shapes, int64_t* in_indptr_shapes,
                                     int64_t* out_indptr_shapes,
                                     void* rng_cpu_states, void* rng_gpu_states) {
    // create a vector of tensors for inputs
    std::vector<mxnet::ext::MXTensor> inputs(num_in);
    // create a vector for sparse inputs
    std::vector<mxnet::ext::MXSparse> in_sparse(num_in);

    for (int i = 0; i < num_in; i++) {
      if (instypes[i] == 0) {
        // Dense representation.
        inputs[i].setTensor(indata[i], (mxnet::ext::MXDType)intypes[i], inshapes[i], indims[i],
                            inIDs[i], mxnet::ext::MXContext(indev_type[i], indev_id[i]),
                            mxnet::ext::kDefaultStorage);
      } else {
        // Sparse representation.
        mxnet::ext::MXStorageType type;
        if (instypes[i] == 1) {
          type = mxnet::ext::kRowSparseStorage;
          in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i], in_indices_shapes[i]);
        } else {
          type = mxnet::ext::kCSRStorage;
          in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i],
                           in_indices_shapes[i], in_indptr[i], in_indptr_shapes[i]);
        }
        inputs[i].setTensor(reinterpret_cast<void*>(&in_sparse[i]), (mxnet::ext::MXDType)intypes[i],
                            inshapes[i], indims[i], inIDs[i],
                            mxnet::ext::MXContext(indev_type[i], indev_id[i]), type);
      }
    }

    // create a vector of tensors for outputs
    std::vector<mxnet::ext::MXTensor> outputs(num_out);
    // create a vector for sparse outputs
    std::vector<mxnet::ext::MXSparse> out_sparse(num_out);

    for (int i = 0; i < num_out; i++) {
      if (outstypes[i] == 0) {
        // Dense representation.
        outputs[i].setTensor(outdata[i], (mxnet::ext::MXDType)outtypes[i], outshapes[i], outdims[i],
                             outIDs[i], mxnet::ext::MXContext(outdev_type[i], outdev_id[i]),
                             mxnet::ext::kDefaultStorage);
      } else {
        // Sparse representation.
        mxnet::ext::MXStorageType type;
        if (outstypes[i] == 1) {
          type = mxnet::ext::kRowSparseStorage;
          out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                            out_indices_shapes[i]);
        } else {
          type = mxnet::ext::kCSRStorage;
          out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                            out_indices_shapes[i], out_indptr[i], out_indptr_shapes[i]);
        }
        outputs[i].setTensor(reinterpret_cast<void*>(&out_sparse[i]),
                             (mxnet::ext::MXDType)outtypes[i],
                             outshapes[i], outdims[i], outIDs[i],
                             mxnet::ext::MXContext(outdev_type[i], outdev_id[i]), type);
      }
    }

    mxnet::ext::OpResource res(cpu_malloc, cpu_alloc, gpu_malloc, gpu_alloc,
                               stream, sparse_malloc, sparse_alloc, rng_cpu_states, rng_gpu_states);

    mxnet::ext::CustomStatefulOp* op_ptr =
      reinterpret_cast<mxnet::ext::CustomStatefulOp*>(state_op);
    if (is_forward) {
      return op_ptr->Forward(&inputs, &outputs, res);
    }
    return op_ptr->Backward(&inputs, &outputs, res);
  }

  /*! \brief returns number of partitioners registered in this library */
  MX_INT_RET _partRegSize() {
    return mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->size();
  }

  /* returns number of strategies registered for partitioner
   * at specified index */
  MX_INT_RET _partRegGetCount(int idx, const char** name) {
    mxnet::ext::CustomPartitioner part =
      mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->get(idx);
    *name = part.name;
    return part.strategies.size();
  }

  /*! \brief returns partitioner registration at specified index */
  MX_VOID_RET _partRegGet(int part_idx, int stg_idx, const char** strategy,
                          mxnet::ext::supportedOps_t* supportedOps,
                          mxnet::ext::createSelector_t* createSelector,
                          mxnet::ext::reviewSubgraph_t* reviewSubgraph, const char** op_name) {
    mxnet::ext::CustomPartitioner part =
      mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->get(part_idx);
    *strategy = part.strategies[stg_idx];
    *op_name = part.op_names[stg_idx];
    *supportedOps = part.getSupportedOps(stg_idx);
    *createSelector = part.getCreateSelector(stg_idx);
    *reviewSubgraph = part.getReviewSubgraph(stg_idx);
  }

  /*! \brief returns status of calling supported ops function from library */
  MX_INT_RET _partCallSupportedOps(mxnet::ext::supportedOps_t supportedOps, const char *json,
                                   int num_ids, int *ids, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts) {
    mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
    // create map of options from list
    std::unordered_map<std::string, std::string> opts;
    for (int i = 0; i < num_opts; i++)
      opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

    // create array of subgraph IDs for operator support
    std::vector<int> _ids(num_ids, -2);
    // call user's supportedOps function
    mxnet::ext::MXReturnValue retval = supportedOps(graph, &_ids, opts);
    if (!retval) return retval;

    // copy bools in ids to ints
    for (int i = 0; i < num_ids; i++)
      ids[i] = _ids[i];

    return retval;
  }

  /*! \brief returns status of calling create selector function from library */
  MX_INT_RET _partCallCreateSelector(mxnet::ext::createSelector_t createSelector, const char *json,
                                     void** selector, const char* const* opt_keys,
                                     const char* const* opt_vals, int num_opts) {
    mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
    // create map of options from list
    std::unordered_map<std::string, std::string> opts;
    for (int i = 0; i < num_opts; i++)
      opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

    // void pointer to hold selector instance created in custom library
    // eventually pointer is populated by instance from custom library
    mxnet::ext::CustomOpSelector** sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector**>(selector);

    // call user's createSelector function
    return createSelector(graph, sel_ptr, opts);
  }

  /*! \brief returns status of calling select function from library */
  MX_VOID_RET _partCallSelect(void* sel_inst, int nodeID, int* selected) {
    mxnet::ext::CustomOpSelector* sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    *selected = sel_ptr->Select(nodeID);
  }

  /*! \brief returns status of calling select input function from library */
  MX_VOID_RET _partCallSelectInput(void* sel_inst, int nodeID,
                                  int input_nodeID, int* selected) {
    mxnet::ext::CustomOpSelector* sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    *selected = sel_ptr->SelectInput(nodeID, input_nodeID);
  }

  /*! \brief returns status of calling select output function from library */
  MX_VOID_RET _partCallSelectOutput(void* sel_inst, int nodeID,
                                    int output_nodeID, int* selected) {
    mxnet::ext::CustomOpSelector* sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    *selected = sel_ptr->SelectOutput(nodeID, output_nodeID);
  }

  /*! \brief returns status of calling filter function from library */
  MX_VOID_RET _partCallFilter(void* sel_inst, int* candidates, int num_candidates,
                              int** keep, int* num_keep) {
    mxnet::ext::CustomOpSelector* sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    std::vector<int> candidates_(num_candidates);
    for (int i=0; i < num_candidates; i++) {
      candidates_[i] = candidates[i];
    }
    std::vector<int> keep_;

    sel_ptr->Filter(candidates_, &keep_);

    *num_keep = keep_.size();
    *keep = static_cast<int*>(malloc(keep_.size() * sizeof(int)));
    for (unsigned i=0; i < keep_.size(); i++)
      (*keep)[i] = keep_[i];
  }

  /*! \brief returns status of calling reset selector function from library */
  MX_VOID_RET _partCallReset(void* sel_inst) {
    mxnet::ext::CustomOpSelector* sel_ptr =
      reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    sel_ptr->Reset();
  }

  /*! \brief returns status of calling review subgraph function from library */
  MX_INT_RET _partCallReviewSubgraph(mxnet::ext::reviewSubgraph_t reviewSubgraph, const char *json,
                                     int subgraph_id, int *accept, const char* const* opt_keys,
                                     const char* const* opt_vals, int num_opts,
                                     char*** attr_keys, char*** attr_vals, int *num_attrs,
                                     const char* const* arg_names, int num_args,
                                     void* const* arg_data, const int64_t* const* arg_shapes,
                                     const int* arg_dims, const int* arg_types,
                                     const size_t* arg_IDs, const char* const* arg_dev_type,
                                     const int* arg_dev_id,
                                     const char* const* aux_names, int num_aux,
                                     void* const* aux_data, const int64_t* const* aux_shapes,
                                     const int* aux_dims, const int* aux_types,
                                     const size_t* aux_IDs, const char* const* aux_dev_type,
                                     const int* aux_dev_id) {
    mxnet::ext::Graph *subgraph = mxnet::ext::Graph::fromString(json);
    bool accept_bool = false;
    // create map of attributes from list
    std::unordered_map<std::string, std::string> opts;
    for (int i = 0; i < num_opts; i++)
      opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

    // create a map of named tensors for args
    std::unordered_map<std::string, mxnet::ext::MXTensor> args;
    for (int i = 0; i < num_args; i++) {
      std::vector<int64_t> shapes;
      for (int j = 0; j < arg_dims[i]; j++)
        shapes.push_back(arg_shapes[i][j]);

      mxnet::ext::MXTensor tensor(arg_data[i], shapes, (mxnet::ext::MXDType)arg_types[i],
                      arg_IDs[i], mxnet::ext::MXContext(arg_dev_type[i], arg_dev_id[i]));
      args[arg_names[i]] = tensor;
    }
    // create a map of named tensors for aux
    std::unordered_map<std::string, mxnet::ext::MXTensor> aux;
    for (int i = 0; i < num_aux; i++) {
      std::vector<int64_t> shapes;
      for (int j = 0; j < aux_dims[i]; j++)
        shapes.push_back(aux_shapes[i][j]);

      mxnet::ext::MXTensor tensor(aux_data[i], shapes, (mxnet::ext::MXDType)aux_types[i],
                                  aux_IDs[i], mxnet::ext::MXContext(aux_dev_type[i],
                                                                    aux_dev_id[i]));
      aux[aux_names[i]] = tensor;
    }

    subgraph->_setParams(&args, &aux);
    mxnet::ext::MXReturnValue retval = reviewSubgraph(subgraph, subgraph_id, &accept_bool,
                                                      opts);
    if (!retval) return retval;

    *accept = accept_bool;

    if (subgraph->attrs.size() > 0) {
      *num_attrs = subgraph->attrs.size();
      // allocate space for attributes
      *attr_keys = static_cast<char**>(malloc (*num_attrs * sizeof(char*)));
      *attr_vals = static_cast<char**>(malloc (*num_attrs * sizeof(char*)));

      // copy attributes
      int i = 0;
      for (auto kv : subgraph->attrs) {
        (*attr_keys)[i] = static_cast<char*>(malloc ((kv.first.size()+1) * sizeof(char)));
        std::string val = kv.second.dump();  // convert JsonVal back to string
        (*attr_vals)[i] = static_cast<char*>(malloc ((val.size()+1) * sizeof(char)));
        snprintf((*attr_keys)[i], kv.first.size()+1, "%s", kv.first.c_str());
        snprintf((*attr_vals)[i], val.size()+1, "%s", val.c_str());
        i++;
      }
    }

    return retval;
  }

  /*! \brief returns number of graph passes registered in this library */
  MX_INT_RET _passRegSize() {
    return mxnet::ext::Registry<mxnet::ext::CustomPass>::get()->size();
  }

  /*! \brief returns pass registration at specified index */
  MX_VOID_RET _passRegGet(int pass_idx, mxnet::ext::graphPass_t* graphPass,
                          const char** pass_name) {
    mxnet::ext::CustomPass pass =
      mxnet::ext::Registry<mxnet::ext::CustomPass>::get()->get(pass_idx);
    *graphPass = pass.pass;
    *pass_name = pass.name;
  }

  /*! \brief returns status of calling graph pass function from library */
  MX_INT_RET _passCallGraphPass(mxnet::ext::graphPass_t graphPass, const char *json,
                                char** out_graph, const char* const* opt_keys,
                                const char* const* opt_vals, int num_opts,
                                const char* pass_name, const char* const* arg_names, int num_args,
                                void* const* arg_data, const int64_t* const* arg_shapes,
                                const int* arg_dims, const int* arg_types,
                                const size_t* arg_IDs, const char* const* arg_dev_type,
                                const int* arg_dev_id, const char* const* aux_names, int num_aux,
                                void* const* aux_data, const int64_t* const* aux_shapes,
                                const int* aux_dims, const int* aux_types,
                                const size_t* aux_IDs, const char* const* aux_dev_type,
                                const int* aux_dev_id, mxnet::ext::nd_malloc_t nd_malloc,
                                const void* nd_alloc) {
    mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
    // create map of attributes from list
    std::unordered_map<std::string, std::string> opts;
    for (int i = 0; i < num_opts; i++)
      opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

    // create a map of named tensors for args
    std::unordered_map<std::string, mxnet::ext::MXTensor> args;
    for (int i = 0; i < num_args; i++) {
      std::vector<int64_t> shapes;
      for (int j = 0; j < arg_dims[i]; j++)
        shapes.push_back(arg_shapes[i][j]);

      mxnet::ext::MXTensor tensor(arg_data[i], shapes, (mxnet::ext::MXDType)arg_types[i],
                                  arg_IDs[i], mxnet::ext::MXContext(arg_dev_type[i],
                                                                    arg_dev_id[i]));
      args[arg_names[i]] = tensor;
    }
    // create a map of named tensors for aux
    std::unordered_map<std::string, mxnet::ext::MXTensor> aux;
    for (int i = 0; i < num_aux; i++) {
      std::vector<int64_t> shapes;
      for (int j = 0; j < aux_dims[i]; j++)
        shapes.push_back(aux_shapes[i][j]);

      mxnet::ext::MXTensor tensor(aux_data[i], shapes, (mxnet::ext::MXDType)aux_types[i],
                                  aux_IDs[i], mxnet::ext::MXContext(aux_dev_type[i],
                                                                    aux_dev_id[i]));
      aux[aux_names[i]] = tensor;
    }

    std::unordered_map<std::string, mxnet::ext::MXTensor> new_args, new_aux;
    mxnet::ext::PassResource res(&new_args, &new_aux, nd_malloc, nd_alloc);
    graph->_setParams(&args, &aux);
    graph->_setPassResource(&res);
    mxnet::ext::MXReturnValue retval = graphPass(graph, opts);
    if (!retval) return retval;

    std::string *tmp = new std::string(graph->toString());
    *out_graph = const_cast<char*>(tmp->c_str());
    return retval;
  }

  /*!
   * \brief Checks if the MXNet version is supported by the library.
   * If supported, initializes the library.
   * \param version MXNet version number passed to library and defined as:
   *                MXNET_VERSION = (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
   * \return Non-zero value on error i.e. library incompatible with passed MXNet version
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) mxnet::ext::MXReturnValue __cdecl
#else
  mxnet::ext::MXReturnValue
#endif
  initialize(int version);
}  // extern "C"
#endif  // MXNET_LIB_API_H_
