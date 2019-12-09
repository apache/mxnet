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
 * register custom ops for library authors
 */

#ifndef MXNET_LIB_API_H_
#define MXNET_LIB_API_H_

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <utility>
#include <stdexcept>

#define MX_LIBRARY_VERSION 1

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
     *  can be NULL, indicating tensor is compact and row-majored.
     */
    int64_t* strides;
    /*! \brief The offset in bytes to the beginning pointer to data */
    uint64_t byte_offset;
  } DLTensor;
#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif

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
};

enum MXReturnValue {
  MX_FAIL = 0,
  MX_SUCCESS = 1,
};

/*!
 * \brief Tensor data structure used by custom operator
 */
struct MXTensor {
  MXTensor() : data_ptr(NULL) {}

  MXTensor(void *data_ptr, const std::vector<int64_t> &shape, MXDType dtype)
  : data_ptr(data_ptr), shape(shape), dtype(dtype) {}

  /*! \brief populate DLTensor fields */
  void setDLTensor() {
    dltensor.data = data_ptr;
    dltensor.ctx.device_type = kDLCPU;
    dltensor.ctx.device_id = 0;
    dltensor.ndim = shape.size();
    dltensor.shape = const_cast<int64_t*>(shape.data());
    dltensor.strides = NULL;
    dltensor.byte_offset = 0;
    dltensor.dtype.lanes = 1;
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
  inline int64_t size() {
    int64_t size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }
    return size;
  }

  // data is flatten 1D repr of tensor, elements are in continuous memory
  // user can access each element using the shape of tensor
  void *data_ptr;

  // shape is in [2,3,4] format to represent high-dim tensor
  std::vector<int64_t> shape;

  // type can only be MXDType enum types
  MXDType dtype;

  // corresponding DLTensor repr of MXTensor
  // easy way to reuse functions taking DLTensor
  DLTensor dltensor;
};

/*!
 * \brief resource malloc function to allocate memory inside Forward/Backward functions
 */
typedef void* (*xpu_malloc_t)(void*, int);

/*!
 * \brief provide resource APIs memory allocation mechanism to Forward/Backward functions
 */
class OpResource {
 public:
  OpResource(xpu_malloc_t cm, void* ca) : cpu_malloc(cm), cpu_alloc(ca) {}

  /*! \brief allocate memory controlled by MXNet */
  void* alloc(int size) {
    return cpu_malloc(cpu_alloc, size);
  }

 private:
  xpu_malloc_t cpu_malloc;
  void* cpu_alloc;
};

/*!
 * \brief Json utility to parse serialized subgraph symbol
 */
/*! \brief Macro to help passing serialized subgraph through attribute dict */
#define SUBGRAPH_SYM_JSON "subgraph_sym_json"

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
  JsonType type;
  int num;
  std::string str;
  std::vector<JsonVal> list;
  std::map<JsonVal, JsonVal> map;
};

/*! \brief functions used for parsing JSON */
struct JsonParser {
  JsonVal parse_to_json(std::string json) {
    unsigned int idx = 0;
    return parse(json, &idx);
  }
  void print_json_val(JsonVal val) {
    std::cout << json_val_string(val) << std::endl;
  }
  // debug function to convert a JSON object to a string
  std::string json_val_string(const JsonVal &val) {
    std::string ret;
    switch (val.type) {
    case ERR:
      ret = "json(Error)";
      break;
    case STR:
      ret = "json(STR:" + val.str + ")";
      break;
    case NUM:
      ret = "json(INT:" + val.str + ")";
      break;
    case LIST:
      ret = "json(LIST:[";
      for (auto &item : val.list)
        ret += json_val_string(item) + ",";
      ret += "])";
      break;
    case MAP:
      ret = "json(MAP:{";
      for (auto &item : val.map)
        ret += json_val_string(item.first) + " : " + json_val_string(item.second) + ",";
      ret += "})";
      break;
    }
    return ret;
  }
  // parse a string JSON object
  JsonVal parse_string(std::string json, unsigned int* idx) {
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
  JsonVal parse_num(std::string json, unsigned int* idx) {
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
  JsonVal parse_list(std::string json, unsigned int* idx) {
    JsonVal ret(LIST);
    while (*idx < json.size()) {
      if (json[*idx] == ']') {
        ++(*idx);
        return ret;
      } else {
        JsonVal item = parse(json, idx);
        if (item.type != ERR)
          ret.list.push_back(item);
      }
    }
    std::cout << "Error! Unable to parse list" << std::endl;
    return JsonVal();
  }
  // parse a map of JSON objects
  JsonVal parse_map(std::string json, unsigned int* idx) {
    JsonVal ret(MAP), key;
    while (*idx < json.size()) {
      if (json[*idx] == '}') {
        ++(*idx);
        return ret;
      } else {
        JsonVal item = parse(json, idx);
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
  JsonVal parse(std::string json, unsigned int *idx) {
    JsonVal ret;
    while (*idx < json.size()) {
      if (json[*idx] == '"') {
        ++(*idx);
        ret = parse_string(json, idx);
      } else if (json[*idx] >= '0' && json[*idx] <= '9') {
        ret = parse_num(json, idx);
      } else if (json[*idx] == '[') {
        ++(*idx);
        ret = parse_list(json, idx);
      } else if (json[*idx] == '{') {
        ++(*idx);
        ret = parse_map(json, idx);
      } else if (json[*idx] == ']' || json[*idx] == '}') {return ret;}
      if (ret.type != ERR) return ret;
      ++(*idx);
    }
    return ret;
  }
};

/*!
 * \brief An abstract class for library author creating stateful op
 * custom library should override Forward and destructor, and has an
 * option to implement Backward
 */
class CustomStatefulOp {
 public:
  virtual MXReturnValue Forward(std::vector<MXTensor> inputs,
                                std::vector<MXTensor> outputs,
                                OpResource op_res) = 0;
  virtual MXReturnValue Backward(std::vector<MXTensor> inputs,
                                 std::vector<MXTensor> outputs,
                                 OpResource op_res) {
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
typedef MXReturnValue (*fcomp_t)(std::map<std::string, std::string>,
                                 std::vector<MXTensor>, std::vector<MXTensor>,
                                 OpResource res);
typedef MXReturnValue (*parseAttrs_t)(std::map<std::string, std::string>,
                                      int*, int*);
typedef MXReturnValue (*inferType_t)(std::map<std::string, std::string>,
                                     std::vector<int>&, std::vector<int>&);
typedef MXReturnValue (*inferShape_t)(std::map<std::string, std::string>,
                                      std::vector<std::vector<unsigned int> >&,
                                      std::vector<std::vector<unsigned int> >&);
typedef MXReturnValue (*mutateInputs_t)(std::map<std::string, std::string>,
                                      std::vector<int>&);
typedef MXReturnValue (*createOpState_t)(std::map<std::string, std::string>,
                                      CustomStatefulOp**);

/*!
 * \brief Class to hold custom operator registration
 */
class CustomOp {
 public:
  explicit CustomOp(const char* op_name) : name(op_name),
    forward(NULL), backward(NULL), parse_attrs(NULL), infer_type(NULL),
    infer_shape(NULL), mutate_inputs(NULL), create_opstate(NULL) {}
  ~CustomOp() {}
  CustomOp& setForward(fcomp_t fcomp) {
    forward = fcomp;
    return *this;
  }
  CustomOp& setBackward(fcomp_t fcomp) {
    backward = fcomp;
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
  CustomOp& setInferShape(inferShape_t func) {
    infer_shape = func;
    return *this;
  }
  CustomOp& setMutateInputs(mutateInputs_t func) {
    mutate_inputs = func;
    return *this;
  }
  CustomOp& setCreateOpState(createOpState_t func) {
    create_opstate = func;
    return *this;
  }

  /*! \brief operator name */
  const char* name;
  /*! \brief operator functions */
  fcomp_t forward;
  fcomp_t backward;
  parseAttrs_t parse_attrs;
  inferType_t infer_type;
  inferShape_t infer_shape;
  mutateInputs_t mutate_inputs;
  createOpState_t create_opstate;
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
  static Registry* get() {
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
    return *(entries[idx]);
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

/*! \brief assign a var to a value */
#define REGISTER_OP(Name) MX_STR_CONCAT(MX_REGISTER_DEF_(Name), __COUNTER__) = \
    Registry<CustomOp>::get()->add(MX_TOSTRING(Name))

/* -------------- BELOW ARE CTYPE FUNCTIONS PROTOTYPES --------------- */

/*!
 * \brief Following are the C type APIs implemented in the external library
 * Each API has a #define string that is used to lookup the function in the library
 * Followed by the function declaration
 */
#define MXLIB_OPREGSIZE_STR "_opRegSize"
typedef int (*opRegSize_t)(void);

#define MXLIB_OPREGGET_STR "_opRegGet"
typedef int (*opRegGet_t)(int, const char**, fcomp_t*, fcomp_t*,
                          parseAttrs_t*, inferType_t*,
                          inferShape_t*, mutateInputs_t*,
                          createOpState_t*);

#define MXLIB_OPCALLFREE_STR "_opCallFree"
typedef int (*opCallFree_t)(void*);

#define MXLIB_OPCALLPARSEATTRS_STR "_opCallParseAttrs"
typedef int (*opCallParseAttrs_t)(parseAttrs_t, const char* const*, const char* const*, int,
                                  int*, int*);

#define MXLIB_OPCALLINFERSHAPE_STR "_opCallInferShape"
typedef int (*opCallInferShape_t)(inferShape_t, const char* const*, const char* const*, int,
                                  unsigned int**, int*, int,
                                  unsigned int***, int**, int);

#define MXLIB_OPCALLINFERTYPE_STR "_opCallInferType"
typedef int (*opCallInferType_t)(inferType_t, const char* const*, const char* const*, int,
                                  int*, int, int*, int);

#define MXLIB_OPCALLFCOMP_STR "_opCallFCompute"
typedef int (*opCallFComp_t)(fcomp_t, const char* const*, const char* const*, int,
                           const int64_t**, int*, void**, int*, int,
                           const int64_t**, int*, void**, int*, int,
                           xpu_malloc_t, void*);

#define MXLIB_OPCALLBKWD_STR "_opCallBackward"
typedef int (*opCallBkwd_t)(fcomp_t, const char* const*, const char* const*, int,
                            const int64_t**, int*, void**, int*, int,
                            const int64_t**, int*, void**, int*, int,
                            xpu_malloc_t, void*);

#define MXLIB_OPCALLMUTATEINPUTS_STR "_opCallMutateInputs"
typedef int (*opCallMutateInputs_t)(mutateInputs_t, const char* const*, const char* const*, int,
                                    int**, int*);

#define MXLIB_OPCALLCREATEOPSTATE_STR "_opCallCreateOpState"
typedef int (*opCallCreateOpState_t)(createOpState_t, const char* const*, const char* const*, int,
                                     void**);

#define MXLIB_OPCALLFSTATEFULCOMP_STR "_opCallFStatefulCompute"
typedef int (*opCallFStatefulComp_t)(bool, void*, const int64_t**, int*, void**, int*, int,
                                     const int64_t**, int*, void**, int*, int,
                                     xpu_malloc_t, void*);

#define MXLIB_INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int);

#define MXLIB_OPVERSION_STR "_opVersion"
typedef int (*opVersion_t)();

extern "C" {
  /*! \brief returns MXNet library version */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opVersion() {
    return MX_LIBRARY_VERSION;
  }

  /*! \brief returns number of ops registered in this library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opRegSize() {
    return Registry<CustomOp>::get()->size();
  }

  /*! \brief returns operator registration at specified index */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) void __cdecl
#else
  void
#endif
  _opRegGet(int idx, const char** name, fcomp_t* fcomp, fcomp_t* fgrad,
            parseAttrs_t* parse, inferType_t* type,
            inferShape_t* shape, mutateInputs_t* mutate,
            createOpState_t* create_op) {
    CustomOp op = Registry<CustomOp>::get()->get(idx);
    *name = op.name;
    *fcomp = op.forward;
    *fgrad = op.backward;
    *parse = op.parse_attrs;
    *type = op.infer_type;
    *shape = op.infer_shape;
    *mutate = op.mutate_inputs;
    *create_op = op.create_opstate;
  }

  /*! \brief calls free from the external library for library allocated arrays */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) void __cdecl
#else
  void
#endif
  _opCallFree(void* ptr) {
    free(ptr);
  }

  /*! \brief returns status of calling parse attributes function for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallParseAttrs(parseAttrs_t parseAttrs, const char* const* keys,
                    const char* const* vals, int num,
                    int* num_in, int* num_out) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    return parseAttrs(attrs, num_in, num_out);
  }

  /*! \brief returns status of calling inferShape function for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallInferShape(inferShape_t inferShape, const char* const* keys,
                    const char* const* vals, int num,
                    unsigned int** inshapes, int* indims, int num_in,
                    unsigned int*** outshapes, int** outdims, int num_out) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
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

    int retval = inferShape(attrs, in_shapes, out_shapes);
    if (!retval)
      return retval;

    // allocate space for output dims, shape
    *outdims = static_cast<int*>(malloc (num_out * sizeof(int)));
    *outshapes = static_cast<unsigned**>(malloc (num_out * sizeof(unsigned*)));

    // copy output shapes
    for (int i = 0; i < num_out; i++) {
      (*outdims)[i] = out_shapes[i].size();
      (*outshapes)[i] = static_cast<unsigned*>(malloc ((*outdims)[i] * sizeof(unsigned)));
      for (int j = 0; j < indims[i]; j++) {
        (*outshapes)[i][j] = out_shapes[i][j];
      }
    }

    return retval;
  }

  /*! \brief returns status of calling inferType function for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallInferType(inferType_t inferType, const char* const* keys,
                   const char* const* vals, int num,
                   int* intypes, int num_in, int* outtypes, int num_out) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
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

    int retval = inferType(attrs, in_types, out_types);
    if (!retval)
      return retval;

    // copy output types
    for (int i = 0; i < num_out; i++) {
      outtypes[i] = out_types[i];
    }

    return retval;
  }

  /*! \brief returns status of calling Forward/Backward function for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallFCompute(fcomp_t fcomp, const char* const* keys,
                  const char* const* vals, int num,
                  const int64_t** inshapes, int* indims,
                  void** indata, int* intypes, int num_in,
                  const int64_t** outshapes, int* outdims,
                  void** outdata, int* outtypes, int num_out,
                  xpu_malloc_t cpu_malloc, void* cpu_alloc) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of tensors for inputs
    std::vector<MXTensor> inputs(num_in);
    for (int i = 0; i < num_in; i++) {
      inputs[i].data_ptr = indata[i];
      inputs[i].dtype = (MXDType)intypes[i];
      for (int j = 0; j < indims[i]; j++) {
        inputs[i].shape.push_back(inshapes[i][j]);
      }
      inputs[i].setDLTensor();
    }

    // create a vector of tensors for outputs
    std::vector<MXTensor> outputs(num_out);
    for (int i = 0; i < num_out; i++) {
      outputs[i].data_ptr = outdata[i];
      outputs[i].dtype = (MXDType) outtypes[i];
      for (int j = 0; j < outdims[i]; j++) {
        outputs[i].shape.push_back(outshapes[i][j]);
      }
      outputs[i].setDLTensor();
    }

    OpResource res(cpu_malloc, cpu_alloc);

    return fcomp(attrs, inputs, outputs, res);
  }

  /*! \brief returns status of calling mutateInputs function for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallMutateInputs(mutateInputs_t mutate, const char* const* keys,
                    const char* const* vals, int num,
                    int** mutate_indices, int* indices_size) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of mutate input indices
    std::vector<int> mut_ind;

    int retval = mutate(attrs, mut_ind);
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
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallCreateOpState(createOpState_t create_op, const char* const* keys,
                       const char* const* vals, int num,
                       void** state_op) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // void pointer to hold custom state op instance created in custom library
    CustomStatefulOp** op_ptr = reinterpret_cast<CustomStatefulOp**>(state_op);
    return create_op(attrs, op_ptr);
  }

  /*! \brief returns status of calling Stateful Forward/Backward for operator from library */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallFStatefulCompute(bool is_forward, void* state_op,
                          const int64_t** inshapes, int* indims,
                          void** indata, int* intypes, int num_in,
                          const int64_t** outshapes, int* outdims,
                          void** outdata, int* outtypes, int num_out,
                          xpu_malloc_t cpu_malloc, void* cpu_alloc) {
    // create a vector of tensors for inputs
    std::vector<MXTensor> inputs(num_in);
    for (int i = 0; i < num_in; i++) {
      inputs[i].data_ptr = indata[i];
      inputs[i].dtype = (MXDType)intypes[i];
      for (int j = 0; j < indims[i]; j++) {
        inputs[i].shape.push_back(inshapes[i][j]);
      }
      inputs[i].setDLTensor();
    }

    // create a vector of tensors for outputs
    std::vector<MXTensor> outputs(num_out);
    for (int i = 0; i < num_out; i++) {
      outputs[i].data_ptr = outdata[i];
      outputs[i].dtype = (MXDType) outtypes[i];
      for (int j = 0; j < outdims[i]; j++) {
        outputs[i].shape.push_back(outshapes[i][j]);
      }
      outputs[i].setDLTensor();
    }
    OpResource res(cpu_malloc, cpu_alloc);
    CustomStatefulOp* op_ptr = reinterpret_cast<CustomStatefulOp*>(state_op);
    if (is_forward) {
      return op_ptr->Forward(inputs, outputs, res);
    }
    return op_ptr->Backward(inputs, outputs, res);
  }

  /*!
   * \brief Checks if the MXNet version is supported by the library.
   * If supported, initializes the library.
   * \param version MXNet version number passed to library and defined as:
   *                MXNET_VERSION = (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
   * \return Non-zero value on error i.e. library incompatible with passed MXNet version
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) MXReturnValue __cdecl
#else
  MXReturnValue
#endif
  initialize(int version);
}
#endif  // MXNET_LIB_API_H_
