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
#include <vector>
#include <map>
#include <string>

#define MX_LIBRARY_VERSION 1

/*!
 * \brief External Tensor data types
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
 * \brief External Tensor data structure
 */
struct MXTensor {
  MXTensor() : data(nullptr) {}

  MXTensor(void *data, const std::vector<int64_t> &shape, MXDType dtype)
  : data{data}, shape{shape}, dtype{dtype} {}

  /*!
   * \brief helper function to cast data pointer
   */
  template<typename data_type>
  data_type* getData() {
    return reinterpret_cast<data_type*>(data);
  }

  void *data;  // not owned
  std::vector<int64_t> shape;
  MXDType dtype;
};

/*!
 * \brief resource malloc function to allocate memory inside fcompute function
 */
typedef void* (*xpu_malloc_t)(void*, int);

/*!
 * \brief Class to provide resource APIs to FCompute
 */
class OpResource {
 public:
  OpResource(xpu_malloc_t xm, void* _xm) : xpu_malloc(xm), _xpu_malloc(_xm) {}

  /*!
   * \brief allocate memory controlled by MXNet
   */
  void* alloc(int size) {
    return xpu_malloc(_xpu_malloc, size);
  }
 private:
  xpu_malloc_t xpu_malloc;
  void* _xpu_malloc;
};

/*!
 * \brief StatefulOp wrapper class to pass to backend OpState
 */
class CustomStatefulOpWrapper {
 public:
  CustomStatefulOpWrapper(void* inst) : instance(inst) {}
  void* get_instance() { return instance; }
 private:
  void* instance;
};

/*!
 * \brief An prototype interface class for library author creating stateful op
 */
class CustomStatefulOp {
 public:
  virtual void Forward() = 0;
  virtual ~CustomStatefulOp() = 0;
};

/*!
 * Custom Operator function templates
 */
typedef MXReturnValue (*fcomp_t)(std::map<std::string, std::string>,
                                 std::vector<MXTensor>, std::vector<MXTensor>,
                                 OpResource res);
typedef MXReturnValue (*parseAttrs_t)(std::map<std::string, std::string>,
                                      int*, int*);
typedef MXReturnValue (*inferType_t)(std::map<std::string, std::string>,
                                     std::vector<int>&, std::vector<int>&);
typedef MXReturnValue (*inferShape_t)(std::map<std::string, std::string>,
                                      std::vector<std::vector<unsigned int>>&,
                                      std::vector<std::vector<unsigned int>>&);
typedef MXReturnValue (*mutateInputs_t)(std::map<std::string, std::string>,
                                      std::vector<int>&);
typedef MXReturnValue (*createOpState_t)(std::map<std::string, std::string>,
                                      CustomStatefulOp**);
typedef MXReturnValue (*fstateful_t)(CustomStatefulOp*, std::vector<MXTensor>,
                                      std::vector<MXTensor>);

/*!
 * \brief Class to hold custom operator registration
 */
class CustomOp {
 public:
  explicit CustomOp(const char* op_name) : name(op_name), fcompute(nullptr),
    fgradient(nullptr), parse_attrs(nullptr), infer_type(nullptr), infer_shape(nullptr),
    mutate_inputs(nullptr), create_op_state(nullptr), fstateful(nullptr) {}
  ~CustomOp() {}
  CustomOp& setForward(fcomp_t fcomp) {
    fcompute = fcomp;
    return *this;
  }
  CustomOp& setGradient(fcomp_t fcomp) {
    fgradient = fcomp;
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
    create_op_state = func;
    return *this;
  }
  CustomOp& setForwardStateful(fstateful_t func) {
    fstateful = func;
    return *this;
  }
  /*! \brief operator name */
  const char* name;
  /*! \brief operator functions */
  fcomp_t fcompute;
  fcomp_t fgradient;
  parseAttrs_t parse_attrs;
  inferType_t infer_type;
  inferShape_t infer_shape;
  mutateInputs_t mutate_inputs;
  createOpState_t create_op_state;
  fstateful_t fstateful;
};

/*!
 * \brief Registry class to registers things (ops, properties)
 *       Singleton class
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

/*
 * Macros to help with string concat
 * Annoyingly, the concat_ and concat macros are necessary to
 * be able to use __COUNTER__ in an identifier name 
 */
#define _STR_CONCAT_(__a, __b) __a ## __b
#define _STR_CONCAT(__a, __b) _STR_CONCAT_(__a, __b)

/*!
 * \brief convert a token to a string
 */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/*!
 * \brief declare a variable with custom name
 */
#define _REGISTER_NAME_(Name) MXNet ## _CustomOp ## _
#define _REGISTER_DEF_(Name) CustomOp _REGISTER_NAME_(Name)

/*!
 * \brief assign a var to a value
 */
#define REGISTER_OP(Name) _STR_CONCAT(_REGISTER_DEF_(Name), __COUNTER__) = \
    Registry<CustomOp>::get()->add(TOSTRING(Name))

/*
 * -------------- BELOW FUNCTIONS ARE USED IN MXNET BACKEND ---------------
 */

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
                          createOpState_t*, fstateful_t*);

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

#define MXLIB_OPCALLMUTATEINPUTS_STR "_opCallMutateInputs"
typedef int (*opCallMutateInputs_t)(mutateInputs_t, const char* const*, const char* const*, int,
                                    int**, int*);

#define MXLIB_OPCALLCREATEOPSTATE_STR "_opCallCreateOpState"
typedef int (*opCallCreateOpState_t)(createOpState_t, const char* const*, const char* const*, int,
                                     void**);

#define MXLIB_OPCALLFSTATEFUL_STR "_opCallFStateful"
typedef int (*opCallFStateful_t)(fstateful_t, void*,
                                 const int64_t**, int*, void**, int*, int,
                                 const int64_t**, int*, void**, int*, int);

#define MXLIB_INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int);

#define MXLIB_OPVERSION_STR "_opVersion"
typedef int (*opVersion_t)();

extern "C" {
  /*!
   * \brief returns MXNet library version 
   */
  #if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opVersion() {
    return MX_LIBRARY_VERSION;
  }

  /*!
   * \brief returns number of ops registered in this library
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opRegSize() {
    return Registry<CustomOp>::get()->size();
  }

  /*!
   * \brief returns operator registration at specified index
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) void __cdecl
#else
  void
#endif
  _opRegGet(int idx, const char** name, fcomp_t* fcomp, fcomp_t* fgrad,
            parseAttrs_t* parse, inferType_t* type,
            inferShape_t* shape, mutateInputs_t* mutate,
            createOpState_t* create_op, fstateful_t* fstateful) {
    CustomOp op = Registry<CustomOp>::get()->get(idx);
    *name = op.name;
    *fcomp = op.fcompute;
    *fgrad = op.fgradient;
    *parse = op.parse_attrs;
    *type = op.infer_type;
    *shape = op.infer_shape;
    *mutate = op.mutate_inputs;
    *create_op = op.create_op_state;
    *fstateful = op.fstateful;
  }

  /*!
   * \brief calls free from the external library for library allocated arrays
   */
  #if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) void __cdecl
#else
  void
#endif
  _opCallFree(void* ptr) {
    free(ptr);
  }

  /*!
   * \brief returns status of calling parse attributes function for operator from library
   */
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

  /*!
   * \brief returns status of calling infer shape function for operator from library
   */
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

  /*!
   * \brief returns status of calling InferType function for operator from library
   */
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

  /*!
   * \brief returns status of calling FCompute function for operator from library
   */

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
                  xpu_malloc_t xpu_malloc, void* _xpu_malloc) {
    // create map of attributes from list
    std::map<std::string, std::string> attrs;
    for (int i = 0; i < num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    // create a vector of tensors for inputs
    std::vector<MXTensor> inputs(num_in);
    for (int i = 0; i < num_in; i++) {
      inputs[i].data = indata[i];
      inputs[i].dtype = (MXDType)intypes[i];
      for (int j = 0; j < indims[i]; j++) {
        inputs[i].shape.push_back(inshapes[i][j]);
      }
    }

    // create a vector of tensors for outputs
    std::vector<MXTensor> outputs(num_out);
    for (int i = 0; i < num_out; i++) {
      outputs[i].data = outdata[i];
      outputs[i].dtype = (MXDType) outtypes[i];
      for (int j = 0; j < outdims[i]; j++) {
        outputs[i].shape.push_back(outshapes[i][j]);
      }
    }

    OpResource res(xpu_malloc, _xpu_malloc);

    return fcomp(attrs, inputs, outputs, res);
  }

  /*!
   * \brief returns status of calling mutate inputs function for operator from library
   */
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

  /*!
   * \brief returns status of calling create stateful op function for operator from library
   */
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

  /*!
   * \brief returns status of calling FStateful function for operator from library
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl
#else
  int
#endif
  _opCallFStateful(fstateful_t fstateful, void* state_op_inst,
                  const int64_t** inshapes, int* indims,
                  void** indata, int* intypes, int num_in,
                  const int64_t** outshapes, int* outdims,
                  void** outdata, int* outtypes, int num_out) {
    // create a vector of tensors for inputs
    std::vector<MXTensor> inputs(num_in);
    for (int i = 0; i < num_in; i++) {
      inputs[i].data = indata[i];
      inputs[i].dtype = (MXDType)intypes[i];
      for (int j = 0; j < indims[i]; j++) {
        inputs[i].shape.push_back(inshapes[i][j]);
      }
    }

    // create a vector of tensors for outputs
    std::vector<MXTensor> outputs(num_out);
    for (int i = 0; i < num_out; i++) {
      outputs[i].data = outdata[i];
      outputs[i].dtype = (MXDType) outtypes[i];
      for (int j = 0; j < outdims[i]; j++) {
        outputs[i].shape.push_back(outshapes[i][j]);
      }
    }

    // pass the stateful op instance to stateful forward in custom library
    CustomStatefulOp* state_op = reinterpret_cast<CustomStatefulOp*>(state_op_inst);
    return fstateful(state_op, inputs, outputs);
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
