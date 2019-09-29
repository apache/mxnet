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
  MXTensor() : data(NULL) {}

  MXTensor(void *data, const std::vector<int64_t> &shape, MXDType dtype)
  : data(data), shape(shape), dtype(dtype) {}

  /*!
   * \brief helper function to cast data pointer
   */
  template<typename data_type>
  inline data_type* getData() {
    return reinterpret_cast<data_type*>(data);
  }

  /*!
   * \brief helper function to get data size
   */
  inline int64_t getDataSize() {
    int64_t size = 1;
    for (unsigned int i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }
    return size;
  }

  // data is flatten 1D repr of tensor, elements are in continuous memory
  // user can access each element using the shape of tensor
  // it may also point to data allocated on gpu
  void *data;

  // shape is in [2,3,4] format to represent high-dim tensor
  std::vector<int64_t> shape;

  // type can only be MXDType enum types
  MXDType dtype;

  // gpu flag to specify the data tensor storage location
  bool is_gpu;
};

/*!
 * \brief resource malloc function to allocate memory inside Forward/Backward functions
 */
typedef void* (*xpu_malloc_t)(void*, int);

/*!
 * \brief Class to provide resource APIs to Forward/Backward functions
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
 * \brief Macro to help passing serialized subgraph through attribute dict
 */
#define SUBGRAPH_SYM_JSON "subgraph_sym_json"

/*!
 * \brief Simple Json parser to parse serialized subgraph symbol
 */

//Types of JSON objects
enum json_type {ERR,STR,NUM,LIST,MAP};
//forward declaration of struct for JSON objects
struct json_val_t;
typedef struct json_val_t json_val;
//definition of struct for JSON objects
typedef struct json_val_t {
  json_val_t() : type(ERR),num(-1),str("") {} //default constructor
  json_val_t(json_type t) : type(t),num(-1),str("") {} //construct a JSON object by type
  json_val_t(std::string s) : type(STR), num(-1), str(s) {} //construct a string JSON object
  json_val_t(int n) : type(NUM), num(n), str(std::to_string(n)) {} //construct a number JSON object
  json_val_t(json_type t, int n, std::string s) : type(t),num(n),str(s) {}  //complex constructor
  bool operator<(const json_val &o) const {
    if(type == STR) return type == o.type && str < o.str; //for string JSON objects compare the string
    if(type == NUM) return type == o.type && num < o.num; ///for number JSON objects compare the number
    if(type == LIST) { //for list JSON objects, compare the size of the list, and then each object in the lists
      if(list.size() != o.list.size()) return false;
      for(unsigned int i=0; i< list.size(); i++)	if(list[i] < o.list[i]) return false; //if we find an object that doesnt match return
      return true; //all objects in lists matched
    }
    if(type == MAP) { //for map JSON objects, compare the size of the map, and then each key/value in the maps
      if(map.size() != o.map.size()) return false;
      for(auto &item : map) {
	if(o.map.find(item.first) == o.map.end()) return false; //if one map is missing a key in another return
	if(item.second < o.map.at(item.first)) return false;
      }
      return true;
    }
    return type < o.type;
  }
  std::string str;
  int num;
  std::vector<json_val> list;
  std::map<json_val, json_val> map;
  json_type type;
} json_val;
//forward declaration of generic parse function
json_val parse(std::string json, unsigned int *idx);
//debug function to convert a JSON object to a string
std::string json_val_string(const json_val &val) {
  std::string ret;
  switch(val.type) {
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
    for(auto &item : val.list)
      ret += json_val_string(item) + ",";
    ret += "])";
    break;
  case MAP:
    ret = "json(MAP:{";
    for(auto &item : val.map)
      ret += json_val_string(item.first) + " : " + json_val_string(item.second) + ",";
    ret += "})";
    break;
  }
  return ret;
}
//debug function to print a JSON object
void print_json_val(json_val val) {
  std::cout << json_val_string(val) << std::endl;
}
//parse a string JSON object
json_val parse_string(std::string json, unsigned int* idx) {
  json_val ret(STR);
  while(*idx < json.size()) {
    if(json[*idx] == '"') {++(*idx); return ret;
    } else {ret.str += json[*idx]; ++(*idx);}
  }
  std::cout << "Error! Unable to parse string" << std::endl;
  return json_val();
}
//parse a number JSON object
json_val parse_num(std::string json, unsigned int* idx) {
  json_val ret(NUM);
  while(*idx < json.size()) {
    if(json[*idx] >= '0' && json[*idx] <= '9') {ret.str += json[*idx]; ++(*idx);
    } else break;
  }
  ret.num = std::stoi(ret.str);
  return ret;
}
//parse a list of JSON objects
json_val parse_list(std::string json, unsigned int* idx) {
  json_val ret(LIST);
  while(*idx < json.size()) {
    if(json[*idx] == ']') {++(*idx); return ret;
    } else {
      json_val item = parse(json,idx);
      if(item.type != ERR)
	ret.list.push_back(item);
    }
  }
  std::cout << "Error! Unable to parse list" << std::endl;
  return json_val();
}
//parse a map of JSON objects
json_val parse_map(std::string json, unsigned int* idx) {
  json_val ret(MAP),key;
  while(*idx < json.size()) {
    if(json[*idx] == '}') { ++(*idx); return ret;
    } else {
      json_val item = parse(json,idx);
      if(key.type == ERR) key = item;
      else {ret.map[key]=item; key.type = ERR;}
    }
  }
  std::cout << "Error! Unable to parse map" << std::endl;
  return json_val();
}
//generic parse function
json_val parse(std::string json, unsigned int *idx) {
  json_val ret;
  while(*idx < json.size()) {
    if(json[*idx] == '"') {++(*idx); ret = parse_string(json,idx);
    } else if(json[*idx] >= '0' && json[*idx] <= '9') {ret = parse_num(json,idx);
    } else if(json[*idx] == '[') {++(*idx); ret = parse_list(json,idx);
    } else if(json[*idx] == '{') {++(*idx); ret = parse_map(json,idx);
    } else if(json[*idx] == ']' || json[*idx] == '}') {return ret;}
    if(ret.type != ERR) return ret;
    else ++(*idx);
  }
  return ret;
}
// Main entry point to parse a string to JSON
json_val parse_json(std::string json) {
  unsigned int idx=0;
  return parse(json,&idx);
}

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
  virtual ~CustomStatefulOp() = 0;
};

CustomStatefulOp::~CustomStatefulOp() {}

/*!
 * \brief StatefulOp wrapper class to pass to backend OpState
 */
class CustomStatefulOpWrapper {
 public:
  explicit CustomStatefulOpWrapper(CustomStatefulOp* inst) : instance(inst) {}
  CustomStatefulOp* get_instance() { return instance; }
 private:
  CustomStatefulOp* instance;
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
  explicit CustomOp(const char* op_name) : name(op_name), forward(NULL),
    backward(NULL), parse_attrs(NULL), infer_type(NULL), infer_shape(NULL),
    mutate_inputs(NULL), create_opstate(NULL) {}
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
   * \brief returns status of calling Forward function for operator from library
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
