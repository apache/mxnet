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
 * Copyright (c) 2015 by Contributors
 * \file lib_api.h
 * \brief APIs to interact with libraries
 */
#ifndef MXNET_LIB_API_H_
#define MXNET_LIB_API_H_

#include <vector>
#include <map>
#include <stdint.h>

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

/*!
 * \brief External Tensor data structure
 */
struct MXTensor {
  MXTensor() { data = nullptr; }
  MXTensor(void *data, const std::vector<int64_t> &shape, MXDType dtype)
  : data{data}, shape{shape}, dtype{dtype} {}

  /*!
   * \brief helper function to cast data pointer
   */
  template<typename data_type>
  data_type* getData() {
    return (data_type*)data;
  }

  void *data; // not owned
  std::vector<int64_t> shape;
  MXDType dtype;
};

/*!
 * Custom Operator function templates
 */
typedef int (*fcomp_t)(std::map<std::string,std::string>,
                       std::vector<MXTensor>, std::vector<MXTensor>);
typedef int (*parseAttrs_t)(std::map<std::string,std::string>,
                            int*, int*);
typedef int (*inferType_t)(std::map<std::string,std::string>,
                           std::vector<int>&, std::vector<int>&);
typedef int (*inferShape_t)(std::map<std::string,std::string>,
                            std::vector<std::vector<unsigned int>>&,
                            std::vector<std::vector<unsigned int>>&);

/*!
 * \brief Class to hold custom operator registration
 */
class CustomOp {
 public:
 CustomOp(const char* op_name) : name(op_name), fcompute(nullptr),
    parse_attrs(nullptr), infer_type(nullptr), infer_shape(nullptr) {}
  ~CustomOp() {}
  CustomOp& setFCompute(fcomp_t fcomp) {
    fcompute = fcomp;
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
  /*! \brief operator name */
  const char* name;
  /*! \brief operator functions */
  fcomp_t fcompute;
  parseAttrs_t parse_attrs;
  inferType_t infer_type;
  inferShape_t infer_shape;
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
#define REGISTER_OP(Name) _STR_CONCAT(_REGISTER_DEF_(Name), __COUNTER__) = Registry<CustomOp>::get()->add(TOSTRING(Name))


/*!
 * \brief Following are the APIs implemented in the external library
 * Each API has a #define string that is used to lookup the function in the library
 * Followed by the function declaration
 */


#define MXLIB_OPREGSIZE_STR "_opRegSize"
typedef int (*opRegSize_t)(void);

#define MXLIB_OPREGGET_STR "_opRegGet"
typedef int (*opRegGet_t)(int, const char**, fcomp_t*,
                          parseAttrs_t*, inferType_t*,
                          inferShape_t*);

#define MXLIB_OPCALLFCOMP_STR "_opCallFCompute"
typedef int (*opCallFComp_t)(fcomp_t, const char* const*, const char* const*, int,
                             const int64_t**, int*, void**, int*, int,
                             const int64_t**, int*, void**, int*, int);

#define MXLIB_INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int);

extern "C" {
  /*!
   * \brief returns number of ops registered in this library
   */
  int _opRegSize() {
    return Registry<CustomOp>::get()->size();
  }

  /*!
   * \brief returns operator registration at specified index
   */
  void _opRegGet(int idx, const char** name, fcomp_t* fcomp,
                 parseAttrs_t* parse, inferType_t* type,
                 inferShape_t* shape) {
    CustomOp op = Registry<CustomOp>::get()->get(idx);
    *name = op.name;
    *fcomp = op.fcompute;
    *parse = op.parse_attrs;
    *type = op.infer_type;
    *shape = op.infer_shape;
  }

  int _opCallFCompute(fcomp_t fcomp, const char* const* keys, const char* const* vals, int num,
                      const int64_t** inshapes, int* indims, void** indata, int* intypes, int num_in,
                      const int64_t** outshapes, int* outdims, void** outdata, int* outtypes, int num_out) {
    //create map of attributes from list
    std::map<std::string,std::string> attrs;
    for(int i=0; i<num; i++) {
      attrs[std::string(keys[i])] = std::string(vals[i]);
    }

    //create a vector of tensors for inputs
    std::vector<MXTensor> inputs(num_in);
    for(int i=0; i<num_in; i++) {
      inputs[i].data = indata[i];
      inputs[i].dtype = (MXDType)intypes[i];
      for(int j=0; j<indims[i]; j++) {
        inputs[i].shape.push_back(inshapes[i][j]);
      }
    }

    //create a vector of tensors for outputs
    std::vector<MXTensor> outputs(num_out);
    for(int i=0; i<num_out; i++) {
      outputs[i].data = outdata[i];
      outputs[i].dtype = (MXDType)outtypes[i];
      for(int j=0; j<outdims[i]; j++) {
        outputs[i].shape.push_back(outshapes[i][j]);
      }
    }

    return fcomp(attrs,inputs,outputs);
  }
  
  /*!
   * \brief Checks if the MXNet version is supported by the library.
   * If supported, initializes the library.
   * \param version MXNet version number passed to library and defined as:
   *                MXNET_VERSION = (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
   * \return Non-zero value on error i.e. library incompatible with passed MXNet version
   */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  __declspec(dllexport) int __cdecl initialize(int);
#else
  int initialize(int);
#endif
}
#endif  // MXNET_LIB_API_H_
