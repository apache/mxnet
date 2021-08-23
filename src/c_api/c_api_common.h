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
 *  Copyright (c) 2015 by Contributors
 * \file c_api_common.h
 * \brief Common C API utils
 */
#ifndef MXNET_C_API_C_API_COMMON_H_
#define MXNET_C_API_C_API_COMMON_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_error.h>
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>

/*!
 * \brief Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END() or API_END_HANDLE_ERROR()
 * The finally clause contains procedure to cleanup states when an error happens.
 */
#ifndef API_BEGIN
#define API_BEGIN MX_API_BEGIN
#endif

#ifndef API_END
#define API_END MX_API_END
#endif

#ifndef API_END_HANDLE_ERROR
#define API_END_HANDLE_ERROR MX_API_END_HANDLE_ERROR
#endif

using namespace mxnet;

/*! \brief entry to to easily hold returning information */
template<typename dtype = int>
struct MXAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning handles */
  std::vector<void *> ret_handles;
  /*! \brief holder for NDArray handles */
  std::vector<NDArray*> ndinputs, ndoutputs;
  /*! \brief result holder for returning shapes */
  mxnet::ShapeVector arg_shapes, out_shapes, aux_shapes;
  /*! \brief result holder for returning type flags */
  std::vector<int> arg_types, out_types, aux_types;
  /*! \brief result holder for returning storage types */
  std::vector<int> arg_storage_types, out_storage_types, aux_storage_types;
  /*! \brief result holder for returning shape dimensions */
  std::vector<uint32_t> arg_shape_ndim, out_shape_ndim, aux_shape_ndim;
  /*! \brief result holder for returning shape dimensions */
  std::vector<int> arg_shape_ndim_ex, out_shape_ndim_ex, aux_shape_ndim_ex;
  /*! \brief result holder for returning shape pointer */
  std::vector<const uint32_t*> arg_shape_data, out_shape_data, aux_shape_data;
  /*! \brief result holder for returning shape pointer */
  std::vector<const dtype*> arg_shape_data_ex, out_shape_data_ex, aux_shape_data_ex;
  /*! \brief uint32_t buffer for returning shape pointer */
  std::vector<uint32_t> arg_shape_buffer, out_shape_buffer, aux_shape_buffer;
  /*! \brief uint32_t buffer for returning shape pointer */
  std::vector<dtype> arg_shape_buffer_ex, out_shape_buffer_ex, aux_shape_buffer_ex;
  /*! \brief bool buffer */
  std::vector<bool> save_inputs, save_outputs;
  // DEPRECATED. Use SetupShapeArrayReturnWithBufferEx instead.
  // helper function to setup return value of shape array
  inline static void SetupShapeArrayReturnWithBuffer(
      const mxnet::ShapeVector &shapes,
      std::vector<uint32_t> *ndim,
      std::vector<const uint32_t*> *data,
      std::vector<uint32_t> *buffer) {
    ndim->resize(shapes.size());
    data->resize(shapes.size());
    size_t size = 0;
    for (const auto& s : shapes) size += s.ndim();
    buffer->resize(size);
    uint32_t *ptr = buffer->data();
    for (size_t i = 0; i < shapes.size(); ++i) {
      ndim->at(i) = shapes[i].ndim();
      data->at(i) = ptr;
      ptr = nnvm::ShapeTypeCast(shapes[i].begin(), shapes[i].end(), ptr);
    }
  }
  // helper function to setup return value of shape array
  inline static void SetupShapeArrayReturnWithBufferEx(
      const mxnet::ShapeVector &shapes,
      std::vector<int> *ndim,
      std::vector<const dtype*> *data,
      std::vector<dtype> *buffer) {
    ndim->resize(shapes.size());
    data->resize(shapes.size());
    size_t size = 0;
    for (const auto& s : shapes) {
      if (s.ndim() > 0) {
        size += s.ndim();
      }
    }
    buffer->resize(size);
    dtype* ptr = buffer->data();
    for (size_t i = 0; i < shapes.size(); ++i) {
      ndim->at(i) = shapes[i].ndim();
      data->at(i) = ptr;
      if (shapes[i].ndim() > 0) {
        ptr = mxnet::ShapeTypeCast(shapes[i].begin(), shapes[i].end(), ptr);
      }
    }
  }
};

// define the threadlocal store.
template<typename dtype = int>
using MXAPIThreadLocalStore = dmlc::ThreadLocalStore<MXAPIThreadLocalEntry<dtype>>;

namespace mxnet {
// copy attributes from inferred vector back to the vector of each type.
template<typename AttrType>
inline void CopyAttr(const nnvm::IndexedGraph& idx,
                     const std::vector<AttrType>& attr_vec,
                     std::vector<AttrType>* in_attr,
                     std::vector<AttrType>* out_attr,
                     std::vector<AttrType>* aux_attr) {
  in_attr->clear();
  out_attr->clear();
  aux_attr->clear();
  for (uint32_t nid : idx.input_nodes()) {
    if (idx.mutable_input_nodes().count(nid) == 0) {
      in_attr->push_back(attr_vec[idx.entry_id(nid, 0)]);
    } else {
      aux_attr->push_back(attr_vec[idx.entry_id(nid, 0)]);
    }
  }
  for (auto& e : idx.outputs()) {
    out_attr->push_back(attr_vec[idx.entry_id(e)]);
  }
}

// stores keys that will be converted to __key__
extern const std::vector<std::string> kHiddenKeys;
}  // namespace mxnet

#endif  // MXNET_C_API_C_API_COMMON_H_
