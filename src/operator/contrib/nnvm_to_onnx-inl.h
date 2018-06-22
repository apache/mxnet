#ifndef MXNET_OPERATOR_CONTRIB_NNVM_TO_ONNX_INL_H_
#define MXNET_OPERATOR_CONTRIB_NNVM_TO_ONNX_INL_H_
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
 * Copyright (c) 2018 by Contributors
 * \file tensorrt-inl.h
 * \brief TensorRT Operator
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>
#include <dmlc/parameter.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>

#include <NvInfer.h>
#include <onnx/onnx.pb.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <string>

#include "./tensorrt-inl.h"
#include "../operator_common.h"
#include "../../common/utils.h"
#include "../../common/serialization.h"

namespace mxnet {
namespace op {
namespace nnvm_to_onnx {

using namespace nnvm;
using namespace ::onnx;
using int64 = ::google::protobuf::int64;

std::unordered_map<std::string, TShape> GetPlaceholderShapes(const ShapeVector& shape_inputs,
    const nnvm::IndexedGraph& ig);

std::unordered_map<std::string, uint32_t> GetOutputLookup(const nnvm::IndexedGraph& ig);

void ConvertPlaceholder(
  const std::string& node_name,
  const std::unordered_map<std::string, TShape>& placeholder_shapes,
  GraphProto* const graph_proto);

void ConvertConstant(GraphProto* const graph_proto,
  const std::string& node_name,
  std::unordered_map<std::string, NDArray>* const shared_buffer);

void ConvertOutput(op::tensorrt::InferenceMap_t* const trt_output_map,
                   GraphProto* const graph_proto,
                   const std::unordered_map<std::string, uint32_t>::iterator& out_iter,
                   const std::string& node_name,
                   const nnvm::Graph& g,
                   const StorageTypeVector& storage_types,
                   const DTypeVector& dtypes);

typedef void (*ConverterFunction)(NodeProto *node_proto,
                                  const NodeAttrs &attrs,
                                  const nnvm::IndexedGraph &ig,
                                  const array_view<IndexedGraph::NodeEntry> &inputs);


// Forward declarations
void ConvertConvolution(
                        NodeProto *node_proto,
                        const NodeAttrs &attrs,
                        const nnvm::IndexedGraph &ig,
                        const array_view<IndexedGraph::NodeEntry> &inputs);


void ConvertPooling(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertActivation(NodeProto *node_proto,
                       const NodeAttrs &attrs,
                       const nnvm::IndexedGraph &ig,
                       const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertFullyConnected(NodeProto *node_proto,
                           const NodeAttrs &attrs,
                           const nnvm::IndexedGraph &ig,
                           const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertSoftmaxOutput(NodeProto *node_proto,
                          const NodeAttrs &attrs,
                          const nnvm::IndexedGraph &ig,
                          const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertFlatten(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertBatchNorm(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertElementwiseAdd(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

TRTParam ConvertNnvmGraphToOnnx(
    const nnvm::Graph &g,
    std::unordered_map<std::string, NDArray> *const shared_buffer);

static const std::unordered_map<std::string, ConverterFunction> converter_map = {
  {"Convolution", ConvertConvolution},
  {"Pooling", ConvertPooling},
  {"Activation", ConvertActivation},
  {"FullyConnected", ConvertFullyConnected},
  {"SoftmaxOutput", ConvertSoftmaxOutput},
  {"Flatten", ConvertFlatten},
  {"BatchNorm", ConvertBatchNorm},
  {"elemwise_add", ConvertElementwiseAdd}};

}  // namespace nnvm_to_onnx
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_OPERATOR_CONTRIB_NNVM_TO_ONNX_INL_H_
