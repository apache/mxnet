#ifndef MXNET_OPERATOR_SUBGRAPH_TENSORRT_NNVM_TO_ONNX_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_TENSORRT_NNVM_TO_ONNX_INL_H_
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
 * \file nnvm_to_onnx-inl.h
 * \brief Conversion from NNVM to ONNX for TensorRT
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include <mxnet/operator.h>
#include <nnvm/pass_functions.h>

#include <onnx/onnx_pb.h>

#include <unordered_map>
#include <vector>
#include <string>

namespace mxnet {
namespace op {
namespace nnvm_to_onnx {

enum ConvDeconvType {Convolution, Deconvolution};

using namespace nnvm;
using namespace ::onnx;
using int64 = ::google::protobuf::int64;

std::unordered_map<std::string, mxnet::TShape> GetPlaceholderShapes(const ShapeVector& shape_inputs,
    const nnvm::IndexedGraph& ig);

std::unordered_map<std::string, int> GetPlaceholderDTypes(const DTypeVector& dtype_inputs,
    const nnvm::IndexedGraph& ig);

std::unordered_map<std::string, uint32_t> GetOutputLookup(const nnvm::IndexedGraph& ig);

void ConvertPlaceholder(
  const std::string& node_name,
  const std::unordered_map<std::string, TShape>& placeholder_shapes,
  const std::unordered_map<std::string, int>& placeholder_dtypes,
  GraphProto* graph_proto);

void ConvertConstant(GraphProto* graph_proto,
  const std::string& node_name,
  const std::unordered_map<std::string, NDArray>* const params_map);

void ConvertOutput(GraphProto* graph_proto,
                   const std::unordered_map<std::string, uint32_t>::iterator& out_iter,
                   const std::string& node_name, const ShapeVector& shapes,
                   const DTypeVector& dtypes, const nnvm::IndexedGraph &ig);

typedef void (*ConverterFunction)(NodeProto *node_proto,
                                  const NodeAttrs &attrs,
                                  const nnvm::IndexedGraph &ig,
                                  const array_view<IndexedGraph::NodeEntry> &inputs);

template <class ConvDeconvParam>
void ConvDeconvConvertHelper(NodeProto *node_proto,
                             const NodeAttrs &attrs,
                             const nnvm::IndexedGraph &ig,
                             const array_view<IndexedGraph::NodeEntry> &inputs,
                             const ConvDeconvParam& param,
                             ConvDeconvType type);

// Forward declarations
void ConvertIdentity(NodeProto* node_proto,
                     const NodeAttrs &attrs,
                     const nnvm::IndexedGraph& ig,
                     const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertConvolution(
                        NodeProto *node_proto,
                        const NodeAttrs &attrs,
                        const nnvm::IndexedGraph &ig,
                        const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertDeconvolution(NodeProto *node_proto,
                        const NodeAttrs &attrs,
                        const nnvm::IndexedGraph &ig,
                        const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertPooling(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertRelu(NodeProto *node_proto,
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

void ConvertDropout(NodeProto *node_proto,
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

void ConvertElementwiseMul(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertElementwiseSub(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertConcatenate(NodeProto *node_proto,
                    const NodeAttrs &attrs,
                    const nnvm::IndexedGraph &ig,
                    const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertClip(NodeProto *node_proto,
                 const NodeAttrs &attrs,
                 const nnvm::IndexedGraph &ig,
                 const array_view<IndexedGraph::NodeEntry> &inputs);

void ConvertPad(NodeProto* node_proto,
                const NodeAttrs & attrs,
                const nnvm::IndexedGraph &ig,
                const array_view<IndexedGraph::NodeEntry> &inputs);

std::string ConvertNnvmGraphToOnnx(const nnvm::Graph &g,
    std::unordered_map<std::string, NDArray>* params_map);

static const std::unordered_map<std::string, ConverterFunction> converter_map = {
  {"_copy", ConvertIdentity},
  {"Activation", ConvertActivation},
  {"BatchNorm", ConvertBatchNorm},
  {"clip", ConvertClip},
  {"Convolution", ConvertConvolution},
  {"Deconvolution", ConvertDeconvolution},
  {"Concat", ConvertConcatenate},
  {"Dropout", ConvertDropout},
  {"elemwise_add", ConvertElementwiseAdd},
  {"elemwise_mul", ConvertElementwiseMul},
  {"elemwise_sub", ConvertElementwiseSub},
  {"Flatten", ConvertFlatten},
  {"FullyConnected", ConvertFullyConnected},
  {"Pad", ConvertPad},
  {"Pooling", ConvertPooling},
  {"relu", ConvertRelu},
  {"SoftmaxOutput", ConvertSoftmaxOutput}
};

typedef void (*PreprocessFunction)(const NodeAttrs &attrs,
                                   const std::vector<nnvm::NodeEntry> &inputs,
                                   std::unordered_map<std::string, NDArray> *params_map);

void PreprocessBatchNorm(const NodeAttrs &attrs,
                         const std::vector<nnvm::NodeEntry> &inputs,
                         std::unordered_map<std::string, NDArray> *params_map);

static const std::unordered_map<std::string, PreprocessFunction> preprocess_map = {
  {"BatchNorm", PreprocessBatchNorm}
};

}  // namespace nnvm_to_onnx
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_OPERATOR_SUBGRAPH_TENSORRT_NNVM_TO_ONNX_INL_H_
