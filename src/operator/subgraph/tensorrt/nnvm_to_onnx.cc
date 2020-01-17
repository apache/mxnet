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
 * \file nnvm_to_onnx.cc
 * \brief Conversion from NNVM to ONNX for TensorRT
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include "./nnvm_to_onnx-inl.h"

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <operator/nn/deconvolution-inl.h>

#include "../../../common/utils.h"
#include "../../../ndarray/ndarray_function.h"
#include "../../pad-inl.h"
#include "../../nn/activation-inl.h"
#include "../../nn/batch_norm-inl.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/fully_connected-inl.h"
#include "../../nn/pooling-inl.h"
#include "../../nn/concat-inl.h"
#include "../../softmax_output-inl.h"
#include "../../tensor/matrix_op-inl.h"

#if MXNET_USE_TENSORRT_ONNX_CHECKER
#include <onnx/checker.h>
#endif  // MXNET_USE_TENSORRT_ONNX_CHECKER

namespace mxnet {
namespace op {
namespace nnvm_to_onnx {

std::string ConvertNnvmGraphToOnnx(
    const nnvm::Graph& g,
    std::unordered_map<std::string, NDArray>* params_map) {

  static std::atomic_ulong subgraph_count = { 0 };

  std::string serialized_onnx_graph;

  const nnvm::IndexedGraph& ig = g.indexed_graph();
  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<ShapeVector>("shape");
  const auto& dtype_inputs = g.GetAttr<DTypeVector>("dtype_inputs");
  const auto& shape_inputs = g.GetAttr<ShapeVector>("shape_inputs");

  ModelProto model_proto;

  // We're currently serializing our models in ONNX 3, opset 8 as it is best supported by the
  // currently linked version of the onnx-tensorrt library.
  // More information on ONNX versions and opsets can be found at:
  // https://github.com/onnx/onnx/blob/master/docs/IR.md

  auto opset_proto = model_proto.add_opset_import();
  const int64 onnx_opset = 8;
  const int64 onnx_major_version = 3;

  // Declare our ONNX versions in our protobuf model.
  opset_proto->set_version(onnx_opset);
  model_proto.set_ir_version(onnx_major_version);

  GraphProto* graph_proto = model_proto.mutable_graph();
  auto subgraph_name_id = subgraph_count.fetch_add(1);
  graph_proto->set_name("MXNetTRTSubgraph" + std::to_string(subgraph_name_id));

  auto placeholder_shapes = GetPlaceholderShapes(shape_inputs, ig);
  auto placeholder_dtypes = GetPlaceholderDTypes(dtype_inputs, ig);
  auto output_lookup = GetOutputLookup(ig);

  for (uint32_t node_idx = 0; node_idx < ig.num_nodes(); ++node_idx) {
      const IndexedGraph::Node& node = ig[node_idx];
      const nnvm::Node* source = node.source;
      // If this is a op
      if (!source->is_variable()) {
        auto mightNeedPreprocessNode = preprocess_map.find(source->op()->name);
        // if this op is defined in preprocess_map
        if (mightNeedPreprocessNode != preprocess_map.end()) {
          mightNeedPreprocessNode->second(source->attrs, source->inputs, params_map);
        }
      }
  }

  uint32_t current_input = 0;
  // Can't do a foreach over IndexedGraph since it doesn't implement begin(), etc.
  for (uint32_t node_idx = 0; node_idx < ig.num_nodes(); ++node_idx) {
    const IndexedGraph::Node& node = ig[node_idx];
    const nnvm::Node* source = node.source;
    const NodeAttrs& attrs = source->attrs;
    const Op* op = source->op();

    std::string node_name = attrs.name;
    // Here, "variable" actually means anything that's not an op i.e. a constant (weights) or a
    // placeholder
    if (source->is_variable()) {
      // Is this a placeholder?
      if (params_map->count(node_name) == 0) {
        // This fixes the problem with a SoftmaxOutput node during inference, but it's hacky.
        // Need to figure out how to properly fix it.
        if (node_name.find("label") != std::string::npos) {
          current_input++;
          continue;
        }
        ConvertPlaceholder(node_name, placeholder_shapes, placeholder_dtypes, graph_proto);
      } else {
        // If it's not a placeholder, then by exclusion it's a constant.
        ConvertConstant(graph_proto, node_name, params_map);
      }  // is_placeholder
    } else {
      // It's an op, rather than a "variable" (constant or placeholder)
      NodeProto* node_proto = graph_proto->add_node();
      node_proto->set_name(node_name);
      if (converter_map.count(op->name) == 0) {
        LOG(FATAL) << "Conversion for node of type " << op->name << " (node "
                   << node_name << ") "
                   << " is not supported yet.";
      }
      // Find function ptr to a converter based on the op name, and invoke the converter. This
      // looks unsafe because find may not succeed, but it does because we're in the operator
      // logic after testing that this node name does not represent a variable.
      converter_map.find(op->name)->second(node_proto, attrs, ig, node.inputs);
      // Add all inputs to the current node (i.e. add graph edges)
      for (const nnvm::IndexedGraph::NodeEntry& entry : node.inputs) {
        std::string in_node_name = ig[entry.node_id].source->attrs.name;
        // As before, we're not adding labels e.g. for SoftmaxOutput, but I wish there was a less
        // hacky way to do it than name matching.
        if (in_node_name.find("label") != std::string::npos) {
          continue;
        }
        node_proto->add_input(in_node_name);
      }
      // The node's output will have the same name as the node name.
      node_proto->add_output(node_name);
      // See if the current node is an output node
      auto out_iter = output_lookup.find(node_name);
      // We found an output
      if (out_iter != output_lookup.end()) {
        ConvertOutput(graph_proto, out_iter, node_name, shapes, dtypes, ig);
      }  // output found
    }    // conversion function exists
  }      // loop over i from 0 to num_nodes

  model_proto.SerializeToString(&serialized_onnx_graph);

#if MXNET_USE_TENSORRT_ONNX_CHECKER
  onnx::checker::check_model(model_proto);
#endif  // MXNET_USE_TENSORRT_ONNX_CHECKER

  return serialized_onnx_graph;
}

void ConvertIdentity(NodeProto* node_proto, const NodeAttrs& attrs,
                     const nnvm::IndexedGraph& /*ig*/,
                     const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Identity");
}

template <class ConvDeconvParam>
void ConvDeconvConvertHelper(NodeProto* node_proto, const NodeAttrs& attrs,
                             const nnvm::IndexedGraph& /*ig*/,
                             const array_view<IndexedGraph::NodeEntry>& /*input*/,
                             const ConvDeconvParam& param,
                             ConvDeconvType type) {
  if (type == ConvDeconvType::Convolution) {
    node_proto->set_op_type("Conv");
  } else {
    node_proto->set_op_type("ConvTranspose");
  }

  const mxnet::TShape kernel = param.kernel;
  const mxnet::TShape stride = param.stride;
  const mxnet::TShape dilate = param.dilate;
  const mxnet::TShape pad = param.pad;
  const uint32_t num_group = param.num_group;
  // const bool no_bias = conv_param.no_bias;
  const dmlc::optional<int> layout = param.layout;

  // dilations
  AttributeProto* const dilations = node_proto->add_attribute();
  dilations->set_name("dilations");
  dilations->set_type(AttributeProto::INTS);
  for (const dim_t kval : dilate) {
    dilations->add_ints(static_cast<int64>(kval));
  }

  // group
  AttributeProto* const group = node_proto->add_attribute();
  group->set_name("group");
  group->set_type(AttributeProto::INT);
  group->set_i(static_cast<int64>(num_group));

  // kernel shape
  AttributeProto* const kernel_shape = node_proto->add_attribute();
  kernel_shape->set_name("kernel_shape");
  kernel_shape->set_type(AttributeProto::INTS);

  for (const dim_t kval : kernel) {
    kernel_shape->add_ints(static_cast<int64>(kval));
  }

  // pads
  AttributeProto* const pads = node_proto->add_attribute();
  pads->set_name("pads");
  pads->set_type(AttributeProto::INTS);

  for (int i =0; i < 2; i++) {
    for (dim_t kval : pad) {
      pads->add_ints(static_cast<int64>(kval));
    }
  }

  // strides
  AttributeProto* const strides = node_proto->add_attribute();
  strides->set_name("strides");
  strides->set_type(AttributeProto::INTS);
  for (const dim_t kval : stride) {
    strides->add_ints(static_cast<int64>(kval));
  }
}

void ConvertConvolution(NodeProto* node_proto, const NodeAttrs& attrs,
                        const nnvm::IndexedGraph& ig,
                        const array_view<IndexedGraph::NodeEntry>& inputs) {
  const auto& conv_param = nnvm::get<op::ConvolutionParam>(attrs.parsed);
  ConvDeconvConvertHelper(node_proto, attrs, ig, inputs, conv_param,
      ConvDeconvType::Convolution);
}  // end ConvertConvolution

void ConvertDeconvolution(NodeProto* node_proto, const NodeAttrs& attrs,
                          const nnvm::IndexedGraph& ig,
                          const array_view<IndexedGraph::NodeEntry>& inputs) {
  const auto& deconv_param = nnvm::get<op::DeconvolutionParam>(attrs.parsed);
  ConvDeconvConvertHelper(node_proto, attrs, ig, inputs, deconv_param,
      ConvDeconvType::Deconvolution);
}  // end ConvertDeconvolution

void ConvertPooling(NodeProto* node_proto, const NodeAttrs& attrs,
                    const nnvm::IndexedGraph& /*ig*/,
                    const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& pooling_param = nnvm::get<op::PoolingParam>(attrs.parsed);

  const mxnet::TShape kernel = pooling_param.kernel;
  const mxnet::TShape stride = pooling_param.stride;
  const mxnet::TShape pad = pooling_param.pad;
  const int pool_type = pooling_param.pool_type;
  const bool global_pool = pooling_param.global_pool;

  if (global_pool) {
    if (pool_type == pool_enum::kMaxPooling) {
      node_proto->set_op_type("GlobalMaxPool");
    } else if (pool_type == pool_enum::kAvgPooling) {
      node_proto->set_op_type("GlobalAveragePool");
    } else {
      LOG(FATAL) << "Pool type of node '" << attrs.name << "' unsupported: " << attrs.name;
    }
    return;
  }

  // kernel_shape
  AttributeProto* const kernel_shape = node_proto->add_attribute();
  kernel_shape->set_name("kernel_shape");
  kernel_shape->set_type(AttributeProto::INTS);
  for (dim_t kval : kernel) {
    kernel_shape->add_ints(static_cast<int64>(kval));
  }

  // pads
  AttributeProto* const pads = node_proto->add_attribute();
  pads->set_name("pads");
  pads->set_type(AttributeProto::INTS);

  // Convert from MXNet symetric pads to ONNX non-symetric by running through padding twice.
  for (int i =0; i < 2; i++) {
    for (dim_t kval : pad) {
      pads->add_ints(static_cast<int64>(kval));
    }
  }

  // strides
  AttributeProto* const strides = node_proto->add_attribute();
  strides->set_name("strides");
  strides->set_type(AttributeProto::INTS);
  for (dim_t kval : stride) {
    strides->add_ints(static_cast<int64>(kval));
  }

  // ceil_mode
  AttributeProto* const ceil_mode = node_proto->add_attribute();
  ceil_mode->set_name("ceil_mode");
  ceil_mode->set_type(AttributeProto::INT);
  ceil_mode->set_i(static_cast<int64>(pooling_param.pooling_convention == pool_enum::kFull));

  if (pool_type == pool_enum::kMaxPooling) {
    node_proto->set_op_type("MaxPool");
  } else if (pool_type == pool_enum::kAvgPooling) {
    node_proto->set_op_type("AveragePool");
  } else {
    LOG(FATAL) << "Pool type of node '" << attrs.name << "' unsupported: " << attrs.name;
  }

  // count_include_pad
  AttributeProto* const count_include_pad = node_proto->add_attribute();
  count_include_pad->set_name("count_include_pad");
  count_include_pad->set_type(AttributeProto::INT);
  if (pooling_param.count_include_pad.has_value()) {
    count_include_pad->set_i(pooling_param.count_include_pad.value());
  } else {
    count_include_pad->set_i(1);
  }
}  // end ConvertPooling

void ConvertRelu(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                 const nnvm::IndexedGraph& /*ig*/,
                 const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Relu");
}

void ConvertActivation(NodeProto* node_proto, const NodeAttrs& attrs,
                       const nnvm::IndexedGraph& /*ig*/,
                       const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& act_param = nnvm::get<op::ActivationParam>(attrs.parsed);
  std::string act_type;
  switch (act_param.act_type) {
    case op::activation::kReLU:
      act_type = "Relu";
      break;
    case op::activation::kSigmoid:
      act_type = "Sigmoid";
      break;
    case op::activation::kTanh:
      act_type = "Tanh";
      break;
    case op::activation::kSoftReLU:
      // act_type = "SoftReLU";
      throw dmlc::Error("SoftReLU is not supported in ONNX");
      break;
    default:
      throw dmlc::Error("Activation of such type doesn't exist");
  }

  node_proto->set_op_type(act_type);
}

void ConvertFullyConnected(NodeProto* node_proto, const NodeAttrs& attrs,
                           const nnvm::IndexedGraph& /*ig*/,
                           const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& act_param = nnvm::get<op::FullyConnectedParam>(attrs.parsed);
  if (act_param.no_bias) {
      node_proto->set_op_type("MatMul");
  } else {
      node_proto->set_op_type("Gemm");

      AttributeProto* const alpha = node_proto->add_attribute();
      alpha->set_name("alpha");
      alpha->set_type(AttributeProto::FLOAT);
      alpha->set_f(1.0f);

      AttributeProto* const beta = node_proto->add_attribute();
      beta->set_name("beta");
      beta->set_type(AttributeProto::FLOAT);
      beta->set_f(1.0f);

      AttributeProto* const transA = node_proto->add_attribute();
      transA->set_name("transA");
      transA->set_type(AttributeProto::INT);
      transA->set_i(0);

      AttributeProto* const transB = node_proto->add_attribute();
      transB->set_name("transB");
      transB->set_type(AttributeProto::INT);
      transB->set_i(1);
  }
}

void ConvertSoftmaxOutput(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                          const nnvm::IndexedGraph& /*ig*/,
                          const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Softmax");

  // Setting by default to 1 since MXNet doesn't provide such an attribute for softmax in its
  // node params. This attribute is only relevant when the input is coerced to 2D, and in that
  // case dimension 0 is assumed to be the batch dimension.
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
}

void ConvertFlatten(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                    const nnvm::IndexedGraph& /*ig*/,
                    const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Flatten");

  // Setting by default to 1 since MXNet doesn't provide such an attribute for Flatten in its
  // node params. This attribute is only relevant when the input is coerced to 2D, and in that
  // case dimension 0 is assumed to be the batch dimension.
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
}

void ConvertBatchNorm(NodeProto* node_proto, const NodeAttrs& attrs,
                      const nnvm::IndexedGraph& /*ig*/,
                      const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("BatchNormalization");
  const auto& param = nnvm::get<op::BatchNormParam>(attrs.parsed);

  AttributeProto* const epsilon = node_proto->add_attribute();
  epsilon->set_name("epsilon");
  epsilon->set_type(AttributeProto::FLOAT);
  epsilon->set_f(static_cast<float>(param.eps));

  AttributeProto* const momentum = node_proto->add_attribute();
  momentum->set_name("momentum");
  momentum->set_type(AttributeProto::FLOAT);
  momentum->set_f(param.momentum);

  AttributeProto* const spatial = node_proto->add_attribute();
  spatial->set_name("spatial");
  spatial->set_type(AttributeProto::INT);
  // MXNet computes mean and variance per feature for batchnorm.  Enabling spatial mode
  // (default in ONNX3) implies running batchnorm on all spatial features so we need to explicitly
  // disable this for MXNet's BatchNorm.
  spatial->set_i(0);
}

void ConvertElementwiseAdd(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                           const nnvm::IndexedGraph& /*ig*/,
                           const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Add");
}

void ConvertElementwiseSub(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                           const nnvm::IndexedGraph& /*ig*/,
                           const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Sub");
}

void ConvertElementwiseMul(NodeProto* node_proto, const NodeAttrs& /*attrs*/,
                           const nnvm::IndexedGraph& /*ig*/,
                           const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Mul");
}

void ConvertConcatenate(NodeProto* node_proto, const NodeAttrs& attrs,
                        const nnvm::IndexedGraph& /*ig*/,
                        const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& _param = nnvm::get<ConcatParam>(attrs.parsed);
  node_proto->set_op_type("Concat");
  node_proto->set_name(attrs.name);
  // axis
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(static_cast<int64_t>(_param.dim));
}

inline TensorProto_DataType ConvertDType(int dtype) {
  switch (dtype) {
    case mshadow::kFloat64:
      return TensorProto_DataType_DOUBLE;
    case mshadow::kFloat32:
      return TensorProto_DataType_FLOAT;
    case mshadow::kFloat16:
      return TensorProto_DataType_FLOAT16;
    case mshadow::kUint8:
      return TensorProto_DataType_UINT8;
    case mshadow::kInt32:
      return TensorProto_DataType_INT32;
    case mshadow::kInt8:
      return TensorProto_DataType_INT8;
    case mshadow::kInt64:
      return TensorProto_DataType_INT64;
    default:
      return TensorProto_DataType_UNDEFINED;
  }
}

std::unordered_map<std::string, TShape> GetPlaceholderShapes(
    const ShapeVector& shape_inputs, const nnvm::IndexedGraph& ig) {
  std::unordered_map<std::string, mxnet::TShape> placeholder_shapes;
  for (uint32_t i = 0; i < shape_inputs.size(); ++i) {
    std::string name = ig[ig.input_nodes()[i]].source->attrs.name;
    mxnet::TShape shp = shape_inputs[i];
    if (!mxnet::op::shape_is_none(shp)) {
      // TODO(@reminisce): confirm
      placeholder_shapes.emplace(name, shp);
    }
  }
  return placeholder_shapes;
}

std::unordered_map<std::string, int> GetPlaceholderDTypes(
    const DTypeVector& dtype_inputs, const nnvm::IndexedGraph& ig) {
  std::unordered_map<std::string, int> placeholder_dtypes;
  for (uint32_t i = 0; i < dtype_inputs.size(); ++i) {
    std::string name = ig[ig.input_nodes()[i]].source->attrs.name;
    int dtype = dtype_inputs[i];
    placeholder_dtypes.emplace(name, dtype);
  }
  return placeholder_dtypes;
}

std::unordered_map<std::string, uint32_t> GetOutputLookup(
    const nnvm::IndexedGraph& ig) {
  std::unordered_map<std::string, uint32_t> output_lookup;
  const std::vector<nnvm::IndexedGraph::NodeEntry>& graph_outputs =
      ig.outputs();
  for (uint32_t i = 0; i < graph_outputs.size(); ++i) {
    const uint32_t id = graph_outputs[i].node_id;
    const IndexedGraph::Node ig_node = ig[id];
    const nnvm::Node* const source = ig_node.source;
    const std::string name = source->attrs.name;
    output_lookup.emplace(name, i);
  }
  return output_lookup;
}

void ConvertPlaceholder(
    const std::string& node_name,
    const std::unordered_map<std::string, TShape>& placeholder_shapes,
    const std::unordered_map<std::string, int>& placeholder_dtypes,
    GraphProto* const graph_proto) {
  auto val_info_proto = graph_proto->add_input();
  auto type_proto = val_info_proto->mutable_type()->mutable_tensor_type();
  auto shape_proto = type_proto->mutable_shape();

  val_info_proto->set_name(node_name);
  auto entry_shape = placeholder_shapes.find(node_name)->second;
  auto entry_dtype = placeholder_dtypes.find(node_name)->second;
  type_proto->set_elem_type(ConvertDType(entry_dtype));
  for (const auto& elem : entry_shape) {
    TensorShapeProto_Dimension* const tsp_dim = shape_proto->add_dim();
    tsp_dim->set_dim_value(static_cast<int64>(elem));
  }
}

void ConvertConstant(
    GraphProto* const graph_proto, const std::string& node_name,
    const std::unordered_map<std::string, NDArray>* const params_map) {
    TensorProto* const initializer_proto = graph_proto->add_initializer();

  // Create initializer for constants
  initializer_proto->set_name(node_name);

  const NDArray nd = params_map->find(node_name)->second;
  const TBlob& blob = nd.data();
  const TShape shape = blob.shape_;
  const auto dtype = ConvertDType(nd.dtype());
  initializer_proto->set_data_type(dtype);

  for (auto& dim : shape) {
    initializer_proto->add_dims(static_cast<int64>(dim));
  }

  auto size = shape.Size();

  if (dtype == TensorProto_DataType_FLOAT) {
    std::shared_ptr<float[]> shared_data_ptr(new float[size]);
    float* const data_ptr = shared_data_ptr.get();
    nd.SyncCopyToCPU(static_cast<void*>(data_ptr), size);

    for (size_t blob_idx = 0; blob_idx < size; ++blob_idx) {
      initializer_proto->add_float_data(data_ptr[blob_idx]);
    }
  } else if (dtype == TensorProto_DataType_FLOAT16) {
    std::shared_ptr<uint16_t[]> shared_data_ptr(new uint16_t[size]);
    uint16_t* const data_ptr = shared_data_ptr.get();
    nd.SyncCopyToCPU(static_cast<void*>(data_ptr), size);
    for (size_t blob_idx = 0; blob_idx < size; ++blob_idx) {
      initializer_proto->add_int32_data(
          reinterpret_cast<int32_t*>(data_ptr)[blob_idx]);
    }
  } else {
    LOG(FATAL) << "dtype not supported for variables: " << node_name;
  }

  // Create inputs for constants.
  ValueInfoProto* const input_proto = graph_proto->add_input();
  input_proto->set_name(node_name);

  input_proto->mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
  for (auto& dim : shape) {
    auto new_dim = input_proto->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim();
    new_dim->set_dim_value(static_cast<int64>(dim));
  }
}

void ConvertOutput(
    GraphProto* const graph_proto,
    const std::unordered_map<std::string, uint32_t>::iterator& out_iter,
    const std::string& node_name, const ShapeVector& shapes,
    const DTypeVector& dtypes, const nnvm::IndexedGraph &ig) {
  uint32_t out_idx = ig.entry_id(ig.outputs()[out_iter->second]);
  int dtype = dtypes[out_idx];
  auto graph_out = graph_proto->add_output();
  auto tensor_type = graph_out->mutable_type()->mutable_tensor_type();
  auto tensor_shape_proto = tensor_type->mutable_shape();
  graph_out->set_name(node_name);

  // Also support fp16.
  tensor_type->set_elem_type(ConvertDType(dtype));

  for (int64_t dim_shp : shapes[out_idx]) {
    TensorShapeProto_Dimension* const tsp_dim = tensor_shape_proto->add_dim();
    tsp_dim->set_dim_value(static_cast<int64>(dim_shp));
  }
}

void ConvertClip(NodeProto* node_proto, const NodeAttrs& attrs,
                 const nnvm::IndexedGraph& /*ig*/,
                 const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& param = nnvm::get<ClipParam>(attrs.parsed);

  node_proto->set_op_type("Clip");

  // max
  AttributeProto* const a_max = node_proto->add_attribute();
  a_max->set_name("max");
  a_max->set_type(AttributeProto::FLOAT);
  a_max->set_f(static_cast<float>(param.a_max));

  // min
  AttributeProto* const a_min = node_proto->add_attribute();
  a_min->set_name("min");
  a_min->set_type(AttributeProto::FLOAT);
  a_min->set_f(static_cast<float>(param.a_min));
}

void ConvertPad(NodeProto* node_proto, const NodeAttrs& attrs,
                const nnvm::IndexedGraph& /*ig*/,
                const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  const auto& param = nnvm::get<PadParam>(attrs.parsed);

  node_proto->set_op_type("Pad");

  // mode
  AttributeProto* const mode = node_proto->add_attribute();
  mode->set_name("mode");
  mode->set_type(AttributeProto::STRING);
  switch (param.mode) {
    case op::pad_enum::kConstant:
      mode->set_s("constant");
      break;
    case op::pad_enum::kEdge:
      mode->set_s("edge");
      break;
    case op::pad_enum::kReflect:
      mode->set_s("reflect");
      break;
    default:
      throw dmlc::Error("Such mode of padding doesn't exist");
  }

  // pads
  AttributeProto* const pads = node_proto->add_attribute();
  pads->set_name("pads");
  pads->set_type(AttributeProto::INTS);

  std::vector<int64> pad_begin;
  std::vector<int64> pad_end;
  for (int st = 0; st < 2; ++st) {
    for (auto it = param.pad_width.begin() + st;
         it != param.pad_width.end(); it += 2) {
      pads->add_ints(static_cast<int64>(*it));
    }
  }

  // value
  AttributeProto* const value = node_proto->add_attribute();
  value->set_name("value");
  value->set_type(AttributeProto::FLOAT);
  value->set_f(param.constant_value);
}

void ConvertDropout(NodeProto* node_proto, const NodeAttrs& attrs,
                    const nnvm::IndexedGraph& /*ig*/,
                    const array_view<IndexedGraph::NodeEntry>& /*inputs*/) {
  node_proto->set_op_type("Dropout");
}

void PreprocessBatchNorm(const NodeAttrs &attrs,
                         const std::vector<nnvm::NodeEntry> &inputs,
                         std::unordered_map<std::string, NDArray> *params_map) {
  const auto& param = nnvm::get<op::BatchNormParam>(attrs.parsed);
  if (param.fix_gamma) {
    // if mxnet is specify fix_gamma, we will need to preprocess the params map
    // to convert the gamma associate with this batch norm layer to 1.
    std::string gammaNodeName = inputs[batchnorm::kGamma].node->attrs.name;
    (*params_map)[gammaNodeName] = 1.0f;
  }
}

}  // namespace nnvm_to_onnx
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
