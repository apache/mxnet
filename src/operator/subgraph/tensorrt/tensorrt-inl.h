#ifndef MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INL_H_
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
 * Copyright (c) 2019-2020 by Contributors
 * \file tensorrt-inl.h
 * \brief TensorRT operation registration
 * \author Marek Kolodziej, Clement Fuji Tsang, Serge Panev
*/

#if MXNET_USE_TENSORRT

#include <onnx-tensorrt/NvOnnxParser.h>

#include <algorithm>
#include <future>
#include <iterator>
#include <utility>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../nn/activation-inl.h"
#include "../../nn/batch_norm-inl.h"
#include "../../nn/concat-inl.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/deconvolution-inl.h"
#include "../../nn/dropout-inl.h"
#include "../../nn/fully_connected-inl.h"
#include "../../nn/pooling-inl.h"
#include "../common.h"
#include "../subgraph_property.h"
#include "nnvm_to_onnx-inl.h"
#include "./onnx_to_tensorrt.h"
#include "./tensorrt_int8_calibrator.h"

namespace mxnet {
namespace op {

using int64 = ::google::protobuf::int64;

struct TRTParam {
  std::unordered_map<std::string, uint32_t> inputs_to_idx;
  std::unordered_map<std::string, uint32_t> outputs_to_idx;
  std::unordered_map<std::string, NDArray> params_map;
  bool fp16_mode;
  bool int8_mode;
  int calibration_iters;
};

struct TRTEngineParam {
  using EnginePtr = onnx_to_tensorrt::unique_ptr<nvinfer1::ICudaEngine>;
  TRTEngineParam(EnginePtr _trt_engine,
                 onnx_to_tensorrt::unique_ptr<nvonnxparser::IParser> _trt_parser,
                 std::unique_ptr<onnx_to_tensorrt::TRT_Logger> _trt_logger,
                 const std::unordered_map<std::string, uint32_t>& input_map,
                 const std::unordered_map<std::string, uint32_t>& output_map,
                 int _max_batch_size,
                 std::unique_ptr<::onnx_to_tensorrt::TRTInt8Calibrator> _calibrator = {},
                 std::future<EnginePtr> _future_int8_engine = {}) {
    trt_engine = std::move(_trt_engine);
    trt_logger = std::move(_trt_logger);
    trt_parser = std::move(_trt_parser);
    calibrator = std::move(_calibrator);
    future_int8_engine = std::move(_future_int8_engine);
    calibration_mode = future_int8_engine.valid();
    max_batch_size = _max_batch_size;
    input_name_to_idx = input_map;
    binding_order = std::make_shared<std::vector<std::pair<uint32_t, bool> > >();
    bindings = std::make_shared<std::vector<void*> >();
    binding_order->reserve(trt_engine->getNbBindings());
    bindings->resize(trt_engine->getNbBindings());
    for (int b = 0; b < trt_engine->getNbBindings(); ++b) {
      const std::string& binding_name = trt_engine->getBindingName(b);
      if (trt_engine->bindingIsInput(b)) {
        binding_order->emplace_back(input_map.at(binding_name), true);
      } else {
        binding_order->emplace_back(output_map.at(binding_name), false);
      }
    }
    trt_executor = onnx_to_tensorrt::InferObject(trt_engine->createExecutionContext());
  }

  void ResetEngine(EnginePtr _trt_engine,
                   bool _calibration_mode = false) {
    trt_executor.reset();
    trt_engine.reset();
    trt_engine = std::move(_trt_engine);
    trt_executor = onnx_to_tensorrt::InferObject(trt_engine->createExecutionContext());
    calibration_mode = _calibration_mode;
  }

  ~TRTEngineParam() {
    if (future_int8_engine.valid()) {
      calibrator->waitAndSetDone();
      future_int8_engine.wait();
    }
  }

  EnginePtr trt_engine;
  onnx_to_tensorrt::unique_ptr<nvinfer1::IExecutionContext> trt_executor;
  onnx_to_tensorrt::unique_ptr<nvonnxparser::IParser> trt_parser;
  std::unique_ptr<onnx_to_tensorrt::TRT_Logger> trt_logger;
  std::shared_ptr<std::vector<std::pair<uint32_t, bool> > > binding_order;
  std::shared_ptr<std::vector<void*> > bindings;

  // needed by the int8 calibrator
  std::unique_ptr<::onnx_to_tensorrt::TRTInt8Calibrator> calibrator;
  std::future<EnginePtr> future_int8_engine;
  bool calibration_mode;
  int max_batch_size;
  std::unordered_map<std::string, uint32_t> input_name_to_idx;
  std::unordered_map<std::string, NDArray> params_map;
};

class TensorrtSelector : public SubgraphSelector {
 public:
  const std::unordered_set<std::string> unconditionalTRTops = {
    "_copy",
    "clip",
    "elemwise_add",
    "elemwise_sub",
    "elemwise_mul",
    "Flatten",
    "Pad",
    "relu",
    "rsqrt",
    "SoftmaxOutput"
  };

  const std::unordered_set<std::string> withWeightsOps = {
    "BatchNorm",
    "Convolution",
    "Deconvolution",
    "FullyConnected"
  };

  bool isTRTCompatible(const nnvm::Node &n) {
    const std::string op_name = n.op()->name;
    if (op_name == "FullyConnected") {
      const auto& param = nnvm::get<FullyConnectedParam>(n.attrs.parsed);
      return !param.no_bias;
    }

    if (op_name == "Pooling") {
      const auto& param = nnvm::get<PoolingParam>(n.attrs.parsed);
      if (param.layout.has_value()) {
        if (param.layout.value() == mshadow::kNHWC) {
          LOG(INFO) << "Warning: NHWC layout (node: " << n.attrs.name
                    << ") is not supported by TensorRT";
          return false;
        } else if (param.layout.value() == mshadow::kNDHWC) {
          LOG(INFO) << "Warning: NDHWC layout (node: " << n.attrs.name
                    << ") is not supported by TensorRT";
          return false;
        }
      }
      if (param.pooling_convention != pool_enum::kValid && !param.global_pool)
        return false;
      if (param.pool_type == pool_enum::kAvgPooling) {
        if ((!param.global_pool) &&
            (!param.count_include_pad.has_value() || param.count_include_pad.value()))
          return false;
        return true;
      } else if (param.pool_type == pool_enum::kMaxPooling) {
        return true;
      } else {
        return false;
      }
    }

    if (op_name == "Convolution") {
      const auto& param = nnvm::get<ConvolutionParam>(n.attrs.parsed);
      if (!param.layout.has_value())
        return true;
      switch (param.layout.value()) {
        case mshadow::kNCHW:
        case mshadow::kNCW:
        case mshadow::kNCDHW:
          return true;
        case mshadow::kNHWC:
          LOG(INFO) << "Warning: NHWC layout (node: " << n.attrs.name
                    << ") is not supported by TensorRT";
          return false;
        case mshadow::kNDHWC:
          LOG(INFO) << "Warning: NDHWC layout (node: " << n.attrs.name
                    << ") is not supported by TensorRT";
          return false;
        default:
          LOG(INFO) << "Warning: Layout (node: " << n.attrs.name
                    << ") is unknown (so unsupported by TensorRT)";
          return false;
      }
    }

    if (op_name == "Deconvolution") {
      const auto& param = nnvm::get<DeconvolutionParam>(n.attrs.parsed);
      if (!param.layout.has_value())
        return true;
      switch (param.layout.value()) {
        case mshadow::kNCHW:
        case mshadow::kNCW:
        case mshadow::kNCDHW:
          return true;
        case mshadow::kNHWC:
          LOG(INFO) << "Warning: NHWC layout (node: " << n.attrs.name
                    << ") is no tsupported by TensorRT";
          return false;
        case mshadow::kNDHWC:
          LOG(INFO) << "Warning: NDHWC layout (node: " << n.attrs.name
                    << ") is not supported by TensorRT";
          return false;
        default:
          LOG(INFO) << "Warning: Layout (node: " << n.attrs.name
                    << ") is unknown (so unsupported by TensorRT)";
          return false;
      }
    }

    if (op_name == "Concat") {
      const auto& param = nnvm::get<ConcatParam>(n.attrs.parsed);
      return (param.dim != 0);
    }

    if (op_name == "Dropout") {
      const auto& param = nnvm::get<DropoutParam>(n.attrs.parsed);
      return param.mode == dropout::kTraining && param.axes.ndim() == 0;
    }

    if (op_name == "Activation") {
      return n.attrs.dict.at("act_type") == "relu" ||
        n.attrs.dict.at("act_type") == "tanh" ||
        n.attrs.dict.at("act_type") == "sigmoid";
    }

    if (op_name == "BatchNorm") {
      const auto& param = nnvm::get<BatchNormParam>(n.attrs.parsed);
      if (param.axis != 1) {
        LOG(INFO) << "Warning: Only Layout NC(D)(H)W are supported by TensorRT "
                  << "(node " << n.attrs.name << ")";
        return false;
      }
      return true;
    }

    if (unconditionalTRTops.count(op_name)) {
      return true;
    }

    return false;
  }

  bool Select(const nnvm::Node &n) override {
    return !n.is_variable() && isTRTCompatible(n);
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (new_node.is_variable()) {
      if (withWeightsOps.count(n.op()->name)) {
        return n.inputs[0].node->attrs.name != new_node.attrs.name;
      } else {
        return false;
      }
    }
    return isTRTCompatible(new_node);
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return isTRTCompatible(new_node);
  }

  std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) override {
    bool found_one = false;
    // TensorRT is interesting with at least 2 operations
    for (auto& n : candidates) {
      if (!n->is_variable()) {
        if (found_one) {
          return candidates;
        } else {
          found_one = true;
        }
      }
    }
    return std::vector<nnvm::Node*>();
  }
};

class TensorrtProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<TensorrtProperty>();
  }

  void PrePartition(const nnvm::Graph& g,
    const std::unordered_map<std::string, std::string>& options_map) override {
    auto it_precision = options_map.find("precision");
    if (it_precision != options_map.end()) {
      fp16_mode_ = false;
      int8_mode_ = false;
      auto precision_string = it_precision->second;
      std::replace(precision_string.begin(), precision_string.end(), '_', ' ');
      std::istringstream iss(precision_string);
      std::unordered_set<std::string> precision_list((std::istream_iterator<std::string>(iss)),
                                                      std::istream_iterator<std::string>());
      for (auto &precision : precision_list) {
        if (precision == "fp16") {
          fp16_mode_  = true;
        } else if (precision == "int8") {
          int8_mode_ = true;
        } else {
          CHECK(precision == "fp32")
            << "TensorRT Op: `precision` only accepts combination of 'fp32`, 'fp16' and 'int8'. "
               "e.g. precision='fp16_int8', precision='int8', precision='fp32_int8'\n"
               "`Notes:\n"
               "Omitting `fp16` or `fp32` is equivalent to `fp32` (`fp32_int8` <=> `int8')\n"
               "fp16` overrides `fp32` (`fp32_fp16` <=> `fp16`)";
        }
      }
    }

    if (int8_mode_) {
      auto it_iters = options_map.find("calibration_iters");
      CHECK(it_iters != options_map.end())
        << "TensorRT Op: `calibration_iters` has to be set when using `int8_mode`.";
      calibration_iters_ = std::stoi(it_iters->second);
    }
    auto& in_arg_names = g.GetAttr<std::vector<std::string>>("in_arg_names");
    auto& in_aux_names = g.GetAttr<std::vector<std::string>>("in_aux_names");
    NDArray **in_args_ptr = g.GetAttr<NDArray**>("in_args");
    NDArray **in_aux_ptr = g.GetAttr<NDArray**>("in_aux");
    in_args_dict_.clear();
    in_aux_dict_.clear();
    // we trust the Python API, len(in_arg_names) == len(in_args_ptr)
    for (unsigned i = 0; i < in_arg_names.size(); ++i) {
      in_args_dict_[in_arg_names[i]] = in_args_ptr[i];
    }
    for (unsigned i = 0; i < in_aux_names.size(); ++i) {
      in_aux_dict_[in_aux_names[i]] = in_aux_ptr[i];
    }
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    nnvm::Symbol new_sym;
    std::unique_copy(sym.outputs.begin(), sym.outputs.end(),
        std::back_inserter(new_sym.outputs), [](
        nnvm::NodeEntry lhs, nnvm::NodeEntry rhs) {
          return lhs.index == rhs.index && lhs.node.get() == rhs.node.get();
        });
    n->attrs.name = "TensorRT" + std::to_string(subgraph_id);
    n->attrs.op = Op::Get("_TensorRT");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));

    // Mapping subgraph params with NDArrays
    TRTParam param;
    param.fp16_mode = fp16_mode_;
    param.int8_mode = int8_mode_;
    param.calibration_iters = calibration_iters_;
    std::ostringstream params_oss;
    for (auto &param_name : new_sym.ListInputNames(nnvm::Symbol::kAll)) {
      NDArray *cache = nullptr;
      auto it_args = in_args_dict_.find(param_name);
      if (it_args != in_args_dict_.end()) {
        cache = it_args->second;
      } else {
        auto it_aux = in_aux_dict_.find(param_name);
        if (it_aux != in_aux_dict_.end()) {
          cache = it_aux->second;
        }
      }
      if (cache != nullptr) {
        param.params_map.emplace(param_name, cache->Copy(Context()));
        param.params_map[param_name].WaitToRead();
        params_oss << param_name << ";";
      }
    }
    auto tensorrt_params_names = params_oss.str();
    if (!tensorrt_params_names.empty()) {
      tensorrt_params_names.pop_back();
    }
    n->attrs.parsed = param;
    n->attrs.dict["subgraph_params_names"] = tensorrt_params_names;
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    return std::make_shared<TensorrtSelector>();
  }

  void ConnectSubgraphOutputs(const nnvm::ObjectPtr subgraph_node, \
                              std::vector<nnvm::NodeEntry*>* output_entries) const override {
    std::vector<nnvm::NodeEntry>& outputs = subgraph_node->attrs.subgraphs[0]->outputs;
    TRTParam& _params = nnvm::get<TRTParam>(subgraph_node->attrs.parsed);
    for (size_t i = 0; i < outputs.size(); i++) {
      auto& o = outputs[i];
      for (auto& e : *output_entries) {
        if (o.index == e->index && o.node.get() == e->node.get()) {
          e->index = i;
          e->node = subgraph_node;
          // TODO(cfujitsang): For future support this would fail
          //                   if the node have multiple outputs
          _params.outputs_to_idx[o.node->attrs.name] = i;
        }
      }
    }
    subgraph_node->attrs.parsed = std::move(_params);
  }

  void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
                             std::vector<nnvm::NodeEntry*>* input_entries,
                             std::vector<nnvm::NodeEntry>* orig_input_entries) const override {
    TRTParam& _params = nnvm::get<TRTParam>(subgraph_node->attrs.parsed);
    subgraph_node->inputs.clear();
    subgraph_node->inputs.resize(orig_input_entries->size());
    for (size_t i = 0; i < orig_input_entries->size(); ++i) {
      subgraph_node->inputs[i] = orig_input_entries->at(i);
      _params.inputs_to_idx[input_entries->at(i)->node->attrs.name] = i;
    }
    subgraph_node->attrs.parsed = std::move(_params);
  }

  void PostPartition(const nnvm::Graph& g) override {
    if (int8_mode_) {
      int n_trt_engines = 0;
      nnvm::DFSVisit(g.outputs, [&n_trt_engines](const std::shared_ptr<nnvm::Node>& n) {
          if (n->attrs.op != nullptr && n->attrs.op->name == "_TensorRT") {
            n_trt_engines++;
          }
      });
      LOG(INFO) << "[TensorRT op] " << n_trt_engines << " INT8 engines have been created. "
            << "They are set in calibration mode for the next " << calibration_iters_
            << " iterations. Please feed calibration data reprensenting the inference data "
            << "(performance is low during calibration).";
    }
  }

 private:
  std::unordered_map<std::string, NDArray*> in_args_dict_, in_aux_dict_;
  bool fp16_mode_;
  bool int8_mode_;
  int calibration_iters_;
};


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INL_H_
