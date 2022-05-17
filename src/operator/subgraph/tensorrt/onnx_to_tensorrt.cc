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
 * \file onnx_to_tensorrt.cc
 * \brief TensorRT integration with the MXNet executor
 * \author Marek Kolodziej, Clement Fuji Tsang
 */

#if MXNET_USE_TENSORRT

#include "./onnx_to_tensorrt.h"

#include <onnx/onnx_pb.h>

#include <NvInfer.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx-tensorrt/NvOnnxParser.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>

using std::cerr;
using std::cout;
using std::endl;

namespace onnx_to_tensorrt {

std::string onnx_ir_version_string(int64_t ir_version = onnx::IR_VERSION) {
  int onnx_ir_major = ir_version / 1000000;
  int onnx_ir_minor = ir_version % 1000000 / 10000;
  int onnx_ir_patch = ir_version % 10000;
  return (std::to_string(onnx_ir_major) + "." + std::to_string(onnx_ir_minor) + "." +
          std::to_string(onnx_ir_patch));
}

void PrintVersion() {
  cout << "Parser built against:" << endl;
  cout << "  ONNX IR version:  " << onnx_ir_version_string(onnx::IR_VERSION) << endl;
  cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
       << NV_TENSORRT_PATCH << endl;
}

std::tuple<unique_ptr<nvinfer1::ICudaEngine>,
           unique_ptr<nvonnxparser::IParser>,
           std::unique_ptr<TRT_Logger> >
onnxToTrtCtx(const std::string& onnx_model,
             int32_t max_batch_size,
             size_t max_workspace_size,
             nvinfer1::ILogger::Severity verbosity,
             bool debug_builder) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  auto trt_logger  = std::unique_ptr<TRT_Logger>(new TRT_Logger(verbosity));
  auto trt_builder = InferObject(nvinfer1::createInferBuilder(*trt_logger));
  const auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto trt_network = InferObject(trt_builder->createNetworkV2(explicitBatch));
  auto trt_parser  = InferObject(nvonnxparser::createParser(*trt_network, *trt_logger));
  ::ONNX_NAMESPACE::ModelProto parsed_model;
  // We check for a valid parse, but the main effect is the side effect
  // of populating parsed_model
  if (!parsed_model.ParseFromString(onnx_model)) {
    throw dmlc::Error("Could not parse ONNX from string");
  }
  if (!trt_parser->parse(onnx_model.c_str(), onnx_model.size())) {
    size_t nerror = trt_parser->getNbErrors();
    for (size_t i = 0; i < nerror; ++i) {
      nvonnxparser::IParserError const* error = trt_parser->getError(i);
      if (error->node() != -1) {
        ::ONNX_NAMESPACE::NodeProto const& node = parsed_model.graph().node(error->node());
        cerr << "While parsing node number " << error->node() << " [" << node.op_type();
        if (!node.output().empty()) {
          cerr << " -> \"" << node.output(0) << "\"";
        }
        cerr << "]:" << endl;
        if (static_cast<int>(verbosity) >= static_cast<int>(nvinfer1::ILogger::Severity::kINFO)) {
          cerr << "--- Begin node ---" << endl;
          cerr << node.DebugString() << endl;
          cerr << "--- End node ---" << endl;
        }
      }
      cerr << "ERROR: " << error->file() << ":" << error->line() << " In function " << error->func()
           << ":\n"
           << "[" << static_cast<int>(error->code()) << "] " << error->desc() << endl;
    }
    throw dmlc::Error("Cannot parse ONNX into TensorRT Engine");
  }
#if NV_TENSORRT_MAJOR >= 8
  auto trt_config = InferObject(trt_builder->createBuilderConfig());
#endif
  if (dmlc::GetEnv("MXNET_TENSORRT_USE_FP16", true)) {
    if (trt_builder->platformHasFastFp16()) {
#if NV_TENSORRT_MAJOR >= 8
      trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
#else
      trt_builder->setFp16Mode(true);
#endif
    } else {
      LOG(WARNING) << "TensorRT can't use fp16 on this platform";
    }
  }
  trt_builder->setMaxBatchSize(max_batch_size);
#if NV_TENSORRT_MAJOR >= 8
  trt_config->setMaxWorkspaceSize(max_workspace_size);
  if (debug_builder) {
    trt_config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
  }
  auto trt_engine = InferObject(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
#else
  trt_builder->setMaxWorkspaceSize(max_workspace_size);
  trt_builder->setDebugSync(debug_builder);
  auto trt_engine = InferObject(trt_builder->buildCudaEngine(*trt_network));
#endif
  return std::make_tuple(std::move(trt_engine), std::move(trt_parser), std::move(trt_logger));
}

}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT
