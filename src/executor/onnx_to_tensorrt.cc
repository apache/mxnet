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
#include <onnx-tensorrt/NvOnnxParserRuntime.h>
#include <onnx-tensorrt/PluginFactory.hpp>
#include <onnx-tensorrt/plugin_common.hpp>

using std::cout;
using std::cerr;
using std::endl;

namespace onnx_to_tensorrt {

struct InferDeleter {
  template<typename T>
    void operator()(T* obj) const {
      if ( obj ) {
        obj->destroy();
      }
    }
};

template<typename T>
inline std::shared_ptr<T> InferObject(T* obj) {
  if ( !obj ) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

std::string onnx_ir_version_string(int64_t ir_version = onnx::IR_VERSION) {
  int onnx_ir_major = ir_version / 1000000;
  int onnx_ir_minor = ir_version % 1000000 / 10000;
  int onnx_ir_patch = ir_version % 10000;
  return (std::to_string(onnx_ir_major) + "." +
    std::to_string(onnx_ir_minor) + "." +
    std::to_string(onnx_ir_patch));
}

void PrintVersion() {
  cout << "Parser built against:" << endl;
  cout << "  ONNX IR version:  " << onnx_ir_version_string(onnx::IR_VERSION) << endl;
  cout << "  TensorRT version: "
    << NV_TENSORRT_MAJOR << "."
    << NV_TENSORRT_MINOR << "."
    << NV_TENSORRT_PATCH << endl;
}

nvinfer1::ICudaEngine* onnxToTrtCtx(
        const std::string& onnx_model,
        int32_t max_batch_size,
        size_t max_workspace_size,
        nvinfer1::ILogger::Severity verbosity,
        bool debug_builder) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  TRT_Logger trt_logger(verbosity);
  auto trt_builder = InferObject(nvinfer1::createInferBuilder(trt_logger));
  auto trt_network = InferObject(trt_builder->createNetwork());
  auto trt_parser  = InferObject(nvonnxparser::createParser(trt_network.get(), trt_logger));
  ::ONNX_NAMESPACE::ModelProto parsed_model;
  // We check for a valid parse, but the main effect is the side effect
  // of populating parsed_model
  if (!parsed_model.ParseFromString(onnx_model)) {
    throw dmlc::Error("Could not parse ONNX from string");
  }

  if ( !trt_parser->parse(onnx_model.c_str(), onnx_model.size()) ) {
      int nerror = trt_parser->getNbErrors();
      for ( int i=0; i < nerror; ++i ) {
        nvonnxparser::IParserError const* error = trt_parser->getError(i);
        if ( error->node() != -1 ) {
          ::ONNX_NAMESPACE::NodeProto const& node =
            parsed_model.graph().node(error->node());
          cerr << "While parsing node number " << error->node()
               << " [" << node.op_type();
          if ( !node.output().empty() ) {
            cerr << " -> \"" << node.output(0) << "\"";
          }
          cerr << "]:" << endl;
          if ( static_cast<int>(verbosity) >= \
            static_cast<int>(nvinfer1::ILogger::Severity::kINFO) ) {
            cerr << "--- Begin node ---" << endl;
            cerr << node.DebugString() << endl;
            cerr << "--- End node ---" << endl;
          }
        }
        cerr << "ERROR: "
             << error->file() << ":" << error->line()
             << " In function " << error->func() << ":\n"
             << "[" << static_cast<int>(error->code()) << "] " << error->desc()
             << endl;
      }
      throw dmlc::Error("Cannot parse ONNX into TensorRT Engine");
  }

  bool fp16 = trt_builder->platformHasFastFp16();

  trt_builder->setMaxBatchSize(max_batch_size);
  trt_builder->setMaxWorkspaceSize(max_workspace_size);
  if ( fp16 && dmlc::GetEnv("MXNET_TENSORRT_USE_FP16_FOR_FP32", false) ) {
    LOG(INFO) << "WARNING: TensorRT using fp16 given original MXNet graph in fp32 !!!";
    trt_builder->setHalf2Mode(true);
  }

  trt_builder->setDebugSync(debug_builder);
  nvinfer1::ICudaEngine* trt_engine = trt_builder->buildCudaEngine(*trt_network.get());
  return trt_engine;
}

}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT
