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
 * \file onnx_to_tensorrt.h
 * \brief TensorRT integration with the MXNet executor
 * \author Marek Kolodziej, Clement Fuji Tsang
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_TENSORRT_ONNX_TO_TENSORRT_H_
#define MXNET_OPERATOR_SUBGRAPH_TENSORRT_ONNX_TO_TENSORRT_H_

#if MXNET_USE_TENSORRT

#include <onnx-tensorrt/NvOnnxParser.h>
#include <NvInfer.h>

#include <fstream>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <tuple>

namespace onnx_to_tensorrt {

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, InferDeleter>;

template <typename T>
inline unique_ptr<T> InferObject(T* obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return unique_ptr<T>(obj, InferDeleter());
}

class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;
  std::ostream* _ostream;

 public:
  TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream& ostream = std::cout)  // NOLINT
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= _verbosity) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
      // clang-format off
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" :
                            severity == Severity::kERROR          ? "  ERROR" :
                            severity == Severity::kWARNING        ? "WARNING" :
                            severity == Severity::kINFO           ? "   INFO" :
                                                                    "UNKNOWN");
      // clang-format on
      (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
    }
  }
};

std::tuple<unique_ptr<nvinfer1::ICudaEngine>,
           unique_ptr<nvonnxparser::IParser>,
           std::unique_ptr<TRT_Logger> >
onnxToTrtCtx(const std::string& onnx_model,
             int32_t max_batch_size                = 32,
             size_t max_workspace_size             = 1L << 30,
             nvinfer1::ILogger::Severity verbosity = nvinfer1::ILogger::Severity::kWARNING,
             bool debug_builder                    = false);
}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT
#endif  // MXNET_OPERATOR_SUBGRAPH_TENSORRT_ONNX_TO_TENSORRT_H_
