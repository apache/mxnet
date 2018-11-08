#ifndef MXNET_EXECUTOR_ONNX_TO_TENSORRT_H_
#define MXNET_EXECUTOR_ONNX_TO_TENSORRT_H_
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
 * \file onnx_to_tensorrt.h
 * \brief TensorRT integration with the MXNet executor
 * \author Marek Kolodziej, Clement Fuji Tsang
 */

#if MXNET_USE_TENSORRT

#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <sstream>
#include <string>

#include "../operator/contrib/tensorrt-inl.h"

namespace onnx_to_tensorrt {

class TRT_Logger : public nvinfer1::ILogger {
        nvinfer1::ILogger::Severity _verbosity;
        std::ostream* _ostream;
 public:
        TRT_Logger(Severity verbosity = Severity::kWARNING,
                   std::ostream& ostream = std::cout)
                : _verbosity(verbosity), _ostream(&ostream) {}
        void log(Severity severity, const char* msg) override {
                if ( severity <= _verbosity ) {
                        time_t rawtime = std::time(0);
                        char buf[256];
                        strftime(&buf[0], 256,
                                 "%Y-%m-%d %H:%M:%S",
                                 std::gmtime(&rawtime));
                        const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" :
                                              severity == Severity::kERROR          ? "  ERROR" :
                                              severity == Severity::kWARNING        ? "WARNING" :
                                              severity == Severity::kINFO           ? "   INFO" :
                                              "UNKNOWN");
                        (*_ostream) << "[" << buf << " " << sevstr << "] "
                                    << msg
                                    << std::endl;
                }
        }
};

nvinfer1::ICudaEngine* onnxToTrtCtx(
        const std::string& onnx_model,
        int32_t max_batch_size = 32,
        size_t max_workspace_size = 1L << 30,
        nvinfer1::ILogger::Severity verbosity = nvinfer1::ILogger::Severity::kWARNING,
        bool debug_builder = false);
}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT

#endif  // MXNET_EXECUTOR_ONNX_TO_TENSORRT_H_
