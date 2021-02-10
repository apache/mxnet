#ifndef MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INT8_CALIBRATOR_H_
#define MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INT8_CALIBRATOR_H_
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
 * Copyright (c) 2020 by Contributors
 * \file tensorrt-inl.h
 * \brief TensorRT operation registration
 * \author Serge Panev
*/

#if MXNET_USE_TENSORRT

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <mxnet/ndarray.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "../common.h"

namespace onnx_to_tensorrt {

// This class provides a 1 element queue to match TFs push model to
// TRTs pull model for calibration. When TRT implements a means for
// a push calibration This class should be updated accordingly

struct TRTInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  // Construct a calibrator for future calibration.
  TRTInt8Calibrator(
      std::unordered_map<std::string, mxnet::NDArray> params_map,
      std::unordered_map<std::string, std::pair<void*, size_t>> input_buffers_,
      int batch_size, int n_iter);

  ~TRTInt8Calibrator();

  int getBatchSize() const override;

  bool getBatch(void* bindings[], const char* names[],
                int num_bindings) override;

  // Feed calibration data to the calibrator, and return true if the data is
  // accepted. Return false if the calibrator has been terminated.
  bool setBatch(const std::unordered_map<std::string, void*>& data,
                const cudaStream_t stream);

  // If not nullptr, calibration is skipped.
  const void* readCalibrationCache(std::size_t& length) override;

  void writeCalibrationCache(const void* ptr, std::size_t length) override;

  // TODO(spanev): determine if we need to serialize it
  const std::string& getCalibrationTableAsString() { return calibration_table_; }

  void setDone();

  void waitAndSetDone();

  bool isCacheEmpty();

  bool lastIter();

 private:
  const int batch_size_;

  // Is calibration finished?
  bool done_;
  std::unordered_map<std::string, mxnet::NDArray> params_map_;
  std::unordered_map<std::string, std::pair<void*, size_t>> input_buffers_;
  bool calib_running_;
  bool batch_is_set_;

  int n_iter_;

  std::string calibration_table_;

  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT
#endif  // MXNET_OPERATOR_SUBGRAPH_TENSORRT_TENSORRT_INT8_CALIBRATOR_H_
