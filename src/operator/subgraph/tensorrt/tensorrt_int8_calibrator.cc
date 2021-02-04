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

#include "./tensorrt_int8_calibrator.h"

#include <atomic>
#include <unordered_map>

namespace onnx_to_tensorrt {

// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator(
    std::unordered_map<std::string, mxnet::NDArray> params_map,
    std::unordered_map<std::string, std::pair<void*, size_t>> input_buffers,
    int batch_size, int n_iter)
    : batch_size_(batch_size),
      done_(false),
      params_map_(params_map),
      input_buffers_(std::move(input_buffers)),
      // Make sure setBatch() waits until getBatch() is called (the first time).
      calib_running_(true),
      batch_is_set_(false),
      n_iter_(n_iter) {}

bool TRTInt8Calibrator::setBatch(const std::unordered_map<std::string, void*>& data,
                                 const cudaStream_t stream) {
  std::unique_lock<std::mutex> lk(mutex_);
  // Wait while the queue is full or calibration is running.
  cv_.wait(lk, [&]{ return (!calib_running_ && !batch_is_set_) || done_; });
  if (done_)
    return false;
  n_iter_--;

  for (const auto& it : data) {
    auto in_it = input_buffers_.find(it.first);
    if (in_it == input_buffers_.end()) {
      LOG(FATAL) << "TensorRT op input name '" << it.first
                 << "' does not match with the buffer names";
    }
    const auto& buff_and_size = in_it->second;
    auto status = cudaMemcpyAsync(buff_and_size.first, it.second, buff_and_size.second,
                                  cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
      LOG(FATAL) << "cudaMemcpy in  TensorRT op for '" << it.first
                 << "' failed with " << status;
    }
  }
  // TODO(spanev): see if we can use something like cudaStreamAddCallback here
  MSHADOW_CUDA_CALL(cudaStreamSynchronize(stream));
  batch_is_set_ = true;
  cv_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) {
  // Wait until new batch arrives
  std::unique_lock<std::mutex> lk(mutex_);
  calib_running_ = false;
  cv_.notify_all();

  cv_.wait(lk, [&]{ return batch_is_set_ || done_; });
  if (done_)
    return false;

  for (int i = 0; i < num_bindings; i++) {
    auto it = input_buffers_.find(names[i]);
    if (it == input_buffers_.end()) {
      LOG(FATAL) << "Calibration engine asked for unknown tensor name '"
                 << names[i] << "' at position " << i;
    }
    bindings[i] = it->second.first;
  }
  batch_is_set_ = false;
  calib_running_ = true;
  return true;
}

const void* TRTInt8Calibrator::readCalibrationCache(std::size_t& length) {
  if (calibration_table_.empty()) {
    return nullptr;
  }
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) {
  calibration_table_ = std::string(static_cast<const char*>(ptr), length);
  LOG(INFO) << "[TensorRT op] Got calibration data for TensorRT op @" << ptr
          << " length=" << length;
}

void TRTInt8Calibrator::setDone() {
  done_ = true;
}

void TRTInt8Calibrator::waitAndSetDone() {
  std::unique_lock<std::mutex> lk(mutex_);
  cv_.wait(lk, [&]{ return (!batch_is_set_ && !calib_running_) || done_; });
  if (!done_) {
    done_ = true;
    cv_.notify_all();
    input_buffers_.clear();
  }
}

bool TRTInt8Calibrator::isCacheEmpty() {
  return calibration_table_.empty();
}

bool TRTInt8Calibrator::lastIter() {
  return n_iter_ == 0;
}

TRTInt8Calibrator::~TRTInt8Calibrator() {
  waitAndSetDone();
  for (auto it : input_buffers_) {
    auto ptr_and_size = it.second;
    MSHADOW_CUDA_CALL(cudaFree(ptr_and_size.first));
  }
}

}  // namespace onnx_to_tensorrt

#endif  // MXNET_USE_TENSORRT
