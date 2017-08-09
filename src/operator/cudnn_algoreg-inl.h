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
 * \file cudnn_algoreg-inl.h
 * \brief
 * \author Bing Xu
 */
#ifndef MXNET_OPERATOR_CUDNN_ALGOREG_INL_H_
#define MXNET_OPERATOR_CUDNN_ALGOREG_INL_H_

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include "../common/cuda_utils.h"
#include "./convolution-inl.h"
#include "./deconvolution-inl.h"
namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

/*!
 * \brief A cuDNN algorithm: an algo number and whether it should be run in TENSOR CORE mode.
 */
template <typename CuDNNAlgoType>
class CuDNNAlgo {
 public:
  CuDNNAlgo() :
      algo_number_(static_cast<CuDNNAlgoType>(0)),
      is_tensor_core_algo_(false) { }
  void Set(CuDNNAlgoType algo, bool is_tensor_core) {
    algo_number_ = algo;
    is_tensor_core_algo_ = is_tensor_core;
  }
  CuDNNAlgoType AlgoNumber() const { return algo_number_; }
  bool IsTensorCoreAlgo() const { return is_tensor_core_algo_; }
  #if CUDNN_MAJOR >= 7
  cudnnMathType_t MathType() {
    return IsTensorCoreAlgo() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
  }
  #endif
 private:
  CuDNNAlgoType algo_number_;
  bool is_tensor_core_algo_;
};

class CuDNNAlgoReg {
 public:
  template <typename Param>
  std::string GetKey(const Param &param, const std::vector<TShape> &in_shape,
                     const std::vector<TShape> &out_shape,
                     cudnnDataType_t cudnn_data_type,
                     cudnnDataType_t cudnn_forward_compute_type,
                     cudnnDataType_t cudnn_backward_compute_type,
                     int sm_arch) {
    std::ostringstream oss;
    oss << "inputs=";
    for (auto &i : in_shape)
      oss << i << ";";
    oss << "outputs=";
    for (auto &i : out_shape)
      oss << i << ";";
    auto dict = param.__DICT__();
    for (auto &k : dict)
      oss << k.first << "=" << k.second << ";";
    oss << "cudnn_data_type=" << cudnn_data_type << ";";
    oss << "cudnn_forward_compute_type=" << cudnn_forward_compute_type << ";";
    oss << "cudnn_backward_compute_type=" << cudnn_backward_compute_type << ";";
    // All GPUs of the same compute capability (SM arch) share an algo selection.
    oss << "sm_arch=" << sm_arch << ";";
    return oss.str();
  }

  bool Find(std::string key,
            CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
            CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
            CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt) {
    std::lock_guard<std::mutex> guard(lock_);
    auto i = reg_.find(key);
    if (i != reg_.end()) {
      *fwd = i->second.fwd;
      *bwd = i->second.bwd;
      *flt = i->second.flt;
      return true;
    }
    return false;
  }

  void Register(std::string key,
                const CuDNNAlgo<cudnnConvolutionFwdAlgo_t> &fwd,
                const CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> &bwd,
                const CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> &flt) {
    std::lock_guard<std::mutex> guard(lock_);
    if (reg_.size() % 50 == 0) {
      LOG(INFO) << "Running performance tests to find the best convolution "
                   "algorithm, "
                   "this can take a while... (setting env variable "
                   "MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)";
      if (reg_.size() >= 1000) {
        LOG(INFO)
            << "If you see this message in the middle of training, you are "
               "probably using bucketing. Consider setting env variable "
               "MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable cudnn tuning.";
      }
    }
    reg_[key].fwd = fwd;
    reg_[key].bwd = bwd;
    reg_[key].flt = flt;
  }

  static CuDNNAlgoReg *Get();

 private:
  struct CudnnAlgorithms {
    CuDNNAlgo<cudnnConvolutionFwdAlgo_t> fwd;
    CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> bwd;
    CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> flt;
  };

  std::mutex lock_;
  std::unordered_map<std::string, CudnnAlgorithms> reg_;
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_ALGOREG_INL_H_
