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
 * Copyright (c) 2015 by Contributors
 * \file cudnn_algoreg-inl.h
 * \brief
 * \author Bing Xu
 */
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_ALGOREG_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_ALGOREG_INL_H_

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include <functional>
#include <utility>
#include "../../../common/cuda_utils.h"
#include "../convolution-inl.h"
#include "../deconvolution-inl.h"
namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

/*!
 * \brief A cuDNN algorithm: an algo number and whether it should be run in TENSOR CORE mode.
 */
template <typename CuDNNAlgoType>
class CuDNNAlgo {
  STATIC_ASSERT_CUDNN_VERSION_GE(7000);

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
  cudnnMathType_t MathType() {
    return IsTensorCoreAlgo() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
  }
 private:
  CuDNNAlgoType algo_number_;
  bool is_tensor_core_algo_;
};

template<typename ParamType>
class CuDNNAlgoReg {
 public:
  using AlgoSetter_t = std::function<void(CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *,
                            CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *,
                            CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *)>;

  void FindOrElseRegister(const ParamType &param,
            const mxnet::ShapeVector &in_shape,
            const mxnet::ShapeVector &out_shape,
            cudnnDataType_t cudnn_data_type,
            cudnnDataType_t cudnn_forward_compute_type,
            cudnnDataType_t cudnn_backward_compute_type,
            int sm_arch,
            bool add_to_weight,
            CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd,
            CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> *bwd,
            CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> *flt,
            const AlgoSetter_t &algo_setter) {
    CHECK(in_shape.size() == 2 || in_shape.size() == 3);
    ParamKey key{param, in_shape[0], in_shape[1], out_shape[0], cudnn_data_type,
                 cudnn_forward_compute_type, cudnn_backward_compute_type, sm_arch, add_to_weight};
    std::lock_guard<std::mutex> guard(lock_);
    auto i = reg_.find(key);
    if (i != reg_.end()) {
      *fwd = i->second.fwd;
      *bwd = i->second.bwd;
      *flt = i->second.flt;
    } else {
      if (param.cudnn_tune.value() && reg_.size() % 50 == 0) {
        LOG(INFO) << "Running performance tests to find the best convolution "
            "algorithm, "
            "this can take a while... (set the environment variable "
            "MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)";
        if (reg_.size() >= 1000) {
          // Many people are very concerned about this warning, so change the warning once.
          if (!is_warning_autotune_) {
            LOG(INFO)
                << "If you see this message in the middle of training, you are "
                    "probably using bucketing. Consider setting env variable "
                    "MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable cudnn tuning.";
            is_warning_autotune_ = true;
          }
        }
      }
      // Call provided function to determine the algos- likely uses cudnnFind() or cudnnGet()
      algo_setter(fwd, bwd, flt);
      // Save result so future lookups hit in this registry
      reg_.insert(std::pair<ParamKey, CudnnAlgorithms>(key, CudnnAlgorithms{*fwd, *bwd, *flt}));
    }
  }

  static CuDNNAlgoReg *Get();

 private:
  struct CudnnAlgorithms {
    CuDNNAlgo<cudnnConvolutionFwdAlgo_t> fwd;
    CuDNNAlgo<cudnnConvolutionBwdDataAlgo_t> bwd;
    CuDNNAlgo<cudnnConvolutionBwdFilterAlgo_t> flt;
  };

  struct ParamKey {
    ParamType param;
    mxnet::TShape data_shape, weight_shape, out_shape;
    cudnnDataType_t cudnn_data_type;
    cudnnDataType_t cudnn_forward_compute_type;
    cudnnDataType_t cudnn_backward_compute_type;
    int sm_arch;
    bool add_to_weight;

    bool operator==(const ParamKey& other) const {
      return this->param == other.param &&
             this->data_shape == other.data_shape &&
             this->weight_shape == other.weight_shape &&
             this->out_shape == other.out_shape &&
             this->cudnn_data_type == other.cudnn_data_type &&
             this->cudnn_forward_compute_type == other.cudnn_forward_compute_type &&
             this->cudnn_backward_compute_type == other.cudnn_backward_compute_type &&
             this->sm_arch == other.sm_arch &&
             this->add_to_weight == other.add_to_weight;
    }
  };

  struct ParamHash {
    size_t operator()(const ParamKey& key) const {
      std::hash<ParamType> hash_param;
      size_t ret = hash_param(key.param);
      ret = dmlc::HashCombine(ret, key.data_shape);
      ret = dmlc::HashCombine(ret, key.weight_shape);
      ret = dmlc::HashCombine(ret, key.out_shape);
      ret = dmlc::HashCombine(ret, static_cast<int>(key.cudnn_data_type));
      ret = dmlc::HashCombine(ret, static_cast<int>(key.cudnn_forward_compute_type));
      ret = dmlc::HashCombine(ret, static_cast<int>(key.cudnn_backward_compute_type));
      ret = dmlc::HashCombine(ret, key.sm_arch);
      ret = dmlc::HashCombine(ret, key.add_to_weight);
      return ret;
    }
  };

  std::mutex lock_;
  std::unordered_map<ParamKey, CudnnAlgorithms, ParamHash> reg_;
  bool is_warning_autotune_ = false;
};

typedef CuDNNAlgoReg<ConvolutionParam> CuDNNConvAlgoReg;
typedef CuDNNAlgoReg<DeconvolutionParam> CuDNNDeconvAlgoReg;

#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_ALGOREG_INL_H_
