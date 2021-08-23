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
 * \file sync_batch_norm-inl.h
 * \brief Synchronized BatchNorm modified from BatchNormV1
 * \author Hang Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <condition_variable>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace syncbatchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta};
enum BatchNormOpOutputs {kOut, kMean, kVar};
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace syncbatchnorm

struct SyncBatchNormParam : public dmlc::Parameter<SyncBatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  int ndev;
  std::string key;
  DMLC_DECLARE_PARAMETER(SyncBatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(ndev).set_default(1)
      .describe("The count of GPU devices");
    DMLC_DECLARE_FIELD(key)
      .describe("Hash key for synchronization, please set the same hash key for same layer, "
                "Block.prefix is typically used as in :class:`gluon.nn.contrib.SyncBatchNorm`.");
  }
};

// Modified from https://github.com/brucechin/SharedTensor
template<class T>
class SharedND {
 private:
  int num_devices_;
  T mean_;
  T *data_;
  bool *flag_;
  bool mean_ready_ = false;
  bool data_inited_ = false;
  std::mutex mutex_;

 public:
  explicit SharedND(int ndev) :num_devices_(ndev) {
    flag_ = new bool[ndev];
    data_ = new T[ndev];
    memset(flag_, false, ndev * sizeof(bool));
  }

  ~SharedND() {
    if (data_inited_) mshadow::FreeSpace(&mean_);
    delete [] flag_;
    delete [] data_;
  }

  void Init(mshadow::Shape<1> shape) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!data_inited_) {
      for (int i = 0; i < num_devices_; i++) {
        data_[i] = mshadow::NewTensor<cpu, real_t>(shape, 0.0f);
      }
      mean_ = mshadow::NewTensor<cpu, real_t>(shape, 0.0f);
      data_inited_ = true;
    }
  }

  T* Retrieve(mshadow::Shape<1> shape, int index) {
    // Retrieve a pointer for copying values
    if (!data_inited_) {
      Init(shape);
    }
    if (flag_[index] == false) {
      return &data_[index];
    } else {
      return nullptr;
    }
  }

  bool SetReady(int index) {
    // Set data ready after copying
    if (flag_[index] == false) {
      flag_[index] = true;
      return true;
    } else {
      return false;
    }
  }

  T Pop(int index) {
    // Pop the mean value after suming up
    std::lock_guard<std::mutex> lock(mutex_);
    while (!MeanReady()) {}
    flag_[index] = false;
    T tmp = mean_;
    ResetMean();
    return tmp;
  }

  bool MeanReady() {
    if (mean_ready_) {
      return true;
    }
    for (int i = 0; i < num_devices_; i++) {
      if (!flag_[i]) {
        return false;
      }
    }
    for (int i = 1; i < num_devices_; i++) {
      data_[0] += data_[i];
    }
    mean_ = data_[0] * 1.0f /  num_devices_;
    mean_ready_ = true;
    return true;
  }

  void ResetMean() {
    for (int i = 0; i < num_devices_; i++) {
      if (flag_[i]) return;
    }
    mean_ready_ = false;
  }
};

template<class T>
class GlobalShared {
 public:
  T* Register(const std::string &key, int ndev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it != registry_.end()) return it->second;
    T *newT = new T(ndev);
    registry_[key] = newT;
    return newT;
  }
  ~GlobalShared() {
    for (auto it = registry_.begin(); it != registry_.end(); it++) {
      T *ptr = it->second;
      delete ptr;
    }
  }
 private:
  std::mutex mutex_;
  std::map<std::string, T*> registry_;
};

template<class T>
class GlobalSharedRank {
 public:
  T Register(const std::string &key, int ndev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it != registry_.end()) {
      T* tmpT = it->second;
      *tmpT = (*tmpT == ndev - 1) ? 0 : *tmpT + 1;
      return *tmpT;
    }
    T *newT = new T(0);
    registry_[key] = newT;
    return *newT;
  }
  ~GlobalSharedRank() {
    for (auto it = registry_.begin(); it != registry_.end(); it++) {
      T *ptr = it->second;
      delete ptr;
    }
  }
 private:
  std::mutex mutex_;
  std::map<std::string, T*> registry_;
};

class Barrier {
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t count_;
  std::size_t total_count_;
 public:
  explicit Barrier(std::size_t count) : count_{count}, total_count_{count} { }
  void Wait() {
    std::unique_lock<std::mutex> lock{mutex_};
    if (--count_ == 0) {
      count_ = total_count_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this] { return count_ == total_count_; });
    }
  }
};

// Global variables for Synchronizations
static GlobalSharedRank<int> global_shared_rank_forward;
static GlobalSharedRank<int> global_shared_rank_backward;
static GlobalShared<Barrier> global_shared_barrier_forward;
static GlobalShared<Barrier> global_shared_barrier_backward;
static GlobalShared<SharedND<mshadow::Tensor<cpu, 1, real_t>>> global_shared_mean;
static GlobalShared<SharedND<mshadow::Tensor<cpu, 1, real_t>>> global_shared_var;
static GlobalShared<SharedND<mshadow::Tensor<cpu, 1, real_t>>> global_shared_grad;
static GlobalShared<SharedND<mshadow::Tensor<cpu, 1, real_t>>> global_shared_prod;

template<typename xpu>
class SyncBatchNorm : public Operator {
 public:
  explicit SyncBatchNorm(SyncBatchNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[syncbatchnorm::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[syncbatchnorm::kData].shape_[1]) /
      static_cast<real_t>(in_data[syncbatchnorm::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[syncbatchnorm::kData].ndim() == 4) {
      data = in_data[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
      out = out_data[syncbatchnorm::kOut].get<xpu, 4, real_t>(s);
    } else {
      index_t num_channels = in_data[syncbatchnorm::kData].ndim() > 1 ?
        in_data[syncbatchnorm::kData].shape_[1] : 1;
      index_t spatial_size = in_data[syncbatchnorm::kData].shape_.ProdShape(2,
          in_data[syncbatchnorm::kData].ndim());
      Shape<4> dshape = Shape4(in_data[syncbatchnorm::kData].shape_[0],
                               num_channels, 1, spatial_size);
      data = in_data[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[syncbatchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    }
    Tensor<xpu, 1> slope = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[syncbatchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[syncbatchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) {
      // get my rank
      Barrier *global_barrier = global_shared_barrier_forward.Register(param_.key, param_.ndev);
      int myRank = global_shared_rank_forward.Register(param_.key, param_.ndev);
      // get the mean and var
      Tensor<xpu, 1> mean = out_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[syncbatchnorm::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[syncbatchnorm::kMean] == kNullOp || req[syncbatchnorm::kMean] == kWriteTo);
      CHECK(req[syncbatchnorm::kVar] == kNullOp || req[syncbatchnorm::kVar] == kWriteTo);
      // E(x) and E(x^2)
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(data));
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedMean =
        global_shared_mean.Register(param_.key, param_.ndev);
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedVar =
        global_shared_var.Register(param_.key, param_.ndev);
      // copy to cpu, push and pull
      Tensor<cpu, 1, real_t>* mean_cpu_ptr = sharedMean->Retrieve(mean.shape_, myRank);
      Tensor<cpu, 1, real_t>* var_cpu_ptr = sharedVar->Retrieve(mean.shape_, myRank);
      mshadow::Copy(*mean_cpu_ptr, mean, s);
      mshadow::Copy(*var_cpu_ptr, var, s);
      sharedMean->SetReady(myRank);
      sharedVar->SetReady(myRank);
      global_barrier->Wait();
      Tensor<cpu, 1, real_t> mean_cpu = sharedMean->Pop(myRank);
      Tensor<cpu, 1, real_t> var_cpu = sharedVar->Pop(myRank);
      // copy back to gpu
      mshadow::Copy(mean, mean_cpu, s);
      mshadow::Copy(var, var_cpu, s);

      var = var-F<mshadow_op::square>(mean);
      Assign(out, req[syncbatchnorm::kOut], broadcast<1>(slope, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(bias, out.shape_));
    } else {
      Assign(out, req[syncbatchnorm::kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    const real_t scale = static_cast<real_t>(out_grad[syncbatchnorm::kOut].shape_[1]) /
      static_cast<real_t>(out_grad[syncbatchnorm::kOut].shape_.Size());
    if (in_data[syncbatchnorm::kData].ndim() == 4) {
      data = in_data[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[syncbatchnorm::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[syncbatchnorm::kData].get<xpu, 4, real_t>(s);
    } else {
      index_t num_channels = out_grad[syncbatchnorm::kOut].ndim() > 1 ?
        out_grad[syncbatchnorm::kOut].shape_[1] : 1;
      index_t spatial_size = out_grad[syncbatchnorm::kOut].shape_.ProdShape(2,
          out_grad[syncbatchnorm::kOut].ndim());
      Shape<4> dshape = Shape4(out_grad[syncbatchnorm::kOut].shape_[0],
                               num_channels, 1, spatial_size);
      data = in_data[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[syncbatchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[syncbatchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    }

    Tensor<xpu, 1> mean = out_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[syncbatchnorm::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[syncbatchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[syncbatchnorm::kBeta].get<xpu, 1, real_t>(s);
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[syncbatchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[syncbatchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    if (ctx.is_train && !param_.use_global_stats) {
      // get my rank
      Barrier *global_barrier = global_shared_barrier_backward.Register(param_.key, param_.ndev);
      int myRank = global_shared_rank_backward.Register(param_.key, param_.ndev);
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[syncbatchnorm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(5, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      // cal
      Tensor<xpu, 1> sumGrad = workspace[3];
      Tensor<xpu, 1> sumProd = workspace[4];
      sumGrad = sumall_except_dim<1>(grad);
      sumProd = sumall_except_dim<1>(grad * (data - broadcast<1>(mean, data.shape_)));
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedGrad =
        global_shared_grad.Register(param_.key, param_.ndev);
      SharedND<mshadow::Tensor<cpu, 1, real_t>> *sharedProd =
        global_shared_prod.Register(param_.key, param_.ndev);
      // copy to cpu, push and pull
      Tensor<cpu, 1, real_t>* grad_cpu_ptr = sharedGrad->Retrieve(sumGrad.shape_, myRank);
      Tensor<cpu, 1, real_t>* prod_cpu_ptr = sharedProd->Retrieve(sumGrad.shape_, myRank);
      mshadow::Copy(*grad_cpu_ptr, sumGrad, s);
      mshadow::Copy(*prod_cpu_ptr, sumProd, s);
      sharedGrad->SetReady(myRank);
      sharedProd->SetReady(myRank);
      global_barrier->Wait();
      Tensor<cpu, 1, real_t> grad_cpu = sharedGrad->Pop(myRank);
      Tensor<cpu, 1, real_t> prod_cpu = sharedProd->Pop(myRank);
      // copy back to gpu
      mshadow::Copy(sumGrad, grad_cpu, s);
      mshadow::Copy(sumProd, prod_cpu, s);

      gvar = -1.0f * sumProd * slope *
        F<mshadow_op::power>(var + param_.eps, -1.5f);
      gmean =  sumGrad * slope;
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      // assign
      if (!param_.fix_gamma) {
        Assign(gslope, req[syncbatchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[syncbatchnorm::kGamma], 0.0f);
      }
      Assign(grad_in, req[syncbatchnorm::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
               broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) *
               scale * (data - broadcast<1>(mean, data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbias, req[syncbatchnorm::kBeta], sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      if (!param_.fix_gamma) {
        Assign(gslope, req[syncbatchnorm::kGamma],
               sumall_except_dim<1>(
                 grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                 F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[syncbatchnorm::kGamma], 0.0f);
      }
      Assign(gbias, req[syncbatchnorm::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[syncbatchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
               1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNorm

template<typename xpu>
Operator *CreateOp(SyncBatchNormParam param, int dtype);


#if DMLC_USE_CXX11
class SyncBatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (mxnet::op::shape_is_none(dshape)) return false;
    in_shape->at(1) = mxnet::TShape(Shape1(dshape[1]));
    in_shape->at(2) = mxnet::TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param = (dtype == kFloat16) ? kFloat32 : dtype;
    for (size_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (size_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        UNIFORM_TYPE_CHECK((*aux_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    int n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (int i = 0; i < n_aux; ++i ) aux_type->push_back(dtype_param);
    int n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (int i = 1; i < n_out; ++i ) out_type->push_back(dtype_param);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SyncBatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_SyncBatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[syncbatchnorm::kOut],
            out_data[syncbatchnorm::kMean],
            out_data[syncbatchnorm::kVar],
            in_data[syncbatchnorm::kData],
            in_data[syncbatchnorm::kGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return nullptr;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
      std::vector<int> *in_type) const override;

  inline const SyncBatchNormParam& getParam() const {
    return param_;
  }

 private:
  SyncBatchNormParam param_;
};  // class SyncBatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_SYNC_BATCH_NORM_INL_H_
