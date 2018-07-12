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
 * \file inplace_abn-inl.h
 * \brief Inplace Activated BatchNorm
 * modified from sync_batch_norm-inl.h
 * \author Hang Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_INPLACE_ABN_INL_H_
#define MXNET_OPERATOR_CONTRIB_INPLACE_ABN_INL_H_

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

namespace inplaceabn {
enum InplaceABNInputs {kData, kGamma, kBeta};
enum InplaceABNOutputs {kOut, kMean, kVar};
enum InplaceABNAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};
}  // namespace inplaceabn

struct InplaceABNParam : public dmlc::Parameter<InplaceABNParam> {
  float eps;
  float momentum;
  float slope;
  bool use_global_stats;
  bool output_mean_var;
  bool sync;
  int ndev;
  std::string key;
  DMLC_DECLARE_PARAMETER(InplaceABNParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(slope).set_default(0.25f)
    .describe("Init slope for the activation. (For leaky and elu only)");
    DMLC_DECLARE_FIELD(sync).set_default(false)
    .describe("Syncrhonize the BatchNorm using global mean and var.");
    DMLC_DECLARE_FIELD(ndev).set_default(1)
    .describe("The count of GPU devices");
    DMLC_DECLARE_FIELD(key)
      .set_default("")
      .describe("Hash key for synchronization");
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
    mshadow::FreeSpace(&mean_);
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
    if (flag_[index] == false) {
      flag_[index] = true;
      return true;
    } else {
      return false;
    }
  }

  T Pop(int index) {
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
class InplaceABN : public Operator {
 public:
  explicit InplaceABN(InplaceABNParam param) {
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
      CHECK_EQ(req[inplaceabn::kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[inplaceabn::kData].shape_[1]) /
      static_cast<real_t>(in_data[inplaceabn::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[inplaceabn::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[inplaceabn::kData].shape_[0],
                               in_data[inplaceabn::kData].shape_[1], 1, 1);
      data = in_data[inplaceabn::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[inplaceabn::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[inplaceabn::kData].get<xpu, 4, real_t>(s);
      out = out_data[inplaceabn::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> gamma = in_data[inplaceabn::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> beta = in_data[inplaceabn::kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[inplaceabn::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[inplaceabn::kMovingVar].get<xpu, 1, real_t>(s);

    // whether use global statistics
    if (ctx.is_train && !param_.use_global_stats) {
      // get the mean and var
      Tensor<xpu, 1> mean = out_data[syncbatchnorm::kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[syncbatchnorm::kVar].get<xpu, 1, real_t>(s);
      CHECK(req[syncbatchnorm::kMean] == kNullOp || req[syncbatchnorm::kMean] == kWriteTo);
      CHECK(req[syncbatchnorm::kVar] == kNullOp || req[syncbatchnorm::kVar] == kWriteTo);
      // E(x) and E(x^2)
      mean = scale * sumall_except_dim<1>(data);
      var = scale * sumall_except_dim<1>(F<mshadow_op::square>(data));
      if (param_.sync) {
        // get my rank
        Barrier *global_barrier = global_shared_barrier_forward.Register(param_.key, param_.ndev);
        int myRank = global_shared_rank_forward.Register(param_.key, param_.ndev);
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
      }

      var = var-F<mshadow_op::square>(mean);
      data = broadcast<1>(gamma, out.shape_) *
             (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)) +
             broadcast<1>(beta, out.shape_)
      // leaky relu forward
      MXNET_ASSIGN_REQ_SWITCH(req[inplaceabn::kOut], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
          s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(param_.slope));
      });
    } else {
      data = broadcast<1>(gamma / F<mshadow_op::square_root>(moving_var + param_.eps),
                          data.shape_) * data +
             broadcast<1>(beta - (gamma * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_)
      // leaky relu forward
      MXNET_ASSIGN_REQ_SWITCH(req[inplaceabn::kOut], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
          s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(param_.slope));
      });
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
    const real_t scale = static_cast<real_t>(out_grad[inplaceabn::kOut].shape_[1]) /
      static_cast<real_t>(out_grad[inplaceabn::kOut].shape_.Size());
    if (in_data[inplaceabn::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[inplaceabn::kOut].shape_[0],
                               out_grad[inplaceabn::kOut].shape_[1], 1, 1);
      // data is the output
      data = out_data[inplaceabn::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[inplaceabn::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[inplaceabn::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      // data is the output
      data = out_data[inplaceabn::kOut].get<xpu, 4, real_t>(s);
      grad = out_grad[inplaceabn::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[inplaceabn::kData].get<xpu, 4, real_t>(s);
    }
    // recover y and dl/dy
    mxnet_op::Kernel<mshadow_op::xelu, xpu>::Launch(
      s, data.size(0) * data.size(1) * data.size(2), data.dptr_, data.dptr_, DType(1.0 / param_.slope));
    mxnet_op::Kernel<mshadow_op::xelu, xpu>::Launch(
      s, grad.size(0) * grad.size(1) * grad.size(2), grad.dptr_, grad.dptr_, DType(param_.slope));

    Tensor<xpu, 1> mean = out_data[inplaceabn::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[inplaceabn::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gamma = in_data[inplaceabn::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> beta = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> ggamma = in_grad[inplaceabn::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbeta = in_grad[inplaceabn::kBeta].get<xpu, 1, real_t>(s);
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[inplaceabn::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[inplaceabn::kMovingVar].get<xpu, 1, real_t>(s);

    if (ctx.is_train && !param_.use_global_stats) {
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[syncbatchnorm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(5, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
      // Tensor<xpu, 1> tmp = workspace[2];

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      // cal
      Tensor<xpu, 1> sumGrad = workspace[3];
      Tensor<xpu, 1> sumProd = workspace[4];
      sumGrad = sumall_except_dim<1>(grad);
      sumProd = sumall_except_dim<1>(grad * (data - broadcast<1>(mean, data.shape_)));
      if (param_.sync) {
        // get my rank
        Barrier *global_barrier = global_shared_barrier_backward.Register(param_.key, param_.ndev);
        int myRank = global_shared_rank_backward.Register(param_.key, param_.ndev);
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
      }
      gvar = (sumProd - sumGrad * mean) * gamma * (-0.5f) *
        F<mshadow_op::power>(var + param_.eps, -1.5f);
      gmean =  sumGrad * gamma;
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      // assign
      Assign(ggamma, req[inplaceabn::kGamma],
             sumall_except_dim<1>(
                 grad * (data - broadcast<1>(mean, data.shape_)) /
                 F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      Assign(grad_in, req[inplaceabn::kData],
             (grad * broadcast<1>(gamma, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) *
              scale * 2.0f * (data - broadcast<1>(mean, data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbeta, req[inplaceabn::kBeta], sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      Assign(ggamma, req[inplaceabn::kGamma],
             sumall_except_dim<1>(
               grad * (data - broadcast<1>(moving_mean, data.shape_)) /
               F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      Assign(gbeta, req[inplaceabn::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[inplaceabn::kData], (grad * broadcast<1>(gamma, data.shape_)) *
             broadcast<1>(
               1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  InplaceABNParam param_;
};  // class InplaceABN

template<typename xpu>
Operator *CreateOp(InplaceABNParam param, int dtype);


#if DMLC_USE_CXX11
class InplaceABNProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
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
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, ListArguments()[i]);
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
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
    auto ptr = new InplaceABNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "InplaceABN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[inplaceabn::kOut],
            out_data[inplaceabn::kOut],
            out_data[inplaceabn::kMean],
            out_data[inplaceabn::kVar],
            //in_data[inplaceabn::kData],
            in_data[inplaceabn::kGamma]
           };
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
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
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const InplaceABNParam& getParam() const {
    return param_;
  }

 private:
  InplaceABNParam param_;
};  // class InplaceABNProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_INPLACE_ABN_INL_H_
