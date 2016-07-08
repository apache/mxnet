/*!
 * Copyright (c) 2015 by Contributors
 * \file warpctc-inl.h
 * \brief warpctc operator
 * \author Liang Xiang
*/
#ifndef PLUGIN_WARPCTC_WARPCTC_INL_H_
#define PLUGIN_WARPCTC_WARPCTC_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <stdio.h>
#include <ctc.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {

namespace warpctc_enum {
  enum CTCOpInputs {kData, kLabel};
  enum CTCOpOutputs {kOut};
}  // namespace warpctc_enum

struct WarpCTCParam : public dmlc::Parameter<WarpCTCParam> {
  int label_length;
  int input_length;
  DMLC_DECLARE_PARAMETER(WarpCTCParam) {
    DMLC_DECLARE_FIELD(label_length)
        .set_default(0)
        .describe("Real label length");
    DMLC_DECLARE_FIELD(input_length)
        .set_default(0)
        .describe("Input length");
  }
};

template<typename xpu>
class WarpCTCOp : public Operator {
 private:
  WarpCTCParam param_;

 public:
  explicit WarpCTCOp(WarpCTCParam p) {
    this->param_ = p;
  }

  ~WarpCTCOp() {
  }

  inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
      throw std::runtime_error(message
                               + (", stat = "
                                  + std::string(ctcGetStatusString(status))));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "CTCOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "CTCOutput Output: [output]";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    TBlob data = in_data[warpctc_enum::kData];
    TBlob out = out_data[warpctc_enum::kOut];
    Tensor<xpu, 2, float> data_tensor = data.FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, float> out_tensor = out.FlatTo2D<xpu, float>(s);
    Softmax(out_tensor, data_tensor);
  }

  std::vector<int> labelLengths(const int * flat_labels, int minibatch,
                                int size, int blank, int * total_length) {
    CHECK_EQ(param_.label_length * minibatch, size)
        << "label size should = label_length * minibatch";
    std::vector<int> ret(minibatch, 0);
    for (int i = 0; i < size; i++) {
      if (flat_labels[i] == blank) {
        continue;
      }
      int b = i / param_.label_length;
      ret[b]++;
      (*total_length)++;
    }
    return ret;
  }

  void removeBlank(const int * flat_labels, int * cpu_labels,
                   int size, int blank) {
    int k = 0;
    for (int i = 0; i < size; i++) {
      if (flat_labels[i] != blank) {
        cpu_labels[k] = flat_labels[i];
        k += 1;
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    TBlob data = in_data[warpctc_enum::kData];
    TBlob label = in_data[warpctc_enum::kLabel];
    CHECK_EQ(data.shape_.ndim(), 2) << "input data shape should be 2 (t*n, p)";
    ctcComputeInfo info;
    if (data.dev_mask_ == cpu::kDevMask) {
      info.loc = CTC_CPU;
      info.num_threads = 1;
    } else if (data.dev_mask_ == gpu::kDevMask) {
#if MXNET_USE_CUDA
      info.loc = CTC_GPU;
      info.stream = ctx.get_stream<gpu>()->stream_;
#endif
    } else {
      LOG(FATAL) << "Unknown device type " << data.dev_mask_;
    }

    int T = param_.input_length;
    int minibatch = data.shape_[0] / T;
    int alphabet_size = data.shape_[1];
    std::vector<int> input_lengths;
    for (int i = 0; i < minibatch; i++) {
      input_lengths.push_back(T);
    }

#if MXNET_USE_CUDA
    cudaError_t cuda_status;
#endif
    float* activations = static_cast<float*>(data.dptr_);
    int* flat_labels = static_cast<int*>(label.dptr_);
    int* cpu_raw_labels = flat_labels;
    float* grads = static_cast<float*>(in_grad[warpctc_enum::kData].dptr_);
    if (data.dev_mask_ == gpu::kDevMask) {
#if MXNET_USE_CUDA
      cpu_raw_labels = reinterpret_cast<int*>(malloc(sizeof(int) * label.Size()));
      cuda_status = cudaMemcpyAsync(cpu_raw_labels, flat_labels,
                                    label.Size()*sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    ctx.get_stream<gpu>()->stream_);
      CHECK_EQ(cuda_status, cudaSuccess) << "cuda memcpy label error";
#endif
    } else {
      LOG(FATAL) << "Unknown device type " << data.dev_mask_;
    }

    int total_label_length = 0;
    std::vector<int> label_lengths = labelLengths(cpu_raw_labels,
                                                  minibatch,
                                                  label.Size(),
                                                  0, &total_label_length);
    int* cpu_labels = reinterpret_cast<int*>(
        malloc(sizeof(int) * total_label_length));
    removeBlank(cpu_raw_labels, cpu_labels, label.Size(), 0);
    free(cpu_raw_labels);

    size_t alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      input_lengths.data(),
                                      alphabet_size,
                                      input_lengths.size(), info,
                                      &alloc_bytes),
                   "Error: get_workspace_size in inf_test");
    void* ctc_workspace;

    if (data.dev_mask_ == cpu::kDevMask) {
      ctc_workspace = malloc(alloc_bytes);
    } else if (data.dev_mask_ == gpu::kDevMask) {
#if MXNET_USE_CUDA
      cuda_status = cudaMalloc(&ctc_workspace, alloc_bytes);
      CHECK_EQ(cuda_status, cudaSuccess) << "cuda malloc worksapce fail";
#endif
    }
    std::vector<float> costs(minibatch);
    throw_on_error(compute_ctc_loss(activations,
                                    grads,
                                    cpu_labels,
                                    label_lengths.data(),
                                    input_lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_workspace,
                                    info),
                   "Error: compute_ctc_loss");

    if (data.dev_mask_ == cpu::kDevMask) {
      free(ctc_workspace);
    } else if (data.dev_mask_ == gpu::kDevMask) {
#if MXNET_USE_CUDA
      cuda_status = cudaFree(ctc_workspace);
      CHECK_EQ(cuda_status, cudaSuccess) << "cuda free workspace fail";
      free(cpu_labels);
#endif
    }
  }
};

template<typename xpu>
Operator* CreateOp(WarpCTCParam type);


#if DMLC_USE_CXX11
class WarpCTCProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  virtual std::vector<std::string> ListOutputs() const {
    return {"output"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    TShape label_shape(dshape.ndim() - 1);
    label_shape[0] = param_.label_length * (dshape[0] / param_.input_length);
    SHAPE_ASSIGN_CHECK(*in_shape, warpctc_enum::kLabel, label_shape);

    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  virtual bool InferType(std::vector<int> *in_type,
                         std::vector<int> *out_type,
                         std::vector<int> *aux_type) const {
    CHECK_LE(in_type->size(), this->ListArguments().size());
    in_type->clear();
    in_type->push_back(mshadow::kFloat32);
    in_type->push_back(mshadow::kInt32);
    out_type->clear();
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new WarpCTCProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "WarpCTC";
  }


  std::vector<int> DeclareBackwardDependency(const std::vector<int> &out_grad,
                                             const std::vector<int> &in_data,
                                             const std::vector<int> &out_data)
      const override {
    return {in_data[warpctc_enum::kData],
          in_data[warpctc_enum::kLabel],
          out_data[warpctc_enum::kOut]};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  WarpCTCParam param_;
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // PLUGIN_WARPCTC_WARPCTC_INL_H_
