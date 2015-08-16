/*!
 * Copyright (c) 2015 by Contributors
 * \file elem_plus-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ELEM_PLUS_INL_H_
#define MXNET_OPERATOR_ELEM_PLUS_INL_H_
namespace mxnet {
namespace op {
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

enum ElemsPlusOpInputs {kData0, kData1, kData2, kData3};
enum ElemsPlusOpOutputs {kOut};

template<typename xpu>
class ElemPlusOp : public Operator {
 public:
  explicit ElemPlusOp(uint32_t size) : size_(size) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), cnt_) << "Invalid Input TBlobs";
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    switch (size_) {
      case 1: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0);
        break;
      }
      case 2: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1);
        break;
      }
      case 3: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1 + in_3);
        break;
      }
      case 4: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> in_3 = in_data[kData3].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kOut], in_0 + in_1 + in_3 + in_4);
        break;
      }
      default: {
        LOG_FATAL;
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), size_);
    CHECK_EQ(out_data.size(), size_);
    switch (size_) {
      case 1: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_0 = out_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out, req[kData0], F<mshadow_op::identity_grad>(in_0));
        break;
      }
      case 2: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_0 = out_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out_0, req[kData0], F<mshadow_op::identity_grad>(in_0));
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_1 = out_data[kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out_1, req[kData1], F<mshadow_op::identity_grad>(in_1));
        break;
      }
      case 3: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_0 = out_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out_0, req[kData0], F<mshadow_op::identity_grad>(in_0));
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_1 = out_data[kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out_1, req[kData1], F<mshadow_op::identity_grad>(in_1));
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_2 = out_data[kData2].FlatTo2D<xpu, real_t>(s);
        Assign(out_2, req[kData2], F<mshadow_op::identity_grad>(in_2));
        break;
      }
      case 4: {
        Tensor<xpu, 2> in_0 = in_data[kData0].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_0 = out_data[kData0].FlatTo2D<xpu, real_t>(s);
        Assign(out_0, req[kData0], F<mshadow_op::identity_grad>(in_0));
        Tensor<xpu, 2> in_1 = in_data[kData1].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_1 = out_data[kData1].FlatTo2D<xpu, real_t>(s);
        Assign(out_1, req[kData1], F<mshadow_op::identity_grad>(in_1));
        Tensor<xpu, 2> in_2 = in_data[kData2].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_2 = out_data[kData2].FlatTo2D<xpu, real_t>(s);
        Assign(out_2, req[kData2], F<mshadow_op::identity_grad>(in_2));
        Tensor<xpu, 2> in_3 = in_data[kData3].FlatTo2D<xpu, real_t>(s);
        Tensor<xpu, 2> out_3 = out_data[kData3].FlatTo2D<xpu, real_t>(s);
        Assign(out_3, req[kData3], F<mshadow_op::identity_grad>(in_3));
        break;
      }
      default: {
        LOG_FATAL;
      }
    }
  }

 private:
  uint32_t size_;
};  // class ElemPlusOp

template<typename xpu>
Operator* CreateElemPlusOp(uint32_t size);

#if DMLC_USE_CXX11
class ElemPlusProp : public OperatorProperty {
 public:
  explicit ElemPlusProp() : size_(0) {}

  explicit ElemPlusProp(uint32_t sz) : size_(sz) {}

  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "size")) size_ = static_cast<uint32_t>(atoi(val));
    CHECK_GE(size_, 0);
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    CHECK_GE(size_, 0);
    CHECK_EQ(in_shape->size(), size_) << "Input should be: " << size_ << \
      "(Given: " << in_shape->size() << ")";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    for (auto i : size_) {
      CHECK_EQ(dshape, in_shape->at(i)) << "Input at " << i << " has different shape";
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  virtual OperatorProperty* Copy() const {
    auto ptr = new ElemPlusProp(size_);
    return ptr;
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    std::vector<int> ret(size_);
    for (auto i : size_) {
      ret[i] = in_data[i];
    }
    return ret;
  }

  virtual std::vector<std::pair<int, int> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<int> &in_grad) const {
    std::vector<std::pair<int, int> > ret;
    for (auto i : size_) {
      ret.emplace_back(in_data[i], in_grad[i]);
    }
    return ret;
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  uint32_t size_;
};  // class ElemPlusProp

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEM_PLUS_INL_H_
