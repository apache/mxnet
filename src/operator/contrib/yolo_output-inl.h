/*!
 * Copyright (c) 2017 by Contributors
 * \file yolo_output-inl.h
 * \brief yolo-v2 output layer
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
#define MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
namespace yoloout_enum {
enum YoloOutputOpInputs {kData, kLabel};
enum YoloOutputOpOutputs {kOut, kTemp};
enum YoloOutputOpResource {kTempSpace};
}  // namespace yoloout_enum

struct YoloOutputParam : public dmlc::Parameter<YoloOutputParam> {
  int num_class;
  int num_anchor;
  float overlap_thresh;
  float object_grad_scale;
  float background_grad_scale;
  float class_grad_scale;
  float coord_grad_scale;
  nnvm::Tuple<float> anchors;
  DMLC_DECLARE_PARAMETER(YoloOutputParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
    .describe("Number of object classes.");
    DMLC_DECLARE_FIELD(num_anchor).set_default(5)
    .set_lower_bound(1)
    .describe("Number of anchors.");
    DMLC_DECLARE_FIELD(overlap_thresh).set_default(0.6)
    .describe("Positive overlap threshold.");
    DMLC_DECLARE_FIELD(object_grad_scale).set_default(1.0)
    .describe("Gradient scale for positive objects.");
    DMLC_DECLARE_FIELD(background_grad_scale).set_default(1.0)
    .describe("Gradient scale for background.");
    DMLC_DECLARE_FIELD(class_grad_scale).set_default(1.0)
    .describe("Gradient scale for positive objects.");
    DMLC_DECLARE_FIELD(coord_grad_scale).set_default(1.0)
    .describe("Gradient scale for positive objects.");
    DMLC_DECLARE_FIELD(anchors)
    .set_default({1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52})
    .describe("Gradient scale for positive objects.");
  }
};  // struct YoloOutputParam

// compute intersection-over-union overlap between two boxes
struct calc_overlap {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out,
      const DType* L1, const DType* T1, const DType* R1, const DType* B1,
      const DType* L2, const DType* T2, const DType* R2, const DType* B2) {
    DType l1 = L1[i];
    DType t1 = T1[i];
    DType r1 = R1[i];
    DType b1 = B1[i];
    DType l2 = L2[i];
    DType t2 = T2[i];
    DType r2 = R2[i];
    DType b2 = B2[i];
    DType w1 = r1 - l1;
    DType h1 = b1 - t1;
    if (w1 <= 0 || h1 <= 0) {
      out[i] = 0;
      return;
    }
    DType area1 = w1 * h1;
    DType w2 = r2 - l2;
    DType h2 = b2 - t2;
    if (w2 <= 0 || h2 <= 0) {
      out[i] = 0;
      return;
    }
    DType area2 = w2 * h2;
    DType left = l1 > l2 ? l1 : l2;
    DType right = r1 < r2 ? r1 : r2;
    DType dw = right - left;
    if (dw <= 0) {
      out[i] = 0;
      return;
    }
    DType top = t1 > t2 ? t1 : t2;
    DType bottom = b1 < b2 ? b1 : b2;
    DType dh = bottom - top;
    if (dh <= 0) {
      out[i] = 0;
      return;
    }
    DType inter_area = dw * dh;
    out[i] = inter_area / (area1 + area2 - inter_area);
  }
};

// create index mask for labels
struct index_mask {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out,
      const DType* x, const DType* y, const index_t width, const index_t height,
      const int stride, const DType on_value) {
    if (x[i] < 0 || y[i] < 0) return;
    int depth = width * height * stride;
    int offset = i * depth;
    int start = static_cast<int>(y[i] * width + x[i]) * stride;
    for (int j = 0; j < stride; ++j) {
      int pos = start + j;
      if (pos >= 0 && pos < depth) {
        out[offset + pos] = on_value;
      }
    }
  }
};

template<typename xpu, typename DType>
class YoloOutputOp : public Operator {
 public:
  explicit YoloOutputOp(YoloOutputParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     using namespace mshadow;
     using namespace mshadow::expr;
     CHECK_EQ(in_data.size(), 2U) << "YoloOutput Input: [data, label]";
     CHECK_EQ(out_data.size(), 2U) << "YoloOutput Output: [output, temp]";
     Stream<xpu> *s = ctx.get_stream<xpu>();
     Tensor<xpu, 4, DType> data = in_data[yoloout_enum::kData]
      .get<xpu, 4, DType>(s);
     Tensor<xpu, 3, DType> out = out_data[yoloout_enum::kOut]
      .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> temp = out_data[yoloout_enum::kTemp]
      .get<xpu, 3, DType>(s);
     Shape<3> tshape = temp.shape_;
     int nc = param_.num_class;
     tshape[2] = nc;
     Tensor<xpu, 4, DType> buffer = ctx.requested[yoloout_enum::kTempSpace]
      .get_space_typed<xpu, 4, DType>(Shape4(2, tshape[0], tshape[1], tshape[2]), s);
     ScalarExp<DType> in_w = ScalarExp<DType>(1.0 / data.shape_[3]);
     ScalarExp<DType> in_h = ScalarExp<DType>(1.0 / data.shape_[2]);

     // change the order of dimensions;
     temp = reshape(transpose(data, Shape4(0, 2, 3, 1)), temp.shape_);
     buffer[1] = slice<2>(temp, 0, nc);
     Softmax(buffer[0], buffer[1]);
     slice<2>(temp, 0, nc) = buffer[0];
     // class id to output
     slice<2>(out, 0, 1) = reshape(reduce_with_axis<red::maximum, true>(
      buffer[0], 2), Shape3(out.shape_[0], out.shape_[1], 1));
     // scores to output
     slice<2>(out, 1, 2) = F<mshadow_op::sigmoid>(slice<2>(
      temp, nc, nc + 1));
     // x = (logistic(pred[0]) + i) / in_w
     tshape[2] = 1;
     slice<2>(out, 2, 3) = in_w * (F<mshadow_op::sigmoid>(slice<2>(
      temp, nc + 1, nc + 2)) +
      reshape(broadcast_with_axis(repmat(range<DType>(
      0, data.shape_[3], 1, param_.num_anchor), data.shape_[2]), -1,
      data.shape_[0]), tshape));
     // y = (logistic(pred[1]) + j) / in_h
     slice<2>(out, 3, 4) = in_h * (F<mshadow_op::sigmoid>(slice<2>(
      temp, nc + 2, nc + 3)) +
      reshape(broadcast_with_axis(range<DType>(0,
      data.shape_[2], 1, data.shape_[3] * param_.num_anchor), -1, data.shape_[0]),
      tshape));
     // anchors
     tshape[2] = 2;
     nnvm::Tuple<DType> anchors(param_.anchors.begin(), param_.anchors.end());
     Tensor<cpu, 1, DType> cpu_bias(anchors.begin(), Shape1(anchors.ndim()));
     Tensor<xpu, 1, DType> xpu_bias = ctx.requested[yoloout_enum::kTempSpace]
      .get_space_typed<xpu, 1, DType>(cpu_bias.shape_, s);
     Copy(xpu_bias, cpu_bias, s);
     slice<2>(temp, nc, nc + 2) = reshape(
      broadcast_with_axis(broadcast_with_axis(reshape(xpu_bias,
      Shape2(param_.num_anchor, 2)), 0, data.shape_[2] * data.shape_[3]),
      -1, data.shape_[0]), tshape);
     // w = exp(pred[2]) * anchor[w] / in_w
     slice<2>(out, 4, 5) = in_w * F<mshadow_op::exp>(slice<2>(temp, nc + 3, nc + 4)) *
      slice<2>(temp, nc, nc + 1);
     // h = exp(pred[3]) * anchor[y] / in_h
     slice<2>(out, 5, 6) = in_h * F<mshadow_op::exp>(slice<2>(temp, nc + 4, nc + 5)) *
      slice<2>(temp, nc + 1, nc + 2);

     // convert output to xmin, ymin, xmax, ymax format
     slice<2>(out, 2, 3) -= ScalarExp<DType>(0.5) * slice<2>(out, 4, 5);
     slice<2>(out, 3, 4) -= ScalarExp<DType>(0.5) * slice<2>(out, 5, 6);
     slice<2>(out, 4, 5) += slice<2>(out, 2, 3);
     slice<2>(out, 5, 6) += slice<2>(out, 3, 4);

      // Tensor<cpu, 3, DType> debug_bias = ctx.requested[yoloout_enum::kTempSpace]
      //  .get_host_space_typed<3, DType>(temp.shape_);
      // Copy(debug_bias, temp, s);
      // for (int i = 0; i < 845; ++i) {
      //   LOG(INFO) << i << ": " << debug_bias[0][i][0 + 20] << ", " << debug_bias[0][i][21];
      // }



    //  // apply softmax to class predictions
    //  Tensor<xpu, 3, DType> softmax_in = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(Shape3(temp.shape_[0], temp.shape_[1],
    //   param_.num_class), s);
    //  Tensor<xpu, 3, DType> softmax_out = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(softmax_in.shape_, s);
    //  softmax_in = slice<2>(temp_space, 0, param_.num_class);
    //  softmax_out = slice<2>(temp, 0, param_.num_class);
    //  Softmax(softmax_out, softmax_in);
    //  // apply logistic to object score and box x and y
    //  slice<2>(temp, param_.num_class, param_.num_class + 3) =
    //   F<mshadow_op::sigmoid>(slice<2>(temp_space, param_.num_class,
    //   param_.num_class + 3));
    //  // predicted coordinates, x = (i + px) / w, y = (j + py) / h
    //  // w = exp(pw) * anchor_w / w, h = exp(ph) * anchor_h / h
    //  Tensor<xpu, 3, DType> bx = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(Shape3(temp.shape_[0], temp.shape_[1],
    //   1), s);
    //  Tensor<xpu, 3, DType> by = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(bx.shape_, s);
    //  Tensor<xpu, 3, DType> bw = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(bx.shape_, s);
    //  Tensor<xpu, 3, DType> bh = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(bx.shape_, s);
    //  bx = slice<2>(temp, param_.num_class + 1, param_.num_class + 2);
    //  by = slice<2>(temp, param_.num_class + 2, param_.num_class + 3);
    //  bw = slice<2>(temp, param_.num_class + 3, param_.num_class + 4);
    //  bh = slice<2>(temp, param_.num_class + 4, param_.num_class + 5);
    //  bx += reshape(broadcast_with_axis(repmat(range<DType>(0, data.shape_[3],
    //    1, param_.num_anchor), data.shape_[2]), -1, data.shape_[0]), bx.shape_);
    //  bx /= static_cast<DType>(data.shape_[3]);  // divide w
    //  by += reshape(broadcast_with_axis(range<DType>(0, data.shape_[2], 1,
    //    data.shape_[3] * param_.num_anchor), -1, data.shape_[0]), by.shape_);
    //  by /= static_cast<DType>(data.shape_[2]);  // divide h
     //
    //  nnvm::Tuple<DType> anchors(param_.anchors.begin(), param_.anchors.end());
    //  Tensor<cpu, 1, DType> cpu_bias(anchors.begin(), Shape1(anchors.ndim()));
    //  Tensor<xpu, 1, DType> xpu_bias = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 1, DType>(cpu_bias.shape_, s);
    //  Copy(xpu_bias, cpu_bias, xpu_bias.stream_);
    //  Tensor<xpu, 3, DType> bias = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 3, DType>(Shape3(temp.shape_[0], temp.shape_[1],
    //   2), s);
    //  bias = reshape(broadcast_with_axis(broadcast_with_axis(reshape(xpu_bias,
    //    Shape2(param_.num_anchor, 2)), 0, data.shape_[2] * data.shape_[3]),
    //    -1, data.shape_[0]), bias.shape_);
    //  Tensor<cpu, 3, DType> debug_bias = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_host_space_typed<3, DType>(bias.shape_);
    //  Copy(debug_bias, bias, bias.stream_);
    //  for (int i = 0; i < 100; ++i) {
    //    LOG(INFO) << i << ": " << debug_bias[0][i][0] << ", " << debug_bias[0][i][1];
    //  }
    //  bw = F<mshadow_op::exp>(bw) * slice<2>(bias, 0, 1);
    //  bw /= static_cast<DType>(data.shape_[3]);  // divide w
    //  bh = F<mshadow_op::exp>(bh) * slice<2>(bias, 1, 2);
    //  bh /= static_cast<DType>(data.shape_[2]);  // divide h
     //
    //  // class id
    //  slice<2>(out, 0, 1) = reshape(reduce_with_axis<red::maximum, true>(
    //    softmax_out, 2), Shape3(out.shape_[0], out.shape_[1], 1));
    //  // score
    //  slice<2>(out, 1, 2) = F<mshadow_op::identity>(
    //   slice<2>(temp, param_.num_class, param_.num_class + 1));
    //  // x, y, w, h
    //  slice<2>(out, 2, 3) = bx;
    //  slice<2>(out, 3, 4) = by;
    //  slice<2>(out, 4, 5) = bw;
    //  slice<2>(out, 5, 6) = bh;
     //
    //  Copy(debug_bias, bias, bias.stream_);
    //  for (int i = 0; i < 100; ++i) {
    //    LOG(INFO) << i << ": " << debug_bias[0][i][0] << ", " << debug_bias[0][i][1];
    //  }
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
    CHECK_EQ(in_data.size(), 2U) << "YoloOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 2U) << "YoloOutput Output: [output, temp]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> label = in_data[yoloout_enum::kLabel].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> grad = in_grad[yoloout_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> temp_out = out_data[yoloout_enum::kTemp].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> out = out_data[yoloout_enum::kOut].get<xpu, 3, DType>(s);
    grad = 0.f;
    index_t num_batch = label.shape_[0];
    index_t num_label = label.shape_[1];
    index_t num_box = out.shape_[1];
    index_t nc = param_.num_class;
    // LOG(INFO) << "Label size: " << num_label;

    // temp space
    Shape<2> label_shape = Shape2(num_batch, num_box);
    Shape<3> softmax_shape = Shape3(num_batch, num_box, nc);
    Shape<4> overlaps_shape = Shape4(9, num_batch, num_box, num_label);
    Shape<3> grad_shape = Shape3(num_batch, num_box, nc + 5);
    Shape<4> label_index_shape = Shape4(2, num_batch, num_label, 1);
    Shape<3> temp_index_mask_shape = Shape3(num_batch, num_label, num_box);
    size_t temp_size_total = label_shape.Size() + 2 * softmax_shape.Size() +
     overlaps_shape.Size() + grad_shape.Size() + label_index_shape.Size() +
     temp_index_mask_shape.Size();
    // LOG(INFO) << "Total size: " << temp_size_total;
    Tensor<xpu, 1, DType> temp_space = ctx.requested[yoloout_enum::kTempSpace]
     .get_space_typed<xpu, 1, DType>(Shape1(temp_size_total), s);
    // LOG(INFO) << "Total dptr: " << temp_space.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 2, DType> temp_label(temp_space.dptr_, label_shape, s);
    // LOG(INFO) << "Label dptr: " << temp_label.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 3, DType> temp_softmax(temp_label.dptr_ + temp_label.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmax dptr: " << temp_softmax.dptr_ << ", " << softmax_shape.Size();
    Tensor<xpu, 3, DType> temp_softmax_grad(temp_softmax.dptr_ + temp_softmax.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmaxgrad dptr: " << temp_softmax_grad.dptr_ << ", " << softmax_shape.Size();
    // [0]-[7] for x1, y1, w1, h1, x2, y2, w2, h2, [8] for overlap
    Tensor<xpu, 4, DType> buffer(temp_softmax_grad.dptr_ + temp_softmax_grad.MSize(),
     overlaps_shape, s);
    // LOG(INFO) << "overlap dptr: " << buffer.dptr_ << ", " << overlaps_shape.Size();
    Tensor<xpu, 3, DType> temp_grad(buffer.dptr_ + buffer.MSize(),
     grad_shape, s);
    Tensor<xpu, 4, DType> label_index(temp_grad.dptr_ + temp_grad.MSize(),
     label_index_shape, s);
    Tensor<xpu, 3, DType> temp_index_mask(label_index.dptr_ + label_index.MSize(),
     temp_index_mask_shape, s);

    Shape<3> tshape = Shape3(num_batch, num_box, num_label);
    for (int i = 0; i < 4; ++i) {
      // gt_x, gt_y, gt_w, gt_h
      buffer[i] = reshape(broadcast_with_axis(slice<2>(label, i + 1, i + 2), 0,
       num_box), tshape);
      // pred_x, pred_y, pred_w, pred_h
      buffer[i + 4] = reshape(broadcast_with_axis(slice<2>(temp_out, i + 2, i + 3),
       1, num_label), tshape);
    }
    mxnet_op::Kernel<calc_overlap, xpu>::Launch(s, tshape.Size(), buffer[8].dptr_,
     buffer[0].dptr_, buffer[1].dptr_, buffer[2].dptr_, buffer[3].dptr_,
     buffer[4].dptr_, buffer[5].dptr_, buffer[6].dptr_, buffer[7].dptr_);

    // objectness grad
    slice<2>(temp_grad, nc, nc + 1) = ScalarExp<DType>(-param_.background_grad_scale) *
     slice<2>(out, 1, 2);
    Shape<3> sshape = Shape3(num_batch, num_box, 1);
    // mask out when iou > thresh
    slice<2>(temp_grad, nc, nc + 1) *= reshape(F<mshadow_op::lt>(
     reduce_with_axis<red::maximum, false>(buffer[8], 2),
     ScalarExp<DType>(param_.overlap_thresh)), sshape);
    // block index i and j regarding each ground-truth
    ScalarExp<DType> halfw = ScalarExp<DType>(0.5 * grad.shape_[3]);
    ScalarExp<DType> halfh = ScalarExp<DType>(0.5 * grad.shape_[2]);
    label_index[0] = F<mshadow_op::floor>(halfw * (slice<2>(label, 1, 2) +
     slice<2>(label, 3, 4)));
    label_index[1] = F<mshadow_op::floor>(halfh * (slice<2>(label, 2, 3) +
     slice<2>(label, 4, 5)));
    temp_index_mask = 0;
    mxnet_op::Kernel<index_mask, xpu>::Launch(s, num_batch * num_label,
     temp_index_mask.dptr_, label_index[0].dptr_, label_index[1].dptr_,
     grad.shape_[3], grad.shape_[2], param_.num_anchor, DType(1));
  }

 private:
  YoloOutputParam param_;
};  // class YoloOutputOp

template<typename xpu>
Operator *CreateOp(YoloOutputParam, int dtype);

#if DMLC_USE_CXX11
class YoloOutputProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "temp"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    if (dshape.ndim() != 4) throw InferShapeError("4-dim data required",
      yoloout_enum::kData);

    // data shape
    CHECK_EQ(param_.anchors.ndim() % 2, 0);
    CHECK_EQ(param_.num_anchor, param_.anchors.ndim() / 2) << "anchor number mismatch";
    int num_channel = param_.num_anchor * (param_.num_class + 1 + 4);
    TShape data_shape = Shape4(dshape[0], num_channel, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, yoloout_enum::kData, data_shape);
    // label shape
    TShape lshape = in_shape->at(yoloout_enum::kLabel);
    CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-(>=5)] tensor";
    CHECK_GT(lshape[1], 0) << "Padded label should > 0";
    CHECK_GE(lshape[2], 5) << "Label width must >=5";
    // output shape
    TShape oshape = Shape3(dshape[0], param_.num_anchor * dshape[2] * dshape[3], 6);
    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(Shape3(dshape[0], param_.num_anchor * dshape[2] * dshape[3],
      param_.num_class + 4 + 1));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new YoloOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_YoloOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[yoloout_enum::kLabel], out_data[yoloout_enum::kOut],
      out_data[yoloout_enum::kTemp]};
  }

  std::vector<ResourceRequest> ForwardResource(
       const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  YoloOutputParam param_;
};  // YoloOutputProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
