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

template<typename DType>
MSHADOW_XINLINE DType Intersect(DType l1, DType r1, DType l2, DType r2) {
  DType left = l1 > l2 ? l1 : l2;
  DType right = r1 < r2 ? r1 : r2;
  DType w = right - left;
  return w > 0 ? w : DType(0);
}

template<typename DType>
MSHADOW_XINLINE DType Area(DType l1, DType t1, DType r1, DType b1) {
  DType width = r1 - l1;
  DType height = b1 - t1;
  if (width <= 0 || height <= 0) return DType(0);
  return width * height;
}

template<typename DType>
MSHADOW_XINLINE DType IOU(DType l1, DType t1, DType r1, DType b1,
  DType l2, DType t2, DType r2, DType b2) {
  DType inter_area = Intersect(l1, r1, l2, r2) * Intersect(t1, b1, t2, b2);
  if (inter_area <= 0) return DType(0);
  DType area1 = Area(l1, t1, r1, b1);
  DType area2 = Area(l2, t2, r2, b2);
  return inter_area / (area1 + area2 - inter_area);
}

// compute intersection-over-union overlap between two boxes
struct calc_overlap {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out,
      const DType* L1, const DType* T1, const DType* R1, const DType* B1,
      const DType* L2, const DType* T2, const DType* R2, const DType* B2) {
    out[i] = IOU(L1[i], T1[i], R1[i], B1[i], L2[i], T2[i], R2[i], B2[i]);
  }
};

// create index mask for labels
// struct index_mask {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(int i, DType* out,
//       const DType* x, const DType* y, const index_t width, const index_t height,
//       const int stride, const DType on_value) {
//     if (x[i] < 0 || y[i] < 0) return;
//     int depth = width * height * stride;
//     int offset = i * depth;
//     int start = static_cast<int>(y[i] * width + x[i]) * stride;
//     for (int j = 0; j < stride; ++j) {
//       int pos = start + j;
//       if (pos >= 0 && pos < depth) {
//         out[offset + pos] = on_value;
//       }
//     }
//   }
// };

// find best anchor box per ground-truth, and calculate grad
struct box_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* grad, DType* out_label,
      const DType* label, const DType* anchor, const DType* pred,
      const index_t label_width, const index_t label_offset,
      const index_t pred_width, const index_t pred_offset,
      const index_t grad_width, const index_t grad_offset,
      const index_t num_anchor, const index_t num_batch,
      const index_t width, const index_t height,
      const float box_scale, const float object_scale) {
    // int n = i % num_batch;
    int b = i / num_batch;
    int offset = i * label_width;
    DType class_id = label[offset];
    if (class_id < 0) return;  // padded label
    offset += label_offset;
    // ground-truth
    DType gl = label[offset];
    DType gt = label[offset + 1];
    DType gr = label[offset + 2];
    DType gb = label[offset + 3];
    DType gx = (gl + gr) / 2;
    DType gy = (gt + gb) / 2;
    DType gw = gr - gl;
    DType gh = gb - gt;
    if (gx < 0 || gy < 0 || gx > 1 || gy > 1) return;  // invalid gt
    if (gw <= 0 || gh <= 0 || gw > 1 || gh > 1) return ;  // invalid gt
    // specific block region only where gt center located
    int col = static_cast<int>(gx * width);
    int row = static_cast<int>(gy * height);
    int best_anchor = 0;
    DType best_ovp = 0;
    // find best anchor
    for (int j = 0; j < num_anchor; ++j) {
      DType aw = anchor[j * 2] / width;
      DType ah = anchor[j * 2 + 1] / height;
      if (aw < 0 || ah < 0) continue;  // invalid param
      DType minw = gw < aw ? gw : aw;
      DType minh = gh < ah ? gh : ah;
      DType ovp = minw * minh;
      ovp = ovp / (gw * gh + aw * ah - ovp);
      if (ovp > best_ovp) {
        best_ovp = ovp;
        best_anchor = j;
      }
    }
    // box prediction and box grad
    offset = (b * width * height * num_anchor + row * width * num_anchor +
      col * num_anchor + best_anchor) * pred_width + pred_offset;
    DType pl = pred[offset];
    DType pt = pred[offset + 1];
    DType pr = pred[offset + 2];
    DType pb = pred[offset + 3];
    DType px = (pl + pr) / 2;
    DType py = (pt + pb) / 2;
    DType pw = pr - pl;
    DType ph = pb - pt;
    int out_offset = (b * width * height * num_anchor + row * width * num_anchor +
      col * num_anchor + best_anchor) * grad_width + grad_offset;
    DType aw = anchor[best_anchor * 2];
    DType ah = anchor[best_anchor * 2 + 1];
    DType scale = box_scale * (2 - gw * gh);
    grad[out_offset] = scale * (gx * width - col - px);  // x
    grad[out_offset + 1] = scale * (gy * height - row - py); // y
    grad[out_offset + 2] = scale * (log(gw * width / aw) - pw);  // w
    grad[out_offset + 3] = scale * (log(gh * height / ah) - ph); // y

    // object grad
    DType iou = IOU(pl, pt, pr, pb, gl, gt, gr, gb);
    --out_offset;  // layout : num_class + 1 + 4
    --offset;
    grad[out_offset] = object_scale * (iou - pred[offset]);

    // class target
    offset = b * width * height * num_anchor + row * width * num_anchor +
      col * num_anchor + best_anchor;
    out_label[offset] = class_id;
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

     // convert output from x, y, w, h to xmin, ymin, xmax, ymax format
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
    index_t num_batch = label.shape_[0];
    index_t num_label = label.shape_[1];
    index_t label_width = label.shape_[2];
    index_t pred_width = out.shape_[2];
    index_t num_box = out.shape_[1];
    index_t in_width = grad.shape_[3];
    index_t in_height = grad.shape_[2];
    index_t nc = param_.num_class;
    index_t grad_width = nc + 5;
    index_t num_anchor = param_.num_anchor;
    const DType ignore_label = static_cast<DType>(-1);
    // LOG(INFO) << "Label size: " << num_label;

    // temp space
    Shape<1> label_shape = Shape1(num_batch * num_box);
    Shape<2> softmax_shape = Shape2(num_batch * num_box, nc);
    Shape<4> overlaps_shape = Shape4(9, num_batch, num_box, num_label);
    Shape<3> grad_shape = Shape3(num_batch, num_box, grad_width);
    Shape<1> anchor_shape = Shape1(num_anchor * 2);
    // Shape<4> label_index_shape = Shape4(2, num_batch, num_label, 1);
    // Shape<3> temp_index_mask_shape = Shape3(num_batch, num_label, num_box);
    size_t temp_size_total = label_shape.Size() + 2 * softmax_shape.Size() +
     overlaps_shape.Size() + grad_shape.Size() + anchor_shape.Size();
    // LOG(INFO) << "Total size: " << temp_size_total;
    Tensor<xpu, 1, DType> temp_space = ctx.requested[yoloout_enum::kTempSpace]
     .get_space_typed<xpu, 1, DType>(Shape1(temp_size_total), s);
    // LOG(INFO) << "Total dptr: " << temp_space.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 1, DType> temp_label(temp_space.dptr_, label_shape, s);
    // LOG(INFO) << "Label dptr: " << temp_label.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 2, DType> temp_softmax(temp_label.dptr_ + temp_label.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmax dptr: " << temp_softmax.dptr_ << ", " << softmax_shape.Size();
    Tensor<xpu, 2, DType> temp_softmax_grad(temp_softmax.dptr_ + temp_softmax.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmaxgrad dptr: " << temp_softmax_grad.dptr_ << ", " << softmax_shape.Size();
    // [0]-[7] for x1, y1, w1, h1, x2, y2, w2, h2, [8] for overlap
    Tensor<xpu, 4, DType> buffer(temp_softmax_grad.dptr_ + temp_softmax_grad.MSize(),
     overlaps_shape, s);
    // LOG(INFO) << "overlap dptr: " << buffer.dptr_ << ", " << overlaps_shape.Size();
    Tensor<xpu, 3, DType> temp_grad(buffer.dptr_ + buffer.MSize(),
     grad_shape, s);
    Tensor<xpu, 1, DType> xpu_bias(temp_grad.dptr_ + temp_grad.MSize(),
     anchor_shape, s);

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
    temp_grad = DType(0);
    slice<2>(temp_grad, nc, nc + 1) = ScalarExp<DType>(-param_.background_grad_scale) *
     slice<2>(out, 1, 2);
    Shape<3> sshape = Shape3(num_batch, num_box, 1);
    // mask out when iou > thresh
    slice<2>(temp_grad, nc, nc + 1) *= reshape(F<mshadow_op::lt>(
     reduce_with_axis<red::maximum, false>(buffer[8], 2),
     ScalarExp<DType>(param_.overlap_thresh)), sshape);

    // find best match for each ground-truth, and calculate grad for box pred
    nnvm::Tuple<DType> anchors(param_.anchors.begin(), param_.anchors.end());
    Tensor<cpu, 1, DType> cpu_bias(anchors.begin(), Shape1(anchors.ndim()));
    Copy(xpu_bias, cpu_bias, s);
    temp_label = ignore_label;  // assign default as ignored
    mxnet_op::Kernel<box_grad, xpu>::Launch(s, num_batch * num_label,
     temp_grad.dptr_, temp_label.dptr_, label.dptr_, xpu_bias.dptr_, out.dptr_,
     label_width, 1, pred_width, 2, grad_width, nc + 1,
     num_anchor, num_batch, in_width, in_height, param_.coord_grad_scale,
     param_.object_grad_scale);

    // softmax loss
    temp_softmax = reshape(slice<2>(temp_out, 0, nc), temp_softmax.shape_);
    SoftmaxGrad(temp_softmax_grad, temp_softmax, temp_label, ignore_label);
    slice<2>(temp_grad, 0, nc) = reshape(temp_softmax_grad, Shape3(num_batch,
     num_box, nc));

    // transpose grad to data shape
    grad = transpose(reshape(temp_grad, Shape4(num_batch, in_height,
      in_width, num_anchor * grad_width)), Shape4(0, 3, 1, 2));
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
