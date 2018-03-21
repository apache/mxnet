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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal.cc
 * \brief
 * \author Xizhou Zhu, Kan Wu
*/

#include "./multi_proposal-inl.h"

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

// bbox prediction and clip to the image borders
inline void BBoxTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                             const mshadow::Tensor<cpu, 3>& deltas,
                             const float im_height,
                             const float im_width,
                             const int real_height,
                             const int real_width,
                             mshadow::Tensor<cpu, 2> *out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  int anchors = deltas.size(0) / 4;
  int heights = deltas.size(1);
  int widths = deltas.size(2);

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int index = 0; index < anchors * heights * widths; ++index) {
    // index_t index = h * (widths * anchors) + w * (anchors) + a;
    int a = index % anchors;
    int w = (index / anchors) % widths;
    int h = index / (widths * anchors);

    float width = boxes[index][2] - boxes[index][0] + 1.0;
    float height = boxes[index][3] - boxes[index][1] + 1.0;
    float ctr_x = boxes[index][0] + 0.5 * (width - 1.0);
    float ctr_y = boxes[index][1] + 0.5 * (height - 1.0);

    float dx = deltas[a*4 + 0][h][w];
    float dy = deltas[a*4 + 1][h][w];
    float dw = deltas[a*4 + 2][h][w];
    float dh = deltas[a*4 + 3][h][w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

    (*out_pred_boxes)[index][0] = pred_x1;
    (*out_pred_boxes)[index][1] = pred_y1;
    (*out_pred_boxes)[index][2] = pred_x2;
    (*out_pred_boxes)[index][3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      (*out_pred_boxes)[index][4] = -1.0;
    }
  }
}

// iou prediction and clip to the image border
inline void IoUTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                            const mshadow::Tensor<cpu, 3>& deltas,
                            const float im_height,
                            const float im_width,
                            const int real_height,
                            const int real_width,
                            mshadow::Tensor<cpu, 2> *out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  int anchors = deltas.size(0) / 4;
  int heights = deltas.size(1);
  int widths = deltas.size(2);

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int index = 0; index < anchors * heights * widths; ++index) {
    // index_t index = h * (widths * anchors) + w * (anchors) + a;
    int a = index % anchors;
    int w = (index / anchors) % widths;
    int h = index / (widths * anchors);

    float x1 = boxes[index][0];
    float y1 = boxes[index][1];
    float x2 = boxes[index][2];
    float y2 = boxes[index][3];

    float dx1 = deltas[a * 4 + 0][h][w];
    float dy1 = deltas[a * 4 + 1][h][w];
    float dx2 = deltas[a * 4 + 2][h][w];
    float dy2 = deltas[a * 4 + 3][h][w];

    float pred_x1 = x1 + dx1;
    float pred_y1 = y1 + dy1;
    float pred_x2 = x2 + dx2;
    float pred_y2 = y2 + dy2;

    pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

    (*out_pred_boxes)[index][0] = pred_x1;
    (*out_pred_boxes)[index][1] = pred_y1;
    (*out_pred_boxes)[index][2] = pred_x2;
    (*out_pred_boxes)[index][3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      (*out_pred_boxes)[index][4] = -1.0f;
    }
  }
}

// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(mshadow::Tensor<cpu, 2> *dets,
                      const float min_size) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int i = 0; i < static_cast<int>(dets->size(0)); ++i) {
    float iw = (*dets)[i][2] - (*dets)[i][0] + 1.0f;
    float ih = (*dets)[i][3] - (*dets)[i][1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      (*dets)[i][0] -= min_size / 2;
      (*dets)[i][1] -= min_size / 2;
      (*dets)[i][2] += min_size / 2;
      (*dets)[i][3] += min_size / 2;
      (*dets)[i][4] = -1.0f;
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace utils {

struct ReverseArgsortCompl {
  const float *val_;
  explicit ReverseArgsortCompl(float *val)
    : val_(val) {}
  bool operator() (float i, float j) {
    return (val_[static_cast<index_t>(i)] >
            val_[static_cast<index_t>(j)]);
  }
};

// copy score and init order
inline void CopyScore(const mshadow::Tensor<cpu, 2>& dets,
                      mshadow::Tensor<cpu, 1> *score,
                      mshadow::Tensor<cpu, 1> *order) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int i = 0; i < static_cast<int>(dets.size(0)); ++i) {
    (*score)[i] = dets[i][4];
    (*order)[i] = i;
  }
}

// sort order array according to score
inline void ReverseArgsort(const mshadow::Tensor<cpu, 1>& score,
                           mshadow::Tensor<cpu, 1> *order) {
  ReverseArgsortCompl cmpl(score.dptr_);
  std::sort(order->dptr_, order->dptr_ + score.size(0), cmpl);
}

// reorder proposals according to order and keep the pre_nms_top_n proposals
// dets.size(0) == pre_nms_top_n
inline void ReorderProposals(const mshadow::Tensor<cpu, 2>& prev_dets,
                             const mshadow::Tensor<cpu, 1>& order,
                             const index_t pre_nms_top_n,
                             mshadow::Tensor<cpu, 2> *dets) {
  CHECK_EQ(dets->size(0), pre_nms_top_n);
  const int dets_size0 = static_cast<int>(dets->size(0));
  const int dets_size1 = static_cast<int>(dets->size(1));
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int k = 0; k < dets_size0 * dets_size1; ++k) {
    int i = k / dets_size1;
    int j = k % dets_size1;
    const index_t index = order[i];
    (*dets)[i][j] = prev_dets[index][j];
  }
}

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2>& dets,
                                  const float thresh,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 1> *area,
                                  mshadow::Tensor<cpu, 1> *suppressed,
                                  mshadow::Tensor<cpu, 1> *keep,
                                  int *out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(area->CheckContiguous(), true);
  CHECK_EQ(suppressed->CheckContiguous(), true);
  CHECK_EQ(keep->CheckContiguous(), true);
  // calculate area
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int i = 0; i < static_cast<int>(dets.size(0)); ++i) {
    (*area)[i] = (dets[i][2] - dets[i][0] + 1) *
                 (dets[i][3] - dets[i][1] + 1);
  }

  // calculate nms
  *out_size = 0;
  for (index_t i = 0; i < dets.size(0) && (*out_size) < static_cast<int>(post_nms_top_n); ++i) {
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = (*area)[i];

    if ((*suppressed)[i] > 0.0f) {
      continue;
    }

    (*keep)[(*out_size)++] = i;
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int j = i + 1; j < static_cast<int>(dets.size(0)); ++j) {
      if ((*suppressed)[j] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[j][0]);
      float yy1 = std::max(iy1, dets[j][1]);
      float xx2 = std::min(ix2, dets[j][2]);
      float yy2 = std::min(iy2, dets[j][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1.0f);
      float h = std::max(0.0f, yy2 - yy1 + 1.0f);
      float inter = w * h;
      float ovr = inter / (iarea + (*area)[j] - inter);
      if (ovr > thresh) {
        (*suppressed)[j] = 1.0f;
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet



namespace mxnet {
namespace op {

template<typename xpu>
class MultiProposalOp : public Operator{
 public:
  explicit MultiProposalOp(MultiProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<cpu, 4> scores = in_data[proposal::kClsProb].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> bbox_deltas = in_data[proposal::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> im_info = in_data[proposal::kImInfo].get<cpu, 2, real_t>(s);

    Tensor<cpu, 2> out = out_data[proposal::kOut].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> out_score = out_data[proposal::kScore].get<cpu, 2, real_t>(s);

    int num_images = scores.size(0);
    int num_anchors = scores.size(1) / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count_anchors = num_anchors * height * width;
    int rpn_pre_nms_top_n =
        (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count_anchors;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count_anchors);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    int workspace_size =
        num_images * (count_anchors * 5 + 2 * count_anchors +
        rpn_pre_nms_top_n * 5 + 3 * rpn_pre_nms_top_n);

    Tensor<cpu, 1> workspace = ctx.requested[proposal::kTempResource].get_space<cpu>(
      Shape1(workspace_size), s);
    int start = 0;
    Tensor<cpu, 3> workspace_proposals(workspace.dptr_ +
            start, Shape3(num_images, count_anchors, 5));
    start += num_images * count_anchors * 5;
    Tensor<cpu, 3> workspace_pre_nms(workspace.dptr_ + start, Shape3(num_images, 2, count_anchors));
    start += num_images * 2 * count_anchors;
    Tensor<cpu, 3> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape3(num_images, rpn_pre_nms_top_n, 5));
    start += num_images * rpn_pre_nms_top_n * 5;
    Tensor<cpu, 3> workspace_nms(workspace.dptr_ + start, Shape3(num_images, 3, rpn_pre_nms_top_n));
    start += num_images * 3 * rpn_pre_nms_top_n;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;

    // Generate anchors
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.ndim() * param_.scales.ndim());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios,
                           param_.scales,
                           &anchors);
    std::memcpy(workspace_proposals.dptr_, &anchors[0], sizeof(float) * anchors.size());

    Tensor<cpu, 2> workspace_proposals0 = workspace_proposals[0];
    // Enumerate all shifted anchors
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int index = 0; index < num_anchors * height * width; ++index) {
      // index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
      int i = index % num_anchors;
      int k = (index / num_anchors) % width;
      int j = index / (width * num_anchors);
      workspace_proposals0[index][0] =
          workspace_proposals0[i][0] + k * param_.feature_stride;
      workspace_proposals0[index][1] =
          workspace_proposals0[i][1] + j * param_.feature_stride;
      workspace_proposals0[index][2] =
          workspace_proposals0[i][2] + k * param_.feature_stride;
      workspace_proposals0[index][3] =
          workspace_proposals0[i][3] + j * param_.feature_stride;
      workspace_proposals0[index][4] = scores[0][i + num_anchors][j][k];
    }

    // Copy shifted anchors to other images
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int t = count_anchors; t < num_images * count_anchors; ++t) {
        int b = t / count_anchors;
        int index = t % count_anchors;
        int i = index % num_anchors;
        int k = (index / num_anchors) % width;
        int j = index / (width * num_anchors);
        for (int w = 0; w < 4; ++w) {
            workspace_proposals[b][index][w] = workspace_proposals[0][index][w];
        }
        workspace_proposals[b][index][4] = scores[b][i + num_anchors][j][k];
    }

    // Assign Foreground Scores for each anchor
    #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int b = 0; b < num_images; ++b) {
      // prevent padded predictions
      int real_height = static_cast<int>(im_info[b][0] / param_.feature_stride);
      int real_width = static_cast<int>(im_info[b][1] / param_.feature_stride);
      CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
      CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

      Tensor<cpu, 2> workspace_proposals_i = workspace_proposals[b];
      Tensor<cpu, 2> workspace_pre_nms_i = workspace_pre_nms[b];
      Tensor<cpu, 2> workspace_ordered_proposals_i =
                       workspace_ordered_proposals[b];
      Tensor<cpu, 2> workspace_nms_i = workspace_nms[b];

      if (param_.iou_loss) {
        utils::IoUTransformInv(workspace_proposals_i, bbox_deltas[b], im_info[b][0], im_info[b][1],
                               real_height, real_width, &(workspace_proposals_i));
      } else {
        utils::BBoxTransformInv(workspace_proposals_i, bbox_deltas[b], im_info[b][0], im_info[b][1],
                                real_height, real_width, &(workspace_proposals_i));
      }
      utils::FilterBox(&workspace_proposals_i, param_.rpn_min_size * im_info[b][2]);

      Tensor<cpu, 1> score = workspace_pre_nms_i[0];
      Tensor<cpu, 1> order = workspace_pre_nms_i[1];

      utils::CopyScore(workspace_proposals_i,
                       &score,
                       &order);
      utils::ReverseArgsort(score,
                            &order);
      utils::ReorderProposals(workspace_proposals_i,
                              order,
                              rpn_pre_nms_top_n,
                              &workspace_ordered_proposals_i);
      int out_size = 0;
      Tensor<cpu, 1> area = workspace_nms_i[0];
      Tensor<cpu, 1> suppressed = workspace_nms_i[1];
      Tensor<cpu, 1> keep = workspace_nms_i[2];
      suppressed = 0;  // surprised!

      utils::NonMaximumSuppression(workspace_ordered_proposals_i,
                                   param_.threshold,
                                   rpn_post_nms_top_n,
                                   &area,
                                   &suppressed,
                                   &keep,
                                   &out_size);

      // fill in output rois and output scores
      #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
      for (int i = 0; i < param_.rpn_post_nms_top_n; ++i) {
        int out_index = b * param_.rpn_post_nms_top_n + i;
        out[out_index][0] = b;
        if (i < out_size) {
          index_t index = keep[i];
          for (index_t j = 0; j < 4; ++j) {
            out[out_index][j + 1] =  workspace_ordered_proposals_i[index][j];
          }
          out_score[out_index][0] = workspace_ordered_proposals_i[index][4];
        } else {
          index_t index = keep[i % out_size];
          for (index_t j = 0; j < 4; ++j) {
            out[out_index][j + 1] = workspace_ordered_proposals_i[index][j];
          }
          out_score[out_index][0] = workspace_ordered_proposals_i[index][4];
        }
      }
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
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  MultiProposalParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<cpu>(MultiProposalParam param) {
  return new MultiProposalOp<cpu>(param);
}

Operator* MultiProposalProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiProposalParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiProposal, MultiProposalProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_score", "NDArray-or-Symbol", "Score of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(MultiProposalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
