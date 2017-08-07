/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_target.cc
 * \brief MultiBoxTarget op
 * \author Joshua Zhang
*/
#include <algorithm>
#include "./multibox_target-inl.h"
#include "../mshadow_op.h"

namespace mshadow {
template<typename DType>
inline void AssignLocTargets(const DType *anchor, const DType *l, DType *dst,
                             const float vx, const float vy,
                             const float vw, const float vh) {
  float al = *(anchor);
  float at = *(anchor+1);
  float ar = *(anchor+2);
  float ab = *(anchor+3);
  float aw = ar - al;
  float ah = ab - at;
  float ax = (al + ar) * 0.5;
  float ay = (at + ab) * 0.5;
  float gl = *(l);
  float gt = *(l+1);
  float gr = *(l+2);
  float gb = *(l+3);
  float gw = gr - gl;
  float gh = gb - gt;
  float gx = (gl + gr) * 0.5;
  float gy = (gt + gb) * 0.5;
  *(dst) = DType((gx - ax) / aw / vx);
  *(dst+1) = DType((gy - ay) / ah / vy);
  *(dst+2) = DType(std::log(gw / aw) / vw);
  *(dst+3) = DType(std::log(gh / ah) / vh);
}

struct SortElemDescend {
  float value;
  int index;

  SortElemDescend(float v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};

template<typename DType>
inline void MultiBoxTargetForward(const Tensor<cpu, 2, DType> &loc_target,
                           const Tensor<cpu, 2, DType> &loc_mask,
                           const Tensor<cpu, 2, DType> &cls_target,
                           const Tensor<cpu, 2, DType> &anchors,
                           const Tensor<cpu, 3, DType> &labels,
                           const Tensor<cpu, 3, DType> &cls_preds,
                           const Tensor<cpu, 4, DType> &temp_space,
                           const float overlap_threshold,
                           const float background_label,
                           const float negative_mining_ratio,
                           const float negative_mining_thresh,
                           const int minimum_negative_samples,
                           const nnvm::Tuple<float> &variances) {
  const DType *p_anchor = anchors.dptr_;
  const int num_batches = labels.size(0);
  const int num_labels = labels.size(1);
  const int label_width = labels.size(2);
  const int num_anchors = anchors.size(0);
  CHECK_EQ(variances.ndim(), 4);
  for (int nbatch = 0; nbatch < num_batches; ++nbatch) {
    const DType *p_label = labels.dptr_ + nbatch * num_labels * label_width;
    const DType *p_overlaps = temp_space.dptr_ + nbatch * num_anchors * num_labels;
    int num_valid_gt = 0;
    for (int i = 0; i < num_labels; ++i) {
      if (static_cast<float>(*(p_label + i * label_width)) == -1.0f) {
        CHECK_EQ(static_cast<float>(*(p_label + i * label_width + 1)), -1.0f);
        CHECK_EQ(static_cast<float>(*(p_label + i * label_width + 2)), -1.0f);
        CHECK_EQ(static_cast<float>(*(p_label + i * label_width + 3)), -1.0f);
        CHECK_EQ(static_cast<float>(*(p_label + i * label_width + 4)), -1.0f);
        break;
      }
      ++num_valid_gt;
    }  // end iterate labels

    if (num_valid_gt > 0) {
      std::vector<bool> gt_flags(num_valid_gt, false);
      std::vector<std::pair<float, int>> max_matches(num_anchors,
        std::pair<float, int>(-1.0f, -1));
      std::vector<char> anchor_flags(num_anchors, -1);  // -1 means don't care
      int num_positive = 0;
      while (std::find(gt_flags.begin(), gt_flags.end(), false) != gt_flags.end()) {
        // ground-truths not fully matched
        int best_anchor = -1;
        int best_gt = -1;
        float max_overlap = 1e-6;  // start with a very small positive overlap
        for (int j = 0; j < num_anchors; ++j) {
          if (anchor_flags[j] == 1) {
            continue;  // already matched this anchor
          }
          const DType *pp_overlaps = p_overlaps + j * num_labels;
          for (int k = 0; k < num_valid_gt; ++k) {
            if (gt_flags[k]) {
              continue;  // already matched this gt
            }
            float iou = static_cast<float>(*(pp_overlaps + k));
            if (iou > max_overlap) {
              best_anchor = j;
              best_gt = k;
              max_overlap = iou;
            }
          }
        }

        if (best_anchor == -1) {
          CHECK_EQ(best_gt, -1);
          break;  // no more good match
        } else {
          CHECK_EQ(max_matches[best_anchor].first, -1.0f);
          CHECK_EQ(max_matches[best_anchor].second, -1);
          max_matches[best_anchor].first = max_overlap;
          max_matches[best_anchor].second = best_gt;
          num_positive += 1;
          // mark as visited
          gt_flags[best_gt] = true;
          anchor_flags[best_anchor] = 1;
        }
      }  // end while

      if (overlap_threshold > 0) {
        // find positive matches based on overlaps
        for (int j = 0; j < num_anchors; ++j) {
          if (anchor_flags[j] == 1) {
            continue;  // already matched this anchor
          }
          const DType *pp_overlaps = p_overlaps + j * num_labels;
          int best_gt = -1;
          float max_iou = -1.0f;
          for (int k = 0; k < num_valid_gt; ++k) {
            float iou = static_cast<float>(*(pp_overlaps + k));
            if (iou > max_iou) {
              best_gt = k;
              max_iou = iou;
            }
          }
          if (best_gt != -1) {
            CHECK_EQ(max_matches[j].first, -1.0f);
            CHECK_EQ(max_matches[j].second, -1);
            max_matches[j].first = max_iou;
            max_matches[j].second = best_gt;
            if (max_iou > overlap_threshold) {
              num_positive += 1;
              // mark as visited
              gt_flags[best_gt] = true;
              anchor_flags[j] = 1;
            }
          }
        }  // end iterate anchors
      }

      if (negative_mining_ratio > 0) {
        const int num_classes = cls_preds.size(1);
        DType *p_cls_preds = cls_preds.dptr_ + nbatch * num_classes * num_anchors;
        CHECK_GT(negative_mining_thresh, 0);
        int num_negative = num_positive * negative_mining_ratio;
        if (num_negative > (num_anchors - num_positive)) {
          num_negative = num_anchors - num_positive;
        }
        if (num_negative > 0) {
          // use negative mining, pick up "best" negative samples
          std::vector<SortElemDescend> temp;
          temp.reserve(num_anchors - num_positive);
          for (int j = 0; j < num_anchors; ++j) {
            if (anchor_flags[j] == 1) {
              continue;  // already matched this anchor
            }
            if (max_matches[j].first < 0) {
              // not yet calculated
              const DType *pp_overlaps = p_overlaps + j * num_labels;
              int best_gt = -1;
              float max_iou = -1.0f;
              for (int k = 0; k < num_valid_gt; ++k) {
                float iou = static_cast<float>(*(pp_overlaps + k));
                if (iou > max_iou) {
                  best_gt = k;
                  max_iou = iou;
                }
              }
              if (best_gt != -1) {
                CHECK_EQ(max_matches[j].first, -1.0f);
                CHECK_EQ(max_matches[j].second, -1);
                max_matches[j].first = max_iou;
                max_matches[j].second = best_gt;
              }
            }
            if (max_matches[j].first < negative_mining_thresh &&
                anchor_flags[j] == -1) {
                // calcuate class predictions
              DType max_val = p_cls_preds[j];
              for (int k = 1; k < num_classes; ++k) {
                DType tmp = p_cls_preds[j + num_anchors * k];
                if (tmp > max_val) max_val = tmp;
              }
              DType sum = 0.f;
              for (int k = 0; k < num_classes; ++k) {
                DType tmp = p_cls_preds[j + num_anchors * k];
                sum += std::exp(tmp - max_val);
              }
              DType prob = std::exp(p_cls_preds[j] - max_val) / sum;
              // loss should be -log(x), but value does not matter, skip log
              temp.push_back(SortElemDescend(-prob, j));
            }
          }  // end iterate anchors

          CHECK_GE(temp.size(), num_negative);
          std::stable_sort(temp.begin(), temp.end());
          for (int i = 0; i < num_negative; ++i) {
            anchor_flags[temp[i].index] = 0;  // mark as negative sample
          }
        }
      } else {
        // use all negative samples
        for (int i = 0; i < num_anchors; ++i) {
          if (anchor_flags[i] != 1) {
            anchor_flags[i] = 0;
          }
        }
      }

      // assign training targets
      DType *p_loc_target = loc_target.dptr_ + nbatch * num_anchors * 4;
      DType *p_loc_mask = loc_mask.dptr_ + nbatch * num_anchors * 4;
      DType *p_cls_target = cls_target.dptr_ + nbatch * num_anchors;
      for (int i = 0; i < num_anchors; ++i) {
        if (anchor_flags[i] == 1) {
          // positive sample
          CHECK_GE(max_matches[i].second, 0);
          // 0 reserved for background
          *(p_cls_target + i) = *(p_label + label_width * max_matches[i].second) + 1;
          int offset = i * 4;
          *(p_loc_mask + offset) = 1;
          *(p_loc_mask + offset + 1) = 1;
          *(p_loc_mask + offset + 2) = 1;
          *(p_loc_mask + offset + 3) = 1;
          AssignLocTargets(p_anchor + i * 4,
            p_label + label_width * max_matches[i].second + 1, p_loc_target + offset,
            variances[0], variances[1], variances[2], variances[3]);
        } else if (anchor_flags[i] == 0) {
          // negative sample
          *(p_cls_target + i) = 0;
          int offset = i * 4;
          *(p_loc_mask + offset) = 0;
          *(p_loc_mask + offset + 1) = 0;
          *(p_loc_mask + offset + 2) = 0;
          *(p_loc_mask + offset + 3) = 0;
        }
      }  // end iterate anchors
    }
  }  // end iterate batches
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiBoxTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxTargetOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiBoxTargetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiBoxTargetParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_MultiBoxTarget, MultiBoxTargetProp)
.describe("Compute Multibox training targets")
.add_argument("anchor", "NDArray-or-Symbol", "Generated anchor boxes.")
.add_argument("label", "NDArray-or-Symbol", "Object detection labels.")
.add_argument("cls_pred", "NDArray-or-Symbol", "Class predictions.")
.add_arguments(MultiBoxTargetParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
