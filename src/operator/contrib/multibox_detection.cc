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
 * Copyright (c) 2016 by Contributors
 * \file multibox_detection.cc
 * \brief MultiBoxDetection op
 * \author Joshua Zhang
*/
#include "./multibox_detection-inl.h"
#include <algorithm>

namespace mshadow {
template<typename DType>
struct SortElemDescend {
  DType value;
  int index;

  SortElemDescend(DType v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};

template<typename DType>
inline void TransformLocations(DType *out, const DType *anchors,
                               const DType *loc_pred, const bool clip,
                               const float vx, const float vy,
                               const float vw, const float vh) {
  // transform predictions to detection results
  DType al = anchors[0];
  DType at = anchors[1];
  DType ar = anchors[2];
  DType ab = anchors[3];
  DType aw = ar - al;
  DType ah = ab - at;
  DType ax = (al + ar) / 2.f;
  DType ay = (at + ab) / 2.f;
  DType px = loc_pred[0];
  DType py = loc_pred[1];
  DType pw = loc_pred[2];
  DType ph = loc_pred[3];
  DType ox = px * vx * aw + ax;
  DType oy = py * vy * ah + ay;
  DType ow = std::exp(pw * vw) * aw / 2;
  DType oh = std::exp(ph * vh) * ah / 2;
  out[0] = clip ? std::max(DType(0), std::min(DType(1), ox - ow)) : (ox - ow);
  out[1] = clip ? std::max(DType(0), std::min(DType(1), oy - oh)) : (oy - oh);
  out[2] = clip ? std::max(DType(0), std::min(DType(1), ox + ow)) : (ox + ow);
  out[3] = clip ? std::max(DType(0), std::min(DType(1), oy + oh)) : (oy + oh);
}

template<typename DType>
inline DType CalculateOverlap(const DType *a, const DType *b) {
  DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
  DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
inline void MultiBoxDetectionForward(const Tensor<cpu, 3, DType> &out,
                                     const Tensor<cpu, 3, DType> &cls_prob,
                                     const Tensor<cpu, 2, DType> &loc_pred,
                                     const Tensor<cpu, 2, DType> &anchors,
                                     const Tensor<cpu, 3, DType> &temp_space,
                                     const float threshold,
                                     const bool clip,
                                     const mxnet::Tuple<float> &variances,
                                     const float nms_threshold,
                                     const bool force_suppress,
                                     const int nms_topk) {
  CHECK_EQ(variances.ndim(), 4) << "Variance size must be 4";
  const int num_classes = cls_prob.size(1);
  const int num_anchors = cls_prob.size(2);
  const int num_batches = cls_prob.size(0);
  const DType *p_anchor = anchors.dptr_;

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  std::vector<DType> outputs(num_anchors * 6);
  for (int nbatch = 0; nbatch < num_batches; ++nbatch) {
    const DType *p_cls_prob = cls_prob.dptr_ + nbatch * num_classes * num_anchors;
    const DType *p_loc_pred = loc_pred.dptr_ + nbatch * num_anchors * 4;
    DType *p_out = out.dptr_ + nbatch * num_anchors * 6;

#pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < num_anchors; ++i) {
      // find the predicted class id and probability
      DType score = -1;
      int id = 0;
      for (int j = 1; j < num_classes; ++j) {
        DType temp = p_cls_prob[j * num_anchors + i];
        if (temp > score) {
          score = temp;
          id = j;
        }
      }

      if (id > 0 && score < threshold) {
        id = 0;
      }

      // [id, prob, xmin, ymin, xmax, ymax]
      outputs[i * 6] = id - 1;
      outputs[i * 6 + 1] = score;
      int offset = i * 4;
      TransformLocations(outputs.data() + i * 6 + 2, p_anchor + offset, p_loc_pred + offset, clip,
                         variances[0], variances[1], variances[2], variances[3]);
    }

    int valid_count = 0;
    for (int i = 0; i < num_anchors; ++i) {
      int offset1 = valid_count * 6;
      int offset2 = i * 6;
      if (outputs[offset2] >= 0) {
        p_out[offset1]     = outputs[offset2];
        p_out[offset1 + 1] = outputs[offset2 + 1];
        p_out[offset1 + 2] = outputs[offset2 + 2];
        p_out[offset1 + 3] = outputs[offset2 + 3];
        p_out[offset1 + 4] = outputs[offset2 + 4];
        p_out[offset1 + 5] = outputs[offset2 + 5];
        ++valid_count;
      }
    }

    if (valid_count < 1 || nms_threshold <= 0 || nms_threshold > 1) continue;

    // sort and apply NMS
    Copy(temp_space[nbatch], out[nbatch], out.stream_);
    // sort confidence in descend order
    std::vector<SortElemDescend<DType>> sorter;
    sorter.reserve(valid_count);
    for (int i = 0; i < valid_count; ++i) {
      sorter.push_back(SortElemDescend<DType>(p_out[i * 6 + 1], i));
    }
    std::stable_sort(sorter.begin(), sorter.end());

    // re-order output
    DType *ptemp = temp_space.dptr_ + nbatch * num_anchors * 6;
    int nkeep = static_cast<int>(sorter.size());
    if (nms_topk > 0 && nms_topk < nkeep) {
      // keep topk detections
      nkeep = nms_topk;
      for (int i = nkeep; i < valid_count; ++i) {
        p_out[i * 6] = -1;
      }
    }
    for (int i = 0; i < nkeep; ++i) {
      for (int j = 0; j < 6; ++j) {
        p_out[i * 6 + j] = ptemp[sorter[i].index * 6 + j];
      }
    }

    // apply nms
    for (int i = 0; i < nkeep; ++i) {
      int offset_i = i * 6;
      if (p_out[offset_i] < 0) continue;  // skip eliminated
      for (int j = i + 1; j < nkeep; ++j) {
        int offset_j = j * 6;
        if (p_out[offset_j] < 0) continue;  // skip eliminated
        if (force_suppress || (p_out[offset_i] == p_out[offset_j])) {
          // when foce_suppress == true or class_id equals
          DType iou = CalculateOverlap(p_out + offset_i + 2, p_out + offset_j + 2);
          if (iou >= nms_threshold) {
            p_out[offset_j] = -1;
          }
        }
      }
    }
  }  // end iter batch
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiBoxDetectionParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxDetectionOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiBoxDetectionProp::CreateOperatorEx(Context ctx,
                                                  mxnet::ShapeVector *in_shape,
                                                  std::vector<int> *in_type) const {
  mxnet::ShapeVector out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiBoxDetectionParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_MultiBoxDetection, MultiBoxDetectionProp)
.describe("Convert multibox detection predictions.")
.add_argument("cls_prob", "NDArray-or-Symbol", "Class probabilities.")
.add_argument("loc_pred", "NDArray-or-Symbol", "Location regression predictions.")
.add_argument("anchor", "NDArray-or-Symbol", "Multibox prior anchor boxes")
.add_arguments(MultiBoxDetectionParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_MultiBoxDetection)
.add_alias("_npx_multibox_detection");

}  // namespace op
}  // namespace mxnet
