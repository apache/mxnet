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
 * \file bounding_box-inl.h
 * \brief bounding box util functions and operators
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_H_
#define MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_H_
#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/sort_op.h"
#include "./bounding_box-common.h"

namespace mxnet {
namespace op {
namespace box_nms_enum {
enum BoxNMSOpInputs {kData};
enum BoxNMSOpOutputs {kOut, kTemp};
enum BoxNMSOpResource {kTempSpace};
}  // box_nms_enum

struct BoxNMSParam : public dmlc::Parameter<BoxNMSParam> {
  float overlap_thresh;
  float valid_thresh;
  int topk;
  int coord_start;
  int score_index;
  int id_index;
  int background_id;
  bool force_suppress;
  int in_format;
  int out_format;
  DMLC_DECLARE_PARAMETER(BoxNMSParam) {
    DMLC_DECLARE_FIELD(overlap_thresh).set_default(0.5)
    .describe("Overlapping(IoU) threshold to suppress object with smaller score.");
    DMLC_DECLARE_FIELD(valid_thresh).set_default(0)
    .describe("Filter input boxes to those whose scores greater than valid_thresh.");
    DMLC_DECLARE_FIELD(topk).set_default(-1)
    .describe("Apply nms to topk boxes with descending scores, -1 to no restriction.");
    DMLC_DECLARE_FIELD(coord_start).set_default(2)
    .describe("Start index of the consecutive 4 coordinates.");
    DMLC_DECLARE_FIELD(score_index).set_default(1)
    .describe("Index of the scores/confidence of boxes.");
    DMLC_DECLARE_FIELD(id_index).set_default(-1)
    .describe("Optional, index of the class categories, -1 to disable.");
    DMLC_DECLARE_FIELD(background_id).set_default(-1)
    .describe("Optional, id of the background class which will be ignored in nms.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
    .describe("Optional, if set false and id_index is provided, nms will only apply"
    " to boxes belongs to the same category");
    DMLC_DECLARE_FIELD(in_format).set_default(box_common_enum::kCorner)
    .add_enum("corner", box_common_enum::kCorner)
    .add_enum("center", box_common_enum::kCenter)
    .describe("The input box encoding type. \n"
        " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
        " \"center\" means boxes are encodes as [x, y, width, height].");
    DMLC_DECLARE_FIELD(out_format).set_default(box_common_enum::kCorner)
    .add_enum("corner", box_common_enum::kCorner)
    .add_enum("center", box_common_enum::kCenter)
    .describe("The output box encoding type. \n"
        " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
        " \"center\" means boxes are encodes as [x, y, width, height].");
  }
};  // BoxNMSParam

inline bool BoxNMSShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  if (mxnet::op::shape_is_none(in_attrs->at(0))
    && mxnet::op::shape_is_none(out_attrs->at(0))) {
    return false;
  }

  mxnet::TShape& ishape = (*in_attrs)[0];
  int indim = ishape.ndim();
  CHECK(indim >= 2)
    << "input must have dim >= 2"
    << " the last two dimensions are num_box and box_width "
    << ishape << " provided";
  int width_elem = ishape[indim - 1];
  int expected = 5;
  if (param.id_index >= 0) {
    expected += 1;
  }
  CHECK_GE(width_elem, expected)
    << "the last dimension must have at least 5 elements"
    << " namely (score, coordinates x 4) "
    << width_elem << " provided, " << expected << " expected.";
  // check indices
  int coord_start = param.coord_start;
  int coord_end = param.coord_start + 3;
  int score_index = param.score_index;
  CHECK(score_index >= 0 && score_index < width_elem)
    << "score_index: " << score_index << " out of range: (0, "
    << width_elem << ")";
  CHECK(score_index < coord_start || score_index > coord_end)
    << "score_index: " << score_index << " conflict with coordinates: ("
    << coord_start << ", " << coord_end << ").";
  CHECK(coord_start >= 0 && coord_end < width_elem)
    << "coordinates: (" << coord_start << ", " << coord_end
    << ") out of range:: (0, " << width_elem << ")";
  if (param.id_index >= 0) {
    int id_index = param.id_index;
    CHECK(id_index >= 0 && id_index < width_elem)
      << "id_index: " << id_index << " out of range: (0, "
      << width_elem << ")";
    CHECK(id_index < coord_start || id_index > coord_end)
      << "id_index: " << id_index << " conflict with coordinates: ("
      << coord_start << ", " << coord_end << ").";
    CHECK_NE(id_index, score_index)
      << "id_index: " << id_index << " conflict with score_index: " << score_index;
  }
  mxnet::TShape oshape = ishape;
  oshape[indim - 1] = 1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ishape);  // out_shape[0] == in_shape
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);  // out_shape[1]
  return true;
}

inline uint32_t BoxNMSNumVisibleOutputs(const NodeAttrs& attrs) {
  return static_cast<uint32_t>(1);
}

template<typename DType, typename FType>
int CopyIf(mshadow::Tensor<cpu, 1, DType> out,
           mshadow::Tensor<cpu, 1, DType> value,
           mshadow::Tensor<cpu, 1, FType> flag) {
  index_t j = 0;
  for (index_t i = 0; i < flag.size(0); i++) {
    if (static_cast<bool>(flag[i])) {
      out[j] = value[i];
      j++;
    }
  }
  return j;
}

struct corner_to_center {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *data, int stride) {
    int index = i * stride;
    DType left = data[index];
    if (left < 0) return;
    DType top = data[index+1];
    DType right = data[index+2];
    DType bot = data[index+3];
    data[index] = (left + right) / 2;
    data[index+1] = (top + bot) / 2;
    data[index+2] = right - left;
    data[index+3] = bot - top;
  }
};

struct center_to_corner {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *data, int stride) {
    int index = i * stride;
    DType x = data[index];
    if (x < 0) return;
    DType y = data[index+1];
    DType width = data[index+2] / 2;
    DType height = data[index+3] / 2;
    data[index] = x - width;
    data[index+1] = y - height;
    data[index+2] = x + width;
    data[index+3] = y + height;
  }
};

template<typename DType>
MSHADOW_XINLINE DType BoxArea(const DType *box, int encode) {
  DType a1 = box[0];
  DType a2 = box[1];
  DType a3 = box[2];
  DType a4 = box[3];
  DType width, height;
  if (box_common_enum::kCorner == encode) {
    width = a3 - a1;
    height = a4 - a2;
  } else {
    width = a3;
    height = a4;
  }
  if (width < 0 || height < 0) {
    return DType(0);
  } else {
    return width * height;
  }
}

/*!
 * \brief compute areas specialized for nms to reduce computation
 *
 * \param i the launched thread index (total thread num_batch * topk)
 * \param out 1d array for areas (size num_batch * num_elem)
 * \param in 1st coordinate of 1st box (buffer + coord_start)
 * \param indices index to areas and in buffer (sorted_index)
 * \param batch_start map (b, k) to compact index by indices[batch_start[b] + k]
 * \param topk effective batch size of boxes, to be mapped to real index
 * \param stride should be width_elem (e.g. 6 including cls and scores)
 * \param encode passed to BoxArea to compute area
 */
struct compute_area {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const int32_t *indices, const int32_t *batch_start,
                                  int topk, int num_elem, int stride, int encode) {
    int b = i / topk;
    int k = i % topk;
    int pos = static_cast<int>(batch_start[b]) + k;
    if (pos >= static_cast<int>(batch_start[b + 1])) return;
    int index = static_cast<int>(indices[pos]);
    int in_index = index * stride;
    out[index] = BoxArea(in + in_index, encode);
  }
};

template<typename DType>
void NMSApply(mshadow::Stream<cpu> *s,
              int num_batch, int topk,
              mshadow::Tensor<cpu, 1, int32_t>* sorted_index,
              mshadow::Tensor<cpu, 1, int32_t>* batch_start,
              mshadow::Tensor<cpu, 3, DType>* buffer,
              mshadow::Tensor<cpu, 1, DType>* areas,
              int num_elem, int width_elem,
              int coord_start, int id_index,
              float threshold, bool force_suppress,
              int in_format) {
  using namespace mxnet_op;
  // go through each box as reference, suppress if overlap > threshold
  // sorted_index with -1 is marked as suppressed
  for (int ref = 0; ref < topk; ++ref) {
    int num_worker = topk - ref - 1;
    if (num_worker < 1) continue;
    Kernel<nms_impl, cpu>::Launch(s, num_batch * num_worker,
      sorted_index->dptr_, batch_start->dptr_, buffer->dptr_, areas->dptr_,
      num_worker, ref, num_elem,
      width_elem, coord_start, id_index,
      threshold, force_suppress, in_format);
  }
}

inline void NMSCalculateBatchStart(mshadow::Stream<cpu> *s,
                                   mshadow::Tensor<cpu, 1, int32_t>* batch_start,
                                   mshadow::Tensor<cpu, 1, int32_t>* valid_batch_id,
                                   int num_batch) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  for (int b = 0; b < num_batch + 1; b++) {
    slice<0>(*batch_start, b, b + 1) = reduce_keepdim<red::sum, false>(
        F<mshadow_op::less_than>(*valid_batch_id, ScalarExp<int32_t>(b)), 0);
  }
}

/*!
   * \brief Assign output of nms by indexing input
   *
   * \param i the launched thread index (total num_batch)
   * \param out output array [cls, conf, b0, b1, b2, b3]
   * \param record book keeping the selected index for backward
   * \param index compact sorted_index, use batch_start to access
   * \param batch_start map(b, k) to compact index by index[batch_start[b] + k]
   * \param k nms topk number
   * \param num number of input boxes in each batch
   * \param stride input stride, usually 6 (id-score-x1-y2-x2-y2)
   */
struct nms_assign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, DType *record, const DType *input,
                                  const int32_t *index, const int32_t *batch_start,
                                  int k, int num, int stride) {
    int count = 0;
    for (int j = 0; j < k; ++j) {
      int pos = static_cast<int>(batch_start[i]) + j;
      if (pos >= static_cast<int>(batch_start[i + 1])) return;
      int location = static_cast<int>(index[pos]);
      if (location >= 0) {
        // copy to output
        int out_location = (i * num + count) * stride;
        int in_location = location * stride;
        for (int s = 0; s < stride; ++s) {
          out[out_location + s] = input[in_location + s];
        }
        // keep the index in the record for backward
        record[i * num + count] = location;
        ++count;
      }
    }
  }
};


struct nms_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *in_grad, const DType *out_grad,
                                  const DType *record, int num, int stride) {
    int index = static_cast<int>(record[i]);
    if (index < 0) return;
    int loc = index * stride;
    int from_loc = i * stride;
    for (int j = 0; j < stride; ++j) {
      in_grad[loc + j] = out_grad[from_loc + j];
    }
  }
};

template<typename xpu>
void BoxNMSForward(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U) << "BoxNMS output: [output, temp]";
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape in_shape = inputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];
  bool class_exist = param.id_index >= 0;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 3, DType> data = inputs[box_nms_enum::kData]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> out = outputs[box_nms_enum::kOut]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> record = outputs[box_nms_enum::kTemp]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, 1), s);

    // prepare workspace
    Shape<1> sort_index_shape = Shape1(num_batch * num_elem);
    Shape<3> buffer_shape = Shape3(num_batch, num_elem, width_elem);
    Shape<1> batch_start_shape = Shape1(num_batch + 1);

    // index
    index_t int32_size = sort_index_shape.Size() * 3 + batch_start_shape.Size();
    index_t dtype_size = sort_index_shape.Size() * 3;
    if (req[0] == kWriteInplace) {
      dtype_size += buffer_shape.Size();
    }
    // ceil up when sizeof(DType) is larger than sizeof(DType)
    index_t int32_offset = (int32_size * sizeof(int32_t) - 1) / sizeof(DType) + 1;
    index_t workspace_size = int32_offset + dtype_size;
    Tensor<xpu, 1, DType> workspace = ctx.requested[box_nms_enum::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(workspace_size), s);
    Tensor<xpu, 1, int32_t> sorted_index(
      reinterpret_cast<int32_t*>(workspace.dptr_), sort_index_shape, s);
    Tensor<xpu, 1, int32_t> all_sorted_index(sorted_index.dptr_ + sorted_index.MSize(),
      sort_index_shape, s);
    Tensor<xpu, 1, int32_t> batch_id(
      all_sorted_index.dptr_ + all_sorted_index.MSize(), sort_index_shape, s);
    Tensor<xpu, 1, int32_t> batch_start(batch_id.dptr_ + batch_id.MSize(), batch_start_shape, s);
    Tensor<xpu, 1, DType> scores(workspace.dptr_ + int32_offset,
      sort_index_shape, s);
    Tensor<xpu, 1, DType> areas(scores.dptr_ + scores.MSize(), sort_index_shape, s);
    Tensor<xpu, 1, DType> classes(areas.dptr_ + areas.MSize(), sort_index_shape, s);
    Tensor<xpu, 3, DType> buffer = data;
    if (req[0] == kWriteInplace) {
      // make copy
      buffer = Tensor<xpu, 3, DType>(areas.dptr_ + areas.MSize(), buffer_shape, s);
      buffer = F<mshadow_op::identity>(data);
    }

    // indecies
    int score_index = param.score_index;
    int coord_start = param.coord_start;
    int id_index = param.id_index;

    // sort topk
    int topk = param.topk < 0? num_elem : std::min(num_elem, param.topk);
    if (topk < 1) {
      out = F<mshadow_op::identity>(buffer);
      record = reshape(range<DType>(0, num_batch * num_elem), record.shape_);
      return;
    }

    // use classes, areas and scores as temporary storage
    Tensor<xpu, 1, DType> all_scores = areas;
    all_scores = reshape(slice<2>(buffer, score_index, score_index + 1), all_scores.shape_);
    all_sorted_index = range<int32_t>(0, num_batch * num_elem);
    Tensor<xpu, 1, DType> all_classes = classes;
    if (class_exist) {
      all_classes = reshape(slice<2>(buffer, id_index, id_index + 1), classes.shape_);
    }

    // filter scores but keep original sorted_index value
    // move valid score and index to the front, return valid size
    Tensor<xpu, 1, DType> valid_box = scores;
    if (class_exist) {
      valid_box = F<mshadow_op::bool_and>(
        F<mshadow_op::greater_than>(all_scores, ScalarExp<DType>(param.valid_thresh)),
        F<mshadow_op::not_equal>(all_classes, ScalarExp<DType>(param.background_id)));
    } else {
      valid_box = F<mshadow_op::greater_than>(all_scores, ScalarExp<DType>(param.valid_thresh));
    }
    classes = F<mshadow_op::identity>(valid_box);
    valid_box = classes;
    int num_valid = mxnet::op::CopyIf(scores, all_scores, valid_box);
    mxnet::op::CopyIf(sorted_index, all_sorted_index, valid_box);

    // if everything is filtered, output -1
    if (num_valid == 0) {
      record = -1;
      out = -1;
      return;
    }
    // mark the invalid boxes before nms
    if (num_valid < num_batch * num_elem) {
      slice<0>(sorted_index, num_valid, num_batch * num_elem) = -1;
    }

    // only sort the valid scores and batch_id
    Shape<1> valid_score_shape = Shape1(num_valid);
    Tensor<xpu, 1, DType> valid_scores(scores.dptr_, valid_score_shape, s);
    Tensor<xpu, 1, int32_t> valid_sorted_index(sorted_index.dptr_, valid_score_shape, s);
    Tensor<xpu, 1, int32_t> valid_batch_id(batch_id.dptr_, valid_score_shape, s);

    // sort index by batch_id then score (stable sort)
    mxnet::op::SortByKey(valid_scores, valid_sorted_index, false);
    valid_batch_id = (valid_sorted_index / ScalarExp<int32_t>(num_elem));
    mxnet::op::SortByKey(valid_batch_id, valid_sorted_index, true);

    // calculate batch_start: accumulated sum to denote 1st sorted_index for a given batch_index
    valid_batch_id = (valid_sorted_index / ScalarExp<int32_t>(num_elem));
    mxnet::op::NMSCalculateBatchStart(s, &batch_start, &valid_batch_id, num_batch);

    // pre-compute areas of candidates
    areas = 0;
    Kernel<compute_area, xpu>::Launch(s, num_batch * topk,
     areas.dptr_, buffer.dptr_ + coord_start, sorted_index.dptr_, batch_start.dptr_,
     topk, num_elem, width_elem, param.in_format);

    // apply nms
    mxnet::op::NMSApply(s, num_batch, topk, &sorted_index,
                        &batch_start, &buffer, &areas,
                        num_elem, width_elem, coord_start,
                        id_index, param.overlap_thresh,
                        param.force_suppress, param.in_format);

    // store the results to output, keep a record for backward
    record = -1;
    out = -1;
    Kernel<nms_assign, xpu>::Launch(s, num_batch,
      out.dptr_, record.dptr_, buffer.dptr_, sorted_index.dptr_, batch_start.dptr_,
      topk, num_elem, width_elem);

    // convert encoding
    if (param.in_format != param.out_format) {
      if (box_common_enum::kCenter == param.out_format) {
        Kernel<corner_to_center, xpu>::Launch(s, num_batch * num_elem,
          out.dptr_ + coord_start, width_elem);
      } else {
        Kernel<center_to_corner, xpu>::Launch(s, num_batch * num_elem,
          out.dptr_ + coord_start, width_elem);
      }
    }
  });
}

template<typename xpu>
void BoxNMSBackward(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape in_shape = outputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 3, DType> out_grad = inputs[box_nms_enum::kOut]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> in_grad = outputs[box_nms_enum::kData]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> record = inputs[box_nms_enum::kTemp + 2]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, 1), s);

    in_grad = 0;
    Kernel<nms_backward, xpu>::Launch(s, num_batch * num_elem, in_grad.dptr_,
      out_grad.dptr_, record.dptr_, num_elem, width_elem);
  });
}

struct BoxOverlapParam : public dmlc::Parameter<BoxOverlapParam> {
  int format;
  DMLC_DECLARE_PARAMETER(BoxOverlapParam) {
    DMLC_DECLARE_FIELD(format).set_default(box_common_enum::kCorner)
    .add_enum("corner", box_common_enum::kCorner)
    .add_enum("center", box_common_enum::kCenter)
    .describe("The box encoding type. \n"
        " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
        " \"center\" means boxes are encodes as [x, y, width, height].");
  }
};  // BoxOverlapParam

inline bool BoxOverlapShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& lshape = (*in_attrs)[0];
  mxnet::TShape& rshape = (*in_attrs)[1];

  CHECK_GE(lshape.ndim(), 2)
    << "lhs must have dim >= 2 "
    << lshape.ndim() << " provided";
  int ldim = lshape[lshape.ndim() - 1];
  CHECK_EQ(ldim, 4)
    << "last dimension of lhs must be 4 "
    << ldim << " provided";
  CHECK_GE(rshape.ndim(), 2)
    << "rhs must have dim >= 2 "
    << rshape.ndim() << " provided";
  int rdim = rshape[rshape.ndim() - 1];
  CHECK_EQ(rdim, 4)
    << "last dimension of rhs must be 4 "
    << rdim << " provided";

  // assign output shape
  mxnet::TShape oshape(lshape.ndim() + rshape.ndim() - 2, -1);
  int idx = 0;
  for (index_t i = 0; i < lshape.ndim() - 1; ++i) {
    oshape[idx++] = lshape[i];
  }
  for (index_t i = 0; i < rshape.ndim() - 1; ++i) {
    oshape[idx++] = rshape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

struct compute_overlap {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs,
                                  const DType *rhs, int num,
                                  int begin, int stride, int encode) {
    int l = i / num;
    int r = i % num;
    int l_index = l * stride + begin;
    int r_index = r * stride + begin;
    DType intersect = Intersect(lhs + l_index, rhs + r_index, encode);
    intersect *= Intersect(lhs + l_index + 1, rhs + r_index + 1, encode);
    if (intersect <= 0) {
      out[i] = DType(0);
      return;
    }
    DType l_area = BoxArea(lhs + l_index, encode);
    DType r_area = BoxArea(rhs + r_index, encode);
    out[i] = intersect / (l_area + r_area - intersect);
  }
};

template<typename xpu>
void BoxOverlapForward(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const BoxOverlapParam& param = nnvm::get<BoxOverlapParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape lshape = inputs[0].shape_;
  mxnet::TShape rshape = inputs[1].shape_;
  int lsize = lshape.ProdShape(0, lshape.ndim() - 1);
  int rsize = rshape.ProdShape(0, rshape.ndim() - 1);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lhs = inputs[0]
     .get_with_shape<xpu, 1, DType>(Shape1(lsize * 4), s);
    Tensor<xpu, 1, DType> rhs = inputs[1]
     .get_with_shape<xpu, 1, DType>(Shape1(rsize * 4), s);
    Tensor<xpu, 1, DType> out = outputs[0]
     .get_with_shape<xpu, 1, DType>(Shape1(lsize * rsize), s);

    Kernel<compute_overlap, xpu>::Launch(s, lsize * rsize, out.dptr_,
     lhs.dptr_, rhs.dptr_, rsize, 0, 4, param.format);
  });
}

template<typename xpu>
void BoxOverlapBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> in_grad_lhs = outputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> in_grad_rhs = outputs[1].FlatTo2D<xpu, DType>(s);
    // TODO(Joshua Zhang): allow backprop?
    in_grad_lhs = 0;
    in_grad_rhs = 0;
  });
}

struct BipartiteMatchingParam : public dmlc::Parameter<BipartiteMatchingParam> {
  bool is_ascend;
  float threshold;
  int topk;
  DMLC_DECLARE_PARAMETER(BipartiteMatchingParam) {
    DMLC_DECLARE_FIELD(is_ascend).set_default(false)
    .describe("Use ascend order for scores instead of descending. "
    "Please set threshold accordingly.");
    DMLC_DECLARE_FIELD(threshold)
    .describe("Ignore matching when score < thresh, if is_ascend=false, "
    "or ignore score > thresh, if is_ascend=true.");
    DMLC_DECLARE_FIELD(topk).set_default(-1)
    .describe("Limit the number of matches to topk, set -1 for no limit");
  }
};  // BipartiteMatchingParam

inline bool MatchingShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  // const BipartiteMatchingParam& param = nnvm::get<BipartiteMatchingParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  mxnet::TShape& dshape = (*in_attrs)[0];

  CHECK_GE(dshape.ndim(), 2)
    << "score matrix must have dim >= 2 "
    << dshape.ndim() << " provided";

  // assign output shape
  mxnet::TShape oshape(dshape.ndim() - 1, -1);
  for (index_t i = 0; i < dshape.ndim() - 1; ++i) {
    oshape[i] = dshape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  oshape[oshape.ndim() - 1] = dshape[dshape.ndim() - 1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);
  return shape_is_known(oshape);
}

struct bipartite_matching {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *row_marker, DType *col_marker,
                                  const DType *scores, const int32_t *sorted_index,
                                  int num_batch, int num_row, int num_col,
                                  float threshold, bool is_ascend, int topk) {
    int stride = num_row * num_col;
    const int32_t *index = sorted_index + i * stride;
    const DType *score = scores + i * stride;
    DType *rmarker = row_marker + i * num_row;
    DType *cmarker = col_marker + i * num_col;
    int count = 0;
    for (int j = 0; j < stride; ++j) {
      int idx = static_cast<int>(index[j]) % stride;
      int r = idx / num_col;
      int c = idx % num_col;
      if (rmarker[r] == -1 && cmarker[c] == -1) {
        if ((!is_ascend && score[j] > threshold) ||
            (is_ascend && score[j] < threshold)) {
          rmarker[r] = c;
          cmarker[c] = r;
          ++count;
          if (topk > 0 && count > topk) {
            break;
          }
        } else {
          // already encounter bad scores
          break;
        }
      }
    }
  }
};

template<typename xpu>
void BipartiteMatchingForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  const BipartiteMatchingParam& param = nnvm::get<BipartiteMatchingParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape dshape = inputs[0].shape_;
  int row = dshape[dshape.ndim() - 2];
  int col = dshape[dshape.ndim() - 1];
  int batch_size = dshape.Size() / row / col;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> scores = inputs[0]
     .get_with_shape<xpu, 1, DType>(Shape1(dshape.Size()), s);
    Tensor<xpu, 2, DType> row_marker = outputs[0]
     .get_with_shape<xpu, 2, DType>(Shape2(batch_size, row), s);
    Tensor<xpu, 2, DType> col_marker = outputs[1]
     .get_with_shape<xpu, 2, DType>(Shape2(batch_size, col), s);
    Shape<1> sort_index_shape = Shape1(dshape.Size());
    index_t workspace_size = sort_index_shape.Size();
    workspace_size += (sort_index_shape.Size() * 2 * sizeof(int32_t) - 1) / sizeof(DType) + 1;
    Tensor<xpu, 1, DType> workspace = ctx.requested[0]
      .get_space_typed<xpu, 1, DType>(Shape1(workspace_size), s);
    Tensor<xpu, 1, DType> scores_copy(workspace.dptr_,
      sort_index_shape, s);
    Tensor<xpu, 1, int32_t> sorted_index(reinterpret_cast<int32_t*>(
      scores_copy.dptr_ + scores_copy.MSize()), sort_index_shape, s);
    Tensor<xpu, 1, int32_t> batch_id(sorted_index.dptr_ + sorted_index.MSize(),
      sort_index_shape, s);

    // sort according to score
    scores_copy = F<mshadow_op::identity>(scores);
    sorted_index = range<int32_t>(0, dshape.Size());
    mxnet::op::SortByKey(scores_copy, sorted_index, param.is_ascend);
    batch_id = (sorted_index / ScalarExp<int32_t>(row * col));
    mxnet::op::SortByKey(batch_id, scores_copy, true);
    batch_id = (sorted_index / ScalarExp<int32_t>(row * col));
    mxnet::op::SortByKey(batch_id, sorted_index, true);

    // bipartite matching, parallelization is limited to batch_size
    row_marker = -1;
    col_marker = -1;
    Kernel<bipartite_matching, xpu>::Launch(s, batch_size, row_marker.dptr_,
     col_marker.dptr_, scores_copy.dptr_, sorted_index.dptr_, batch_size, row, col,
     param.threshold, param.is_ascend, param.topk);
  });
}

template<typename xpu>
void BipartiteMatchingBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> in_grad = outputs[0].FlatTo2D<xpu, DType>(s);
    // TODO(Joshua Zhang): allow backprop?
    in_grad = 0;
  });
}


inline bool BoxEncodeShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 6U);
  CHECK_EQ(out_attrs->size(), 2U);
  mxnet::TShape& sshape = (*in_attrs)[0];
  mxnet::TShape& mshape = (*in_attrs)[1];
  mxnet::TShape& ashape = (*in_attrs)[2];
  mxnet::TShape& rshape = (*in_attrs)[3];

  CHECK_EQ(sshape.ndim(), 2)
    << "samples shape must have dim == 2, "
    << sshape.ndim() << " provided";

  CHECK_GE(mshape.ndim(), 2)
    << "matches shape must have dim == 2, "
    << mshape.ndim() << " provided";

  CHECK_GE(ashape.ndim(), 3)
    << "matches shape must have dim == 3, "
    << ashape.ndim() << " provided";
  int ldim = ashape[ashape.ndim() - 1];
  CHECK_EQ(ldim, 4)
    << "last dimension of anchors must be 4, "
    << ldim << " provided";

  CHECK_GE(rshape.ndim(), 3)
    << "refs shape must have dim == 3, "
    << ashape.ndim() << " provided";
  ldim = rshape[rshape.ndim() - 1];
  CHECK_EQ(ldim, 4)
    << "last dimension of anchors must be 4, "
    << ldim << " provided";

  // asign input shape
  SHAPE_ASSIGN_CHECK(*in_attrs, 4, mshadow::Shape1(4));
  SHAPE_ASSIGN_CHECK(*in_attrs, 5, mshadow::Shape1(4));

  // assign output shape
  mxnet::TShape oshape = ashape;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);
  return shape_is_known(oshape);
}

struct box_encode {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out_targets, DType *out_masks,
                                  const DType *samples, const DType *matches,
                                  const DType *anchors, const DType *refs,
                                  const DType *means, const DType *stds,
                                  const int m, const int n) {
    index_t j = i / n;
    index_t match = matches[i];
    // xmin: 0, ymin:1, xmax: 2, ymax: 3
    // x:0, y:1, w:2, h:3
    index_t ref_index = (j * m + match) * 4;
    DType ref_xmin = refs[ref_index + 0];
    DType ref_ymin = refs[ref_index + 1];
    DType ref_width = refs[ref_index + 2] - ref_xmin;
    DType ref_height = refs[ref_index + 3] - ref_ymin;
    DType ref_x = ref_xmin + ref_width * 0.5;
    DType ref_y = ref_ymin + ref_height * 0.5;
    index_t a_index = i * 4;
    DType a_xmin = anchors[a_index + 0];
    DType a_ymin = anchors[a_index + 1];
    DType a_width = anchors[a_index + 2] - a_xmin;
    DType a_height = anchors[a_index + 3] - a_ymin;
    DType a_x = a_xmin + a_width * 0.5;
    DType a_y = a_ymin + a_height * 0.5;
    DType valid = samples[i] > 0.5 ? 1.0 : 0.0;
    out_masks[a_index + 0] = valid;
    out_masks[a_index + 1] = valid;
    out_masks[a_index + 2] = valid;
    out_masks[a_index + 3] = valid;
    out_targets[a_index + 0] = valid > static_cast<DType>(0.5) ?
        ((ref_x - a_x) / a_width - static_cast<DType>(means[0])) /
        static_cast<DType>(stds[0]) : static_cast<DType>(0.0);
    out_targets[a_index + 1] = valid > static_cast<DType>(0.5) ?
        ((ref_y - a_y) / a_height - static_cast<DType>(means[1])) /
        static_cast<DType>(stds[1]) : static_cast<DType>(0.0);
    out_targets[a_index + 2] = valid > static_cast<DType>(0.5) ?
        (log(ref_width / a_width) - static_cast<DType>(means[2])) /
        static_cast<DType>(stds[2]) : static_cast<DType>(0.0);
    out_targets[a_index + 3] = valid > static_cast<DType>(0.5) ?
        (log(ref_height / a_height) - static_cast<DType>(means[3])) /
        static_cast<DType>(stds[3]) : static_cast<DType>(0.0);
  }
};

template<typename xpu>
void BoxEncodeForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 6U);
  CHECK_EQ(outputs.size(), 2U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // samples, matches, anchors, refs, means, stds
  mxnet::TShape anchor_shape = inputs[2].shape_;
  int loop_size = anchor_shape.ProdShape(0, 2);
  int b = anchor_shape[0];
  int n = anchor_shape[1];
  int m = inputs[3].shape_[1];
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> samples = inputs[0]
     .get_with_shape<xpu, 2, DType>(Shape2(b, n), s);
    Tensor<xpu, 2, DType> matches = inputs[1]
     .get_with_shape<xpu, 2, DType>(Shape2(b, n), s);
    Tensor<xpu, 3, DType> anchors = inputs[2]
     .get_with_shape<xpu, 3, DType>(Shape3(b, n, 4), s);
    Tensor<xpu, 3, DType> refs = inputs[3]
     .get_with_shape<xpu, 3, DType>(Shape3(b, m, 4), s);
    Tensor<xpu, 1, DType> means = inputs[4]
     .get_with_shape<xpu, 1, DType>(Shape1(4), s);
    Tensor<xpu, 1, DType> stds = inputs[5]
     .get_with_shape<xpu, 1, DType>(Shape1(4), s);
    Tensor<xpu, 3, DType> out_targets = outputs[0]
     .get_with_shape<xpu, 3, DType>(Shape3(b, n, 4), s);
    Tensor<xpu, 3, DType> out_masks = outputs[1]
     .get_with_shape<xpu, 3, DType>(Shape3(b, n, 4), s);

    Kernel<box_encode, xpu>::Launch(s, loop_size, out_targets.dptr_,
     out_masks.dptr_, samples.dptr_, matches.dptr_, anchors.dptr_,
     refs.dptr_, means.dptr_, stds.dptr_, m, n);
  });
}

struct BoxDecodeParam : public dmlc::Parameter<BoxDecodeParam> {
  float std0;
  float std1;
  float std2;
  float std3;
  float clip;
  int format;
  DMLC_DECLARE_PARAMETER(BoxDecodeParam) {
    DMLC_DECLARE_FIELD(std0).set_default(1.0)
    .describe("value to be divided from the 1st encoded values");
    DMLC_DECLARE_FIELD(std1).set_default(1.0)
    .describe("value to be divided from the 2nd encoded values");
    DMLC_DECLARE_FIELD(std2).set_default(1.0)
    .describe("value to be divided from the 3rd encoded values");
    DMLC_DECLARE_FIELD(std3).set_default(1.0)
    .describe("value to be divided from the 4th encoded values");
    DMLC_DECLARE_FIELD(clip).set_default(-1.0)
    .describe("If larger than 0, bounding box target will be clipped to this value.");
    DMLC_DECLARE_FIELD(format).set_default(box_common_enum::kCenter)
    .add_enum("corner", box_common_enum::kCorner)
    .add_enum("center", box_common_enum::kCenter)
    .describe("The box encoding type. \n"
              " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
              " \"center\" means boxes are encodes as [x, y, width, height].");
  }
};  // BoxDecodeParam

inline bool BoxDecodeShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& dshape = (*in_attrs)[0];
  mxnet::TShape& ashape = (*in_attrs)[1];

  CHECK_EQ(dshape.ndim(), 3)
    << "data shape must have dim == 3, "
    << dshape.ndim() << " provided";
  int ldim = dshape[dshape.ndim() - 1];
  CHECK_EQ(ldim, 4)
    << "last dimension of data must be 4, "
    << ldim << " provided";

  CHECK_GE(ashape.ndim(), 3)
    << "anchors shape must have dim == 3, "
    << ashape.ndim() << " provided";
  ldim = ashape[ashape.ndim() - 1];
  CHECK_EQ(ldim, 4)
    << "last dimension of anchors must be 4, "
    << ldim << " provided";

  // assign output shape
  mxnet::TShape oshape = dshape;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

template<int anchor_encode, bool has_clip>
struct box_decode {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *x,
                                  const DType *anchors, const DType std0,
                                  const DType std1, const DType std2,
                                  const DType std3, const DType clip,
                                  const int n) {
    index_t index = i * 4;
    index_t a_index = (i % n) * 4;
    DType a_x = anchors[a_index + 0];
    DType a_y = anchors[a_index + 1];
    DType a_width = anchors[a_index + 2];
    DType a_height = anchors[a_index + 3];
    if (box_common_enum::kCorner == anchor_encode) {
      // a_x = xmin, a_y = ymin, a_width = xmax, a_height = ymax
      a_width = a_width - a_x;
      a_height = a_height - a_y;
      a_x = a_x + a_width * 0.5;
      a_y = a_y + a_height * 0.5;
    }
    DType ox = x[index + 0] * std0 * a_width + a_x;
    DType oy = x[index + 1] * std1 * a_height + a_y;
    DType dw = x[index + 2] * std2;
    DType dh = x[index + 3] * std3;
    if (has_clip) {
        dw = dw < clip ? dw : clip;
        dh = dh < clip ? dh : clip;
    }
    dw = exp(dw);
    dh = exp(dh);
    DType ow = dw * a_width * 0.5;
    DType oh = dh * a_height * 0.5;
    out[index + 0] = ox - ow;
    out[index + 1] = oy - oh;
    out[index + 2] = ox + ow;
    out[index + 3] = oy + oh;
  }
};

template<typename xpu>
void BoxDecodeForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  mxnet::TShape x_shape = inputs[0].shape_;
  int b = x_shape[0];
  int n = x_shape[1];
  int loop_size = b * n;
  const BoxDecodeParam& param = nnvm::get<BoxDecodeParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 3, DType> data = inputs[0]
     .get_with_shape<xpu, 3, DType>(Shape3(b, n, 4), s);
    Tensor<xpu, 3, DType> anchors = inputs[1]
     .get_with_shape<xpu, 3, DType>(Shape3(1, n, 4), s);
    Tensor<xpu, 3, DType> out = outputs[0]
     .get_with_shape<xpu, 3, DType>(Shape3(b, n, 4), s);
    if (box_common_enum::kCorner == param.format && param.clip > 0.0) {
      Kernel<box_decode<box_common_enum::kCorner, true>, xpu>::Launch(s, loop_size,
        out.dptr_, data.dptr_, anchors.dptr_, static_cast<DType>(param.std0),
        static_cast<DType>(param.std1), static_cast<DType>(param.std2),
        static_cast<DType>(param.std3), static_cast<DType>(param.clip), n);
    } else if (box_common_enum::kCenter == param.format && param.clip > 0.0) {
      Kernel<box_decode<box_common_enum::kCenter, true>, xpu>::Launch(s, loop_size,
        out.dptr_, data.dptr_, anchors.dptr_, static_cast<DType>(param.std0),
        static_cast<DType>(param.std1), static_cast<DType>(param.std2),
        static_cast<DType>(param.std3), static_cast<DType>(param.clip), n);
    } else if (box_common_enum::kCorner == param.format && param.clip <= 0.0) {
      Kernel<box_decode<box_common_enum::kCorner, false>, xpu>::Launch(s, loop_size,
        out.dptr_, data.dptr_, anchors.dptr_, static_cast<DType>(param.std0),
        static_cast<DType>(param.std1), static_cast<DType>(param.std2),
        static_cast<DType>(param.std3), static_cast<DType>(param.clip), n);
    } else {
      Kernel<box_decode<box_common_enum::kCenter, false>, xpu>::Launch(s, loop_size,
        out.dptr_, data.dptr_, anchors.dptr_, static_cast<DType>(param.std0),
        static_cast<DType>(param.std1), static_cast<DType>(param.std2),
        static_cast<DType>(param.std3), static_cast<DType>(param.clip), n);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_H_
