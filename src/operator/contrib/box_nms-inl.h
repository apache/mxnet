/*!
 *  Copyright (c) 2017 by Contributors
 * \file box_nms-inl.h
 * \brief Non-maximum suppression for bounding boxes
 * \author Joshua Zhang
 */
#ifndef MXNET_OPERATOR_CONTRIB_BOX_NMS_INL_H_
#define MXNET_OPERATOR_CONTRIB_BOX_NMS_INL_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <nnvm/tuple.h>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/sort_op.h"

namespace mxnet {
namespace op {
namespace box_nms_enum {
enum BoxNMSOpInputs {kData};
enum BoxNMSOpOutputs {kOut, kTemp};
enum BoxNMSOpBoxType {kCorner, kCenter};
enum BoxNMSOpResource {kTempSpace};
}  // box_nms_enum

struct BoxNMSParam : public dmlc::Parameter<BoxNMSParam> {
  float overlap_thresh;
  int topk;
  int coord_start;
  int score_index;
  int id_index;
  bool force_suppress;
  int in_format;
  int out_format;
  DMLC_DECLARE_PARAMETER(BoxNMSParam) {
    DMLC_DECLARE_FIELD(overlap_thresh).set_default(0.5)
    .describe("Overlapping(IoU) threshold to suppress object with smaller score.");
    DMLC_DECLARE_FIELD(topk).set_default(-1)
    .describe("Apply nms to topk boxes with descending scores, -1 to no restriction.");
    DMLC_DECLARE_FIELD(coord_start).set_default(2)
    .describe("Start index of the consecutive 4 coordinates.");
    DMLC_DECLARE_FIELD(score_index).set_default(1)
    .describe("Index of the scores/confidence of boxes.");
    DMLC_DECLARE_FIELD(id_index).set_default(-1)
    .describe("Optional, index of the class categories, -1 to disable.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
    .describe("Optional, if set false and id_index is provided, nms will only apply"
    " to boxes belongs to the same category");
    DMLC_DECLARE_FIELD(in_format).set_default(box_nms_enum::kCorner)
    .add_enum("corner", box_nms_enum::kCorner)
    .add_enum("center", box_nms_enum::kCenter)
    .describe("The input box encoding type. \n"
        " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
        " \"center\" means boxes are encodes as [x, y, width, height].");
    DMLC_DECLARE_FIELD(out_format).set_default(box_nms_enum::kCorner)
    .add_enum("corner", box_nms_enum::kCorner)
    .add_enum("center", box_nms_enum::kCenter)
    .describe("The output box encoding type. \n"
        " \"corner\" means boxes are encoded as [xmin, ymin, xmax, ymax],"
        " \"center\" means boxes are encodes as [x, y, width, height].");
  }
};  // BoxNMSParam

inline bool BoxNMSShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  if (in_attrs->at(0).ndim() == 0U && out_attrs->at(0).ndim() == 0U) {
    return false;
  }

  TShape& ishape = (*in_attrs)[0];
  int indim = ishape.ndim();
  CHECK(indim >= 2)
    << "input must have dim >= 2"
    << " the last two dimensions are num_box and box_width "
    << ishape << " provided";
  int width_elem = ishape[indim - 1];
  int expected = 5;
  if (param.id_index > 0) {
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
  TShape oshape = ishape;
  oshape[indim - 1] = 1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ishape);  // out_shape[0] == in_shape
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);  // out_shape[1]
  return true;
}

inline uint32_t BoxNMSNumVisibleOutputs(const NodeAttrs& attrs) {
  return static_cast<uint32_t>(1);
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

struct compute_area {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const DType *indices, int topk, int num_elem,
                                  int stride, int encode) {
    int b = i / topk;
    int k = i % topk;
    int index = static_cast<int>(indices[b * num_elem + k]);
    int in_index = index * stride;
    DType a1 = in[in_index++];
    DType a2 = in[in_index++];
    DType a3 = in[in_index++];
    DType a4 = in[in_index];
    DType width, height;
    if (box_nms_enum::kCorner == encode) {
      width = a3 - a1;
      height = a4 - a2;
    } else {
      width = a3;
      height = a4;
    }
    if (width < 0 || height < 0) {
      out[index] = 0;
    } else {
      out[index] = width * height;
    }
  }
};

template<typename DType>
MSHADOW_XINLINE DType Intersect(const DType *a, const DType *b, int encode) {
  DType a1 = a[0];
  DType a2 = a[2];
  DType b1 = b[0];
  DType b2 = b[2];
  DType w;
  if (box_nms_enum::kCorner == encode) {
    DType left = a1 > b1 ? a1 : b1;
    DType right = a2 < b2 ? a2 : b2;
    w = right - left;
  } else {
    DType aw = a2 / 2;
    DType bw = b2 / 2;
    DType al = a1 - aw;
    DType ar = a1 + aw;
    DType bl = b1 - bw;
    DType br = b1 + bw;
    DType left = bl > al ? bl : al;
    DType right = br < ar ? br : ar;
    w = right - left;
  }
  return w > 0 ? w : DType(0);
}

/*!
   * \brief Implementation of the non-maximum suppression operation
   *
   * \param i the launched thread index
   * \param index sorted index in descending order
   * \param input the input of nms op
   * \param areas pre-computed box areas
   * \param k nms topk number
   * \param ref compare reference position
   * \param num number of input boxes in each batch
   * \param stride input stride, usually 6 (id-score-x1-y1-x2-y2)
   * \param offset_box box offset, usually 2
   * \param thresh nms threshold
   * \param force force suppress regardless of class id
   * \param offset_id class id offset, used when force == false, usually 0
   * \param encode box encoding type, corner(0) or center(1)
   * \tparam DType the data type
   */
struct nms_impl {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *index, const DType *input,
                                  const DType *areas, int k, int ref, int num,
                                  int stride, int offset_box, int offset_id,
                                  float thresh, bool force, int encode) {
    int b = i / k;  // batch
    int pos = i % k + ref + 1;  // position
    if (index[b * num + ref] < 0) return;  // reference has been suppressed
    if (index[b * num + pos] < 0) return;  // self been suppressed
    int ref_offset = static_cast<int>(index[b * num + ref]) * stride + offset_box;
    int pos_offset = static_cast<int>(index[b * num + pos]) * stride + offset_box;
    if (!force && offset_id >=0) {
      int ref_id = static_cast<int>(input[ref_offset - offset_box + offset_id]);
      int pos_id = static_cast<int>(input[pos_offset - offset_box + offset_id]);
      if (ref_id != pos_id) return;  // different class
    }
    DType intersect = Intersect(input + ref_offset, input + pos_offset, encode);
    intersect *= Intersect(input + ref_offset + 1, input + pos_offset + 1, encode);
    int ref_area_offset = static_cast<int>(index[b * num + ref]);
    int pos_area_offset = static_cast<int>(index[b * num + pos]);
    DType iou = intersect / (areas[ref_area_offset] + areas[pos_area_offset] -
      intersect);
    if (iou > thresh) {
      index[b * num + pos] = -1;
    }
  }
};

struct nms_assign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, DType *record, const DType *input,
                                  const DType *index, int k, int num, int stride) {
    int count = 0;
    for (int j = 0; j < k; ++j) {
      int location = static_cast<int>(index[i * num + j]);
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
  TShape in_shape = inputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 3, DType> data = inputs[box_nms_enum::kData]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> out = outputs[box_nms_enum::kOut]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<xpu, 3, DType> record = outputs[box_nms_enum::kTemp]
     .get_with_shape<xpu, 3, DType>(Shape3(num_batch, num_elem, 1), s);

    // prepare workspace
    Shape<1> sort_index_shape = Shape1(num_batch * num_elem);
    Shape<3> buffer_shape = Shape3(num_batch, num_elem, width_elem);
    index_t workspace_size = 4 * sort_index_shape.Size();
    if (req[0] == kWriteInplace) {
      workspace_size += buffer_shape.Size();
    }
    Tensor<xpu, 1, DType> workspace = ctx.requested[box_nms_enum::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(workspace_size), s);
    Tensor<xpu, 1, DType> sorted_index(workspace.dptr_, sort_index_shape, s);
    Tensor<xpu, 1, DType> scores(sorted_index.dptr_ + sorted_index.MSize(),
      sort_index_shape, s);
    Tensor<xpu, 1, DType> batch_id(scores.dptr_ + scores.MSize(), sort_index_shape,
      s);
    Tensor<xpu, 1, DType> areas(batch_id.dptr_ + batch_id.MSize(), sort_index_shape, s);
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
    scores = reshape(slice<2>(buffer, score_index, score_index + 1), scores.shape_);
    sorted_index = range<DType>(0, num_batch * num_elem);
    mshadow::SortByKey(scores, sorted_index, false);
    batch_id = F<mshadow_op::floor>(sorted_index / ScalarExp<DType>(num_elem));
    mshadow::SortByKey(batch_id, scores, true);
    batch_id = F<mshadow_op::floor>(sorted_index / ScalarExp<DType>(num_elem));
    mshadow::SortByKey(batch_id, sorted_index, true);

    // pre-compute areas of candidates
    areas = 0;
    Kernel<compute_area, xpu>::Launch(s, num_batch * topk, areas.dptr_,
     buffer.dptr_ + coord_start, sorted_index.dptr_, topk, num_elem, width_elem,
     param.in_format);

    // apply nms
    // go through each box as reference, suppress if overlap > threshold
    // sorted_index with -1 is marked as suppressed
    for (int ref = 0; ref < topk; ++ref) {
      int num_worker = topk - ref - 1;
      if (num_worker < 1) continue;
      Kernel<nms_impl, xpu>::Launch(s, num_batch * num_worker, sorted_index.dptr_,
        buffer.dptr_, areas.dptr_, num_worker, ref, num_elem, width_elem,
        coord_start, id_index, param.overlap_thresh, param.force_suppress, param.in_format);
    }

    // store the results to output, keep a record for backward
    record = -1;
    out = -1;
    Kernel<nms_assign, xpu>::Launch(s, num_batch, out.dptr_, record.dptr_,
      buffer.dptr_, sorted_index.dptr_, topk, num_elem, width_elem);

    // convert encoding
    if (param.in_format != param.out_format) {
      if (box_nms_enum::kCenter == param.out_format) {
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
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  TShape in_shape = outputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
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
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOX_NMS_INL_H_
