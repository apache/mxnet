/*!
 *  Copyright (c) 2016 by Contributors
 * \file ordering_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <vector>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./sort_op.h"
#include "./indexing_op.h"

namespace mshadow {
template<typename xpu, int src_dim, typename DType, int dst_dim>
inline Tensor<xpu, dst_dim, DType> inplace_reshape(Tensor<xpu, src_dim, DType> src,
                                                   Shape<dst_dim> target_shape) {
  CHECK_EQ(src.CheckContiguous(), true);
  return Tensor<xpu, dst_dim, DType>(src.dptr_, target_shape, src.stream_);
}
};


namespace mxnet {
namespace op {
// These enums are only visible within this header
namespace topk_enum {
enum TopKReturnType {kReturnValue, kReturnIndices, kReturnMask, kReturnBoth};
}  // topk_enum

struct TopKParam : public dmlc::Parameter<TopKParam> {
  dmlc::optional<int> axis;
  int k;
  int ret_typ;
  bool is_ascend;
  DMLC_DECLARE_PARAMETER(TopKParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose the top k indices."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(k).set_default(1)
    .describe("Number of top elements to select,"
              " should be always smaller than or equal to the element number in the given axis."
              " A global sort is performed if set k < 1.");
    DMLC_DECLARE_FIELD(ret_typ).set_default(topk_enum::kReturnIndices)
    .add_enum("value", topk_enum::kReturnValue)
    .add_enum("indices", topk_enum::kReturnIndices)
    .add_enum("mask", topk_enum::kReturnMask)
    .add_enum("both", topk_enum::kReturnBoth)
    .describe("The return type.\n"
        " \"value\" means to return the top k values,"
        " \"indices\" means to return the indices of the top k values,"
        " \"mask\" means to return a mask array containing 0 and 1. 1 means the top k values."
        " \"both\" means to return a list of both values and indices of top k elements.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(false)
      .describe("Whether to choose k largest or k smallest elements."
                " Top K largest elements will be chosen if set to false.");
  }
};

struct SortParam : public dmlc::Parameter<SortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  DMLC_DECLARE_PARAMETER(SortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
  }
};

struct ArgSortParam : public dmlc::Parameter<ArgSortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  DMLC_DECLARE_PARAMETER(ArgSortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
  }
};

inline void ParseTopKParam(const TShape& src_shape, const TopKParam& param, TShape *target_shape,
                           int *batch_size, int *element_num, int *axis, int *k,
                           bool *do_transpose, bool *is_ascend) {
  *do_transpose = false;
  *k = param.k;
  *is_ascend = param.is_ascend;
  // get batch_size, axis and element_num
  if (!static_cast<bool>(param.axis)) {  // No axis given
    *axis = 0;
    *batch_size = 1;
    *element_num = src_shape.Size();
  } else {
    *axis = param.axis.value();
    if (*axis < 0) {
      *axis += src_shape.ndim();
    }
    CHECK(*axis >= 0 && *axis < static_cast<int>(src_shape.ndim()))
                                                  << "Invalid axis! axis should be between 0 and "
                                                  << src_shape.ndim() << ", found axis=" << *axis;
    *batch_size = src_shape.Size() / src_shape[*axis];
    *element_num = src_shape[*axis];
    if (*axis != static_cast<int>(src_shape.ndim()) - 1) {
      *do_transpose = true;
    }
  }
  // get k
  if (param.k <= 0) {
    *k = *element_num;
  }
  // get target_shape
  if (!static_cast<bool>(param.axis)) {
    if (param.ret_typ != topk_enum::kReturnMask) {
      *target_shape = mshadow::Shape1(*k);
    } else {
      *target_shape = src_shape;
    }
  } else {
    *target_shape = src_shape;
    if (param.ret_typ != topk_enum::kReturnMask) {
      (*target_shape)[*axis] = *k;
    }
  }
  CHECK(*k >= 1 && *k <= *element_num) << "k must be smaller than "
                                      << *element_num << ", get k = " << *k;
}

/*!
   * \brief Implementation of the TopK operation
   *
   *
   * \param ctx the running context
   * \param resource temporary resource handler
   * \param src the Source blob
   * \param ret the destination blobs
   * \param k the K elements to keep
   * \param param the topk parameters
   * \tparam xpu the device type.
   */
template<typename xpu>
void TopKImpl(RunContext ctx,
              Resource resource,
              const TBlob& src,
              const std::vector<TBlob>& ret,
              const TopKParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  for (auto ret_ele : ret) {
    CHECK_EQ(ret_ele.type_flag_, src.type_flag_);
  }
  // 1. Parse and initialize information
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 1, real_t> workspace;
  Tensor<xpu, 1, real_t> sorted_dat, indices, batch_id, sel_indices;
  Tensor<xpu, 2, real_t> mask_val;
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  TShape target_shape;
  ParseTopKParam(src.shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  Tensor<xpu, 3, real_t> dat = src.FlatTo3D<xpu, real_t>(axis, axis, s);
  if (param.ret_typ == topk_enum::kReturnMask) {
    workspace =
      resource.get_space_typed<xpu, 1, real_t>(Shape1(src.Size() * 3 + 2 * batch_size * k), s);
  } else {
    workspace = resource.get_space_typed<xpu, 1, real_t>(mshadow::Shape1(src.Size() * 3), s);
  }
  sorted_dat = Tensor<xpu, 1, real_t>(workspace.dptr_,
                                      Shape1(src.Size()), s);  // contain sorted dat
  indices = Tensor<xpu, 1, real_t>(workspace.dptr_ + src.Size(),
                                   Shape1(src.Size()), s);  // indices in the original matrix
  batch_id = Tensor<xpu, 1, real_t>(workspace.dptr_ + 2 * src.Size(),
                                    Shape1(src.Size()), s);  // batch id in the original matrix
  if (do_transpose) {
    sorted_dat = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(src.Size()));
  } else {
    sorted_dat = reshape(dat, Shape1(src.Size()));
  }
  indices = range<real_t>(0, batch_size * element_num);
  CHECK_EQ(sorted_dat.CheckContiguous(), true);
  CHECK_EQ(indices.CheckContiguous(), true);
  if (param.ret_typ == topk_enum::kReturnMask) {
    sel_indices = Tensor<xpu, 1, real_t>(workspace.dptr_ + 3 * src.Size(),
                                         Shape1(batch_size * k), s);
    mask_val = Tensor<xpu, 2, real_t>(workspace.dptr_ + 3 * src.Size() + batch_size * k,
                                      Shape2(batch_size * k, 1), s);
    mask_val = scalar<real_t>(1);
    CHECK_EQ(sel_indices.CheckContiguous(), true);
    CHECK_EQ(mask_val.CheckContiguous(), true);
  }

  // 2. Perform inplace batch sort using the `SortByKey` in MShadow
  // After sorting, each batch in `sorted_dat` will be sorted in the corresponding order
  //   and the `indices` will contain the corresponding index in `sorted_dat`
  // Sort the data and keep record of the correspondence to global indices.
  mxnet::op::SortByKey(sorted_dat, indices, is_ascend);
  // Calculate the corresponding batch indices of the elements
  batch_id = F<mshadow_op::floor>(indices / static_cast<real_t>(element_num));
  // Since the SortByKey performs stable sort, the second SortByKey will reorder
  //   the sorted_dat based on the order of the batch_id
  mxnet::op::SortByKey(batch_id, sorted_dat, true);
  // Reorder the indices
  batch_id = F<mshadow_op::floor>(indices / static_cast<real_t>(element_num));
  mxnet::op::SortByKey(batch_id, indices, true);

  // 3. Assign results to the ret blob
  if (param.ret_typ == topk_enum::kReturnMask) {
    Tensor<xpu, 2, real_t> ret_mask =
      ret[0].get_with_shape<xpu, 2, real_t>(Shape2(ret[0].Size(), 1), s);
    ret_mask = scalar<real_t>(0);
    sel_indices = reshape(slice<1>(
                              inplace_reshape(indices,
                                              Shape2(batch_size,
                                                    element_num)), 0, k),
                            Shape1(batch_size * k));
    if (do_transpose) {
      TShape src_shape = src.shape_.FlatTo3D(axis);
      CHECK_EQ(sel_indices.CheckContiguous(), true);
      sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                      Shape3(0, 2, 1));
    }
    IndexFill(ret_mask, sel_indices, mask_val);
  } else if (param.ret_typ == topk_enum::kReturnIndices) {
    indices -= batch_id * static_cast<real_t>(element_num);
    if (do_transpose) {
      Tensor<xpu, 3, real_t> ret_indices = ret[0].FlatTo3D<xpu, real_t>(axis, axis, s);
      ret_indices = transpose(
                      slice<2>(inplace_reshape(indices,
                                               Shape3(ret_indices.shape_[0],
                                                      ret_indices.shape_[2],
                                                      element_num)),
                               0, k),
                      Shape3(0, 2, 1));
    } else {
      Tensor<xpu, 2, real_t> ret_indices =
        ret[0].get_with_shape<xpu, 2, real_t>(Shape2(batch_size, k), s);
      ret_indices = slice<1>(inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k);
    }
  } else {
    indices -= batch_id * static_cast<real_t>(element_num);
    if (do_transpose) {
      Tensor<xpu, 3, real_t> ret_value = ret[0].FlatTo3D<xpu, real_t>(axis, axis, s);
      Tensor<xpu, 3, real_t> ret_indices = ret[1].FlatTo3D<xpu, real_t>(axis, axis, s);
      ret_value = transpose(
                   slice<2>(inplace_reshape(sorted_dat,
                                    Shape3(ret_value.shape_[0], ret_value.shape_[2], element_num)),
                            0, k),
                   Shape3(0, 2, 1));
      ret_indices = transpose(
                      slice<2>(inplace_reshape(indices,
                                               Shape3(ret_indices.shape_[0],
                                                      ret_indices.shape_[2],
                                                      element_num)),
                               0, k),
                      Shape3(0, 2, 1));
    } else {
      Tensor<xpu, 2, real_t> ret_value =
        ret[0].get_with_shape<xpu, 2, real_t>(Shape2(batch_size, k), s);
      Tensor<xpu, 2, real_t> ret_indices =
        ret[1].get_with_shape<xpu, 2, real_t>(Shape2(batch_size, k), s);
      ret_value = slice<1>(inplace_reshape(sorted_dat, Shape2(batch_size, element_num)), 0, k);
      ret_indices = slice<1>(inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k);
    }
  }
}

template<typename xpu>
void TopK(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  // TODO(sxjscience) We can support inplace in the future
  CHECK_EQ(req[0], kWriteTo) << "TopK does not support inplace";
  TopKImpl<xpu>(ctx.run_ctx, ctx.requested[0], inputs[0], outputs, param);
}

template<typename xpu>
void Sort(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Sort does not support inplace";
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  TopKImpl<xpu>(ctx.run_ctx, ctx.requested[0], inputs[0], outputs, topk_param);
}

template<typename xpu>
void ArgSort(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "ArgSort does not support inplace";
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnIndices;
  TopKImpl<xpu>(ctx.run_ctx, ctx.requested[0], inputs[0], outputs, topk_param);
}

template<typename xpu>
void TopKBackward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_NE(req[0], kWriteInplace);
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.run_ctx.get_stream<xpu>();
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  CHECK(param.ret_typ == topk_enum::kReturnValue || param.ret_typ == topk_enum::kReturnBoth);
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  TShape target_shape;
  ParseTopKParam(outputs[0].shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  Tensor<xpu, 1, real_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, real_t>(Shape1(batch_size * k * 2 + batch_size), s);
  Tensor<xpu, 1, real_t> sel_indices =
    Tensor<xpu, 1, real_t>(workspace.dptr_, Shape1(batch_size * k), s);
  Tensor<xpu, 1, real_t> batch_shift =
    Tensor<xpu, 1, real_t>(workspace.dptr_ + batch_size * k, Shape1(batch_size), s);
  Tensor<xpu, 1, real_t> dummy_index =
    Tensor<xpu, 1, real_t>(workspace.dptr_ + batch_size * k + batch_size,
                           Shape1(batch_size * k), s);
  Tensor<xpu, 2, real_t> out_grad =
    inputs[0].get_with_shape<xpu, 2, real_t>(Shape2(inputs[0].shape_.Size(), 1), s);
  Tensor<xpu, 2, real_t> in_grad =
    outputs[0].get_with_shape<xpu, 2, real_t>(Shape2(outputs[0].shape_.Size(), 1), s);
  batch_shift = range<real_t>(0, batch_size, 1) * element_num;
  if (do_transpose) {
    Tensor<xpu, 1, real_t> indices = inputs[2].FlatTo1D<xpu, real_t>(s);
    TShape src_shape = outputs[0].shape_.FlatTo3D(axis);
    sel_indices = reshape(transpose(
                            broadcast_to(inplace_reshape(batch_shift,
                                                         Shape3(src_shape[0], src_shape[2], 1)),
                                         TShape(Shape3(src_shape[0], src_shape[2], k))),
                            Shape3(0, 2, 1)),
                          Shape1(batch_size * k));
    sel_indices += indices;
    sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                    Shape3(0, 2, 1));
  } else {
    Tensor<xpu, 2, real_t> indices =
      inputs[2].get_with_shape<xpu, 2, real_t>(Shape2(batch_size, k), s);
    sel_indices = reshape(indices +
                          broadcast_to(inplace_reshape(batch_shift, Shape2(batch_size, 1)),
                                       TShape(Shape2(batch_size, k))),
                          Shape1(batch_size * k));
  }
  CHECK_EQ(sel_indices.CheckContiguous(), true);
  if (kWriteTo == req[0]) {
    in_grad = scalar<real_t>(0);
    IndexFill(in_grad, sel_indices, out_grad);
  } else if (kAddTo == req[0]) {
    // TODO(sxjscience) We can use AddTakeGrad in the future.
    // However, the current implementation of AddTakeGrad is not so efficient.
    dummy_index = range<real_t>(0, sel_indices.shape_.Size());
    mxnet::op::AddTakeGradLargeBatch(in_grad, sel_indices, dummy_index, out_grad);
  } else if (kNullOp == req[0]) {
    return;
  } else {
    LOG(FATAL) << "Not Implemented!";
  }
}

inline uint32_t TopKNumOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    return static_cast<uint32_t>(1);
  } else {
    return static_cast<uint32_t>(2);
  }
}

inline uint32_t TopKNumVisibleOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnBoth) {
    return static_cast<uint32_t>(2);
  } else {
    return static_cast<uint32_t>(1);
  }
}

inline bool TopKType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

inline bool TopKShapeImpl(const TopKParam& param,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    CHECK_EQ(out_attrs->size(), 1U);
  } else {
    CHECK_EQ(out_attrs->size(), 2U);
  }
  TShape& in_shape = (*in_attrs)[0];
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  TShape target_shape;
  ParseTopKParam(in_shape, param,
    &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, target_shape);
  }
  return true;
}

inline bool TopKShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  return TopKShapeImpl(param, in_attrs, out_attrs);
}

inline bool SortShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  return TopKShapeImpl(topk_param, in_attrs, out_attrs);
}

inline bool ArgSortShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnIndices;
  return TopKShapeImpl(topk_param, in_attrs, out_attrs);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
