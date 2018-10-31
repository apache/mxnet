/*!
 *  Copyright (c) 2017 by Contributors
 * \file nnvm/top/tensor.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef NNVM_TOP_TENSOR_H_
#define NNVM_TOP_TENSOR_H_

#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace top {

struct ConcatenateParam : public dmlc::Parameter<ConcatenateParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(ConcatenateParam) {
    DMLC_DECLARE_FIELD(axis).set_default(1)
    .describe("the axis to be concated.");
  }
};

struct ExpandDimsParam : public dmlc::Parameter<ExpandDimsParam> {
  int axis;
  int num_newaxis;
  DMLC_DECLARE_PARAMETER(ExpandDimsParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("the axis to be expanded.");
    DMLC_DECLARE_FIELD(num_newaxis).set_lower_bound(1).set_default(1)
    .describe("Number of new axis to be inserted.");
  }
};

struct SplitParam : public dmlc::Parameter<SplitParam> {
  // numpy convention, only support indices, not support list.
  Tuple<int> indices_or_sections;
  int axis;
  // additional hint whether it is equal_split mode
  // deduced from indices_or_sections
  bool equal_split;

  DMLC_DECLARE_PARAMETER(SplitParam) {
    DMLC_DECLARE_FIELD(indices_or_sections)
        .describe("Number of outputs to be splitted");
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0).set_default(1)
        .describe("the axis to be splitted.");
  }
};


struct TakeParam : public dmlc::Parameter<TakeParam> {
  dmlc::optional<int> axis;

  DMLC_DECLARE_PARAMETER(TakeParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>())
        .describe("the axis over which to select values.");
  }
};

struct StridedSliceParam : public dmlc::Parameter<StridedSliceParam> {
  // numpy convention, only support indices, not support list.
  Tuple<int64_t> begin;
  Tuple<int64_t> end;
  Tuple<int64_t> stride;

  DMLC_DECLARE_PARAMETER(StridedSliceParam) {
    DMLC_DECLARE_FIELD(begin)
        .describe("Indices for begin of slice");
    DMLC_DECLARE_FIELD(end)
        .describe("Indices for end of the slice");
    DMLC_DECLARE_FIELD(stride).set_default(Tuple<int64_t>())
        .describe("Stride values of the slice");
  }
};

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kInt16 = 7,
  kUint16 = 8,
  kUint32 = 9,
  kUint64 = 10,
};

enum IndicatorRuleFlag {
  kGT0 = 0,
  kLT0 = 1,
  kMax = 2,
  kMin = 3,
};

#define DMLC_DECLARE_DTYPE_FIELD(name)                              \
  DMLC_DECLARE_FIELD(name)                                          \
  .add_enum("float16", kFloat16)                                    \
  .add_enum("float32", kFloat32)                                    \
  .add_enum("float64", kFloat64)                                    \
  .add_enum("uint8",  kUint8)                                       \
  .add_enum("uint16", kUint16)                                      \
  .add_enum("uint32", kUint32)                                      \
  .add_enum("uint64", kUint64)                                      \
  .add_enum("int8",  kInt8)                                         \
  .add_enum("int16", kInt16)                                        \
  .add_enum("int32", kInt32)                                        \
  .add_enum("int64", kInt64)

struct CastParam : public dmlc::Parameter<CastParam> {
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_DTYPE_FIELD(dtype)
    .describe("Output data type.");
  }
};

struct IndicatorParam : public dmlc::Parameter<IndicatorParam> {
  TShape axis;
  bool exclude;
  DMLC_DECLARE_PARAMETER(IndicatorParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
    .describe(R"code(The axis or axes along which to perform the indicator rule.

        The default, `axis=()`, will compute over all elements into a
        scalar array with shape `(1,)`.

        If `axis` is int, rule is applied on a particular axis.

        If `axis` is a tuple of ints, rule is applied on all the axes
        specified in the tuple.

        If `exclude` is true, rule will be applied on the axes that are
        NOT in axis instead.)code");
    DMLC_DECLARE_FIELD(exclude).set_default(false)
    .describe("Whether to apply rule on axis that are NOT in axis instead.");
  }
};

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  Tuple<int64_t> shape;

  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(shape);
  }
};

struct SqueezeParam : public dmlc::Parameter<SqueezeParam> {
  TShape axis;

  DMLC_DECLARE_PARAMETER(SqueezeParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
    .describe("The axis to squeeze in the input tensor.");
  }
};

struct ScalarParam : public dmlc::Parameter<ScalarParam> {
  double scalar;

  DMLC_DECLARE_PARAMETER(ScalarParam) {
    DMLC_DECLARE_FIELD(scalar);
  }
};

struct FillValueParam : public dmlc::Parameter<FillValueParam> {
  double fill_value;

  DMLC_DECLARE_PARAMETER(FillValueParam) {
    DMLC_DECLARE_FIELD(fill_value)
    .describe("Scalar value to be filled");
  }
};

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;

  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

struct FlipParam : public dmlc::Parameter<FlipParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0)
    .describe("the axis to be reveresed.");
  }
};

struct BroadcastToParam : public dmlc::Parameter<BroadcastToParam> {
  TShape shape;

  DMLC_DECLARE_PARAMETER(BroadcastToParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape())
      .describe("The shape of the desired array."
                " We can set the dim to zero if it's same as the original."
                " E.g `A = broadcast_to(B, shape=(10, 0, 0))` ");
  }
};

struct ReduceParam : public dmlc::Parameter<ReduceParam> {
  TShape axis;
  bool keepdims;
  bool exclude;

  DMLC_DECLARE_PARAMETER(ReduceParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

struct InitOpWithScalarParam : public dmlc::Parameter<InitOpWithScalarParam> {
  TShape shape;
  int dtype;
  double fill_value;

  DMLC_DECLARE_PARAMETER(InitOpWithScalarParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape());
    DMLC_DECLARE_DTYPE_FIELD(dtype).set_default(kFloat32)
      .describe("Target data type.");
    DMLC_DECLARE_FIELD(fill_value).describe("Scalar value to fill");
  }
};

struct InitOpParam : public dmlc::Parameter<InitOpParam> {
  TShape shape;
  int dtype;

  DMLC_DECLARE_PARAMETER(InitOpParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape());
    DMLC_DECLARE_DTYPE_FIELD(dtype).set_default(kFloat32)
      .describe("Target data type.");
  }
};

struct ElementWiseReduceParam : public dmlc::Parameter<ElementWiseReduceParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseReduceParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
      .describe("Number of inputs to be reduced.");
  }
};

struct MatMulParam : public dmlc::Parameter<MatMulParam> {
  bool transpose_a;
  bool transpose_b;

  DMLC_DECLARE_PARAMETER(MatMulParam) {
    DMLC_DECLARE_FIELD(transpose_a)
      .describe("If true then transpose the first input before dot.")
      .set_default(false);
    DMLC_DECLARE_FIELD(transpose_b)
      .describe("If true then transpose the second input before dot.")
      .set_default(false);
  }
};

struct ClipParam : public dmlc::Parameter<ClipParam> {
  double a_min, a_max;
  DMLC_DECLARE_PARAMETER(ClipParam) {
    DMLC_DECLARE_FIELD(a_min)
      .describe("Minimum value such that value smaller then this will be clipped.");
    DMLC_DECLARE_FIELD(a_max)
      .describe("Maximum value such that value larger then this will be clipped.");
  }
};

struct SliceLikeParam : public dmlc::Parameter<SliceLikeParam> {
  Tuple<int> axis;
  DMLC_DECLARE_PARAMETER(SliceLikeParam) {
    DMLC_DECLARE_FIELD(axis).set_default(Tuple<int>())
      .describe("List of axes on which input data will be sliced according to the "
                "corresponding size of the second input. By default will slice "
                "on all axes. Negative axes are supported.");
  }
};

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_TENSOR_H_
