/*!
 * Copyright (c) 2017 by Contributors
 * \file control_flow.h
 * \brief Function definitions of operators for controlling flow
 */
#ifndef MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_
#define MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

/*! \brief Choose elements from x or y depending on condition.
 * The condition, x, and y have the same shape.
 * The returned array is formed by elements from x or y
 * depending on the elements of condition.
 */
template<int req>
struct where {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const CType* cond,
                                  const DType* x, const DType* y) {
    KERNEL_ASSIGN(out[i], req, (0 != cond[i]? x[i] : y[i]));
  }
};

/*! \brief Choose elements from x or y depending on condition
 * The condition is a vector whose size is the same as the
 * x's first dim size.
 * The returned array is formed by rows from x or y depending on
 * the condition's elements.
 */
template<int req>
struct where_batch {
  // DType is the output data type
  // CType is the condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const CType* cond,
                                  const DType* x, const DType* y, int M) {
    KERNEL_ASSIGN(out[i], req, (0 != cond[i/M]? x[i] : y[i]));
  }
};

/*!
 * \brief Template for calculating grad[x] and grad[y].
 * template argument req is OpReqType; negate indicates
 * whether the output is grad_x (negate=true)
 * or grad_y (negate=false).
 */
template<int req, bool negate>
struct where_backward {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* grad_out,
                                  const DType* grad_in,
                                  const CType* cond) {
    KERNEL_ASSIGN(grad_out[i], req,
      ((0 == cond[i])^negate)? grad_in[i] : static_cast<DType>(0));
  }
};

/*!
 * \brief Template for calculating grad[x] and grad[y].
 * template argument req is OpReqType; negate indicates
 * whether the output is grad_x (negate=true)
 * or grad_y (negate=false).
 * The condition is a vector whose size is the same as the
 * x's first dim size.
 */
template<int req, bool negate>
struct where_batch_backward {
  // DType is the output data type
  // CType is condition data type
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* grad_out,
                                  const DType* grad_in,
                                  const CType* cond, int M) {
    KERNEL_ASSIGN(grad_out[i], req,
      ((0 == cond[i/M])^negate)? grad_in[i] : static_cast<DType>(0));
  }
};

inline bool WhereOpShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape>* in_attrs,
                         std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);

  // condition shape
  const TShape& cshape = (*in_attrs)[0];
  // x shape
  const TShape& xshape = (*in_attrs)[1];
  // y shape
  const TShape& yshape = (*in_attrs)[2];

  CHECK_EQ(xshape, yshape) << "x must have the same shape as y";
  CHECK_LE(cshape.ndim(), xshape.ndim())
    << "condition's dimension=" << cshape.ndim()
    << " cannot be greater than x's dimension=" << xshape.ndim();

  // handle 0-dim arrays
  if (cshape.ndim() == 0 && xshape.ndim() == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape());
    return true;
  }

  // condition, x, and y cannot be 0-dim arrays
  // if the program has reached here
  CHECK_GT(cshape.ndim(), 0) << "condition array cannot be 0-dim if x is not 0-dim";
  CHECK_GT(xshape.ndim(), 0) << "x array cannot 0-dim if condition is not 0-dim";

  // condition and x have the same shape
  if (cshape.ndim() == xshape.ndim()) {
    CHECK_EQ(cshape, xshape) << "condition and x have the same dimension but different shapes";
  } else {
    CHECK_EQ(cshape.ndim(), 1) << "condition must either have the same shape as x or be a vector";
    CHECK_EQ(cshape[0], xshape[0]) << "condition's first dim size ("
      << cshape[0] << ") must be equal to x's first dim size (" << xshape[0] << ")";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, xshape);
  return true;
}

inline bool WhereOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);

  std::vector<int> in_attrs_xy = {(*in_attrs)[1], (*in_attrs)[2]};
  if (!ElemwiseType<2, 1>(attrs, &in_attrs_xy, out_attrs)) {
    return false;
  }
  (*in_attrs)[1] = in_attrs_xy[0];
  (*in_attrs)[2] = in_attrs_xy[1];
  return true;
}

template<typename xpu>
void WhereOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& cond = inputs[0];
  const TBlob& x = inputs[1];
  const TBlob& y = inputs[2];
  const TBlob& out = outputs[0];
  if (out.Size() == 0) return;
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.type_flag_, CType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        if (cond.shape_ == x.shape_) {
          Kernel<where<req_type>, xpu>::Launch(s, out.Size(), out.dptr<DType>(),
                                               cond.dptr<CType>(), x.dptr<DType>(),
                                               y.dptr<DType>());
        } else {
          Kernel<where_batch<req_type>, xpu>::Launch(s, out.Size(), out.dptr<DType>(),
                                                     cond.dptr<CType>(), x.dptr<DType>(),
                                                     y.dptr<DType>(), x.Size()/cond.Size());
        }
      });
    });
  });
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to condition, x, and y. The gradient
 * with respect to condition is always 0. The gradient
 * with respect to x and y depends on the corresponding
 * elements in the condition.
 * The inputs are gradient with respect to the output
 * of the operator, condition, x, and y.
 * The outputs are gradients with respect to
 * condition, x, and y.
 */
template<typename xpu>
void WhereOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(req.size(), 2);
  CHECK_EQ(outputs.size(), 2);
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& grad_in = inputs[0];
  const TBlob& cond = inputs[1];
  const TBlob& grad_x = outputs[0];
  const TBlob& grad_y = outputs[1];
  if (grad_in.Size() == 0) return;
  MSHADOW_TYPE_SWITCH(grad_in.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(cond.type_flag_, CType, {
      bool same_shape = (cond.shape_ == grad_in.shape_);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type_x, {
        if (same_shape) {
          Kernel<where_backward<req_type_x, true>, xpu>::Launch(s, grad_in.Size(),
            grad_x.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>());
        } else {
          Kernel<where_batch_backward<req_type_x, true>, xpu>::Launch(s, grad_in.Size(),
            grad_x.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>(),
            grad_in.Size()/cond.Size());
        }
      });
      MXNET_ASSIGN_REQ_SWITCH(req[1], req_type_y, {
        if (same_shape) {
          Kernel<where_backward<req_type_y, false>, xpu>::Launch(s, grad_in.Size(),
            grad_y.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>());
        } else {
          Kernel<where_batch_backward<req_type_y, false>, xpu>::Launch(s, grad_in.Size(),
            grad_y.dptr<DType>(), grad_in.dptr<DType>(), cond.dptr<CType>(),
            grad_in.Size()/cond.Size());
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CONTROL_FLOW_OP_H_
