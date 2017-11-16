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
  CHECK_EQ(in_attrs->size(), 3U)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  TShape tshape((*in_attrs)[1]);
  if (!shape_assign(&tshape, (*in_attrs)[2])) return false;
  if (!shape_assign(&tshape, (*out_attrs)[0])) return false;
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, tshape);
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, tshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);

  if ((*in_attrs)[0].ndim() == tshape.ndim()) {
    if (!shape_assign(&tshape, (*in_attrs)[0])) return false;
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tshape);
    return true;
  } else if ((*in_attrs)[0].ndim() == 1) {
    return (*in_attrs)[0].Size() == static_cast<size_t>(tshape[0]);
  }
  return false;
}

inline bool WhereOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U)
    << "where operator takes 3 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1U);

  int dtype = -1;
  if (!type_assign(&dtype, (*in_attrs)[1])) return false;
  if (!type_assign(&dtype, (*in_attrs)[2])) return false;
  if (!type_assign(&dtype, (*out_attrs)[0])) return false;
  if (-1 == dtype) return false;

  TYPE_ASSIGN_CHECK(*in_attrs, 1, dtype);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, dtype);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, dtype);

  return true;
}

template<typename xpu>
void WhereOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
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
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
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
