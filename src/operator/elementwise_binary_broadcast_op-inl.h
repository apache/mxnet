/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_broadcast_op-inl.h
 * \brief Function defintion of elementwise binary operators with broadcast
 *
 * For example,
 *
 *   [1, 2] + [1,  = [2, 3;
 *             2 ]    3, 4]
 *
 * The shapes broacast of the above example
 *
 *   A      (2d tensor):  1 x 2
 *   B      (1d tensor):  2 x 1
 *   Result (2d tensor):  2 x 2
 *
 * More examples
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 1 x 5
 *   Result (3d tensor):  15 x 3 x 5
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 3 x 1
 *   Result (3d tensor):  15 x 3 x 5
 *
 * Here are examples of shapes that do not broadcast:
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (3d tensor):  15 x 1 x 5  # the diminsions for broadcasting should be continous
 *
 *   A      (1d tensor):  3
 *   B      (1d tensor):  4 # trailing dimensions do not match
 *
 *   A      (2d tensor):  1 x 2 x 1
 *   B      (3d tensor):  8 x 4 x 3 # second from last dimensions mismatched
 *
 * When no broadcast is need, it fails back to elementwise_binary_op-inl.h
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline bool IsBroadcastNeeded_(const TShape& lhs,
                              const TShape& rhs) {
  // force ndim to be equal. do not smartly padding dims with 1s, which may
  // confuse users
  CHECK_EQ(lhs.ndim(), rhs.ndim());
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    if (lhs[i] != rhs[i]) return true;
  }
  return false;
}


inline TShape BinaryBroadcastShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  if (!IsBroadcastNeeded_(lhs, rhs)) return lhs;
  std::vector<index_t> ret(lhs.ndim());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(lhs[i], rhs[i]);
  }
  // check
  for (int h = 0; h < 2; ++h) {
    const TShape& inp = h == 0 ? lhs : rhs;
    int contdim = 0;
    for (size_t i = 0; i < inp.ndim(); ++i) {
      if (inp[i] != 1) {
        CHECK_EQ(inp[i], ret[i]) << "broadcast error on index " << i << ". "
                                 << "lhs = " << lhs << "; rhs = " << rhs;
      }
      if (inp[i] == ret[i]) {
        if (i == 0 || inp[i-1] != ret[i-1]) ++contdim;
      }
    }
    CHECK_LE(contdim, 1) << "broadcast dimensions are not continuous. "
                         << "lhs = " << lhs << "; rhs = " << rhs;
  }
  return TShape(ret.begin(), ret.end());
}

inline void GetBroadcastShape_(const TShape& lhs,
                               const TShape& rhs,
                               TShape* ret_reshaped,
                               int* lhs_broadcast_axis,
                               int* rhs_broadcast_axis) {
  TShape ret = BinaryBroadcastShape_(lhs, rhs, EnvArguments());
  int n = static_cast<int>(ret.ndim());
  int pos[4] = {0, n, n, n};
  for (int h = 0; h < 2; ++h) {
    const TShape& inp = h == 0 ? lhs : rhs;
    for (int i = 0; i < n; ++i) {
      if (inp[i] == ret[i]) {
        pos[h*2] = i; break;
      }
    }
    for (int i = n; i > 0; --i) {
      if (inp[i-1] == ret[i-1]) {
        pos[h*2+1] = i; break;
      }
    }
  }
  bool no_broadcast_lhs = pos[0] == 0 && pos[1] == n;
  bool no_broadcast_rhs = pos[2] == 0 && pos[3] == n;
  int pos_ordered[4] = {0, -1, -1, n};
  if (no_broadcast_lhs && no_broadcast_rhs) {
    // no broadcast
    LOG(FATAL) << "no broadcast is needed";
  } else if (no_broadcast_lhs && !no_broadcast_rhs) {
    // only broadcast rhs
    *rhs_broadcast_axis = 1;
    *lhs_broadcast_axis = -1;
    pos_ordered[1] = pos[2];
    pos_ordered[2] = pos[3];
  } else if (!no_broadcast_lhs && no_broadcast_rhs) {
    // only broadcast lhs
    *rhs_broadcast_axis = -1;
    *lhs_broadcast_axis = 1;
    pos_ordered[1] = pos[0];
    pos_ordered[2] = pos[1];
  } else {
    // broadcast both lhs and rhs
    int p;
    if (pos[0] <= pos[2]) {
      CHECK(pos[0] == 0 && pos[1] == pos[2] && pos[3] == n)
        << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 0;
      *rhs_broadcast_axis = 1;
      p = pos[1];
    } else {
      CHECK(pos[2] == 0 && pos[3] == pos[0] && pos[1] == n)
        << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 1;
      *rhs_broadcast_axis = 0;
      p = pos[0];
    }
    std::vector<index_t> dim(2, 1);
    for (int i = 0; i < p; ++i) dim[0] *= ret[i];
    for (int i = p; i < n; ++i) dim[1] *= ret[i];
    *ret_reshaped = TShape(dim.begin(), dim.end());
    return;
  }
  std::vector<index_t> dim(3, 1);
  for (int i = 0; i < 3; ++i) {
    for (int j = pos_ordered[i]; j < pos_ordered[i+1]; ++j) {
      dim[i] *= ret[j];
    }
  }
  *ret_reshaped = TShape(dim.begin(), dim.end());
}


template<typename xpu, typename OP>
void BinaryBroadcastForward_(const TBlob& lhs,
                             const TBlob& rhs,
                             const EnvArguments& env,
                             TBlob *ret,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mshadow::expr;
  using mshadow::Shape;
  using mshadow::Shape1;
  using mshadow::Tensor;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";

  if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
        Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req,
                        F<OP>(lhs.FlatTo2D<xpu, DType>(s),
                              rhs.FlatTo2D<xpu, DType>(s)));
      });
    return;
  }

  TShape ret_reshaped;
  int lhs_broadcast_axis;
  int rhs_broadcast_axis;
  GetBroadcastShape_(lhs.shape_, rhs.shape_, &ret_reshaped,
                     &lhs_broadcast_axis, &rhs_broadcast_axis);
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      if (lhs_broadcast_axis >= 0) {
        // broadcast lhs
        Tensor<xpu, 1, DType> mlhs =
            lhs.get_with_shape<xpu, 1, DType>(Shape1(lhs.shape_.Size()), s);
        if (rhs_broadcast_axis >= 0) {
          // broadcast both
          Tensor<xpu, 1, DType> mrhs =
              rhs.get_with_shape<xpu, 1, DType>(Shape1(rhs.shape_.Size()), s);

          Shape<2> ret_mshape = ret_reshaped.get<2>();
          Tensor<xpu, 2, DType> out =
              ret->get_with_shape<xpu, 2, DType>(ret_mshape, s);
          if (lhs_broadcast_axis == 0) {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(broadcast<0>(mlhs, ret_mshape),
                                  broadcast<1>(mrhs, ret_mshape)));
          } else {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(broadcast<1>(mlhs, ret_mshape),
                                  broadcast<0>(mrhs, ret_mshape)));
          }
        } else {
          // only lhs
          Shape<3> ret_mshape = ret_reshaped.get<3>();
          Tensor<xpu, 3, DType> out =
              ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          Tensor<xpu, 3, DType> mrhs =
              rhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          if (lhs.shape_.Size() == 1) {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(broadcast_scalar(mlhs, ret_mshape), mrhs));
          } else {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(broadcast<1>(mlhs, ret_mshape), mrhs));
          }
        }
      } else {
        Tensor<xpu, 1, DType> mrhs =
            rhs.get_with_shape<xpu, 1, DType>(mshadow::Shape1(rhs.shape_.Size()), s);
        if (rhs_broadcast_axis >= 0) {
          // only rhs
          Shape<3> ret_mshape = ret_reshaped.get<3>();
          Tensor<xpu, 3, DType> out =
              ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          Tensor<xpu, 3, DType> mlhs =
              lhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          if (lhs.shape_.Size() == 1) {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(mlhs, broadcast_scalar(mrhs, ret_mshape)));
          } else {
            ASSIGN_DISPATCH(out, req,
                            F<OP>(mlhs, broadcast<1>(mrhs, ret_mshape)));
          }
        } else {
          LOG(FATAL) << "no broadcast is needed";
        }
      }
    });
}


template<typename xpu, typename LHS_OP, typename RHS_OP>
void BinaryBroadcastBackward_(const OutputGrad& out_grad,
                              const EnvArguments& env,
                              TBlob* lhs_grad,
                              TBlob* rhs_grad,
                              OpReqType req_lhs_grad,
                              OpReqType req_rhs_grad,
                              RunContext ctx) {
  using namespace mshadow::expr;
  using mshadow::Shape;
  using mshadow::Shape1;
  using mshadow::Shape2;
  using mshadow::Tensor;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
        Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
        ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
      });
    return;
  }

  TShape ret_reshaped;
  int lhs_broadcast_axis;
  int rhs_broadcast_axis;
  GetBroadcastShape_(lhs_grad->shape_, rhs_grad->shape_, &ret_reshaped,
                     &lhs_broadcast_axis, &rhs_broadcast_axis);
  index_t lhs_size = lhs_grad->shape_.Size();
  index_t rhs_size = rhs_grad->shape_.Size();

  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      if (lhs_broadcast_axis >= 0) {
        Tensor<xpu, 1, DType> mlhs_grad =
            lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);
        if (rhs_broadcast_axis >= 0) {
          // broadcast both
          Tensor<xpu, 2, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 2, DType>(ret_reshaped.get<2>(), s);
          Tensor<xpu, 1, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          if (lhs_broadcast_axis == 0) {
            ASSIGN_DISPATCH(
                mlhs_grad, req_lhs_grad, sumall_except_dim<0>(F<LHS_OP>(mout_grad)));
            ASSIGN_DISPATCH(
                mrhs_grad, req_rhs_grad, sumall_except_dim<1>(F<RHS_OP>(mout_grad)));
          } else {
            ASSIGN_DISPATCH(
                mlhs_grad, req_lhs_grad, sumall_except_dim<1>(F<LHS_OP>(mout_grad)));
            ASSIGN_DISPATCH(
                mrhs_grad, req_rhs_grad, sumall_except_dim<0>(F<RHS_OP>(mout_grad)));
          }
        } else {
          // only broadcast lhs
          Tensor<xpu, 3, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 3, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          ASSIGN_DISPATCH(
              mlhs_grad, req_lhs_grad, sumall_except_dim<1>(F<LHS_OP>(mout_grad)));
          ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
        }
      } else {
        if (rhs_broadcast_axis >= 0) {
          // only broadcast rhs
          Tensor<xpu, 3, DType> mlhs_grad =
              lhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 1, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          Tensor<xpu, 3, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
          ASSIGN_DISPATCH(
              mrhs_grad, req_rhs_grad, sumall_except_dim<1>(F<RHS_OP>(mout_grad)));
        } else {
          LOG(FATAL) << "no broadcast is needed";
        }
      }
    });
}

template<typename xpu>
void BroadcastMulBackward_(const OutputGrad& out_grad,
                            const Input0& lhs,
                            const Input1& rhs,
                            const EnvArguments& env,
                            TBlob* lhs_grad,
                            TBlob* rhs_grad,
                            OpReqType req_lhs_grad,
                            OpReqType req_rhs_grad,
                            RunContext ctx) {
  using namespace mshadow::expr;
  using mshadow::Shape;
  using mshadow::Shape1;
  using mshadow::Shape2;
  using mshadow::Tensor;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
        Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
        CHECK_NE(req_rhs_grad, kWriteInplace);
        ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, mlhs_data * mout_grad);
        ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mrhs_data * mout_grad);
      });
    return;
  }

  TShape ret_reshaped;
  int lhs_broadcast_axis;
  int rhs_broadcast_axis;
  GetBroadcastShape_(lhs_grad->shape_, rhs_grad->shape_, &ret_reshaped,
                     &lhs_broadcast_axis, &rhs_broadcast_axis);
  index_t lhs_size = lhs_grad->shape_.Size();
  index_t rhs_size = rhs_grad->shape_.Size();

  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      if (lhs_broadcast_axis >= 0) {
        Tensor<xpu, 1, DType> mlhs_data =
            lhs.data.get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);
        Tensor<xpu, 1, DType> mlhs_grad =
            lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);

        if (rhs_broadcast_axis >= 0) {
          // broadcast both
          Tensor<xpu, 2, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 2, DType>(ret_reshaped.get<2>(), s);
          Tensor<xpu, 1, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          Tensor<xpu, 1, DType> mrhs_data =
              rhs.data.get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          if (lhs_broadcast_axis == 0) {
            ASSIGN_DISPATCH(
                mlhs_grad, req_lhs_grad, sumall_except_dim<0>(
                    mout_grad * broadcast<1>(mrhs_data, ret_reshaped.get<2>())));
            ASSIGN_DISPATCH(
                mrhs_grad, req_rhs_grad, sumall_except_dim<1>(
                    mout_grad * broadcast<0>(mlhs_data, ret_reshaped.get<2>())));
          } else {
            ASSIGN_DISPATCH(
                mlhs_grad, req_lhs_grad, sumall_except_dim<1>(
                    mout_grad * broadcast<0>(mrhs_data, ret_reshaped.get<2>())));
            ASSIGN_DISPATCH(
                mrhs_grad, req_rhs_grad, sumall_except_dim<0>(
                    mout_grad * broadcast<1>(mlhs_data, ret_reshaped.get<2>())));
          }
        } else {
          // only broadcast lhs
          Tensor<xpu, 3, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 3, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 3, DType> mrhs_data =
              rhs.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);

          ASSIGN_DISPATCH(
              mlhs_grad, req_lhs_grad, sumall_except_dim<1>(mout_grad * mrhs_data));
          if (lhs_size == 1) {
            ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                            mout_grad * broadcast_scalar(mlhs_data, ret_reshaped.get<3>()));
          } else {
            ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                            mout_grad * broadcast<1>(mlhs_data, ret_reshaped.get<3>()));
          }
        }
      } else {
        if (rhs_broadcast_axis >= 0) {
          // only broadcast rhs
          Tensor<xpu, 3, DType> mlhs_grad =
              lhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 3, DType> mlhs_data =
              lhs.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          Tensor<xpu, 1, DType> mrhs_grad =
              rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          Tensor<xpu, 1, DType> mrhs_data =
              rhs.data.get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          Tensor<xpu, 3, DType> mout_grad =
              out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);

          if (rhs_size == 1) {
            ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                            mout_grad * broadcast_scalar(mrhs_data, ret_reshaped.get<3>()));
          } else {
            ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                            mout_grad * broadcast<1>(mrhs_data, ret_reshaped.get<3>()));
          }
          ASSIGN_DISPATCH(
              mrhs_grad, req_rhs_grad, sumall_except_dim<1>(mout_grad * mlhs_data));
        } else {
          LOG(FATAL) << "no broadcast is needed";
        }
      }
    });
}

template<typename xpu>
void BroadcastDivBackward_(const OutputGrad& out_grad,
  const Input0& lhs,
  const Input1& rhs,
  const EnvArguments& env,
  TBlob* lhs_grad,
  TBlob* rhs_grad,
  OpReqType req_lhs_grad,
  OpReqType req_rhs_grad,
  RunContext ctx) {
  using namespace mshadow::expr;
  using mshadow::Shape;
  using mshadow::Shape1;
  using mshadow::Shape2;
  using mshadow::Tensor;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      F<mshadow_op::negation>(mout_grad * mlhs_data)/
                      F<mshadow_op::square>(mrhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mout_grad /  mrhs_data);    });
    return;
  }

  TShape ret_reshaped;
  int lhs_broadcast_axis;
  int rhs_broadcast_axis;
  GetBroadcastShape_(lhs_grad->shape_, rhs_grad->shape_, &ret_reshaped,
    &lhs_broadcast_axis, &rhs_broadcast_axis);
  index_t lhs_size = lhs_grad->shape_.Size();
  index_t rhs_size = rhs_grad->shape_.Size();

  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    if (lhs_broadcast_axis >= 0) {
      Tensor<xpu, 1, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);
      Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);

      if (rhs_broadcast_axis >= 0) {
        // broadcast both
        Shape<2> rshape = ret_reshaped.get<2>();
        Tensor<xpu, 2, DType> mout_grad =
          out_grad.data.get_with_shape<xpu, 2, DType>(rshape, s);
        Tensor<xpu, 1, DType> mrhs_grad =
          rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
        Tensor<xpu, 1, DType> mrhs_data =
          rhs.data.get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
        if (lhs_broadcast_axis == 0) {
          ASSIGN_DISPATCH(
            mlhs_grad, req_lhs_grad, sumall_except_dim<0>(
            mout_grad / broadcast<1>(mrhs_data, rshape)));
          ASSIGN_DISPATCH(
            mrhs_grad, req_rhs_grad, sumall_except_dim<1>(
            F<mshadow_op::negation>(mout_grad * broadcast<0>(mlhs_data, rshape)) /
            F<mshadow_op::square>(broadcast<1>(mrhs_data, rshape))));
        } else {
          ASSIGN_DISPATCH(
            mlhs_grad, req_lhs_grad, sumall_except_dim<1>(
            mout_grad / broadcast<0>(mrhs_data, rshape)));
          ASSIGN_DISPATCH(
            mrhs_grad, req_rhs_grad, sumall_except_dim<0>(
            F<mshadow_op::negation>(mout_grad * broadcast<1>(mlhs_data, rshape)) /
            F<mshadow_op::square>(broadcast<0>(mrhs_data, rshape))));
        }
      } else {
        // only broadcast lhs
        Shape<3> rshape = ret_reshaped.get<3>();
        Tensor<xpu, 3, DType> mout_grad =
          out_grad.data.get_with_shape<xpu, 3, DType>(rshape, s);
        Tensor<xpu, 3, DType> mrhs_grad =
          rhs_grad->get_with_shape<xpu, 3, DType>(rshape, s);
        Tensor<xpu, 3, DType> mrhs_data =
          rhs.data.get_with_shape<xpu, 3, DType>(rshape, s);

        ASSIGN_DISPATCH(
          mlhs_grad, req_lhs_grad, sumall_except_dim<1>(mout_grad / mrhs_data));
        if (lhs_size == 1) {
          ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
            F<mshadow_op::negation>(mout_grad * broadcast_scalar(mlhs_data, rshape)) /
            F<mshadow_op::square>(mrhs_data));
        } else {
          ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
            F<mshadow_op::negation>(mout_grad * broadcast<1>(mlhs_data, rshape)) /
            F<mshadow_op::square>(mrhs_data));
        }
      }
    } else {
      if (rhs_broadcast_axis >= 0) {
        // only broadcast rhs
        Shape<3> rshape = ret_reshaped.get<3>();
        Tensor<xpu, 3, DType> mlhs_grad = lhs_grad->get_with_shape<xpu, 3, DType>(rshape, s);
        Tensor<xpu, 3, DType> mlhs_data = lhs.data.get_with_shape<xpu, 3, DType>(rshape, s);
        Tensor<xpu, 1, DType> mrhs_grad =
          rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
        Tensor<xpu, 1, DType> mrhs_data =
          rhs.data.get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
        Tensor<xpu, 3, DType> mout_grad =
          out_grad.data.get_with_shape<xpu, 3, DType>(rshape, s);

        if (rhs_size == 1) {
          ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
            mout_grad / broadcast_scalar(mrhs_data, rshape));
          ASSIGN_DISPATCH(
            mrhs_grad, req_rhs_grad, sumall_except_dim<1>(
            F<mshadow_op::negation>(mout_grad * mlhs_data) /
            F<mshadow_op::square>(broadcast_scalar(mrhs_data, rshape))));
        } else {
          ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
            mout_grad / broadcast<1>(mrhs_data, rshape));
          ASSIGN_DISPATCH(
            mrhs_grad, req_rhs_grad, sumall_except_dim<1>(
            F<mshadow_op::negation>(mout_grad * mlhs_data) /
            F<mshadow_op::square>(broadcast<1>(mrhs_data, rshape))));
        }
      } else {
        LOG(FATAL) << "no broadcast is needed";
      }
    }
  });
}


MXNET_REGISTER_SIMPLE_OP(broadcast_plus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::plus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::identity>, kNoInplace)
.describe("lhs add rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_minus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::minus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::negation>, kNoInplace)
.describe("lhs minus rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_mul, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::mul>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastMulBackward_<XPU>, kNoInplace)
.describe("lhs multiple rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_div, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::div>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastDivBackward_<XPU>, kNoInplace)
.describe("lhs divide rhs with broadcast");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
