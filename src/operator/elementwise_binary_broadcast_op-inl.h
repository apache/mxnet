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
 *   A      (2d tensor):      2
 *   B      (1d tensor):  2 x 1
 *   Result (2d tensor):  2 x 2
 *
 * More examples
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):           5
 *   Result (3d tensor):  15 x 3 x 5
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):       3 x 1
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
 *   A      (2d tensor):      2 x 1
 *   B      (3d tensor):  8 x 4 x 3 # second from last dimensions mismatched
 *
 * When no broadcast is need, it fails back to elementwise_binary_op-inl.h
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"
#include "./elementwise_binary_op-inl.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline bool IsBroadcastNeeded_(const TShape& lhs,
                              const TShape& rhs) {
  if (lhs.ndim() != rhs.ndim()) return true;
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    if (lhs[i] != rhs[i]) return true;
  }
  return false;
}

inline void GetAlignedShape_(const TShape& lhs,
                             const TShape& rhs,
                             std::vector<index_t>* lhs_aligned,
                             std::vector<index_t>* rhs_aligned,
                             std::vector<index_t>* ret) {
  index_t new_dim = std::max(lhs.ndim(), rhs.ndim());
  lhs_aligned->resize(new_dim, 1);
  rhs_aligned->resize(new_dim, 1);
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    (*lhs_aligned)[i] = lhs[i];
  }
  for (index_t i = 0; i < rhs.ndim(); ++i) {
    (*rhs_aligned)[i] = rhs[i];
  }
  ret->resize(new_dim);
  for (index_t i = 0; i < new_dim; ++i) {
    (*ret)[i] = std::max((*lhs_aligned)[i], (*rhs_aligned)[i]);
  }
}


inline TShape BinaryBroadcastShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  if (!IsBroadcastNeeded_(lhs, rhs)) return lhs;
  std::vector<index_t> lhs_aligned, rhs_aligned, ret;
  GetAlignedShape_(lhs, rhs, &lhs_aligned, &rhs_aligned, &ret);
  // check
  for (int h = 0; h < 2; ++h) {
    std::vector<index_t>inp = h == 0 ? lhs_aligned : rhs_aligned;
    int contdim = 0;
    for (size_t i = 0; i < inp.size(); ++i) {
      if (inp[i] != 1) {
        CHECK_EQ(inp[i], ret[i]) << "broadcast error on index " << i << ". "
                                 << "lhs = " << lhs << "; rhs = " << rhs;
      }
      if (inp[i] == ret[i]) {
        if (i == 0 || inp[i-1] != ret[i-1]) ++contdim;
      }
      CHECK_EQ(contdim, 1) << "broadcast dimensions are not continuous. "
                           << "lhs = " << lhs << "; rhs = " << rhs;
    }
  }
  return TShape(ret.begin(), ret.end());
}

inline void GetBroadcastShape_(const TShape& lhs,
                               const TShape& rhs,
                               TShape* ret_reshaped,
                               int* lhs_broadcast_axis,
                               int* rhs_broadcast_axis) {
  // 0 : lhs, 1 : rhs
  std::vector<std::vector<index_t> > aligned(2);
  std::vector<index_t> ret;
  GetAlignedShape_(lhs, rhs, &aligned[0], &aligned[1], &ret);

  std::vector<std::vector<int> > pos(2);
  int n = static_cast<int>(aligned[0].size());
  for (int h = 0; h < 2; ++h) {
    for (int i = 0; i < n; ++i) {
      if (aligned[h][i] != ret[i]) {
        pos[h].push_back(i); break;
      }
    }
    for (int i = n; i > 1; --i) {
      if (aligned[h][i-1] != ret[i-1]) {
        pos[h].push_back(i); break;
      }
    }
  }
  if (pos[0].size() && pos[1].size()) {
    // broadcast both lhs and rhs
    if (pos[0][0] <= pos[1][0]) {
      CHECK(pos[0][0] == 0 && pos[0][1] == pos[1][0] && pos[1][1] == n)
          << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 0;
      *rhs_broadcast_axis = 1;
    } else {
      CHECK(pos[1][0] == 0 && pos[1][1] == pos[0][0] && pos[0][1] == n)
          << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 1;
      *rhs_broadcast_axis = 0;
    }
    std::vector<index_t> dim(2, 1);
    for (int i = 0; i < pos[0][1]; ++i) dim[0] *= ret[i];
    for (int i = pos[0][1]; i < n; ++i) dim[1] *= ret[i];
    *ret_reshaped = TShape(dim.begin(), dim.end());
  } else {
    // if or not broadcast lhs
    bool b_lhs = !pos[0].empty();
    int pos_new[4] = {0, b_lhs ? pos[0][0] : pos[1][0],
                      b_lhs ? pos[0][1] : pos[1][1], n};
    std::vector<index_t> dim(3, 1);
    for (int i = 0; i < 3; ++i) {
      for (int j = pos_new[i]; j < pos_new[i+1]; ++j) {
        dim[i] *= ret[j];
      }
    }
    *ret_reshaped = TShape(dim.begin(), dim.end());
    if (b_lhs) {
      *lhs_broadcast_axis = 1;
      *rhs_broadcast_axis = -1;
    } else {
      *rhs_broadcast_axis = 1;
      *lhs_broadcast_axis = -1;
    }
  }
}


template<typename xpu, typename OP>
void BinaryBroadcastForward_(const TBlob& lhs,
                             const TBlob& rhs,
                             const EnvArguments& env,
                             TBlob *ret,
                             OpReqType req,
                             RunContext ctx) {
  if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    BinaryForward_<xpu, OP>(lhs, rhs, env, ret, req, ctx);
    return;
  }
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";

  TShape ret_reshaped;
  int lhs_broadcast_axis;
  int rhs_broadcast_axis;
  GetBroadcastShape_(lhs.shape_, rhs.shape_, &ret_reshaped,
                     &lhs_broadcast_axis, &rhs_broadcast_axis);
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      if (lhs_broadcast_axis >= 0) {
        // broadcast lhs
        mshadow::Tensor<xpu, 1, DType> mlhs =
            lhs.get_with_shape<xpu, 1, DType>(mshadow::Shape1(lhs.shape_.Size()), s);
        if (rhs_broadcast_axis >= 0) {
          // broadcast both
          mshadow::Tensor<xpu, 1, DType> mrhs =
              rhs.get_with_shape<xpu, 1, DType>(mshadow::Shape1(rhs.shape_.Size()), s);

          mshadow::Shape<2> ret_mshape = ret_reshaped.get<2>();
          mshadow::Tensor<xpu, 2, DType> out =
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
          mshadow::Shape<3> ret_mshape = ret_reshaped.get<3>();
          mshadow::Tensor<xpu, 3, DType> out =
              ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          mshadow::Tensor<xpu, 3, DType> mrhs =
              rhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          ASSIGN_DISPATCH(out, req,
                          F<OP>(broadcast<1>(mlhs, ret_mshape), mrhs));
        }
      } else {
        mshadow::Tensor<xpu, 1, DType> mrhs =
            rhs.get_with_shape<xpu, 1, DType>(mshadow::Shape1(rhs.shape_.Size()), s);
        if (rhs_broadcast_axis >= 0) {
          // only rhs
          mshadow::Shape<3> ret_mshape = ret_reshaped.get<3>();
          mshadow::Tensor<xpu, 3, DType> out =
              ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          mshadow::Tensor<xpu, 3, DType> mlhs =
              lhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          ASSIGN_DISPATCH(out, req,
                          F<OP>(mlhs, broadcast<1>(mrhs, ret_mshape)));
        } else {
          LOG(FATAL) << "should not reached";
        }
      }
    });
}

template<typename xpu>
void PlusBroadcastBackward_(const OutputGrad& out_grad,
                            const EnvArguments& env,
                            TBlob* lhs_grad,
                            TBlob* rhs_grad,
                            OpReqType req_lhs_grad,
                            OpReqType req_rhs_grad,
                            RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<mshadow_op::identity>(mout_grad));
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<mshadow_op::identity>(mout_grad));
    });
}

MXNET_REGISTER_SIMPLE_OP(_plus_broadcast, XPU)
.set_symbol_op_name("_PlusBroadcast")
.set_function(XPU::kDevMask, BinaryBroadcastForward_<XPU, mshadow::op::plus>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, PlusBroadcastBackward_<XPU>, kInplaceOutLhs)
.describe("Add lhs and rhs with broadcast");


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
