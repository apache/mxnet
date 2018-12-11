/*!
 * Copyright (c) 2016 by Contributors
 * \file q_fully_connected.cc
 * \brief Quantized FC operator
 * \author HPI-DeepLearning
*/
#include "./q_fully_connected-inl.h"
#include "./xnor_cpu.h"


namespace mshadow {
  using namespace mxnet::op::xnor_cpu;

  inline void _QFullyConnectedForward(int m, int n, int k,
                                      const Tensor<cpu, 2, float> &data,
                                      Tensor<cpu, 1, float> &workspace,
                                      BINARY_WORD* binary_weights_col,
                                      Tensor<cpu, 2, float> &out) {
    CHECK_EQ(data.size(1) % BITS_PER_BINARY_WORD, 0) << "input channel number for Q_fully_connected layer is not divisible by "
                                                     << BITS_PER_BINARY_WORD;
    //check matrix dims:
    //  out should have dims (m, k)
    CHECK_EQ((int)out.size(0), m);
    CHECK_EQ((int)out.size(1), k);

    CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * m);
    BINARY_WORD* binary_row = (BINARY_WORD*) workspace.dptr_;

    get_binary_row(data.dptr_, binary_row, m*n);

    out = 0;    

    xnor_gemm(m, k, n/BITS_PER_BINARY_WORD,
              binary_row, n/BITS_PER_BINARY_WORD,
              binary_weights_col, k,
              out.dptr_, k);
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<cpu, 2, float> &data,
                                     Tensor<cpu, 1, float> &workspace,
                                     BINARY_WORD* wmat_binarized,
                                     Tensor<cpu, 2, float> &out) {
    _QFullyConnectedForward(m, n, k, data, workspace, wmat_binarized, out);
  }

  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<cpu, 2, float> &data,
                                     Tensor<cpu, 1, float> &workspace,
                                     const Tensor<cpu, 2, float> &wmat,
                                     Tensor<cpu, 2, float> &out) {
    BINARY_WORD binary_col[n * k/BITS_PER_BINARY_WORD];
    get_binary_col_unrolled(wmat.dptr_, &binary_col[0], n, k);

    _QFullyConnectedForward(m, n, k, data, workspace, binary_col, out);
  }


  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<cpu, 2, DType> &data,
                                     Tensor<cpu, 1, DType> &workspace,
                                     BINARY_WORD* wmat_binarized,
                                     Tensor<cpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }

  template<typename DType>
  inline void QFullyConnectedForward(int m, int n, int k,
                                     const Tensor<cpu, 2, DType> &data,
                                     Tensor<cpu, 1, DType> &workspace,
                                     const Tensor<cpu, 2, DType> &wmat,
                                     Tensor<cpu, 2, DType> &out) {
    CHECK(false) << "only float supported";
  }

}

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(QFullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new QFullyConnectedOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QFullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(QFullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(QFullyConnected, QFullyConnectedProp)
.describe(R"(Apply matrix multiplication to input then add a bias.
It maps the input of shape `(batch_size, input_dim)` to the shape of
`(batch_size, num_hidden)`. Learnable parameters include the weights
of the linear transform and an optional bias vector.)")
.add_argument("data", "NDArray-or-Symbol", "Input data to the Q_FullyConnectedOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(QFullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
