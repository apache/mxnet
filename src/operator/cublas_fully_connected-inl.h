/*!
 * \file cublas_fully_connected-inl.h
 * \brief fully connect operator and symbol with direct use of cuBLAS
 */
#ifndef MXNET_OPERATOR_CUBLAS_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_CUBLAS_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "./fully_connected-inl.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDA && CUDA_VERSION >= 8000

/**
 * \brief This is the implementation of fully connected operator for cuBLAS.
 */
template<typename DType>
class CuBLASFullyConnectedOp : public Operator {
 public:
  explicit CuBLASFullyConnectedOp(FullyConnectedParam p,
                                  const Context& ctx) :
      param_(p),
      algo_(CUBLAS_GEMM_DFALT) {
    using namespace mshadow;
    cublas_algo_precision_ = DataType<DType>::kFlag;

    // Silently fail-over to pseudo-fp16 on Maxwell or earlier GPUs, or
    // if the library has been built with MSHADOW_USE_PASCAL == 0
    if (DataType<DType>::kFlag == kFloat16 &&
        (!SupportsFloat16Compute(ctx.dev_id) || MSHADOW_USE_PASCAL == 0))
      cublas_algo_precision_ = kFloat32;

    #if CUDA_VERSION >= 9000
      // TensorCore algos are allowed on fp16-I/O gemms if permitted by the global policy.
      if (DataType<DType>::kFlag == kFloat16 && GetEnvAllowTensorCore())
        algo_ = CUBLAS_GEMM_DFALT_TENSOR_OP;
    #endif
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using mshadow::Shape2;
    using mshadow::expr::repmat;

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";

    // Note: data is in row-major order.  Assume a data input with batch size 'B'
    // and a number of input features 'IF', and an output with batch size 'B'
    // and output features 'OF'.  Then the shape of this operation's I/O's are:
    //
    // Data input: B x IF
    // Weights input: OF x IF
    // Output: B x OF
    //
    // Now consider an operation dot(A,B) -> C, where
    //
    // A has shape (m,k)
    // B has shape (k,n)
    // C has shape (m,n)
    //
    // Matching C's shape to our Output, we have m=B and n=OF. What remains is k=IF.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( B x IF, IF x OF ) -> B x OF
    // dot ( Data, Weights.transpose() ) -> Output
    //

    int m = 0, k = 0, n = 0;
    GetForwardMKN(in_data[fullc::kData], in_data[fullc::kWeight], out_data[fullc::kOut],
                  &m, &k, &n);

    mshadow::Tensor<gpu, 2, DType> data =
        in_data[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);
    mshadow::Tensor<gpu, 2, DType> wmat =
        in_data[fullc::kWeight].get<gpu, 2, DType>(s);
    mshadow::Tensor<gpu, 2, DType> out =
        out_data[fullc::kOut].get_with_shape<gpu, 2, DType>(Shape2(m, n), s);

    // Performs fully_connected-inl.h line:     out = dot(data, wmat.T());

    ExpandedDot(s, false, true, kWriteTo,
                data, wmat, out,
                cublas_algo_precision_,
                algo_);

    if (!param_.no_bias) {
      mshadow::Tensor<gpu, 1, DType> bias = in_data[fullc::kBias].get<gpu, 1, DType>(s);
      out += repmat(bias, data.size(0));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using mshadow::Shape2;
    using mshadow::expr::sum_rows;

    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";

    // For back-prop to weights:
    //
    // Data input: B x IF
    // "Output" gradient (an input): B x OF
    // Weights gradient (an output): OF x IF
    //
    // Matching C's shape to the Weights grad, we have m=OF and n=IF. What remains is k=B.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( OF x B, B x IF ) -> OF x IF
    // dot ( OutGradient.transpose(), Data ) -> Weights

    int m = 0, k = 0, n = 0;
    GetForwardMKN(in_data[fullc::kData], in_data[fullc::kWeight], out_grad[fullc::kOut],
                  &m, &k, &n);

    mshadow::Tensor<gpu, 2, DType> data =
        in_data[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);
    mshadow::Tensor<gpu, 2, DType> wmat =
        in_data[fullc::kWeight].get<gpu, 2, DType>(s);
    mshadow::Tensor<gpu, 2, DType> grad =
        out_grad[fullc::kOut].get_with_shape<gpu, 2, DType>(Shape2(m, n), s);

    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    mshadow::Tensor<gpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<gpu, 2, DType>(s);

    // Performs fully_connected-inl.h: Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));

    ExpandedDot(s, true, false, req[fullc::kWeight],
                grad, data, gwmat,
                cublas_algo_precision_,
                algo_);

    // gradient of bias

    if (!param_.no_bias) {
      mshadow::Tensor<gpu, 1, DType> gbias = in_grad[fullc::kBias].get<gpu, 1, DType>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }

    // gradient of data

    // "Output" gradient (an input): B x OF
    // Weights : OF x IF
    // Data gradient (an output): B x IF
    //
    // Matching C's shape to the Data gradient output, we have m=B and n=IF. What remains is k=OF.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( B x OF, OF x IF ) -> B x IF
    // dot ( OutGradient, Weights ) -> Data Gradient

    mshadow::Tensor<gpu, 2, DType> gdata =
        in_grad[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);

    // Performs fully_connected-inl.h line: Assign(gdata, req[fullc::kData], dot(grad, wmat));
    ExpandedDot(s, false, false, req[fullc::kData],
                grad, wmat, gdata,
                cublas_algo_precision_,
                algo_);
  }

  /*!
   * \brief Returns whether the cublas library supports the fully-connected
   * operation described by `param`.
   */
  static bool Supports(FullyConnectedParam param,
                                  const Context& ctx) {
    // This operator uses cublasGemmEx(), which is only supported on cuda
    // compute architectures >= 5.0.  The FullyConnectedParam argument
    // is currently not considered, although this may change in the future.

    return ComputeCapabilityMajor(ctx.dev_id) >= 5;
  }

 private:

  // Returns the matrix multiply parameters m, k, and n of the forward inference operation:
  //
  // (m x k) matrix-multiply (k x n) -> (m x n)
  //
  // Similar to the code in fully_connected-inl.h, the TBlob shapes are effectively flattened if
  // they are not 2D.
  static void GetForwardMKN(const TBlob &data,
                            const TBlob &weights,
                            const TBlob &output,
                            int *m_ptr, int *k_ptr, int *n_ptr) {
    const TShape& ishape = data.shape_;
    const TShape& wshape = weights.shape_;
    const TShape& oshape = output.shape_;

    int m = ishape[0];
    int k = ishape.ProdShape(1, ishape.ndim());
    int n = wshape[0];  // Weight matrix is transposed in forward inference
    // Check consistency of input and output shapes
    int k2 = wshape.ProdShape(1, wshape.ndim());
    int m2 = oshape[0];
    int n2 = oshape.ProdShape(1, oshape.ndim());

    CHECK_EQ(m, m2) << "In FullyConnected GEMM, 'data' matrix rows (" << m << ")"
                    << " must match output matrix rows (" << m2 << ")";
    CHECK_EQ(k, k2) << "In FullyConnected GEMM, 'data' matrix cols (" << k << ")"
                    << " must match 'weight' matrix rows (" << k2 << ")";
    CHECK_EQ(n, n2) << "In FullyConnected GEMM, 'data' matrix cols (" << n << ")"
                    << " must match output matrix cols (" << n2 << ")";
    *m_ptr = m;
    *k_ptr = k;
    *n_ptr = n;
  }

  // The following matrix multiply facility has additional features over mshadow's
  // 'dot' operator.  It allows specification of the precision of the accumulation
  // via the `algo_precision` and also the algo number passed to gemmEx (which
  // dictates whether TensorCore is allowed).

  // Perform the matrix multiplication (a.k.a. 'dot') on the supplied Tensors,
  // converting between the row-major specification of this routine's interface/Tensors
  // and the column-major interface of the underlying cuBLAS gemm API.
  static void ExpandedDot(mshadow::Stream<gpu> *s, bool transposeA, bool transposeB,
                          OpReqType output_request,
                          const mshadow::Tensor<gpu, 2, DType> &A,
                          const mshadow::Tensor<gpu, 2, DType> &B,
                          const mshadow::Tensor<gpu, 2, DType> &C,
                          int algo_precision,
                          cublasGemmAlgo_t algo) {
    int m = transposeA ? A.shape_[1] : A.shape_[0];
    int k = transposeA ? A.shape_[0] : A.shape_[1];
    int n = transposeB ? B.shape_[0] : B.shape_[1];
    // Check consistency of input and output shapes by grabbing n, m and k a different way.
    int k2 = transposeB ? B.shape_[1] : B.shape_[0];
    int m2 = C.shape_[0];
    int n2 = C.shape_[1];

    CHECK_EQ(m, m2) << "In FullyConnected GEMM, 'data' matrix rows (" << m << ")"
                    << " must match output matrix rows (" << m2 << ")";
    CHECK_EQ(k, k2) << "In FullyConnected GEMM, 'data' matrix cols (" << k << ")"
                    << " must match 'weight' matrix rows (" << k2 << ")";
    CHECK_EQ(n, n2) << "In FullyConnected GEMM, 'data' matrix cols (" << n << ")"
                    << " must match output matrix cols (" << n2 << ")";

    // We now juggle the arguments of the matrix multiply to account for the
    // fact that the data as generated by mxnet is in row-major order, yet the
    // BLAS gemm kernel assumes they are in column-major order.
    //
    // Let .T() represent the transpose operation below.
    //
    // If A matrix-multiply B -> C, then B.T() matrix-multiply A.T() -> C.T()
    //
    // The ramifications of this are that in order for the gemm kernel to generate
    // the correct result we need to do 2 things:
    //
    // 1. swap the input ordering (so matrix B is the first operand)
    // 2. swap the dimensions of the matrices.
    //
    // The effect of these two steps effectively swaps n and m, leaving k the same.
    // Keep in mind that any transposition that needs to be done moves with the
    // input, so the transpose arguments are swapped as well.
    // In order to operate on submatrices in a "row-major world", one needs the
    // row-stride, which is the second dimension of the two for each matrix.

    int lda = B.shape_[1];
    int ldb = A.shape_[1];
    int ldc = C.shape_[1];

    CublasGemm(s, transposeB, transposeA, n, m, k,
               output_request,
               B,
               A,
               C,
               lda, ldb, ldc,
               algo_precision,
               algo);
  }

  // A wrapper for the full-featured matrix multiply kernel cublasGemmEx() available
  // since CUDA 8 that permits data-type, compute-type and algorithm selection.
  static void CublasGemm(mshadow::Stream<gpu> *s, bool transposeA, bool transposeB,
                         int m, int n, int k,
                         OpReqType output_request,
                         const mshadow::Tensor<gpu, 2, DType> &A,
                         const mshadow::Tensor<gpu, 2, DType> &B,
                         const mshadow::Tensor<gpu, 2, DType> &C,
                         int lda, int ldb, int ldc,
                         int algo_precision,
                         cublasGemmAlgo_t algo) {
    using namespace mxnet::common::cuda;

    // Compute precision specifies the precision of the gemm math,
    // assigned arbitrarily and then set as specified.
    cudaDataType_t compute_precision = CUDA_R_32F;
    MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
      compute_precision = mshadow::DataType<CType>::kCudaFlag;
    })
    cudaDataType_t data_precision = mshadow::DataType<DType>::kCudaFlag;
    const void *alpha = one(algo_precision);
    const void *beta = zero(algo_precision);

    switch (output_request) {
      case kNullOp:
        break;
      case kAddTo:
        // Change beta to point to 1 (the alpha value) rather than 0 to achieve summation.
        beta = alpha;
      case kWriteTo:
      case kWriteInplace: {
          auto blas_handle = mshadow::Stream<gpu>::GetBlasHandle(s);

          // The cuBLAS algo number reflects the "uses TensorCore" property.
          // Ensure the cuBLAS handle's math mode reflects the mode needed by the algo.
          #if CUDA_VERSION >= 9000
            auto handle_math_mode = CUBLAS_DEFAULT_MATH;
            CUBLAS_CALL(cublasGetMathMode(blas_handle, &handle_math_mode));
            // The math mode of the handle needs to be set in sync with the math mode that
            // is baked into the algo number.  Algo numbers at or above CUBLAS_GEMM_DFALT_TENSOR
            // currently have names that include "TENSOR_OP".
            auto desired_math_mode = (algo >= CUBLAS_GEMM_DFALT_TENSOR_OP) ? CUBLAS_TENSOR_OP_MATH
                                                                           : CUBLAS_DEFAULT_MATH;
            if (handle_math_mode != desired_math_mode)
              CUBLAS_CALL(cublasSetMathMode(blas_handle, desired_math_mode));
          #endif

          auto err = CUBLAS_STATUS_SUCCESS;
          err = cublasGemmEx(blas_handle,
                             CublasTransposeOp(transposeA), CublasTransposeOp(transposeB),
                             m, n, k,
                             alpha,
                             A.dptr_, data_precision, lda,  // A operand
                             B.dptr_, data_precision, ldb,  // B operand
                             beta,
                             C.dptr_, data_precision, ldc,  // C operand
                             compute_precision,
                             algo);
          CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas gemmEx fail.";

          #if CUDA_VERSION >= 9000
            // Revert to preexisting math mode.
            if (handle_math_mode != desired_math_mode)
              CUBLAS_CALL(cublasSetMathMode(blas_handle, handle_math_mode));
          #endif
        }
        break;
      default:
        LOG(FATAL) << "not reached";
    }
  }

  FullyConnectedParam param_;
  // The algo number to be passed to the cublasGemmEx calls.
  cublasGemmAlgo_t algo_;
  int cublas_algo_precision_;
};  // class CuBLASFullyConnectedOp
#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUBLAS_FULLY_CONNECTED_INL_H_
