// mexnet.cc

#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS 	0
#else
#define MSHADOW_USE_CBLAS 	1
#endif
#endif

#define MSHADOW_USE_CUDA 	0
#define MSHADOW_USE_MKL 	0
#define MSHADOW_RABIT_PS 	0
#define MSHADOW_DIST_PS 	0

#if defined(__ANDROID__) || defined(__MXNET_JS__)
#define MSHADOW_USE_SSE         0
#endif

#define MXNET_USE_OPENCV 	0
#define MXNET_PREDICT_ONLY 	1
#define DISABLE_OPENMP 1

//  NDArray
#include "src/ndarray/ndarray_function.cc"
#include "src/ndarray/ndarray.cc"

// Engine
#include "src/engine/engine.cc"
#include "src/engine/naive_engine.cc"

// Executor
#include "src/executor/graph_executor.cc"
#include "src/executor/attach_op_execs_pass.cc"
#include "src/executor/attach_op_resource_pass.cc"
#include "src/executor/inplace_addto_detect_pass.cc"

// Operators
#include "src/operator/operator.cc"
#include "src/nnvm/legacy_json_util.cc"
#include "src/nnvm/legacy_op_util.cc"

//  Tensor Op
#include "src/operator/tensor/sample_op.cc"
#include "src/operator/tensor/matrix_op.cc"
#include "src/operator/tensor/init_op.cc"
#include "src/operator/tensor/indexing_op.cc"
#include "src/operator/tensor/elemwise_unary_op.cc"
#include "src/operator/tensor/elemwise_sum.cc"
#include "src/operator/tensor/elemwise_binary_scalar_op_extended.cc"
#include "src/operator/tensor/elemwise_binary_scalar_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_extended.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_basic.cc"
#include "src/operator/tensor/broadcast_reduce_op_value.cc"
#include "src/operator/tensor/broadcast_reduce_op_index.cc"

//  Common Op
#include "src/operator/activation.cc"
#include "src/operator/batch_norm.cc"
#include "src/operator/concat.cc"
#include "src/operator/convolution.cc"
#include "src/operator/deconvolution.cc"
#include "src/operator/dropout.cc"
#include "src/operator/fully_connected.cc"
#include "src/operator/leaky_relu.cc"
#include "src/operator/make_loss.cc"
#include "src/operator/pooling.cc"
#include "src/operator/regression_output.cc"
#include "src/operator/softmax_output.cc"

// Storages
#include "src/storage/storage.cc"
#include "src/resource.cc"

//  C_API
#include "src/c_api/c_predict_api.cc"
#include "src/c_api/c_api_error.cc"

//  MKLDNN
#if MXNET_USE_MKL2017==1
#include "mkl_cppwrapper.cc"
#include "mkl_memory.cc"
#endif