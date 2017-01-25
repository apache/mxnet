// mexnet.cc
#if defined(__ANDROID__) || defined(__MXNET_JS__)
#define MSHADOW_USE_SSE         0
#endif

#include "src/ndarray/ndarray_function.cc"
#include "src/ndarray/ndarray.cc"

#include "src/engine/engine.cc"
#include "src/engine/naive_engine.cc"
#include "src/engine/profiler.cc"

#include "src/executor/graph_executor.cc"
#include "src/executor/attach_op_execs_pass.cc"
#include "src/executor/attach_op_resource_pass.cc"
#include "src/executor/inplace_addto_detect_pass.cc"

#include "src/nnvm/legacy_json_util.cc"
#include "src/nnvm/legacy_op_util.cc"

#include "src/operator/operator.cc"
#include "src/operator/operator_util.cc"
#include "src/operator/activation.cc"
#include "src/operator/batch_norm.cc"
#include "src/operator/concat.cc"
#include "src/operator/convolution.cc"
#include "src/operator/deconvolution.cc"
#include "src/operator/dropout.cc"
#include "src/operator/fully_connected.cc"
#include "src/operator/leaky_relu.cc"
#include "src/operator/pooling.cc"
#include "src/operator/softmax_activation.cc"
#include "src/operator/softmax_output.cc"
#include "src/operator/tensor/elemwise_binary_broadcast_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_op_basic.cc"
#include "src/operator/tensor/elemwise_binary_scalar_op_basic.cc"
#include "src/operator/tensor/elemwise_unary_op.cc"
#include "src/operator/tensor/matrix_op.cc"

#include "src/storage/storage.cc"

#include "src/resource.cc"
#include "src/initialize.cc"

#include "src/c_api/c_predict_api.cc"
#include "src/c_api/c_api_symbolic.cc"
#include "src/c_api/c_api_ndarray.cc"
#include "src/c_api/c_api_error.cc"
