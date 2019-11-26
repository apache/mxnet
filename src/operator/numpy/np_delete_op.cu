/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_delete_op.cu
 * \brief GPU Implementation of numpy delete operations
 */

#include "./np_delete_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_delete)
.set_attr<FComputeEx>("FComputeEx<gpu>", NumpyDeleteCompute<gpu>);

}
}