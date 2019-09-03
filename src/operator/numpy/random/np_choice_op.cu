#include "./np_choice_op.h"
#include <thrust/swap.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace mxnet {
namespace op {

template<>
void _swap<gpu>(int64_t& a, int64_t& b) {
    thrust::swap(a, b);
}

template<>
void _sort<gpu>(float* key, int64_t* data, index_t length) {
    thrust::device_ptr<float> dev_key(key);
    thrust::device_ptr<int64_t> dev_data(data);
    thrust::sort_by_key(dev_key, dev_key + length, dev_data, thrust::greater<float>());
}

NNVM_REGISTER_OP(_npi_choice)
.set_attr<FCompute>("FCompute<gpu>", NumpyChoiceForward<gpu>);

}  // namespace op 
}  // namespace mxnet