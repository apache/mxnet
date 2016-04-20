/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param) {
  return new ConcatOp<cpu>(param);
}

Operator* ConcatProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.add_argument("data", "Symbol[]", "List of tensors to concatenate")
.add_arguments(ConcatParam::__FIELDS__())
.set_key_var_num_args("num_args")
#if DMLC_USE_CXX11
.describe(R"doc(Perform an feature concat on channel dim (defaut is 1) over all
the inputs.

Examples
--------
>>> import mxnet as mx
>>> data = mx.nd.array(range(6)).reshape((2,1,3))
>>> print "input shape = %s\ndata = \n%s" % (data.shape, data.asnumpy())
input shape = (2L, 1L, 3L)
data =
[[[ 0.  1.  2.]]
 [[ 3.  4.  5.]]]

>>> # concat two variables on different dimensions
>>> a = mx.sym.Variable('a')
>>> b = mx.sym.Variable('b')
>>> for dim in range(3):
...     cat = mx.sym.Concat(a, b, dim=dim)
...     exe = cat.bind(ctx=mx.cpu(), args={'a':data, 'b':data})
...     exe.forward()
...     out = exe.outputs[0]
...     print "concat at dim = %d\nshape = %s\nresults = \n%s" % (dim, out.shape, out.asnumpy())
concat at dim = 0
shape = (4L, 1L, 3L)
results =
[[[ 0.  1.  2.]]
 [[ 3.  4.  5.]]
 [[ 0.  1.  2.]]
 [[ 3.  4.  5.]]]
concat at dim = 1
shape = (2L, 2L, 3L)
results =
[[[ 0.  1.  2.]
  [ 0.  1.  2.]]
 [[ 3.  4.  5.]
  [ 3.  4.  5.]]]
concat at dim = 2
shape = (2L, 1L, 6L)
results =
[[[ 0.  1.  2.  0.  1.  2.]]
 [[ 3.  4.  5.  3.  4.  5.]]]
)doc");
#else
.describe("Perform an feature concat on channel dim (defaut is 1) over all");
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
