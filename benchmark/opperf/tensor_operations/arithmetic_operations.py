import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test

"""Performance benchmark tests for MXNet NDArray Arithmetic Operations
1. Add
2. Sub
3. Mul
4. Div
5. Mod
6. Pow
7. Neg
8. iadd (In place Add with +=)
9. isub (In place Sub with -=)
10. imul (In place Mul with *=)
11. idiv (In place Div with /=)
12. imod (In place Mod with %=)

TODO:
1. As part of default tests, add broadcast operations for all below benchmarks. Ex: 1024 * 1024 OP 1024 * 1
2. Logging - Info, Error and Debug
"""

run_performance_test(nd.add, run_backward=True, dtype='float32', ctx=mx.cpu(),
                     inputs={"lhs": (1024, 1024),
                             "rhs": (1024, 1024)},
                     warmup=10, runs=50)
