import tvm
from tvmop.utils import AllTypes

def defop_vadd(dtype: AllTypes()):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]