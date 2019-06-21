import tvm
from .. import defop, AllTypes

@defop(name="vadd", target="cpu", dtype=AllTypes)
def defop_vadd(dtype):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]


@defop(name="vmul", target="cpu", dtype=["float32"])
def vmul(dtype):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] * B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]
