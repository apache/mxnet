import tvm
from .. import defop, AllTypes

@defop(name="vadd", target="cpu", dtype=AllTypes)
def vadd(dtype):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]

@defop(name="cuda_vadd", target="gpu", dtype="float32")
def vadd_gpu(dtype):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]

@defop(name="vmul", target="cpu", dtype=["float32"])
def vmul(dtype):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    B = tvm.placeholder((n,), name='B', dtype=dtype)
    C = tvm.compute(A.shape, lambda i: A[i] * B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]
