import tvm
from .. import defop, AllTypes

@defop(name="vadd", target="cpu", auto_broadcast=True, dtype=AllTypes)
def vadd(dtype):
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A', dtype=dtype)
    B = tvm.placeholder((n0, n1), name='B', dtype=dtype)
    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    s = tvm.create_schedule(C.op)

    return s, [A, B, C]

@defop(name="cuda_vadd", target="cuda", auto_broadcast=True, dtype="float32")
def vadd_gpu(dtype):
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A', dtype=dtype)
    B = tvm.placeholder((n0, n1), name='B', dtype=dtype)
    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    s = tvm.create_schedule(C.op)
    s[C].bind(C.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[C].bind(C.op.axis[1], tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]
