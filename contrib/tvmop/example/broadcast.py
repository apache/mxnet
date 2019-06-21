import tvm
from .. import defop, AllTypes

@defop(name="bcast_add", target="cpu", auto_broadcast=True, dtype=AllTypes)
def defop_bcast_add(dtype):
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A', dtype=dtype)
    B = tvm.placeholder((n0, n1), name='B', dtype=dtype)
    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    s = tvm.create_schedule(C.op)

    return s, [A, B, C]
