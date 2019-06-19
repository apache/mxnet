import tvm
from tvmop.utils import Types, AllTypes

def defop_bcast_add(dtype: AllTypes()):
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A', dtype=dtype)
    B = tvm.placeholder((n0, n1), name='B', dtype=dtype)
    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="broadcast")
    Bb = tvm.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="broadcast")

    s = tvm.create_schedule(C.op)

    fadd = tvm.build(s, [A, B, C], target='c', name='myadd', binds={A:Ab, B:Bb})
    print(fadd.get_source())

    return s, [(A, Ab), (B, Bb), C]
