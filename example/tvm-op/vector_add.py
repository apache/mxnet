import tvm

def defop_vadd():
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]

def defop_cuda_vadd():
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]

def defop_bcast_add():
    m = tvm.var("m")
    n = tvm.var("n")
    o = tvm.var("o")

    A = tvm.placeholder((m,), name='A')
    B = tvm.placeholder((n,), name='B')

    @tvm.hybrid.script
    def if_then_else(a, b, oshape):
        c = output_tensor((oshape,), 'float32')
        for i in range(oshape):
            if i % 2 == 0:
                c[i] = a[i]
            else:
                c[i] = b[i]
        return c

    C = if_then_else(A, B, o)
    s = tvm.create_schedule(C.op)
    func_lower = tvm.lower(s, [A, B, C], name="bcast_add", simple_mode=True)
    print(func_lower)
    return s, [A, B, C]
