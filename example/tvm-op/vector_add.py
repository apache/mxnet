import tvm

def defop_vadd():
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    return s, [A, B, C]

# def defop_cuda_vadd():
#     n = tvm.var("n")
#     A = tvm.placeholder((n,), name='A')
#     B = tvm.placeholder((n,), name='B')
#     C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
#     s = tvm.create_schedule(C.op)
#     bx, tx = s[C].split(C.op.axis[0], factor=64)
#     s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
#     s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
#     return s, [A, B, C]

def defop_ifelse():
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
    func_lower = tvm.lower(s, [A, B, C], name="ifelse", simple_mode=True)
    print(func_lower)
    return s, [A, B, C]


def defop_bcast_add():
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A')
    B = tvm.placeholder((n0, n1), name='B')

    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                         name="Ab",
                         elem_offset=0,
                         offset_factor=16,
                         strides=[tvm.var("sa"), 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                         name="Bb",
                         elem_offset=0,
                         offset_factor=16,
                         strides=[tvm.var("sb"), 1])
    s = tvm.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], target='c', name='myadd', binds={A:Ab, B:Bb})
    print(fadd.get_source())
    return s, [(A, Ab), (B, Bb), C]
