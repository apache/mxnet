import mxnet as mx

a = mx.narray.create((3000,4000))
b = mx.narray.create((3000,4000))
a.numpy[:] = 10
b.numpy[:] = 11
print(a.numpy)

c = b * a

cc = mx.op.mul(b, a)

print(c.context)
print(cc.numpy)

d = c.copyto(mx.Context('cpu', 0))

print(d.numpy)

with mx.Context('gpu', 0) as ctx:
    # gpu operations
    print mx.current_context()
    print ctx
    a_gpu = a.copyto(ctx)
    b_gpu = b.copyto(ctx)
    c_gpu = b * a

d_cpu = c_gpu.copyto(mx.current_context())
print d_cpu.numpy

