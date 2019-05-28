import mxnet as mx
import tvm

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)

tgt = "llvm"
tgt_host = None
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

from tvm.contrib import cc
from tvm.contrib import util
temp = "/home/ubuntu/tvm-compiler/build"
fadd.save(temp + "/myadd.o")
cc.create_shared(temp + "/myadd.so", [temp + "/myadd.o"])
# if tgt == "cuda":
#     fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
# if tgt.startswith('opencl'):
#     fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
# cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
# print(temp.listdir())


a = mx.nd.array([1, 2, 3, 4, 5], ctx=mx.cpu(0))
b = mx.nd.array([5, 4, 3, 2, 1], ctx=mx.cpu(0))
c = mx.nd.tvm_vector_add(a, b)
print("a =", a.asnumpy())
print("b =", b.asnumpy())
print("a + b =", c.asnumpy())