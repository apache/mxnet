import os
os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

import mxnet as mx
import time
from mxnet.ndarray import np

print("tvm ffi...")
a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
print(a)
print("legacy ffi...")
a = np.zeros((3, 4), ctx=mx.cpu())
print(a)

print("tvm ffi dummy...")
repeat = 10000
a = np.zeros0((3, 4), ctx="cpu(0)", dtype='float64')
start = time.time()
for i in range(repeat):
    a = np.zeros0((3, 4), ctx="cpu(0)", dtype='float64')
end = time.time()
print("time = {}".format((end - start) / repeat))

print("legacy ffi...")
repeat = 10000
a = np.zeros((3, 4))
start = time.time()
for i in range(repeat):
    a = np.zeros((3, 4))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm ffi...")
repeat = 10000
a = np.zeros1((3, 4))
start = time.time()
for i in range(repeat):
    a = np.zeros1((3, 4))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("legacy ffi dtype...")
repeat = 10000
a = np.zeros((3, 4), dtype='float64')
start = time.time()
for i in range(repeat):
    a = np.zeros((3, 4), dtype='float64')
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm ffi dtype...")
repeat = 10000
a = np.zeros1((3, 4), dtype='float64')
start = time.time()
for i in range(repeat):
    a = np.zeros1((3, 4), dtype='float64')
end = time.time()
print("time = {}".format((end - start) / repeat))

print("legacy ffi ctx dtype...")
repeat = 10000
a = np.zeros((3, 4), ctx="cpu(0)", dtype='float64')
start = time.time()
for i in range(repeat):
    a = np.zeros((3, 4), ctx="cpu(0)", dtype='float64')
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm ffi ctx dtype...")
repeat = 10000
a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
start = time.time()
for i in range(repeat):
    a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
end = time.time()
print("time = {}".format((end - start) / repeat))

####################### tensordot #########################

print("####  tensordot verification ####")
print("scalar axis...")
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot1(a, b)
print(c)

print("tuple axes...")
a = np.ones((2, 3))
b = np.ones((3, 2))
c = np.tensordot1(a, b, ((1, 0), (0, 1)))
print(c)


print("####  tensordot benchmark ####")
print("legacy ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot(a, b, ((1, 0), (0, 1)))
start = time.time()
for i in range(repeat):
    c = np.tensordot(a, b, ((1, 0), (0, 1)))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot1(a, b, ((1, 0), (0, 1)))
start = time.time()
for i in range(repeat):
    c = np.tensordot1(a, b, ((1, 0), (0, 1)))
end = time.time()
print("time = {}".format((end - start) / repeat))

print("tvm dummy ffi...")
repeat = 10000
a = np.ones((2, 2))
b = np.ones((2, 2))
c = np.tensordot0(a, b, ((1, 0), (0, 1)))
start = time.time()
for i in range(repeat):
    c = np.tensordot0(a, b, ((1, 0), (0, 1)))
end = time.time()
print("time = {}".format((end - start) / repeat))
