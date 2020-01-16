import os
os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

import mxnet as mx
import time
from mxnet.ndarray import np


def benchmark(func, *args, **kwargs):
    func(*args, **kwargs)
    repeat = 10000
    start = time.time()
    for i in range(repeat):
        x = func(*args, **kwargs)
    end = time.time()
    return (end - start) / repeat


# print("tvm ffi...")
# a = np.zeros1((3, 4), ctx="cpu(0)", dtype='float64')
# print(a)
# print("legacy ffi...")
# a = np.zeros((3, 4), ctx=mx.cpu())
# print(a)

print("########################{:^32}########################".format("zeros benchmark"))

overhead = benchmark(np.zeros, (3, 4))
print("{:>16}: {:>10}".format("legacy", overhead))

overhead = benchmark(np.zeros1, (3, 4))
print("{:>16}: {:>10}".format("tvm", overhead))

overhead = benchmark(np.zeros, (3, 4), dtype='float64')
print("{:>16}: {:>10}".format("legacy dtype", overhead))

overhead = benchmark(np.zeros1, (3, 4), dtype='float64')
print("{:>16}: {:>10}".format("tvm dtype", overhead))

overhead = benchmark(np.zeros, (3, 4), dtype='float64', ctx='cpu(0)')
print("{:>16}: {:>10}".format("legacy dtype ctx", overhead))

overhead = benchmark(np.zeros1, (3, 4), dtype='float64', ctx='cpu(0)')
print("{:>16}: {:>10}".format("tvm dtype ctx", overhead))

####################### tensordot #########################

# print("####  tensordot verification ####")
# print("scalar axis...")
# a = np.ones((2, 2))
# b = np.ones((2, 2))
# c = np.tensordot1(a, b)
# print(c)

# print("tuple axes...")
# a = np.ones((2, 3))
# b = np.ones((3, 2))
# c = np.tensordot1(a, b, ((1, 0), (0, 1)))
# print(c)


print("########################{:^32}########################".format("tensordot benchmark"))
a = np.ones((2, 2))
b = np.ones((2, 2))

overhead = benchmark(np.tensordot, a, b, ((1, 0), (0, 1)))
print("{:>16}: {:>10}".format("legacy", overhead))

overhead = benchmark(np.tensordot1, a, b, ((1, 0), (0, 1)))
print("{:>16}: {:>10}".format("tvm", overhead))
