# import mxnet as mx
# import numpy as np
# a = mx.np.arange(4*4*4).reshape((4,4,4))

# out = mx.np.array_split(a, axis=2, indices_or_sections=3)

# # print(a)
# # print("\n\n\n")
# # print(out)


# b = np.arange(4*4*4).reshape((4,4,4))

# out2 = np.array_split(b, axis=2, indices_or_sections=3)

# for o1, o2 in zip(out, out2):
#     print(o2 - o1)



import mxnet
import mxnet.gluon.nn as nn
import mxnet.numpy as np
import time


dims = [128, 512, 1024, 4096]
print("shape;axis;time")
for ndim in range (2):
   for dim1 in dims:
     for dim2 in dims:
        shape = (dim1, dim2) if ndim == 0 else (32, dim1, dim2)
        a = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        for axis in range(2 + ndim):
            for section in range(1, 4):
                tic = time.time()
                for i in range(100):
                    out = np.array_split(a, axis=axis, indices_or_sections=section)
                    [o.wait_to_read() for o in out]
                toc = time.time()
                print(f"{shape};{axis};{section};{toc-tic}")