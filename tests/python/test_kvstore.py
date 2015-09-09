# pylint: skip-file
import mxnet as mx

num_devs = 3
devs = [mx.Context('cpu', i) for i in range(num_devs)]
mx.kvstore.init_devices(devs)

s = (4,4)

# init
a = mx.narray.empty(s,devs[0])
a[:] = 1.0
mx.kvstore.init((3, a))

# push
# B = [mx.narray.empty(s,d) for d in devs]
# for b in B:
#     b[:] = 2.0
#     mx.kvstore.push((3, b))

# pull
C = [mx.narray.empty(s,d) for d in devs]
for c in C:
    mx.kvstore.pull((3, c))
    print c.asnumpy()
mx.kvstore.stop()
