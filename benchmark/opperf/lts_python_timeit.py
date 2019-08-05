import mxnet as mx
import time
import numpy as np

# sizes = [10, 50, 100,200,500]
# iters = [10000,1000,500,200,20]
val = [(1000,10)]#,(1000,100),(1000,100,100),(1000,100,100,100)]
times = []
for iterations in range(75):
    data = []
    # s = sizes[size]
    # print(s)
    # for i in range(iters[size]):
    for i in range(len(val)):
        x = mx.nd.ones(val[i])
        mx.nd.waitall()
        start = time.time()
        y = mx.nd.transpose(x)
        mx.nd.waitall()
        data.append((time.time() - start)*1000)
        #print(data[-1])                                                                                                                                                                            
    times.append(data)

print('mxnet version: %s' % mx.__version__)
for s in range(len(val)):
    print('--------------------')
    print('size: %s' % str(val[s]))
    print('p50: %4.2f ms' % np.percentile(times[s],50))
    print('p90: %4.2f ms' % np.percentile(times[s],90))
    print('p99: %4.2f ms' % np.percentile(times[s],99))