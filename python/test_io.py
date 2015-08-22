#pylint: skip-file
import mxnet as mx

dataiter = mx.io.DataIter()
#a.createfromcfg('/home/tianjun/mxnet/mxnet/MNIST.conf')
dataiter.createbyname('mnist')
dataiter.setparam('path_img', "/home/tianjun/data/mnist/train-images-idx3-ubyte")
dataiter.setparam('path_label', "/home/tianjun/data/mnist/train-labels-idx1-ubyte")
dataiter.setparam('shuffle', '1')
dataiter.setparam('seed_data', '2')
dataiter.setparam('batch_size', '100')

dataiter.init()

dataiter.beforefirst()

for i in range(100):
    dataiter.next()
    info = "Batch %d" % (i)
    print info
    label = dataiter.getdata()
    print label.numpy
