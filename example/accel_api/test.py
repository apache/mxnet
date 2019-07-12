import mxnet as mx

ctx = mx.context.load_acc('libmyacc.so')
print(ctx)
