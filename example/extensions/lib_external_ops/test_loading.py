import mxnet as mx
import os

# check if operator exists
if hasattr(mx.nd,'min_ex'):
    raise Exception('Operator already loaded')
else:
    print('Operator not registered yet')

# test loading library
if (os.name=='posix'):
    path = os.path.abspath('build/libexternal_lib.so')
    mx.library.load(path,False)

# execute operator
print(mx.nd.min_ex())
print('Operator executed successfully')
