import mxnet.numpy as np
import numpy as _np

a=np.array([1,2,3])
b=2.0
print('dnp (a,b)', np.less(a,b), 'onp (a,b)', _np.less(a.asnumpy(),b))
print('dnp (b,a)', np.less(b,a), 'onp (b,a)', _np.less(b,a.asnumpy()))
print('\n')
print('dnp (a,b)', np.greater_equal(a,b), 'onp (a,b)', _np.greater_equal(a.asnumpy(),b))
print('dnp (b,a)', np.greater_equal(b,a), 'onp (b,a)', _np.greater_equal(b,a.asnumpy()))
print('\n')
print('dnp (a,b)', np.less_equal(a,b), 'onp (a,b)', _np.less_equal(a.asnumpy(),b))
print('dnp (b,a)', np.less_equal(b,a), 'onp (b,a)', _np.less_equal(b,a.asnumpy()))