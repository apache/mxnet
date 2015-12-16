import sys
sys.path.insert(0, '../unittest')
from test_operator import *

if __name__ == '__main__':
	test_softmax_with_shape((3,4), mx.gpu())
    test_multi_softmax_with_shape((3,4,5), mx.gpu())